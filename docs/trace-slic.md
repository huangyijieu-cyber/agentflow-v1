# TRACE 状态信用与分类型 SLIC 配对

版本命名、差异、参数对照和代码快照统一维护于桌面文档 `webwalker-trace版本记录.md`。**v3a＝原 v2a1，v3b＝原 v2a2**。最新实验 **v3d** 沿用 v3c 的 Replay 容量 512、采样 ×4 和严格 V 匹配，主要设置为 `k_pairs_per_group=None`（YAML 为 `null`），允许同题同类型工具跨 Turn 配对并取消组内数量上限。QA 信用沿用 v3b 的动态范围回填。当前 `alfworld` 分支在 `41f0552` 恢复 v2a 评分后，按用户新指令应用排序轮转；None 跨 Turn 设置保持不变。QA 历史实验定义及 `websearch` 分支保持原样。

**v3c、v3d 均开启严格价值匹配**：`outcome_match_value=true`、`match_epsilon=0.05`，analysis／final 也必须满足 `abs(V_i - V_j) <= 0.05`，并保留可信终局成功／失败标签。此处记录实验实际设置，基础配置默认值仍为 `false`。

QA 的状态、终局和回填语义见下文；ALFWorld 的真实后状态与统一环境动作信用见 [ALFWorld 接入说明](alfworld-trace-goals.md)。当前分支训练配置默认 `k_pairs_per_group=null`，允许同题跨 Turn 且无组内 Pair 上限；显式设为正整数时，工具/环境动作仅允许同题同原始 Turn id 配对。底层库 `TraceConfig()` 未传配置时仍默认 2。

本实现按决策类型选择偏好标签：analysis 和实际最终回答使用可验证的最终任务结果，工具 Turn 使用 K-step GT 信用及可选的终局优势回填。配对质量优先；没有合格正负两侧时跳过，不为凑数量放宽条件。SLIC Loss 和原 GRPO 训练公式保持原样；v3c／v3d 的 Replay 调整记录为用户实验设置。

算法待办已完成实现与 CPU 集成回归，对应提交 `6eb6215` 已推送到 `trace` 分支。其后的指标精简也纳入 `trace` 分支，具体版本见 Git 提交记录；GPU/NPU 端训练效果尚未验证。版本状态见单独维护的 `webwalker-trace版本记录.md`。

## 状态与 GT 评分

从 `v3a` 起，QA 各类型 Turn 的 `trace_state_context` 统一使用问题与动作执行前的工具历史快照，不因切换到 final 而增删分析文本或改变评分模板。快照沿用生成时读取的有界工具历史，不扩大记忆窗口；实际生成 Prompt 保持原样。评分时统一附加：

```text
Answer the question directly and concisely.
```

使用冻结参考模型、温度 1，对固定 GT 做 teacher forcing，只平均 GT Token 的 LogP：

$$
\bar\ell_{g,t}=\frac{1}{|y^*|}\sum_j\log\pi_{\mathrm{ref}}(y_j^*\mid S_{g,t},y_{<j}^*),
\qquad
V_{g,t}=\log\frac{-\bar\ell_{g,0}+\epsilon}{-\bar\ell_{g,t}+\epsilon}.
$$

GT 仅进入独立评分请求，不写入 Rollout 生成上下文；多个别名固定选择第一个非空答案。相同 token 化的状态和 GT 在评分批次内去重。缺失 GT、GT 超长或评分上下文超长时跳过评分，不截断来制造信用。

工具动作的基本增益仍来自同一 Rollout 中真实相邻 Turn 的动作前状态：

$$
\delta_t=V_{t+1}-V_t
=\log\frac{-\bar\ell_t+\epsilon}{-\bar\ell_{t+1}+\epsilon}.
$$

原始 Turn 编号存在缺口、状态缺失或评分非有限时，不跨越缺口拼接。最后可见状态本身并不等于已验证终局；最终回答没有自然 GT 后状态，也不虚构 `delta_final`。

## 按类型选择标签

| 决策 | 标签来源 | GT 价值／信用是否必需 |
| --- | --- | --- |
| `analysis` | 所属 Rollout 的可验证最终成功／失败 | 默认不要求；不接收工具 K-step 信用或回填 |
| 实际用于最终验证的回答（`final_answer`、`direct` 或 `base`） | 可验证最终成功／失败 | 默认不要求自然后继或 GT 信用 |
| 工具规划 Turn | K-step 局部信用加符合条件的终局优势回填 | 要求有效状态价值和有效窗口信用 |

最终结果必须可信。verifier 调用／解析失败、外部中断或没有实际完成的最终回答，不应伪装成模型失败；对应结果不能用于奖励 Pair 或终局回填。优先采用明确的 verifier 成功布尔标记；数值奖励的兼容判别由 `outcome_success_threshold` 控制，不能把正的格式分一概当作成功。二值 `0/1` 中，失败的 `0` 是 Rejected 类，不要求存在负数奖励。

## K=3 与实际 final 边界

K 表示包含当前实际 Turn 在内的总窗口，默认 K=3。实际 final 回答本身占一个窗口位置，analysis 不接收工具信用。只对窗口内确实存在的工具转移增益求折扣平均：

$$
c_t^{(K)}=\frac{\sum_{u\in\mathcal W_t^{\mathrm{tool}}}\gamma^{u-t}\delta_u}
{\sum_{u\in\mathcal W_t^{\mathrm{tool}}}\gamma^{u-t}},
\quad
\mathcal W_t=[t,\min(t+K-1,\text{末可见 Turn})].
$$

final 的执行结果单独用于终局项，不作为 GT delta 混入局部平均。QA 按当前 batch 同题（同 `uid`）所有有效工具 Turn 的回填前局部信用，以及所有可信 Rollout 的终局优势，分别统计正负范围：

$$
C_x^+=\max_{g,t}\max(c_{g,t}^{(K)},0),\quad
C_x^-=\max_{g,t}\max(-c_{g,t}^{(K)},0),\qquad
A_x^+=\max_g\max(A_g^{\mathrm{out}},0),\quad
A_x^-=\max_g\max(-A_g^{\mathrm{out}},0).
$$

保持零点和正负号，动态映射终局优势，不再乘固定回填系数：

$$
\widehat A_g=
\begin{cases}
(A_g^{\mathrm{out}}/A_x^+)C_x^+,& A_g^{\mathrm{out}}>0,\\
(A_g^{\mathrm{out}}/A_x^-)C_x^-,& A_g^{\mathrm{out}}<0,\\
0,& A_g^{\mathrm{out}}=0.
\end{cases}
$$

没有某个方向的有效工具信用时，该方向映射为零；空工具范围或全零范围不产生回填。工具范围在信用阈值、价值匹配和 Pair 数量筛选前统计，包括未触及 final 或终局不可信 Rollout 中的有效工具窗口；排除外部故障、阻断窗口、无效／重复／被丢弃／无 Response 的动作及 final 后输出。分母仅在对应优势非零时使用，无新增 epsilon 或幅度超参数。范围对齐不保证每个弱局部信用的正负号保持不变。

窗口触及实际已完成且验证可信的 final 时，对工具 Turn 回填：

$$
\widetilde c_t=c_t^{(K)}+
\mathbf1[\mathcal W_t\text{触及可信 final}]
\gamma^{f-t}\widehat A_g,
$$

其中 f 为真实 final 的原始 Turn 编号，折扣距离与总窗口的计数一致。用于 TRACE 回填的终局优势仅在有可信终局结果的同题 Rollout 中计算，并按 Rollout 去重。复用的是原 GRPO 的中心化、是否除以标准差及单例组特例等计算口径；可信 Rollout 的统计人口可能与实际 GRPO batch 不同，因此两条路径的优势数值不保证相同。它不是直接把原始奖励 R 加到信用上，也不会回写或修改原 GRPO 优势。局部信用、回填项及回填后信用分别保留。

例如 `analysis → tool A → tool B → tool C → final`，K=3：

| 工具 Turn | 窗口 | 是否回填 |
| --- | --- | --- |
| A | A、B、C | 否 |
| B | B、C、final | 是，`gamma**2 * A_hat` |
| C | C、final | 是，`gamma * A_hat` |

因此本项目的 K=3 最多向 final 前两个工具 Turn 回填。这是明确包含 final 的总 Turn 窗口约定；与 TRACE 论文按工具转移计 K 的约定不同，不能同时声称回填最后三个工具 Turn。

## 外部故障与窗口隔离

网络请求超时、连接／DNS 失败、服务不可用、网关或外部限流等最终未恢复故障，应标为外部故障；对应 Turn 不能进入 Chosen 或 Rejected。优先识别结构化执行状态及已知适配器错误格式，不扫描网页正文中的普通 error 文本。

正常无搜索结果不是网络故障；模型非法工具名／参数也不能自动归为外部故障。内部重试后成功取得结果，按最终成功状态处理。

包含外部故障转移的工具 K-step 窗口信用不可用，不参与 Pair。不把故障 delta 填成 0，也不删除故障后连接原本不相邻的状态。故障导致终局无法可靠验证时，analysis／final 奖励 Pair 与工具终局回填也不可使用该结果。

## 配对、排序与 Loss

候选必须同 `uid`、同决策类型和同标签来源。`k_pairs_per_group` 为正整数时，QA 工具和 ALFWorld 环境动作还必须具有相同原始 Turn id，允许来自不同 Rollout，不允许跨 Turn 配对；若设为 2，则每题每 Turn 最多 2 对，多个 Turn 各有独立额度。设为 `null` 时，取消上述 Turn 分组和组内数量限制，允许同题同类型动作跨 Turn 配对，仍受全步 `max_pairs_per_step`（默认 256）限制。QA analysis 和实际最终回答保留原 outcome 分组及标签；`null` 也取消这些组的数量上限，但不混合决策类型或标签来源。两侧均过滤无效响应、执行故障、重复动作、被丢弃的 Prompt 和异常数值。SLIC baseline 使用与配对一致的分组及 V 邻域；原始 Turn 编号、信用窗口、有效性和其它 Pair 筛选条件不变。跨 Turn 模式没有额外的不同 Rollout 限制，同一 Rollout 的不同 Turn 也可能配对。

工具候选要求：

$$
|V_i-V_j|\leq\epsilon_{\mathrm{match}},\qquad
\widetilde c_i>\delta_{\mathrm{pos}},\qquad
\widetilde c_j<-\delta_{\mathrm{neg}},\qquad
\widetilde c_i-\widetilde c_j\geq\epsilon_{\mathrm{gap}}.
$$

analysis／final 依据可信成功／失败分池。基础配置的 `outcome_match_value` 默认为 `false`；v3c、v3d 实验显式设为 `true`，额外要求两侧 V 有效且 `abs(V_i - V_j) <= 0.05`，baseline 同步采用 V 邻域。工具 Turn 仍要求严格正负有效信用，没有负样本不以终局 reward 兜底。两条路线都保留响应长度比、相同动作排除和相应数量上限。

当前实现采用排序轮转：每组 Chosen 按偏好分数降序，Rejected 按分数升序；同分时按样本 index 升序。第 k 个 Chosen 只提案给 `negative[k % len(negative)]`，每个 Chosen 一次，负例耗尽后循环。提案未通过 V、信用差、长度或动作不同等过滤时跳过该 Chosen，不尝试其它负例。工具使用有效信用作为偏好分数，analysis／final 使用终局 reward。通过过滤后的优先级、阈值和 cap 保持不变。

通过条件的候选优先按对应偏好分数差降序选择：工具用回填后信用差，奖励路线用最终奖励差。价值距离、旧策略平均 Token LogP 距离仅依次打破平局；奖励路线无有效价值时不制造假值。每组、每步上限在分布式补齐之前应用。全局截断沿用候选的原始差值排序，尚未校准两种来源的分数尺度；触及全局上限时需观察各来源选中占比。

Pair 两侧保留各自原始生成 Prompt、响应及原生 token IDs。仍使用现有 per-token calibration 与 Chosen SFT；不将 GT 信用塞入 Loss，不改变原 GRPO 优势。GT 可预测性接近并不证明证据相同或动作可交换，需要独立状态抽查与验证任务收益确认配对质量。

## Replay

此次核对的远程代码基线为 `d010f3c`：启动配置默认 `REPLAY_BUFFER_SIZE=512`，trainer 未配置时的缺省值为 8192，采样目标为 `16 * round(enq_ema)`。用户确认的 **v3c、v3d** 实验设置均为容量 **512**、采样目标 `4 * round(enq_ema)`，再沿用存量限制和分布式对齐；EMA 为 40、存量充足且可整除时采样 160 对。本次仅记录实验设置，未修改基线训练代码。

## 代码组织

实现统一位于 `agentflow/agentflow/trace/` 子包：`credit.py` 包含原信用计算与 GT 评分，`execution.py` 负责故障和可信终局，`metrics.py` 负责指标，`__init__.py` 为轻量包入口。原 4 个平铺文件整理为 4 个包内文件，生产文件总数不变。该结构整理保持 `trace` 分支的公式、默认参数、筛选规则和训练行为不变。

## 配置

全部位于 `data.trace`；基础声明在 `agentflow/verl/config.yaml`。两份训练启动 YAML 提供同名 `TRACE_*` 环境变量覆盖，Hydra 参数不使用 `+`。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `enabled` | `true` | 仅算法含 `slic` 时启用；关闭恢复 legacy 配对 |
| `epsilon` | `0.1` | GT NLL gap offset |
| `k` | `3` | 包含当前 Turn 和可能的 final 的总窗口大小 |
| `gamma` | `0.8` | 窗口和终局回填折扣 |
| `terminal_fill_enabled` | `true` | 启用可信终局优势回填 |
| `terminal_fill_scale` | `2.0` | 仅保留非 QA 任务原回填系数；QA 忽略此值，使用同题动态范围 |
| `outcome_match_value` | `false` | analysis／final 是否额外要求 GT 价值匹配 |
| `outcome_success_threshold` | `0.5` | 兼容数值奖励时的成功阈值；明确 verifier 布尔值优先 |
| `match_epsilon` | `0.05` | 工具执行前价值差上限；同初态时平滑 NLL 比不超过 `exp(0.05)≈1.051` |
| `positive_threshold` | `0.01` | 工具有效信用严格正阈值 |
| `negative_threshold` | `0.01` | 工具有效信用严格负阈值的绝对值 |
| `gap_epsilon` | `0.02` | 工具正负信用差下限 |
| `length_ratio_max` | `1.8` | 两侧响应最大长度比 |
| `k_pairs_per_group` | 当前训练配置 `null`；库默认 `2` | 正整数：工具/环境动作同题同原始 Turn 配对并按组限额；`null`：同题同类型跨 Turn，无组内限额。QA analysis/final 路由不变；全步上限始终保留 |
| `max_pairs_per_step` | `256` | 每步原始 Pair 上限 |
| `max_gt_tokens` | `256` | GT token 上限，超长跳过 |
| `gt_alias_strategy` | `first` | 固定第一个非空 GT 别名 |
| `answer_instruction` | 上文统一指令 | 独立 GT 评分提示 |

TRACE 论文提供 epsilon=0.1、gamma=0.8、K-step 归一化、终局回填 scale=2 的参考设置；当前 QA 使用同题动态范围缩放，按类型配对、包含 final 的窗口计数和阈值也是项目设计，不能称为精确复现。[TRACE §3 与附录 A.1](https://arxiv.org/html/2607.13988v1)

```bash
TRACE_MATCH_EPSILON=0.05 python train-roma/train_agent.py
python train-roma/train_agent.py data.trace.terminal_fill_enabled=false
# Same-question cross-turn pairing without a group cap; keeps the global cap.
export TRACE_K_PAIRS_PER_GROUP=None  # null is also accepted; use 2 to restore same-turn pairs.
```

评分依赖冻结参考模型，可能增加显存与前向成本。状态来自实际角色可见输入，历史窗口与模板变化仍可能影响 GT 信用；本轮未重构评分视图。训练指标定义见 [指标说明](trace-slic-metrics.md)。

指标精简后，工具有效信用统一输出为 `trace/credit_*`，保留独立的 `local_credit_*` 与 `terminal_fill_*` 供比较；不再重复输出 `effective_credit_*` 或按类型拆分的工具信用。Pair 保留工具价值差、信用差、分来源数量与工具匹配率，奖励路线的定义恒定统计及混合来源比例已移除。该调整只精简日志，不改变信用、配对门槛、Loss、Replay 采样或配置；小价值距离与大信用差仍不能独立证明真实可比性。
