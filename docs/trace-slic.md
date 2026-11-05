# TRACE 状态信用与分类型 SLIC 配对

版本命名、差异、参数对照和代码快照统一维护于桌面文档 `webwalker-trace版本记录.md`。**v3a＝原 v2a1，v3b＝原 v2a2**。最新实验 **v3d** 沿用 v3c 的 Replay 容量 512、采样 ×4 和严格 V 匹配，主要设置为 `k_pairs_per_group=None`（YAML 为 `null`），允许同题同类型工具跨 Turn 配对并取消组内数量上限。QA 信用沿用 v3b 的动态范围回填。当前 `alfworld` 分支在 `41f0552` 恢复 v2a 评分、`ad412f6` 应用轮转后，本次修正为先 V 匹配再轮转；None 跨 Turn 设置保持不变。`websearch` 同步修正配对顺序，QA 历史实验定义及已有标签保留。

**v3c、v3d 均开启严格价值匹配**：`outcome_match_value=true`、`match_epsilon=0.05`，analysis／final 也必须满足 `abs(V_i - V_j) <= 0.05`，并保留可信终局成功／失败标签。此处记录实验实际设置，基础配置默认值仍为 `false`。

QA 的状态、终局和回填语义见下文；ALFWorld 的真实后状态与统一环境动作信用见 [ALFWorld 接入说明](alfworld-trace-goals.md)。当前分支训练配置默认 `k_pairs_per_group=null`，允许同题跨 Turn 且无组内 Pair 上限；显式设为正整数时，工具/环境动作仅允许同题同原始 Turn id 配对。底层库 `TraceConfig()` 未传配置时仍默认 2。

本实现按决策类型选择偏好标签：analysis 和实际最终回答使用可验证的最终任务结果，工具 Turn 使用 K-step GT 信用及可选的终局优势回填。配对质量优先；没有合格正负两侧时跳过，不为凑数量放宽条件。SLIC Loss 和原 GRPO 训练公式保持原样；v3c／v3d 的 Replay 调整记录为用户实验设置。

算法待办已完成实现与 CPU 集成回归，对应提交 `6eb6215` 已推送到 `trace` 分支。其后的指标精简也纳入 `trace` 分支，具体版本见 Git 提交记录；GPU/NPU 端训练效果尚未验证。版本状态见单独维护的 `webwalker-trace版本记录.md`。

## 状态与 GT 评分

从 `v3a` 起，QA 各类型 Turn 的 `trace_state_context` 统一使用问题与动作执行前的工具历史快照，不因切换到 final 而增删分析文本或改变评分模板。快照沿用生成时读取的有界工具历史，不扩大记忆窗口；实际生成 Prompt 保持原样。评分时统一附加：

```text
Answer the question directly and concisely.
```

使用当前训练 step 内冻结的参考模型、温度 1，对固定 GT 做 teacher forcing，只平均 GT Token 的 LogP。默认固定初始权重；可通过 [参考模型更新间隔](#参考模型更新) 按 step 刷新：

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

final 的执行结果单独用于终局项，不作为 GT delta 混入局部平均。QA 和 ALFWorld 按当前 batch 同题（同 `uid`）所有有效工具/环境动作 Turn 的回填前局部信用，以及所有可信 Rollout 的终局优势，分别统计正负范围：

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

ALFWorld 复用上述同题正负范围缩放，但保留环境动作边界：最后一个动作具有真实后继状态，其 delta 参与局部信用，K=3 最多回填最后三个动作，距离分别为 2、1、0。其全部动作均走 TRACE 标签，不新增 QA 的 final/analysis 路由。固定 `terminal_fill_scale` 对这两类任务均忽略，关闭回填统一使用 `terminal_fill_enabled=false`。

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

当前 ALFWorld `v2_d` / `v2_e` 实验均设置 `pair_round_robin=false`；本分支 v2_e 训练配置已将默认值固定为 false，库调用 `TraceConfig()` 与 WebSearch 的默认值仍为 true。

`pair_round_robin=true` 时，先 V 匹配再轮转：Chosen 按偏好分数降序、同分 index 升序处理；对每个 Chosen 先筛出 `abs(V_chosen - V_negative) <= match_epsilon` 的兼容负例池。epsilon 为 0 时精确匹配，为正时使用真实数值邻域，不按小数位取整，也不传递合并邻域。兼容列表按负例偏好分升序、index 升序排列；以有序负例索引元组作为组内池身份。属于同一池的 Chosen 按上述正例顺序局部编号 k（从 0 开始），严格分配 `compatible[k % len(compatible)]`，即该池的 `pos[k] -> neg[k % len(neg)]`。各池游标独立，分配后立即前进，信用差、长度、动作不同等后续过滤失败也不回退或补选。epsilon=0 时按各精确 V 池循环；epsilon>0 时相同兼容列表共享游标，不同列表即使有重叠也独立，不传递合并。每个 Chosen 最多提案一次，无兼容池才计 V 拒绝。工具使用有效信用、analysis／final 使用终局 reward 作为偏好分；`outcome_match_value=false` 时 outcome 路线仍跳过 V 匹配，组内完整负例池使用一个游标。本次不改变该开关、原阈值、合格候选的后续优先级和 cap。

`pair_round_robin=false` 时，枚举原组内全部合格正例与负例的组合，同一个 Chosen 可与多个不同 Rejected 配对；逐对应用相同的 V 门槛、信用差、长度、动作不同等过滤。它与 `k_pairs_per_group` 独立：后者仍控制同 Turn/跨 Turn 分组及组内 cap，None 跨 Turn 无组内 cap、2 同 Turn 最多 2 对；全步默认 256 对始终保留。

通过条件的候选优先按对应偏好分数差降序选择：工具用回填后信用差，奖励路线用最终奖励差。价值距离、旧策略平均 Token LogP 距离仅依次打破平局；奖励路线无有效价值时不制造假值。每组、每步上限在分布式补齐之前应用。全局截断沿用候选的原始差值排序，尚未校准两种来源的分数尺度；触及全局上限时需观察各来源选中占比。

Pair 两侧保留各自原始生成 Prompt、响应及原生 token IDs。仍使用现有 per-token calibration 与 Chosen SFT；不将 GT 信用塞入 Loss，不改变原 GRPO 优势。GT 可预测性接近并不证明证据相同或动作可交换，需要独立状态抽查与验证任务收益确认配对质量。

## Replay

此次核对的远程代码基线为 `d010f3c`：启动配置默认 `REPLAY_BUFFER_SIZE=512`，trainer 未配置时的缺省值为 8192，采样目标为 `16 * round(enq_ema)`。用户确认的 **v3c、v3d** 实验设置均为容量 **512**、采样目标 `4 * round(enq_ema)`，再沿用存量限制和分布式对齐；EMA 为 40、存量充足且可整除时采样 160 对。本次仅记录实验设置，未修改基线训练代码。

## 参考模型更新

ALFWorld 和 QA 共用 `data.trace.reference_update_interval`，仅在算法包含
`slic` 且 `data.trace.enabled=true` 时生效。支持 `grpo+slic`、`gigpo+slic`。

在 `train-roma/config.yaml` 或 `train-roma/config_grpo+slic.yaml` 中配置：

```yaml
env:
  TRACE_REFERENCE_UPDATE_INTERVAL: ${env:TRACE_REFERENCE_UPDATE_INTERVAL,0}
```

也可在原有启动命令前设置环境变量：

```bash
export TRACE_REFERENCE_UPDATE_INTERVAL=10    # 每完成 10 个训练 step 刷新
# export TRACE_REFERENCE_UPDATE_INTERVAL=None  # 每完成一个训练 step 刷新
# export TRACE_REFERENCE_UPDATE_INTERVAL=0     # 固定初始模型，默认值，保留原实验行为
```

直接使用 Hydra 参数时，写 `data.trace.reference_update_interval=null` 表示每步刷新。
`None`/`null` 字符串也会正规化为空值；负数、布尔值和小数不合法。

| 配置 | step 1 的参考 | step 2 的参考 | step 11 的参考 |
| --- | --- | --- | --- |
| `0` | 初始模型 | 初始模型 | 初始模型 |
| `10` | 初始模型 | 初始模型 | step 10 更新后的 actor |
| `None` / `null` / `1` | 初始模型 | step 1 更新后的 actor | step 10 更新后的 actor |

“step” 是完成整个训练更新的 step，不是 turn、rollout、微批次或单次优化器更新。
只有收集数据、因 stale 队列尚未满足条件提前返回时，不更新参考。
一个 step 内的 GT LogP、初始状态 V 基准、局部信用和终局回填全部使用同一个参考版本。
待 GRPO/GiGPO 和该步 SLIC 更新都完成后，才复制最终 actor 参数，供下一步使用。

此处 checkpoint 指 actor 的内存权重快照，不依赖 `trainer.save_freq`。
FSDP/FSDP2 的 actor 和 reference 同 rank 同进程时，直接复制已分片参数；不会经
driver 收集整模型，不会每步写磁盘，不复制优化器、不改 actor 梯度。参考模型继续
保持 eval 模式、评分温度 1。更新后清除共享前缀评分的验证状态，重新验证新权重。
不兼容的模型分片布局会明确报错；目前不支持 LoRA/ref_in_actor 和 Megatron 刷新。

刷新的是现有的独立 reference：TRACE GT 评分使用它；如果另行开启 KL reward 或
KL loss，它们也使用该参考模型。默认两种 KL 都关闭。SLIC loss、配对阈值、配对
方式、replay 采样与容量不变；已有 replay pair 保留其选入时的偏好标签，不重新评分。

### 保存和恢复

普通训练 checkpoint 开启时，会在 `global_step_N/trace_reference/` 一并保存参考模型
分片和 `trace_reference.json`，记录更新间隔与参考来源 step。例如 interval=10、
训练保存于 step 17 时，actor 是 step 17，参考仍是 step 10，恢复时分别加载。
该参考存储随 actor 的 checkpoint 保留策略清理。

恢复时必须使用相同 interval。旧 checkpoint 没有参考快照时，只有恰好处于更新边界
才能从恢复后的 actor 重建参考；否则报错，避免把较新的 actor 冒充旧参考。`0` 模式
沿用原恢复行为。远程 HDFS 恢复仍受原训练器的限制。

### 指标

- `trace/reference_step`：本 step 评分实际使用的权重来源，0 为初始模型。
- `trace/reference_step_next`：本 step 完成后参考来源，供下一 step 使用。
- `trace/reference_refreshed`：本 step 是否执行刷新。
- `timing_s/reference_refresh`：刷新阶段耗时，计入 step 总耗时。

在 `alfworld` 分支本次更新中，评分历史已恢复正式 v2_e（`ee5da31`）的原生 memory 格式，
同时保留独立 `ALFWORLD_TRACE_HISTORY_LENGTH`（默认 4、None 全部、0 不包含历史）和
同题局部信用范围内的动态终局回填。设置刷新间隔为 0 可隔离参考更新这一实验变量。

## 代码组织

TRACE 核心统一位于 `agentflow/agentflow/trace/`，按职责组织：

| 文件 | 职责 |
| --- | --- |
| `credit.py` | 配置、GT 评分输入、V、信用和 Pair 选择 |
| `execution.py` | 执行状态、外部错误和可信终局判断 |
| `metrics.py` | 训练诊断指标，不参与筛选或 Loss |
| `prefix_scoring.py` | 精确前缀树、attention mask、参考模型执行与分布式一致性 |
| `alfworld.py` | PDDL 规则转换、目标核验与评分状态渲染 |
| `alfworld_goal_store.py` | 数据集游戏映射、SQLite/Parquet 离线准备与运行时读取 |

参考模型更新调度及 checkpoint 保存/恢复保留在原 `agentflow/verl/trainer.py`；分片复制与 RPC 保留在原 `agentflow/verl/diy/workers/fsdp_workers.py`。ALFWorld 的两个轻量环境适配类统一放在 `envs/alfworld/trace_wrapper.py`，避免异步子进程导入带初始化副作用的主环境文件。

Probe 的交互入口、评分后端和 HTML 各自保留；采样缓存和评分可独立使用。配置、执行、数据准备与 replay 测试归入相应已有测试文件。此整理只移动实现、更新内部引用，保留 CLI、配置值、数据格式、公式、筛选规则和训练行为。

## 配置

全部位于 `data.trace`；基础声明在 `agentflow/verl/config.yaml`。两份训练启动 YAML 提供同名 `TRACE_*` 环境变量覆盖，Hydra 参数不使用 `+`。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `enabled` | `true` | 仅算法含 `slic` 时启用；关闭恢复 legacy 配对 |
| `epsilon` | `0.1` | GT NLL gap offset |
| `k` | `3` | 包含当前 Turn 和可能的 final 的总窗口大小 |
| `gamma` | `0.8` | 窗口和终局回填折扣 |
| `terminal_fill_enabled` | `true` | 启用可信终局优势回填 |
| `terminal_fill_scale` | `2.0` | 仅保留其它旧任务的回填系数；QA/ALFWorld 忽略此值，使用同题动态范围 |
| `outcome_match_value` | `false` | analysis／final 是否额外要求 GT 价值匹配 |
| `outcome_success_threshold` | `0.5` | 兼容数值奖励时的成功阈值；明确 verifier 布尔值优先 |
| `match_epsilon` | `0.05` | 工具执行前价值差上限；同初态时平滑 NLL 比不超过 `exp(0.05)≈1.051` |
| `positive_threshold` | `0.01` | 工具有效信用严格正阈值 |
| `negative_threshold` | `0.01` | 工具有效信用严格负阈值的绝对值 |
| `gap_epsilon` | `0.02` | 工具正负信用差下限 |
| `length_ratio_max` | `1.8` | 两侧响应最大长度比 |
| `pair_round_robin` | ALFWorld v2_e 训练 `false`；库 `true` | 先 V 兼容池内轮转；`false` 枚举全部正负组合。与分组和组 cap 独立 |
| `k_pairs_per_group` | 当前训练配置 `null`；库默认 `2` | 正整数：工具/环境动作同题同原始 Turn 配对并按组限额；`null`：同题同类型跨 Turn，无组内限额。QA analysis/final 路由不变；全步上限始终保留 |
| `max_pairs_per_step` | `256` | 每步原始 Pair 上限 |
| `max_gt_tokens` | `256` | GT token 上限，超长跳过 |
| `gt_alias_strategy` | `first` | 固定第一个非空 GT 别名 |
| `answer_instruction` | 上文统一指令 | 独立 GT 评分提示 |
| `reference_update_interval` | `0` | `0` 固定初始参考；正整数 N 每完成 N 个训练 step 更新；`None`/`null` 每步使用最新 actor 快照 |

TRACE 论文提供 epsilon=0.1、gamma=0.8、K-step 归一化、终局回填 scale=2 的参考设置；当前 QA 使用同题动态范围缩放，按类型配对、包含 final 的窗口计数和阈值也是项目设计，不能称为精确复现。[TRACE §3 与附录 A.1](https://arxiv.org/html/2607.13988v1)

```bash
TRACE_MATCH_EPSILON=0.05 python train-roma/train_agent.py
python train-roma/train_agent.py data.trace.terminal_fill_enabled=false
# Same-question cross-turn pairing without a group cap; keeps the global cap.
export TRACE_K_PAIRS_PER_GROUP=None  # null is also accepted; use 2 to restore same-turn pairs.
export TRACE_PAIR_ROUND_ROBIN=false  # ALFWorld v2_d/v2_e: all combinations; same filters/caps.
# export TRACE_PAIR_ROUND_ROBIN=true   # Enable V-matched pool-local modulo rotation.
```

评分依赖冻结参考模型，可能增加显存与前向成本。状态来自实际角色可见输入，历史窗口与模板变化仍可能影响 GT 信用；本轮未重构评分视图。训练指标定义见 [指标说明](trace-slic.md#training-metrics)。

指标精简后，工具有效信用统一输出为 `trace/credit_*`，保留独立的 `local_credit_*` 与 `terminal_fill_*` 供比较；不再重复输出 `effective_credit_*` 或按类型拆分的工具信用。Pair 保留工具价值差、信用差、分来源数量与工具匹配率，奖励路线的定义恒定统计及混合来源比例已移除。该调整只精简日志，不改变信用、配对门槛、Loss、Replay 采样或配置；小价值距离与大信用差仍不能独立证明真实可比性。

<a id="shared-prefix-scoring"></a>

## 共享前缀 GT 评分

本实现迁移 IGPO 论文 Figure 3 的计算方式：相同前缀只前向一次，多个 GT 分支通过自定义四维 attention mask 相互隔离。它不是把独立序列合并为普通 batch，也不将 TRACE 换成 IGPO 的奖励公式。

参考代码固定到官方 [IGPO `64165e2` 的自定义注意力实现](https://github.com/GuoqingWang1/IGPO/blob/64165e2741ed8801f977948c8128080ce87b4101/scrl/llm_agent/vectorized_gt_logprob.py)。该仓库正式 generation 入口实际使用另一条普通批量评分路径；本次迁移的是论文中的共享计算机制。

### 完整状态如何共享

ALFWorld 每个 turn 的当前事实、观察及配置的历史窗口会变化，不能假设后一轮完整文本是前一轮的追加。本实现直接使用现有评分器已经生成的 `prompt_ids` 和 `target_ids` 构建精确 token 前缀树，既共享同题的规则与初始事实，也共享任何确实一致的更长前缀。分包只影响计算组织，不改文本、token、GT 长度检查或评分目标。

每个物理节点保存 token、父节点和逻辑深度。对于 query 节点 q，只允许它关注自身及祖先节点；其它分支、未来状态和其它 GT 均不可见。RoPE 的 position ID 使用逻辑深度，而非节点在拼接数组中的物理位置。

每个 GT token 从其逻辑父节点的 logits 取概率。特别是 GT 的第一个 token 使用对应上下文最后一个节点的 logits，不能使用数组中前一个分支末尾的 logits。这一处理避免官方示意实现中直接使用 `gt_start - 1` 的对齐风险。

逐 token LogP 回填到原请求行后，继续使用原 `mean_gt_logps`、V、K-step credit、终局回填与配置指定的配对逻辑。完全相同的状态仍先使用原有去重。

### 配置

正式 `train-roma/config.yaml` 和 `config_grpo+slic.yaml` 默认请求共享评分。已有自定义配置若未设置新字段，仍采用原批量评分方式。对应环境变量为：

```bash
export TRACE_SHARED_PREFIX_SCORING=true
export TRACE_SHARED_PREFIX_MAX_TOKENS=8192
export TRACE_SHARED_PREFIX_MAX_MASK_MIB=256
export TRACE_SHARED_PREFIX_VERIFY=true
```

设 `TRACE_SHARED_PREFIX_SCORING=false` 恢复原 GT 批量评分。四个字段位于 `data.trace.shared_prefix_*`；不需要重新生成 GT Parquet，不需要改变 GRPO、rollout 或 actor 的 SP/microbatch。

token 数和 mask 预算共同限制每个树包的大小；超限不能截断输入或伪造评分。mask 预算仅覆盖四维 additive mask，不是整个模型峰值内存；eager attention 还需要与 head 数相关的中间张量。默认规划也限制树包的稠密 attention 规模，避免把几乎不共享前缀的长序列强行合并。

### Reference 与分布式边界

新入口仅用于冻结 reference 的 GT 评分，原 actor forward/backward、rollout、动作 LogP 和配对流程继续沿用原路径。树评分对现有 reference FSDP 模块临时选择支持任意 4D mask 的 eager attention，结束后恢复配置。首版支持纯文本 Qwen2/Qwen3 的标准 RoPE；不支持的模型、LoRA、fused forward 或其它不满足条件的配置会明确记录回退原因。

启动时可以保留 SP=2；但树评分绕开原 remove-padding/SP 序列切片，SP 组内各 rank 对完整树重复前向，只有原收集 rank 的结果回传。它不意味着树评分仍享有 SP=2 的序列并行收益。actor 训练的 SP 行为不变。

不同 rank 的树包数量可能不同。模型前向前统一协商计划与最大轮数，不足者执行不回填结果的 dummy 前向，确保 FSDP 参数通信顺序一致。规划或能力回退在所有相关 rank 上统一决策，不能让一个 rank 中途独自改走另一条通信路径。

首次共享评分默认与原评分路径作逐 GT token 数值对照，因此第一次会额外执行原路径，可能暂时增加首步耗时。数值不匹配时统一使用原结果，并在后续停止重复尝试该共享路径。后续成功验证的共享评分无需每步再执行原评分。BF16 与不同 attention backend 不承诺逐 bit 相同；有效性判断与最终返回精度应遵循原评分口径。

另一个原有性能因素是 reference 的 FSDP 包装会强制 CPU offload，即使 YAML 的 `ref.fsdp_config.param_offload=false`；因此之前 GT 评分的 113 秒也可能包含参数搬运。本次不改这项既有行为。共享评分减少前向次数时，也可能减少部分参数搬运。

### 验证与性能解读

新测试覆盖树路径、分支隔离、位置编码、首 GT token、不同 GT 长度、预算分包、配置路由和回退。真实小型 Qwen2/Qwen3 在 CPU 上逐 token 对照独立 teacher forcing；这不等于服务器 4B 模型、Ascend 或多机 HCCL 的完整训练验证。

本地随机小型 Qwen3 合成基准中，16 条请求共享 256-token 前缀，原始 token 总数 4288，树节点数 448，节省约 89.6%；逐 GT token LogP 最大差为 0。该微基准不能推导生产环境加速倍数。稠密 4D attention 仍与树包长度平方相关，实际速度还受共享程度、分包、SP 重复前向、reference 参数搬运和内存影响。

本地两进程 Gloo 测试也已通过：两端本地 pack 数不同，需要一端补 dummy 前向，仍能完成首批校验、后续免重复校验、跨 rank 一致回退，并返回一致的指标。它使用 CPU attention 模型模拟每次 FSDP 前向中的 collective，不代表真实 Ascend/HCCL/FSDP 已完成验证。可在服务器仓库中单独复现：

```bash
TRACE_TEST_PREFIX_DISTRIBUTED=1 python -m unittest discover \
  -s agentflow/tests -p 'test_trace_prefix_distributed.py'
```

服务器应观察下面的指标，并比较第 2 步以后的 `trace/scoring_seconds`。不能只看到开关为 true 就认定已经共享计算。

| 指标 | 含义 |
| --- | --- |
| `trace/prefix_requested` | trainer 是否请求新路径 |
| `trace/prefix_used` | 本批是否真正使用树评分结果 |
| `trace/prefix_saved_tokens` | 相对独立评分少算的 token 数，已排除 SP 复制计数 |
| `trace/prefix_original_tokens` / `tree_tokens` | 独立输入总 token 数与树节点总数 |
| `trace/prefix_forward_rounds` | 各 rank 对齐后的树前向轮数 |
| `trace/prefix_dummy_forwards` | 为保持通信顺序而增加、不回填结果的前向 |
| `trace/prefix_attention_cost_ratio` | 树包稠密 attention 格数与独立请求格数的比值 |
| `trace/prefix_validation_checked` / `passed` / `max_diff` | 是否执行数值校验、是否通过、最大逐 token 差值 |
| `trace/prefix_fallback_*` | 是否因模型、预算、无共享收益或数值校验等原因回退 |

所有 fallback 指标在各 rank 上统一。`prefix_used=0` 时不能把该批耗时解释为共享前缀评分的性能；如果 `fallback_planning=1`，应先检查输入长度与两个包预算，不能通过截断完整状态来强行启用。

<a id="training-metrics"></a>

## 训练指标

指标接入原训练 metrics 字典，默认 console logger 输出，无需额外开关。比例为 0–1。早期指标精简只调整统计；当前 `alfworld` 与 `websearch` 均支持 `TRACE_PAIR_ROUND_ROBIN`（`data.trace.pair_round_robin`），true 使用先 V 匹配再轮转，false 枚举全部正负组合；当前 ALFWorld v2_d/v2_e 实验及 v2_e 训练默认均为 false，WebSearch 默认仍为 true。指标分母按模式区分如下。

`trace` 来源是工具局部信用加终局回填，`outcome` 来源是 analysis／实际最终回答的可信成功／失败标签。两种来源的 Pair 数分开观察；工具信用使用保留的 `credit_*` 名称，已移除与其完全相同的 `effective_credit_*`。

### 优先观察

| 指标 | 含义 | 如何判断 |
| --- | --- | --- |
| `val/reward`、`val/completion_rate`、`val/valid_rollout_rate` | 有效已完成轨迹的平均 reward，以及完成／有效轨迹占排队轨迹比例 | 完成率和有效率稳定时，reward 才便于比较；只有二值奖励时均值直接对应成功率 |
| `trace/credit_coverage` | 有效信用工具 Turn 数 / 预期计算信用的工具 Turn 数 | 应结合评分缺失、故障窗口原因判断；analysis／final 不进入分母 |
| `trace/local_credit_p10/p50/p90` | 未回填的 K-step 工具信用分布 | 观察纯局部信号；不要求正负各半 |
| `trace/terminal_fill_p10/p50/p90` | 终局优势回填项分布 | 与局部信用一起看，判断回填是否主导 |
| `trace/credit_p10/p50/p90` | 局部信用加回填后的有效工具信用 | 工具 Pair 使用此信号；幅度越大不等于效果越好 |
| `trace/pair/trace_selected_unique`、`outcome_selected_unique` | 分布式补齐前的工具／奖励来源 Pair 数 | 区分两种监督构成；数量不作为质量目标 |
| `trace/pair/tool_value_match_rate` | true：存在 V 兼容池的工具 Chosen 尝试比例；false：通过 V 的工具组合比例 | 两种模式分母分别为 Chosen 尝试数、正负组合数，不能直接横比 |
| `trace/pair/value_gap_p90/max`、`credit_gap_min` | 选中工具 Pair 的状态价值差、有效信用差 | 应满足 `match_epsilon`、`gap_epsilon`；小价值差、大信用差仅为代理信号，不能证明状态可比或动作因果贡献 |
| `trace/external_error_turns`、`credit_windows_blocked_external` | 外部故障 Turn 数及受影响信用窗口数 | 区分基础设施问题与模型退化 |
| `trace/outcome_invalid_rollouts` | 没有可信终局结果的 Rollout 数 | 增长时检查实际 final、verifier 和外部中断；不能直接当作失败负样本 |
| `trace/replay/sample_unique_pair_rate`、`sample_age_p90` | 采样多样性、采样 Pair 年龄的 90 分位数 | 观察过量重复抽样和旧数据占用 |
| `trace/replay/slic_updated` | 本 step 是否实际完成 SLIC 更新 | 计划更新且有可靠数据时应为 1；无合格 Pair 时跳过符合设计 |

### 状态评分与信用

以下为当前保留的状态指标。`count` 仅计有效有限观测；没有观测时不生成假均值或分位数。

| 指标 | 定义 |
| --- | --- |
| `trace/turns_total` | 本批全部保留 Turn 数 |
| `trace/score_coverage` | 有效且不大于零的 GT LogP Turn 数 / 全部保留 Turn 数；奖励路线不要求 GT 评分 |
| `trace/gt_logp_count`、`gt_logp_mean` | 有效 GT 平均 Token LogP 的数量、跨 Turn 均值 |
| `trace/credit_expected_turns` | 预期计算信用的工具行数，排除 analysis／final |
| `trace/credit_coverage` | 预期工具行中获得有效信用的比例；状态缺失和外部故障仍保留在分母 |
| `trace/credit_{count,mean,p10,p50,p90}` | 有效工具信用，包含启用的终局回填 |
| `trace/local_credit_{count,mean,p10,p50,p90}` | 未回填的 K-step 局部工具信用 |
| `trace/terminal_fill_{count,mean,p10,p50,p90}` | 工具终局优势回填项，含未触发回填或优势为零的有限零值 |
| `trace/positive_rate`、`negative_rate` | 全部有限工具信用超过严格正阈值、低于严格负阈值的比例；尚未做动作质量过滤。`credit_count>0` 时，中性比例可由 `1-positive_rate-negative_rate` 推得；无信用样本时两者均输出 0，不据此推断中性占比 |
| `trace/terminal_reached_turns` | 信用窗口触及可信实际 final 的工具 Turn 数 |
| `trace/terminal_fill_nonzero_turns` | 回填项非零且有限的工具 Turn 数；触及终局但优势为零时不会计入 |
| `trace/credit_windows_blocked` | 因所有原因失效的工具信用窗口数，包括外部故障、编号缺口和所需状态缺失 |
| `trace/credit_windows_blocked_external` | 因外部故障失效的窗口数，是总失效窗口的子集 |
| `trace/external_error_turns` | 标记为外部故障的 Turn 数 |
| `trace/outcome_invalid_rollouts` | 整条 Rollout 没有任何可信终局标签行时才计入；可信 final 后的无标签附加输出不使该 Rollout 变为无效 |
| `trace/post_final_turns` | 实际用于验证的 final 之后的输出行数；这些行不能继承最终标签或进入 Pair |

正常验证成功和失败都可以是可信终局；verifier 调用／解析失败、外部中断等才会使结果不可信。优先使用明确的 verifier 成功标记，不能根据正格式奖励推断任务成功。

终局优势按可信终局 Rollout 去重统计，其均值／标准差人口可能与原 GRPO batch 不同；复用中心化和归一化口径不代表两条路径优势数值相同，原 GRPO 优势未被修改。

### 评分准备与成本

- `trace/missing_gt`、`missing_state`、`invalid_gt_length`、`overlong_state`：按检查顺序记录第一个输入拒绝原因，彼此互斥；超长状态和 GT 跳过，不截断。
- `trace/scoring_requests`：去重后的实际逻辑评分请求数，分布式补齐前计数。
- `trace/scoring_invalid_result_turns`：已映射评分请求却得到无效 LogP 的 Turn 数；共享请求失败可能影响多行。
- `trace/scoring_seconds`：GT reference RPC 的墙钟时间。
- `trace/step_time_fraction`：评分请求构造、reference 评分及信用汇总区间占训练 step 的比例；不包含所有 Pair／Replay 工作或所有新增 reference 开销。

### 配对筛选与质量

以下统一省略 `trace/pair/` 前缀：

| 指标 | 定义 |
| --- | --- |
| `candidates_sign` | true：有负例时为正例数，无负例时为 0，含无 V 兼容池的尝试；false：正例数 × 负例数。均按同题/类型/来源的组求和；正整数 cap 还按 Turn 分组，None 不按 Turn 分组 |
| `groups_no_positive`、`groups_no_negative` | 缺少合格正侧／负侧的组数；中性组可能同时计入两者 |
| `both_sign_group_rate` | 同时具备正负两侧的合格组数 / 有合格行的组数；无合格组时为 0 |
| `tool_value_match_rate` | true：存在 V 兼容池的工具 Chosen 数 / 工具 Chosen 尝试数；false：通过 V 的工具组合数 / 全部工具候选组合数；无相应分母时为 0 |
| `rejected_value_match` | true：需要 V 匹配却无兼容负例的 Chosen 尝试数；false：不满足 V 门槛的组合数。outcome 路线是否要求 V 仍由原 `outcome_match_value` 控制 |
| `rejected_credit_gap` | 因工具有效信用差不足被拒绝的组合数；奖励路线不使用此门槛 |
| `rejected_length` | 因响应长度比过大被拒绝的组合数 |
| `rejected_same_action` | 因两侧动作身份相同被拒绝的组合数 |
| `rejected_group_cap`、`rejected_global_cap` | 通过质量过滤后，受到每组／每步数量上限限制的组合数 |
| `selected_unique` | 补齐前最终选中的 Pair 数；true 中每个 Chosen 至多一次、Rejected 可循环；false 中 Chosen 可与多个不同 Rejected 配对 |
| `trace_selected_unique`、`outcome_selected_unique` | 工具／奖励来源最终选中的 Pair 数 |
| `question_coverage` | 最终 Pair 覆盖问题数 / 当前批次已有 Turn 的不同问题数；未返回任何 Turn 的问题不在分母 |
| `cross_turn_rate` | 最终 Pair 两侧原始 Turn 编号不同的比例；工具/环境动作同 Turn 模式应为 0，当前 None 跨 Turn 模式可非零；QA outcome 路由保留原行为 |
| `value_gap_{p50,p90,max}` | 最终工具 Pair 两侧动作前状态价值差，仅统计工具来源 |
| `credit_gap_{min,p50,p90}` | 最终工具 Pair 两侧有效信用差，仅统计工具来源 |
| `{type}/selected_unique` | 各决策类型最终选中的 Pair 数 |
| `padding_duplicates` | 为分布式分区补齐而重复的 Pair 条数 |

六个 `rejected_*` 原因按检查顺序互斥，可核对 `candidates_sign = 六个拒绝计数之和 + selected_unique`。累计通过阶段已删除；`groups_eligible`、`groups_both_signs`、`trace_candidates_sign`、`trace_after_value_match` 仅作为计算两个比例的内部计数，不再输出日志。

`pair_round_robin=true` 时，每个 Chosen 先按 `abs(V_chosen - V_negative) <= match_epsilon` 找兼容负例；0 为精确匹配，正值为真实邻域，不取整或传递合并。兼容列表按负例偏好分升序、index 升序排列，以其有序索引元组标识组内池。同池 Chosen 按偏好分降序、index 升序从 0 局部编号 k，严格分配 `compatible[k % len(compatible)]`；每池独立游标，分配后即前进，后续质量过滤失败也不回退或补选。0 对应各精确 V 池；正值时相同兼容列表共享游标，不同列表即使重叠也独立。每个 Chosen 最多分配一次；`outcome_match_value=false` 时 outcome 路线跳过 V，组内全负例池使用一个游标。未生成的其它正负组合不计入候选或拒绝数。`pair_round_robin=false` 恢复全组合计数，`candidates_sign` 为正负例数乘积之和，V 拒绝按组合计数；两模式分母不能直接横比；分布式补齐及跨步 Replay 仍可能重复同一 Pair。通过过滤的工具 Pair 按有效信用差、奖励路线按最终奖励差降序选择，再按价值距离、旧策略平均 Token LogP 距离打破平局。全局上限仍基于候选原始差值排序，两种来源尚无尺度校准；触及全局上限时需观察分来源选中数量。精简日志没有改变这一选择逻辑。

`trace/invalid_action_turns`、`trace/duplicate_turns`、`trace/dropped_prompt_turns` 是行级过滤原因，彼此可能重叠。原 `replay/n_new_pairs` 可能包含补齐条数，不能替代 `selected_unique`。

工具信用与终局是否一致，仅保留 `trace/pair/trace/` 下的：

- `outcome_pairs_count`：双方都有可信成功标记的工具 Pair 数；无可信观测时为 0，且不输出下列比例。
- `chosen_failed_rollout_rate`：上述工具 Pair 中 Chosen 来自最终失败 Rollout 的比例。
- `rejected_successful_rollout_rate`：上述工具 Pair 中 Rejected 来自最终成功 Rollout 的比例。
- `same_outcome_rate`：上述工具 Pair 两侧最终结果相同的比例。

这些比例描述局部标签与最终结果的关系，不能充当因果正确率；奖励来源本来就按成功／失败配对，不再重复输出其定义决定的终局比例。新路径只采用可信 verifier 成功标记；无 `label_source` 的历史诊断行保留原 reward > 0 解释。

### Replay、验证与原有 Loss

- `trace/replay/sampled_pairs`：实际采样条数，不能等同于优化次数。
- `trace/replay/sample_unique_pair_rate`：采样的不同原始 Pair 数 / 采样条数；原始身份含出生 step，不代表动作语义已去重。
- `trace/replay/sample_age_p90`：采样 Pair 年龄的 90 分位数，单位训练 step。
- `trace/replay/slic_updated`：本 step 实际完成 SLIC 更新的 0/1 标记。
- 原 `replay/buffer_size/n_dpo_batch`、`dpo_loss/sft_loss/calibration_loss/quality_gap/rewards/accuracy` 保留。后两者表示拟合偏好标签的程度，不等于任务成功率。
- `rewards/chosen/rejected/margin` 使用整段概率，长响应易下溢，需要结合 per-token gap、损失分项及梯度解释。
- `val/queued_rollouts/completed_rollouts/valid_rollouts`、`completion_rate/valid_rollout_rate` 保留；两种比例分母均为排队轨迹数。`val/reward` 不把未返回轨迹自动记失败。

本轮不修改 SLIC Loss、Replay 更新强度、GRPO 优势算法及配置。CPU 回归不能替代 GPU/NPU 训练收益验证；小状态距离与大信用差仍需独立状态抽查和验证集收益确认。
