# TRACE 状态信用与分类型 SLIC 配对

版本说明：本页描述 **v2_a**，对应 `data.trace.pair_score_mode=trace_fill`。`trace-2b` 分支默认使用 `igpo` 模式，完整公式、配置与指标见 [v2_b 使用说明](trace-2b.md)。

本实现按决策类型选择偏好标签：analysis 和实际最终回答使用可验证的最终任务结果，工具 Turn 使用 K-step GT 信用及可选的终局优势回填。配对质量优先；没有合格正负两侧时跳过，不为凑数量放宽条件。SLIC Loss、原 GRPO 训练公式和 Replay 更新剂量保持原样。

算法待办已完成实现与 CPU 集成回归，对应提交 `6eb6215` 已推送到 `trace` 分支。其后的指标精简也纳入 `trace` 分支，具体版本见 Git 提交记录；GPU/NPU 端训练效果尚未验证。修改状态见 [修改记录](trace-slic-change-log.md)。

## 状态与 GT 评分

每个可训练 Turn 保存自己的 `trace_state_context`：从实际 Prompt 中去掉当前决策指令、Task 和输出格式要求，保留该角色原有问题、工具信息、分析或交互历史。统一附加：

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

final 的执行结果单独用于终局项，不作为 GT delta 混入局部平均。窗口触及实际已完成且验证可信的 final 时，对工具 Turn 回填：

$$
\widetilde c_t=c_t^{(K)}+
\mathbf1[\mathcal W_t\text{触及可信 final}]
\lambda_{\mathrm{term}}\gamma^{f-t}A_g^{\mathrm{out}},
$$

其中 f 为真实 final 的原始 Turn 编号，折扣距离与总窗口的计数一致。用于 TRACE 回填的终局优势仅在有可信终局结果的同题 Rollout 中计算，并按 Rollout 去重。复用的是原 GRPO 的中心化、是否除以标准差及单例组特例等计算口径；可信 Rollout 的统计人口可能与实际 GRPO batch 不同，因此两条路径的优势数值不保证相同。它不是直接把原始奖励 R 加到信用上，也不会回写或修改原 GRPO 优势。局部信用、回填项及回填后信用分别保留。

例如 `analysis → tool A → tool B → tool C → final`，K=3：

| 工具 Turn | 窗口 | 是否回填 |
| --- | --- | --- |
| A | A、B、C | 否 |
| B | B、C、final | 是，系数 `lambda * gamma**2` |
| C | C、final | 是，系数 `lambda * gamma` |

因此本项目的 K=3 最多向 final 前两个工具 Turn 回填。这是明确包含 final 的总 Turn 窗口约定；与 TRACE 论文按工具转移计 K 的约定不同，不能同时声称回填最后三个工具 Turn。

## 外部故障与窗口隔离

网络请求超时、连接／DNS 失败、服务不可用、网关或外部限流等最终未恢复故障，应标为外部故障；对应 Turn 不能进入 Chosen 或 Rejected。优先识别结构化执行状态及已知适配器错误格式，不扫描网页正文中的普通 error 文本。

正常无搜索结果不是网络故障；模型非法工具名／参数也不能自动归为外部故障。内部重试后成功取得结果，按最终成功状态处理。

包含外部故障转移的工具 K-step 窗口信用不可用，不参与 Pair。不把故障 delta 填成 0，也不删除故障后连接原本不相邻的状态。故障导致终局无法可靠验证时，analysis／final 奖励 Pair 与工具终局回填也不可使用该结果。

## 配对、排序与 Loss

候选必须同 `uid`、同决策类型和同标签来源，允许不同 Rollout、不同 Turn 序号。两侧均过滤无效响应、执行故障、重复动作、被丢弃的 Prompt 和异常数值。

工具候选要求：

$$
|V_i-V_j|\leq\epsilon_{\mathrm{match}},\qquad
\widetilde c_i>\delta_{\mathrm{pos}},\qquad
\widetilde c_j<-\delta_{\mathrm{neg}},\qquad
\widetilde c_i-\widetilde c_j\geq\epsilon_{\mathrm{gap}}.
$$

analysis／final 依据可信成功／失败分池，默认不施加 GT 价值距离门槛；可用 `outcome_match_value=true` 显式开启。工具 Turn 仍要求严格正负有效信用，没有负样本不以终局 reward 兜底。两条路线都保留响应长度比、相同动作排除和数量上限。

通过条件的候选优先按对应偏好分数差降序选择：工具用回填后信用差，奖励路线用最终奖励差。价值距离、旧策略平均 Token LogP 距离仅依次打破平局；奖励路线无有效价值时不制造假值。每组、每步上限在分布式补齐之前应用。全局截断沿用候选的原始差值排序，尚未校准两种来源的分数尺度；触及全局上限时需观察各来源选中占比。

Pair 两侧保留各自原始生成 Prompt、响应及原生 token IDs。仍使用现有 per-token calibration 与 Chosen SFT；不将 GT 信用塞入 Loss，不改变原 GRPO 优势。GT 可预测性接近并不证明证据相同或动作可交换，需要独立状态抽查与验证任务收益确认配对质量。

## 配置

全部位于 `data.trace`；基础声明在 `agentflow/verl/config.yaml`。两份训练启动 YAML 提供同名 `TRACE_*` 环境变量覆盖，Hydra 参数不使用 `+`。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `enabled` | `true` | 仅算法含 `slic` 时启用；关闭恢复 legacy 配对 |
| `epsilon` | `0.1` | GT NLL gap offset |
| `k` | `3` | 包含当前 Turn 和可能的 final 的总窗口大小 |
| `gamma` | `0.8` | 窗口和终局回填折扣 |
| `terminal_fill_enabled` | `true` | 启用可信终局优势回填 |
| `terminal_fill_scale` | `2.0` | 回填系数，可独立消融 |
| `outcome_match_value` | `false` | analysis／final 是否额外要求 GT 价值匹配 |
| `outcome_success_threshold` | `0.5` | 兼容数值奖励时的成功阈值；明确 verifier 布尔值优先 |
| `match_epsilon` | `0.05` | 工具执行前价值差上限；同初态时平滑 NLL 比不超过 `exp(0.05)≈1.051` |
| `positive_threshold` | `0.01` | 工具有效信用严格正阈值 |
| `negative_threshold` | `0.01` | 工具有效信用严格负阈值的绝对值 |
| `gap_epsilon` | `0.02` | 工具正负信用差下限 |
| `length_ratio_max` | `1.8` | 两侧响应最大长度比 |
| `k_pairs_per_group` | `2` | 每题、类型及来源组的 Pair 上限 |
| `max_pairs_per_step` | `256` | 每步原始 Pair 上限 |
| `max_gt_tokens` | `256` | GT token 上限，超长跳过 |
| `gt_alias_strategy` | `first` | 固定第一个非空 GT 别名 |
| `answer_instruction` | 上文统一指令 | 独立 GT 评分提示 |

TRACE 论文提供 epsilon=0.1、gamma=0.8、K-step 归一化、终局回填 scale=2 的参考设置，但本项目按类型配对、包含 final 的窗口计数和阈值均是项目设计，不能称为精确复现。[TRACE §3 与附录 A.1](https://arxiv.org/html/2607.13988v1)

```bash
TRACE_MATCH_EPSILON=0.05 python train-roma/train_agent.py
python train-roma/train_agent.py data.trace.terminal_fill_enabled=false
```

评分依赖冻结参考模型，可能增加显存与前向成本。状态来自实际角色可见输入，历史窗口与模板变化仍可能影响 GT 信用；本轮未重构评分视图。训练指标定义见 [指标说明](trace-slic-metrics.md)。

指标精简后，工具有效信用统一输出为 `trace/credit_*`，保留独立的 `local_credit_*` 与 `terminal_fill_*` 供比较；不再重复输出 `effective_credit_*` 或按类型拆分的工具信用。Pair 保留工具价值差、信用差、分来源数量与工具匹配率，奖励路线的定义恒定统计及混合来源比例已移除。该调整只精简日志，不改变信用、配对门槛、Loss、Replay 采样或配置；小价值距离与大信用差仍不能独立证明真实可比性。
