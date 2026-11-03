# TRACE-SLIC 训练观察指标

指标接入原训练 metrics 字典，默认 console logger 输出，无需额外开关。比例为 0–1。早期指标精简只调整统计；当前 `alfworld` 与 `websearch` 均支持 `TRACE_PAIR_ROUND_ROBIN`（`data.trace.pair_round_robin`），默认 true 保持先 V 匹配再轮转，false 枚举全部正负组合；指标分母按模式区分如下。

`trace` 来源是工具局部信用加终局回填，`outcome` 来源是 analysis／实际最终回答的可信成功／失败标签。两种来源的 Pair 数分开观察；工具信用使用保留的 `credit_*` 名称，已移除与其完全相同的 `effective_credit_*`。

## 优先观察

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

## 状态评分与信用

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

## 评分准备与成本

- `trace/missing_gt`、`missing_state`、`invalid_gt_length`、`overlong_state`：按检查顺序记录第一个输入拒绝原因，彼此互斥；超长状态和 GT 跳过，不截断。
- `trace/scoring_requests`：去重后的实际逻辑评分请求数，分布式补齐前计数。
- `trace/scoring_invalid_result_turns`：已映射评分请求却得到无效 LogP 的 Turn 数；共享请求失败可能影响多行。
- `trace/scoring_seconds`：GT reference RPC 的墙钟时间。
- `trace/step_time_fraction`：评分请求构造、reference 评分及信用汇总区间占训练 step 的比例；不包含所有 Pair／Replay 工作或所有新增 reference 开销。

## 配对筛选与质量

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

## Replay、验证与原有 Loss

- `trace/replay/sampled_pairs`：实际采样条数，不能等同于优化次数。
- `trace/replay/sample_unique_pair_rate`：采样的不同原始 Pair 数 / 采样条数；原始身份含出生 step，不代表动作语义已去重。
- `trace/replay/sample_age_p90`：采样 Pair 年龄的 90 分位数，单位训练 step。
- `trace/replay/slic_updated`：本 step 实际完成 SLIC 更新的 0/1 标记。
- 原 `replay/buffer_size/n_dpo_batch`、`dpo_loss/sft_loss/calibration_loss/quality_gap/rewards/accuracy` 保留。后两者表示拟合偏好标签的程度，不等于任务成功率。
- `rewards/chosen/rejected/margin` 使用整段概率，长响应易下溢，需要结合 per-token gap、损失分项及梯度解释。
- `val/queued_rollouts/completed_rollouts/valid_rollouts`、`completion_rate/valid_rollout_rate` 保留；两种比例分母均为排队轨迹数。`val/reward` 不把未返回轨迹自动记失败。

本轮不修改 SLIC Loss、Replay 更新强度、GRPO 优势算法及配置。CPU 回归不能替代 GPU/NPU 训练收益验证；小状态距离与大信用差仍需独立状态抽查和验证集收益确认。
