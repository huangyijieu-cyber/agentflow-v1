# TRACE-SLIC 训练观察指标

指标接入原训练 metrics 字典，默认 console logger 输出，无需额外开关。比例为 0–1。没有有效观测的分布只输出 count，不生成假均值或分位数。

本轮有两条标签路线：`trace` 为工具局部信用加终局回填，`outcome` 为 analysis／实际最终回答的可验证奖励标签。观察总体统计时必须同时看来源，不能把奖励正负样本伪装成 GT 信用。

## 优先观察

| 指标 | 分母与含义 | 如何判断 |
| --- | --- | --- |
| `val/reward`、`val/completion_rate`、`val/valid_rollout_rate` | 有效已完成验证轨迹的 reward 均值；完成／有效轨迹数除以排队数 | 完成率与有效率稳定时，reward 改善才具有可比性。只有二值奖励时均值可直接理解为成功率 |
| `trace/score_coverage` | 有效 GT LogP Turn 数 / 全部保留 Turn 数 | 应稳定；奖励路线不要求 GT，不能把低评分覆盖直接当成所有 Pair 不可用 |
| `trace/credit_coverage` | 有效信用工具 Turn 数 / 预期计算工具信用的 Turn 数 | 新流程分母通过 tool-only mask 排除 analysis／final；外部故障和评分缺失仍应结合原因计数看 |
| `trace/local_credit_p10/p50/p90` | 不含终局回填的 K-step 工具信用分位数 | 观察纯局部信号，不要求正负各半 |
| `trace/terminal_fill_p10/p50/p90` | 工具终局回填项分位数 | 与 local 信用一起看，检查回填是否主导；零填充值可能来自未触及 final 或终局优势为零 |
| `trace/effective_credit_p10/p50/p90` | 局部信用加终局回填后的有效工具信用 | 工具 Pair 使用这一信号；不能与仅 K=1 的历史信用直接做幅度比较 |
| `trace/pair/trace_selected_unique`、`outcome_selected_unique` | 补齐前各来源的 Pair 数 | 区分工具监督与终局奖励监督；数量只是运行诊断，不作为质量目标 |
| `trace/pair/tool_value_match_rate` | 工具正负候选中通过价值匹配的比例 | 评估工具 margin 的实际筛选；混合来源的全局 value_match_rate 含奖励路线跳过匹配，不宜代替它 |
| `trace/pair/value_gap_p90/max`、`credit_gap_min` | 选中工具 Pair 的动作前价值差、有效信用差 | 应符合工具 match_epsilon、gap_epsilon；小距离或大差距不独立证明真实可比性 |
| `trace/pair/question_coverage` | Pair 覆盖的问题数 / 本批已有 Turn 的问题数 | 仅描述覆盖，不为提高覆盖放宽可靠性条件；未返回任何 Turn 的问题不在分母 |
| `trace/external_error_turns`、`trace/credit_windows_blocked_external` | 外部故障 Turn 数、因故障阻断的信用窗口数 | 用于区分基础设施问题与模型质量，不能把过滤量上升直接当策略退化 |
| `trace/outcome_invalid_rollouts` | 没有可信终局结果的 Rollout 数 | 增长时检查外部中断、实际 final 与 verifier 完成情况；不应自动转为失败负样本 |
| `trace/replay/sample_unique_pair_rate` | 采样的不同原始 Pair 数 / 采样条数 | 区分真实多样性与分布式补齐／重复抽样 |
| `trace/replay/slic_updated` | 本 step 是否完成 SLIC worker 更新 | 计划更新且有可靠数据时应为 1；没有合格 Pair 时跳过是允许行为 |

## 状态、信用与终局信号

分布后缀：`count` 为有效有限样本数；`mean/std` 为均值和标准差；`p10/p50/p90` 为分位数。所有分位数应与 count 一起解释。

- `trace/gt_logp_{count,mean,std,p10,p50,p90}`：有效且不大于零的 GT 平均 Token LogP。
- `trace/value_{count,mean,std,p10,p50,p90}`：相对各自初态的 GT 状态价值。
- `trace/credit_{count,mean,std,p10,p50,p90}`：有效工具信用，含已启用的终局回填；与 `effective_credit` 在新流程中对应。
- `trace/local_credit_{count,mean,std,p10,p50,p90}`：未回填的 K-step 工具信用。
- `trace/terminal_fill_{count,mean,std,p10,p50,p90}`：工具终局优势回填项；`terminal_fill_nonzero_turns` 是非零有限项的数量，不能代表所有触及终局的窗口数。
- `trace/effective_credit_{count,mean,std,p10,p50,p90}`：回填后的工具信用。
- `trace/credit_abs_{count,p90}`：有效工具信用绝对值幅度；不追求越大越好。
- `trace/{positive,negative,neutral}_{turns,rate}`：全部有限工具信用按严格阈值分类，尚未做 Pair 候选的动作质量过滤。
- `trace/{type}/credit_{count,mean,p10,p50,p90}` 及三类 `turns/rate`：按决策类型拆分。analysis／final 采用奖励标签，其 GT credit count 为 0 是正常的。
- `trace/label_source/{source}/turns`：来源行数，供观察工具与奖励监督构成。

覆盖率分母：

- `trace/turns_total`：本批全部保留 Turn 数。
- `trace/terminal_turns`：每条 Rollout 最后可见 Turn 数；这个旧名称不表示确认环境正常终止。
- `trace/credit_expected_turns`：新流程中预期计算工具信用的行数，排除奖励路线。
- `trace/credit_missing_nonterminal`：预期工具行中未得到有效信用的数量。
- `trace/unscored_credit_turns`：全部信用缺失行数，包含本就用奖励标签的 analysis／final，不应追求为零。
- `trace/outcome_invalid_turns`：没有可信终局标签的行数，逐 Turn 统计；实际 final 后的可选输出也计入。`trace/outcome_invalid_rollouts`：整条 Rollout 没有任何可信终局标签行时才计入；可信 final 之后的无标签输出不会把该 Rollout 判为无效。
- `trace/credit_windows_blocked`：所有原因导致不可用的信用窗口数，例如外部故障、编号缺口或所需状态缺失；`trace/credit_windows_blocked_external` 只计外部故障原因。两者不能混用。

正常的成功／失败验证都会产生可信结果；verifier 调用或解析失败、外部中断等才是不可信。应优先使用明确的 verifier success 标记，不能根据正的格式奖励推断任务成功。

终局回填的优势均值／标准差仅来自可信终局 Rollout 的去重统计。该人口可能与原 GRPO batch 不同；复用中心化与归一化口径不表示两者优势数值相同，原 GRPO 优势未被回写。

## 评分准备与成本

`trace/missing_gt`、`missing_state`、`invalid_gt_length`、`overlong_state` 是按顺序检查的首个拒绝原因，彼此互斥。超长状态和 GT 跳过，不截断。`scoring_invalid_result_turns` 是发出请求后取得无效 LogP 的 Turn 数；共享请求失败可能影响多行。

`scoring_requests` 为去重后请求数；`deduplicated_states` 为复用行数；`scoring_dedup_rate` 为复用行数 / 成功映射行数。`scoring_prompt_tokens`、`scoring_gt_tokens` 是去重后、分布式补齐前的逻辑 token 数。

`scoring_seconds` 是 GT reference RPC 的墙钟时间；`total_seconds` 还包括评分请求构造和信用汇总；`step_time_fraction` 为该区间占训练 step 比例，不包含所有新增参考模型成本，也不包含所有 Pair／Replay 工作。

## 配对筛选漏斗

统一前缀 `trace/pair/`，以下为绝对组合数：

```text
candidates_sign
  -> after_value_match
  -> after_credit_gap
  -> after_length_filter
  -> after_distinct_action
  -> selected_unique
```

奖励路线默认跳过价值匹配和 TRACE 信用差门槛，但仍计入对应通过阶段，以保持漏斗守恒。因此全局 `value_match_rate=after_value_match/candidates_sign` 包含 bypass；看工具匹配应使用 `trace_after_value_match / trace_candidates_sign`，Trainer 额外汇总为 `tool_value_match_rate`。

分来源候选计数：`trace_candidates_sign`、`trace_after_value_match`、`outcome_candidates_sign`、`outcome_after_value_match`。

互斥拒绝原因：

- `rejected_value_match`：需要价值匹配但未通过。
- `rejected_credit_gap`：工具有效信用差不足；不用于奖励路线。
- `rejected_length`：响应长度比过大。
- `rejected_same_action`：两侧动作身份相同。
- `rejected_group_cap`、`rejected_global_cap`：质量条件通过，但受到数量上限限制。

可核对 `candidates_sign = 全部 rejected 之和 + selected_unique`。`selection_rate` 含上限影响，不是质量正确率。排序先按本路线偏好分数差降序，再按价值距离和旧策略平均 Token LogP 距离依次打破平局。候选统一排序后应用全局上限，两种来源尚无尺度校准；若 `rejected_global_cap>0`，需检查来源占比是否受到影响。

行与组计数：`rows_total/rows_eligible`、`positive_rows/negative_rows/neutral_rows`、`groups_total/groups_eligible/groups_both_signs/groups_no_positive/groups_no_negative`。合格行中的正负比例与全局信用分类不同；奖励行按成功／失败分池，不应把其比例解释成 GT 正负信用。两个 no-sign 组计数可以重叠。

`trace/invalid_action_turns`、`duplicate_turns`、`dropped_prompt_turns` 为行级原因，彼此可能重叠。`padding_duplicates` 单列补齐重复数；`trace/unique_pairs` 与 `selected_unique` 均为补齐前数量。原 `replay/n_new_pairs` 可能含补齐，不代表真实多样性。

## 最终 Pair 的来源、质量与结果

- `trace_selected_unique`、`outcome_selected_unique`：各来源选中 Pair 数；`label_source_mismatch_pairs` 应为 0。
- `value_gap_{count,p50,p90,max}`、`credit_gap_{count,p50,p90,min}`、`chosen_credit_{count,mean}`、`rejected_credit_{count,mean}`：仅统计 TRACE 工具 Pair，不把 reward 标签转成假 GT 信用。
- `outcome_reward_gap_{count,p50,p90,min}`：奖励路线两侧最终 reward 差，单位与 TRACE 对数信用不同。
- `length_ratio_{count,p90}`：全部 Pair 响应长度比。
- `questions_covered/question_coverage`、`chosen_rows_unique/rejected_rows_unique/rows_unique`：问题及行覆盖，辅助检查极端样本复用。
- `cross_turn_rate/cross_rollout_rate`：跨原始 Turn 编号／Rollout 比例，不以越高越好为目标。
- `{type}/selected_unique`：各类型贡献的 Pair 数。

`outcome_pairs_count` 是有双方可信成功标记的 Pair 数。`chosen_failed_rollout_rate`、`rejected_successful_rollout_rate`、`same_outcome_rate` 描述局部标签与最终结果的关系，不是因果正确率。新路径优先 verifier 布尔值；无 `label_source` 的旧诊断行仍按历史 reward > 0 解释。

同样的终局描述按来源输出在 `trace/pair/trace/`、`trace/pair/outcome/` 下。奖励路线本来就按成功／失败选对，其不一致率天然接近 0；新增奖励 Pair 会改变整体比率，所以跨版本比较优先看工具来源的描述。

## Replay、验证与原有 Loss

- `trace/replay/sampled_pairs/unique_sampled_pairs/sample_unique_pair_rate`：实际采样量及不同原始 Pair 数，不能等同于优化次数。
- `trace/replay/sample_age_p50/p90/max`：采样年龄，单位训练 step。
- `trace/replay/buffer_unique_pairs`：缓存不同原始 Pair 数；身份包含出生 step，不代表动作语义已去重。
- `trace/replay/slic_updated`：实际完成 SLIC 更新的 0/1 标记。
- 原 `replay/buffer_size/n_dpo_batch`、`dpo_loss/sft_loss/calibration_loss/quality_gap/rewards/accuracy` 保留。后两者是模型拟合偏好标签的程度，不是任务成功率。
- `rewards/chosen/rejected/margin` 使用整段概率，长响应易下溢，优先结合 per-token gap、损失分项与梯度解释。
- `val/queued_rollouts/completed_rollouts/valid_rollouts`、`completion_rate/valid_rollout_rate` 保留；两种率分母都是排队轨迹数。`val/reward` 不把未返回轨迹自动记失败。

本轮未修改现有 SLIC Loss 的累计 SFT 或 Replay 倍率；其尺度和更新强度仍需独立对照。CPU 回归只验证指标分母、来源隔离、故障计数与数值行为，不能替代 GPU/NPU 训练效果验证。
