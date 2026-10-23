# TRACE-SLIC 训练观察指标

以下指标已接入原训练 metrics 字典，默认 console logger 会输出，无需另开开关。所有比例均为 0–1。分布没有有效样本时只输出 count，不伪造 0 分位数；应连同 count 看图。

## 优先观察

| 指标 | 分母与含义 | 如何解读 |
| --- | --- | --- |
| `val/reward`、`val/completion_rate` | 有非空 triplet 的已完成验证轨迹 reward 均值；完成轨迹数 / 排队轨迹数 | 主要效果指标。只有二值 0/1 reward 时 reward 均值才是成功率；完成率变化会造成样本选择偏差 |
| `trace/score_coverage` | 有效 GT 平均 LogP 的 Turn 数 / 全部保留 Turn 数 | 通常应较高；下降时看 missing_gt、missing_state、invalid_gt_length、overlong_state、scoring_invalid_result_turns |
| `trace/credit_coverage` | 有有限信用的非末 Turn 数 / 保留的非末 Turn 数 | 天然没有后继的末 Turn 已排除；下降通常意味着缺初态、缺评分或原 Turn 编号断开 |
| `trace/credit_p10`、`trace/credit_p50`、`trace/credit_p90` | 所有有限信用的分位数 | 看信号幅度和正负两端；均值可能因正负抵消而接近 0，不能单看均值 |
| `trace/pair/positive_rate`、`trace/pair/negative_rate`、`trace/pair/neutral_rate` | 通过有效性、长度非空及数值检查的候选行中，各信用类别的比例 | 不要求正负各半。长期没有负信用时，严格配对自然不会产生 Pair，应核对状态、分布与阈值 |
| `trace/pair/both_sign_group_rate` | 同时有正、负候选的同题同类型组 / 有有效候选行的组 | 全局正负都有，但此值低，说明同一问题和类型内缺少可比较的两侧 |
| `trace/pair/value_match_rate` | 通过价值匹配的正负组合 / 同题同类型的全部正负组合 | 低时优先检查价值匹配阈值是否过紧、进度分布是否重叠 |
| `trace/pair/selected_unique`、`trace/pair/question_coverage` | 每步最终原始 Pair 数；被选中 Pair 覆盖的问题数 / 本批已有 Turn 的不同 uid 数 | 看训练数据是否持续供给、是否只覆盖少数问题；计数发生在分布式补齐之前，未返回轨迹不在问题覆盖率分母中 |
| `trace/pair/value_gap_p90`、`trace/pair/credit_gap_min` | 最终 Pair 的执行前价值差 P90、信用差最小值 | 应分别不超过 match_epsilon、不低于 gap_epsilon；用于核对配对约束 |
| `trace/pair/cross_turn_rate`、`trace/pair/cross_rollout_rate` | 最终 Pair 中两端 turn 序号不同、rollout 不同的比例 | 确认跨 Turn 配对是否实际发生；不以越高越好作为目标 |
| `trace/replay/sample_unique_pair_rate` | 本次采样的不同原始 Pair 数 / 采样条数 | 区分真实 Pair 多样性与补齐、重复抽样。明显偏低时，batch 很大也可能只是在反复用少数 Pair |
| `trace/step_time_fraction` | TRACE 评分及信用汇总区间耗时 / 整个训练 step 耗时 | 观察该段逻辑的成本；不包含单独 action-reference 前向，也不代表全部新增参考模型成本 |

## 配对筛选漏斗

以下均以 `trace/pair/` 为前缀，是每步绝对计数：

```text
candidates_sign
  -> after_value_match
  -> after_credit_gap
  -> after_length_filter
  -> after_distinct_action
  -> selected_unique
```

各阶段拒绝计数互斥：

- `rejected_value_match`：正负信用都有，但价值差过大。
- `rejected_credit_gap`：价值相近，但信用差不足。
- `rejected_length`：前两项通过，但响应长度比过大。
- `rejected_same_action`：前述通过，但两端为同一动作。
- `rejected_group_cap`：通过全部质量条件后，受组内对数上限限制。
- `rejected_global_cap`：通过全部质量条件后，受每步总对数上限限制。

可核对 `candidates_sign = 所有 rejected 之和 + selected_unique`。`selection_rate` 是 selected_unique / candidates_sign，包含采样上限影响，不能将其低值全部归因于阈值过严。

行与组的分母同时输出：`rows_total`、`rows_eligible`、`positive_rows`、`negative_rows`、`neutral_rows`、`groups_total`、`groups_eligible`、`groups_both_signs`、`groups_no_positive`、`groups_no_negative`。两个 no-sign 组计数可以重叠，例如全中性组同时缺两侧，不能将两者相加当作互斥过滤数。

`padding_duplicates` 单列为分布式对齐增加的条数；原有 `replay/n_new_pairs` 含这些条数，评估真实产量优先用 `trace/pair/selected_unique`。

## 信用是否区别于轨迹最终成败

以下针对最终选中的原始 Pair 计算，分母为 `trace/pair/outcome_pairs_count`：

| 指标 | 含义 |
| --- | --- |
| `trace/pair/chosen_failed_rollout_rate` | Chosen 的最终轨迹 reward <= 0 的比例 |
| `trace/pair/rejected_successful_rollout_rate` | Rejected 的最终轨迹 reward > 0 的比例 |
| `trace/pair/same_outcome_rate` | 两端轨迹具有相同成功/失败标记的比例 |

这些指标直接观察新信用标签何时区别于轨迹 reward；它们不是因果正确率，也不应以越高越好为目标。reward 只用于指标汇总，不参与 TRACE 标签或选对。这里沿用 reward > 0 的成功标记；非二值任务需结合实际 reward 语义解读。

`chosen_credit_mean`、`rejected_credit_mean`、`credit_gap_p50/p90/min`、`value_gap_p50/p90/max`、`length_ratio_p90` 用于看最终 Pair 的质量分布；`rows_unique`、`chosen_rows_unique`、`rejected_rows_unique`、`questions_covered` 用于看样本重复程度。

## 按决策类型拆分

`trace/{type}/credit_count`、`credit_mean`、`credit_p10/p50/p90`、`positive_rate`、`negative_rate`、`neutral_rate` 按实际出现的 analysis、tool_plan、final_answer 等类型输出。这里的比例分母是该类型的全部有限信用，不做动作有效性过滤；全局 `trace/positive_rate` 等也使用这一口径。进入 Pair 候选后的比例应看 `trace/pair/*_rate`。

`trace/pair/{type}/selected_unique` 列出每种类型贡献的 Pair 数。单独生成的最终回答通常没有下一状态，信用缺失，因此 final_answer 无 Pair 是当前定义的正常边界。类型分布变化本身也可能影响总体信用分位数。

## 实际训练与成本

- `trace/replay/slic_updated`：本 step 是否完成一次 SLIC worker 更新，0 或 1。
- `trace/replay/sampled_pairs`、`unique_sampled_pairs`：实际传给 SLIC 的 Pair 条数及不同原始 Pair 数；不是 PPO epoch 展开后的优化次数。
- `trace/replay/sample_age_p50/p90/max`：采样 Pair 的年龄，单位为训练 step，定义为 current_step - born_step。
- `trace/replay/buffer_unique_pairs`：缓存中的不同原始 Pair 数。去重身份包含问题、双方 rollout/turn 和出生 step，因此能识别分布式补齐产生的重复条目。
- 原有 `replay/buffer_size`、`replay/n_dpo_batch` 保留；队列非空和采样条数较大不等于样本多样。
- `trace/scoring_seconds`：GT 参考模型调用的墙钟耗时，包含 RPC 等待；`trace/total_seconds` 还包含请求构造与信用汇总。
- `trace/scoring_requests`、`scoring_dedup_rate`：去重后评分请求数及复用比例；`scoring_prompt_tokens`、`scoring_gt_tokens` 是去重后、分布式补齐前的逻辑 token 数，不是硬件实际计算量。
- `trace/turns_total`、`gt_logp_count`、`value_count`、`credit_count`、`credit_expected_turns`、`credit_missing_nonterminal`、`terminal_turns` 提供覆盖率分母。这里末 Turn 指每条 Rollout 当前可见的最大原始 Turn 编号，不表示已确认环境正常终止。末 Turn 的 NaN 会进入旧 `unscored_credit_turns`，排障应优先用排除了末 Turn 的 `credit_missing_nonterminal`。

原有 SLIC `quality_gap` 是当前模型 Chosen 平均 token LogP 减去 Rejected 的差值，`rewards/accuracy` 是该差值大于 0 的 Pair 比例。它们衡量模型对标签的拟合，不能替代任务成功率。`rewards/chosen/rejected/margin` 使用整段概率，长响应容易下溢，优先看 quality_gap、calibration_loss 与 rewards/accuracy。

验证时同时保留 `val/queued_rollouts`、`completed_rollouts`、`valid_rollouts`、`completion_rate`、`valid_rollout_rate`；完成数可含空轨迹，有效数要求非空 triplet，两种比例的分母都是排队轨迹数。`val/reward` 不将未完成轨迹自动记为失败，因此不是全量任务成功率。`val/turn_count` 是可训练 triplet 数，不是工具调用次数；结合 `val/sum_response_length` 观察交互长度和生成成本。

## 建议的排查顺序

1. Pair 长期为 0：先看 score_coverage、credit_coverage，再看候选 negative_rate 和 both_sign_group_rate。
2. 两侧都有但仍无 Pair：沿漏斗找首个大量减少的阶段；先确认数据分布，再考虑该阶段的阈值。
3. Pair 足够但没有训练改善：看 slic_updated、采样唯一率和年龄，再结合 quality_gap 与验证 reward 判断标签拟合是否转化为任务收益。
4. 训练 reward 上升但验证不升：确认验证完成率与格式提取率稳定，再检查 Pair 的问题覆盖和类型分布。

当前指标通过 CPU 回归校验分母、筛选守恒、补齐去重及采样不变性；尚未进行真实 GPU/NPU 训练观测。
