# TRACE-SLIC 修改记录

记录日期：2026-10-26。

先前算法待办已完成实现及 CPU 集成回归，提交 `6eb6215` 已推送到 `trace` 分支。后续指标精简于 2026-10-27 纳入 `trace` 分支，具体版本见 Git 提交记录；GPU/NPU 端训练效果尚未验证。

## 2026-10-27：v2_b / trace-2b

原 v2 命名为 v2_a。基于 `5fbe4cc` 的 `trace-2b` 新增 IGPO 风格工具优势：LogP 增量、IG 与终局奖励分别按同题标准化，再全未来折扣累计；替换旧 K=3 与 terminal fill，不重复计算终局奖励。analysis／实际 final 仍按可信终局结果配对。

默认 `pair_score_mode=igpo`，设为 `trace_fill` 可恢复 v2_a。保留状态匹配 margin=0.05、两侧质量过滤与原 GRPO／SLIC Loss／Replay 设置，新增全未来故障隔离及独立优势阈值。179 项 TRACE 和 8 项原生 Token IDs CPU 回归通过，未验证 GPU/NPU 训练收益。完整公式、参数与指标见 [v2_b 说明](trace-2b.md)及[设计方案](trace-2b-design.md)。下文保留 v2_a 的历史记录。

## 设计原则

- 配对质量优先于数量和覆盖；没有可靠正负两侧时跳过，不为凑数放宽条件。
- 小价值距离、大信用差只是代理信号，不能独立证明状态可比或动作的因果贡献。
- analysis 与实际最终回答使用可信最终任务结果；工具使用局部信用与符合条件的终局回填。
- 外部故障不作为模型正负样本，其影响不能沿 K-step 窗口传播。

## 状态概览

| 修改项 | 状态 |
| --- | --- |
| 状态匹配 margin 由 0.1 调整为 0.05 | 已改：实现及回归完成 |
| 通过匹配门槛后，优先选择信用差最大的 Pair | 已改：实现及回归完成 |
| analysis 按可信最终可验证奖励配对 | 已改：实现及回归完成 |
| 实际最终回答按可信最终可验证奖励配对 | 已改：实现及回归完成 |
| 工具信用 K=3，final 计入总窗口 | 已改：实现及回归完成 |
| 窗口触及真实 final 时回填终局优势，排除 analysis | 已改：实现及回归完成 |
| 外部故障分类、双侧过滤及 K-step 窗口隔离 | 已改：实现及回归完成 |
| verifier／终局可信性与普通、profiling 数据链路 | 已改：实现及回归完成 |
| 分来源、局部／终局／有效信用及故障过滤指标 | 已改：实现及回归完成 |

## 此前已完成

### margin=0.05

- [x] Python 默认值、基础 YAML 和两份启动 YAML 同步 `match_epsilon=0.05`。
- [x] `epsilon=0.1`、`gap_epsilon=0.02` 保持原值，它们与状态匹配 margin 是不同参数。

同初态时，该匹配距离限制平滑 NLL 较大值不超过较小值的 `exp(0.05)≈1.051` 倍。用户已选择先试 0.05；这是实验值，不是已证明的质量最优值。steps 10–20 旧数据使用 K=1、无回填，不能直接为新候选分布定论。

### 信用差最大优先

- [x] 筛选条件通过后，使用 `(-credit_gap, value_gap, old_logp_gap, chosen_index, rejected_index)` 排序。
- [x] 保留价值匹配和质量条件，补充排序测试。

工具路线采用有效信用差，奖励路线采用最终 reward 差；各组内均优先选择差值大的 Pair。候选统一排序后应用全局上限，尚未对两种差值做跨来源尺度校准；若 `rejected_global_cap>0`，需同时观察两种来源的选中占比。

## 本轮已完成的实现与回归

### analysis／最终回答使用奖励路线

- [x] analysis 不再用“生成分析前后 GT 增长”确定正负，也不接受 K-step 或终局回填。
- [x] 实际用于验证的最终回答按可信成功／失败配对，覆盖 `final_answer`、`direct`、`base` 的真实末端回答。
- [x] 最终回答没有后继状态也可配对，不虚构末端 GT 状态。
- [x] 默认 `outcome_match_value=false`，奖励路线不要求 GT value／credit 有效，不施加工具的正负信用阈值或 `gap_epsilon`。
- [x] 仍要求同题、同类型、同标签来源、动作及响应有效、长度比和不同动作等条件；按最终 reward 差优先选择。
- [x] 优先采用明确 verifier 成功标记；二值 0/1 的 0 是失败类，正格式奖励不能冒充成功。数值兼容阈值配置为 `outcome_success_threshold=0.5`。

### 工具 K=3 与终局回填

- [x] 将默认 K 从 1 改为 3，gamma=0.8，局部增益采用权重归一化。
- [x] K 包含当前实际 Turn，final 本身占一个位置。因此 K=3 的终局回填最多覆盖 final 前两个工具 Turn，不把 K 解释成“当前以外再三步”或“只计三个工具转移”。
- [x] 工具局部增益仍使用下一真实 Turn 的动作前状态；final 无 GT 后状态，不补造 `delta_final`。
- [x] 窗口覆盖实际完成且验证可信的 final 才回填，不能以最后可见 Turn、最后工具 Turn或截断点代替。
- [x] 工具有效信用为局部 K-step 信用加 `lambda * gamma**(final_index-current_index) * A_out`，按有效信用严格正负和信用差选对。
- [x] 终局项仅在可信终局的同题 Rollout 中去重统计，复用 GRPO 的中心化、是否标准差归一化及单例组特例等计算口径。统计人口可能与原 GRPO batch 不同，优势数值不保证相同；不直接把原始 R 加到 credit，不回写原 GRPO 优势。
- [x] 新增配置 `terminal_fill_enabled=true`、`terminal_fill_scale=2.0`，并透传两份启动 YAML。
- [x] 不将该信用回写到原 GRPO 训练优势；SLIC Loss 保持现有实现。

计数例：`analysis → A → B → C → final`，K=3 时 A 的窗口是 A/B/C，无终局回填；B 的窗口 B/C/final，回填折扣 gamma²；C 的窗口 C/final，折扣 gamma。

### 外部故障与可信终局

- [x] 区分网络请求超时、DNS／连接错误、限流与网关服务故障，模型非法调用，以及正常无搜索结果。
- [x] 覆盖 Google `Google Search tried ... but failed. Last error: ...`、Base Generator `Error generating response: ...` 和 Wikipedia 页面结构中的请求错误；不得扫描普通网页正文中的 error 词。
- [x] 最终未恢复的外部故障 Turn 禁止进入 Chosen 和 Rejected；内部重试后成功按最终状态判断。
- [x] 包含故障转移的信用窗口标为不可用；不把故障 delta 置 0、不删掉故障后拼接不相邻状态。
- [x] verifier 调用或解析失败、外部中断且没有可靠最终验证结果时，不能创建 analysis／final 奖励标签或 terminal fill。
- [x] 普通与 profiling Rollout 均透传外部故障、实际 final、终局有效性及 verifier 成功标记。
- [x] 保留现有 GRPO 奖励路径；本轮不将外部中断自动改写成其它 GRPO 奖励。

### 指标与验证（按精简后口径）

- [x] 分开输出 local_credit、terminal_fill、credit；`credit` 即有效信用，不再重复输出 effective_credit，analysis／final 不伪装 GT 信用。
- [x] 区分 trace／outcome Pair 数、工具匹配率与工具信用／终局结果的关系；奖励路线的固定奖励差和定义恒定比例已删除。
- [x] 明确工具信用覆盖率分母，排除天然不使用 GT 信用的 analysis／final。
- [x] 输出外部故障、受影响窗口、不可信终局与回填触发计数。
- [x] CPU 集成回归覆盖类型分流、K 计数、final 触发、故障隔离、正常无结果、重试成功、verifier 失败和两入口元数据。
- [x] 核对 3 份 YAML、新环境变量及 Python 默认值一致性。

## 本轮范围之外

SLIC 累计 SFT 的长度尺度、Replay 的 16 倍采样、跨 Prompt 似然校正等属于诊断建议，不在当前待办实施范围。本轮不更改 Loss、Replay 更新强度或原 GRPO 优势算法。

## 指标精简（2026-10-27 提交）

只精简日志及对应测试，不修改状态价值、信用、Pair 筛选和排序、SLIC Loss、GRPO 优势、Replay 采样或配置。小价值距离、大信用差仍只是代理信号，不代表已经证明状态可比。

| 删除的指标或指标族 | 原因与替代观察项 |
| --- | --- |
| `trace/effective_credit_*` | 与 `trace/credit_*` 完全相同，统一保留后者 |
| `trace/{type}/credit_*`、按类型的正／负／中性计数及比例 | 仅工具使用 GT 信用；重复总体工具统计，analysis／final 的空统计没有诊断价值 |
| `trace/value_*` | 不同问题和初态的总体分布难以判断可比性；保留最终工具 Pair 的 `value_gap_*` |
| `trace/gt_logp_{std,p10,p50,p90}` | 保留 GT LogP 数量和均值确认评分水平；匹配质量直接看 Pair 距离 |
| `trace/{credit,local_credit,terminal_fill}_std`、`trace/credit_abs_*` | 已有三组信用的均值与分位数，删除重复幅度统计 |
| `trace/{positive,negative,neutral}_turns`、`trace/neutral_rate` | 保留有限信用数与正／负比例；中性比例可推导 |
| `trace/label_source/{source}/turns` | 输入行构成不等于实际监督构成；保留分来源选中 Pair 数 |
| `trace/unscored_credit_turns`、`trace/credit_missing_nonterminal` | 前者混入天然无信用的 analysis／final；覆盖率与预期工具行数已足够解释缺失 |
| `trace/terminal_turns`、`trace/outcome_invalid_turns` | 最后可见 Turn 不一定是真实 final，final 后附加输出也会无标签；改看实际边界与无可信结果的 Rollout 数 |
| `trace/unique_pairs` | 重复 `trace/pair/selected_unique` |
| `trace/pair/after_*` 累计通过计数 | 可由候选数与六种互斥拒绝计数还原；工具匹配内部计数只用于计算比例，不外发 |
| `trace/pair/{rows_total,rows_eligible,groups_total,groups_eligible,groups_both_signs}` | 保留缺正／缺负组数、`both_sign_group_rate` 及实际 Pair 数；必要分母留作内部统计 |
| `trace/pair/{positive,negative,neutral}_{rows,rate}` | 混合工具信用标签与终局奖励标签，不能解释为统一 GT 信用分布 |
| `trace/pair/{trace_candidates_sign,trace_after_value_match,outcome_candidates_sign,outcome_after_value_match}` | 工具计数仅内部派生匹配率；奖励路线默认跳过匹配，重复通过数没有独立信息 |
| `trace/pair/value_match_rate`、`trace/pair/selection_rate` | 混合来源匹配率含奖励 bypass；选择率同时受数量上限影响，保留工具匹配率和拒绝原因 |
| `trace/pair/{chosen_rows_unique,rejected_rows_unique,rows_unique,questions_covered}` | 精简覆盖计数，保留 `question_coverage` 与 Replay 采样多样性 |
| `trace/pair/cross_rollout_rate` | 保留跨原始 Turn 比例描述配对形式；跨 Rollout 比例不是质量目标 |
| `trace/pair/value_gap_count`、`credit_gap_count` | 只统计最终工具 Pair，使用 `trace_selected_unique` 确认样本量 |
| `trace/pair/{chosen_credit,rejected_credit}_*` | 有效信用及信用差分布已覆盖主要诊断，删除重复两侧统计 |
| `trace/pair/length_ratio_*` | 长度比仍有硬过滤并保留 `rejected_length`，无需重复输出通过样本分布 |
| `trace/pair/outcome_reward_gap_*` | 二值任务通常固定为 1，奖励路线直接保留可信成功／失败分流与选中数 |
| 总体 `trace/pair/{outcome_pairs_count,chosen_failed_rollout_rate,rejected_successful_rollout_rate,same_outcome_rate}` | 混合两种标签来源难以解释；只保留 `trace/pair/trace/` 下的工具终局关系 |
| `trace/pair/outcome/{outcome_pairs_count,chosen_failed_rollout_rate,rejected_successful_rollout_rate,same_outcome_rate}`、`label_source_mismatch_pairs` | 前者由奖励配对定义决定，后者是选择逻辑不变量，依赖测试保障 |
| `trace/{deduplicated_states,scoring_dedup_rate,scoring_prompt_tokens,scoring_gt_tokens}` | 删除细碎评分开销统计，保留请求数、评分异常、耗时和 step 占比 |
| `trace/total_seconds` | 已有训练 step 时间及 `step_time_fraction`；单独保留 RPC 耗时定位评分成本 |
| `trace/replay/{unique_sampled_pairs,buffer_unique_pairs,sample_age_p50,sample_age_max}` | 保留采样量、不同 Pair 比例、年龄 p90 与实际更新标记，减少重复统计 |

保留指标的完整定义见 [指标说明](trace-slic-metrics.md)。精简后 TRACE CPU 回归 122 项通过，覆盖保留指标语义、已删指标不再输出、有限数值、空样本、来源隔离、漏斗守恒、信用和配对结果不变。文档变更通过 `git diff --check`。

## 算法提交验证（`6eb6215`）

- TRACE CPU 回归 118 项通过，覆盖状态评分、类型分流、K=3 与 final 计数、可信终局、故障窗口隔离、重试恢复、两入口元数据、Pair 排序和指标统计。
- 原生 token IDs 回归 8 项通过；3 份 YAML、新环境变量及 Python 默认值已核对一致。
- 实际 final 后的可选输出不会使可信 Rollout 被计入 `outcome_invalid_rollouts`；后续指标精简已移除 Turn 级无效计数。
- `git diff --check` 通过；尚无真实 GPU/NPU 训练效果结论。
