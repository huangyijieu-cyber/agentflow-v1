# TRACE-SLIC 修改记录

记录日期：2026-10-26。

先前待办已完成实现及 CPU 集成回归，下表按最终实现更新。修改纳入 `trace` 分支，具体版本见 Git 提交记录；GPU/NPU 端训练效果尚未验证。

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

### 指标与验证

- [x] 分开输出 local_credit、terminal_fill、effective_credit，analysis／final 不伪装 GT 信用。
- [x] 区分 trace／outcome Pair 数、工具匹配率、奖励差分布和终局结果描述。
- [x] 明确工具信用覆盖率分母，排除天然不使用 GT 信用的 analysis／final。
- [x] 输出外部故障、受影响窗口、不可信终局与回填触发计数。
- [x] CPU 集成回归覆盖类型分流、K 计数、final 触发、故障隔离、正常无结果、重试成功、verifier 失败和两入口元数据。
- [x] 核对 3 份 YAML、新环境变量及 Python 默认值一致性。

## 本轮范围之外

SLIC 累计 SFT 的长度尺度、Replay 的 16 倍采样、跨 Prompt 似然校正等属于诊断建议，不在当前待办实施范围。本轮不更改 Loss、Replay 更新强度或原 GRPO 优势算法。

## 本地验证

- TRACE CPU 回归 118 项通过，覆盖状态评分、类型分流、K=3 与 final 计数、可信终局、故障窗口隔离、重试恢复、两入口元数据、Pair 排序和指标统计。
- 原生 token IDs 回归 8 项通过；3 份 YAML、新环境变量及 Python 默认值已核对一致。
- 实际 final 后的可选输出不会使可信 Rollout 被计入 `outcome_invalid_rollouts`；Turn 级无效计数仍逐行保留。
- `git diff --check` 通过；尚无真实 GPU/NPU 训练效果结论。
