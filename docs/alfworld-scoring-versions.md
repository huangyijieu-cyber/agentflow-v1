# ALFWorld v2a / v2b 评分版本

本页命名仅用于 ALFWorld。仓库已有的 `v2a`、`v2b` 是旧 QA 标签，保持原样；ALFWorld 使用独立标签 `alfworld-v2a`、`alfworld-v2b`。

| 项目 | ALFWorld v2a | ALFWorld v2b |
| --- | --- | --- |
| 版本位置 | `alfworld-v2a`，提交 `3f7abe9` | `alfworld-v2b`，当前实现 |
| `result` | 保存原 PDDL 成功条件 | 复用原值与哈希，不重转 |
| 成功条件位置 | 固定 teacher-forcing Response | Prompt 中明确标为 requirements |
| 固定评分 Response | 每题不同的完整成功状态陈述 | 所有题统一的未来成功句 |
| 指令 | 描述当前状态与目标的关系 | 预测剩余预算内最终是否成功 |
| 剩余动作预算 | 未显式输入 | 使用原模拟器与外层上限的较小值，逐动作递减 |
| 完整动作/观察历史 | 不输入 | 不输入 |
| QA、actor、rollout 组装、信用公式、配对及损失 | 原实现 | 保持原实现 |

## v2b 实际评分格式

```text
Task: {task}

Success conditions (requirements, not observed facts):
{原 Parquet result，完整保留}

Scoring view: FULL ENVIRONMENT STATE ...

Applicable environment rules ...
{原 PDDL 动作前提与效果}

{事实语义说明}

Initial state (before this episode's first action):
{完整初始事实}

{当前状态由初始事实与下列差分共同确定的说明}

Currently true facts added to the initial state:
{新增事实}

Initial facts that are now false:
{删除事实}

Current observation (additional textual evidence):
{当前观察}

The desired goal is not an observed achievement. A true initial flag does not prove an action happened during this episode.

Remaining action budget:
{剩余动作次数}

Predict whether the agent will satisfy all success conditions before the remaining action budget is exhausted.
Base your prediction on the verified current state, the actions still required, and the applicable environment rules.
Conditions that are not satisfied yet may be achieved through subsequent actions. Do not assume that success is guaranteed or that future actions will be perfect.
If all success conditions already hold, count the task as completed; no further actions are required.
State your prediction about the final outcome of this episode.
```

固定 Response：

```text
The agent will successfully complete the task within the remaining action budget.
```

该 Response 不包含 turn 编号或预算数值，每个状态使用完全相同的 token 目标。Prompt 经 reference tokenizer 的聊天模板编码，温度为 1，仅平均 Response tokens 的 LogP。共享前缀评分与原 reference fallback 使用同一份文本和目标。

## 数据与升级

`trace_gt` 仍保留原成功条件，daemon 继续核验其与 Parquet `result` 一致。`trace_goal_metadata` 增加 `scoring_version=alfworld-v2b-future-success-v1` 与固定 `scoring_response`，通过既有元数据链传递。trainer 校验版本后，仅替换 ALFWorld 实际评分目标；QA 仍评分原正确答案。

已有 Parquet、SQLite 和原 PDDL 文件无需重新生成。更新后重启训练及 rollout 进程；旧 v2a rollout 不能直接混入 v2b。Probe 允许复用保存的完整事实、规则及经核验的实际剩余预算重建新 Prompt，不重新采样、不覆盖原轨迹。旧版本仅按外层 sampling.max_steps 计算的预算可能高于环境实际上限，缺少真实预算依据时需要 `enrich --replay` 重放原动作确认，而非重新采样。已有 Probe 自定义 Prompt/Response 保留，需要在配置窗口选择新预测指令与固定 Response 才是在测试 v2b。

评分 Prompt 不读取未来奖励或最终 `won` 标签；这些信息仍只用于状态核验与原有终局信用回填。未来成功句的 LogP 是模型判断的代理信号，尚未校准为实际成功概率，也不保证每个有效准备动作都获得正信用。
