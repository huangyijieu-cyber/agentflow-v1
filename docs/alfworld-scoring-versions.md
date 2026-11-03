# ALFWorld v2a / v2b 评分版本

当前 `alfworld` 分支使用 **v2a 评分＋先 V 匹配再排序轮转**：使用 Parquet `result` 的成功状态描述评分；当前 full-state Prompt 新增 actor 原生最近历史窗口（默认 4 条），不是累计历史。`k_pairs_per_group=None`，同题可跨 turn。`41f0552` 恢复全组合候选，`ad412f6` 应用先轮转后检查 V；此前已按用户纠正，先得到每个 Chosen 的有序 V 兼容负例列表，再在相同列表对应的池内严格使用 `pos[k] -> neg[k % len(neg)]`，每池独立游标；不同列表即使重叠也不合并。`websearch` 同步修正配对顺序，历史实验定义和标签保留，不新增版本名。新开关 `TRACE_PAIR_ROUND_ROBIN`（`data.trace.pair_round_robin`）默认 true，保持上述轮转；false 恢复全部正负组合，评分文本及其它过滤、额度不变。它与控制分组和组内 cap 的 `TRACE_K_PAIRS_PER_GROUP` 独立。具体分配与过滤规则见 [SLIC 配对](trace-slic.md)。

本页命名仅用于 ALFWorld。仓库已有的裸 `v2a`、`v2b` 是旧 QA 标签。历史 `alfworld/v2a`、`alfworld/v2b` 及 `alfworld-v2a`、`alfworld-v2b` 标签保持原位置；旧归档 v2a/v2b 快照曾包含默认 cap=2 和排序轮转，与用户确认的实验配对定义不同。当前恢复版本请使用 `alfworld` 分支，详见 [版本归档](../ALFWORLD_VERSIONS.md)。

下表保留历史 v2a/v2b 定义；原 v2a 标签无历史窗口。当前分支在 v2a Response 基础上新增的 actor 最近窗口见下一节，不另起版本名。此次评分窗口适配只更新 `alfworld`，QA / `websearch` 评分保持原样。

| 项目 | 历史 ALFWorld v2a | 历史 ALFWorld v2b |
| --- | --- | --- |
| 版本位置 | 历史成功条件评分源码 `3f7abe9`，原 v2a 标签 | 历史未来成功句评分实现 |
| `result` | 保存原 PDDL 成功条件 | 复用原值与哈希，不重转 |
| 成功条件位置 | 固定 teacher-forcing Response | Prompt 中明确标为 requirements |
| 固定评分 Response | 每题不同的完整成功状态陈述 | 所有题统一的未来成功句 |
| 指令 | 描述当前状态与目标的关系 | 预测剩余预算内最终是否成功 |
| 剩余动作预算 | 未显式输入 | 使用原模拟器与外层上限的较小值，逐动作递减 |
| 完整动作/观察历史 | 不输入 | 不输入 |
| 此前实验配对定义 | None、同题跨 turn、全组合候选、全步默认 256 对 | 同 v2a；旧归档标签未完全符合此定义 |
| actor、rollout 组装、信用公式及损失 | 本次恢复未另作修改 | 未来成功句评分改动不改变这些公式 |

## 当前分支的评分格式

Prompt 包含任务、环境规则及谓词语义、完整初始事实、当前事实相对初始事实的新增/删除、当前观察，并追加 actor 实际收到的原生最近历史窗口，最后附加以下指令；不输入累计全历史或 v2b 的剩余动作预算：

```text
Based on the applicable environment rules, the initial state, and the current full environment state above, describe the current state in relation to the task goal. Desired goal conditions are not evidence that they currently hold.
```

原生窗口直接复用 `_build_prompt()` 中 `SimpleMemory.fetch(history_length)` 的文本，默认最近 4 条已存记录，自定义窗口则跟随实际设置。每条保存动作前 observation 和解析后的 action，保留原绝对编号及换行，例如：

```text
Recent observations and actions (same history window as the actor):
[Observation 12: '{observation before action 12}', Action 12: '{parsed action 12}']
[Observation 13: '{observation before action 13}', Action 13: '{parsed action 13}']
```

初始化或 history_length<=0 时没有历史段。格式/命令校验失败和外部异常未存 memory 时沿用原窗口，模拟器正常返回但动作无效的记录仍遵循原有写入逻辑。这里不是取 `trace_action_history` 的最后 4 条；后者记录动作后的 observation，与原生窗口的时序不同。actor Prompt、memory 写入及环境初始化均保持原样。

固定 Response 直接读取本题的 Parquet `result`，即确定性转换的完整 PDDL 成功状态陈述。同题全部状态使用同一 Response，仅平均其 tokens 的 teacher-forcing LogP；不将目标原文作为 v2b 的 Success conditions 区块提前放进 Prompt。共享前缀与原 reference fallback 继续使用相同输入和目标。

已有 GT Parquet 无需重新转换。更新后重启训练与 rollout 进程，使用新评分契约的 rollout/replay，避免混入旧信用。新 Probe 缓存保存原生快照和版本；旧缓存缺快照时，只有完整事实仍不足以恢复 actor 实际窗口，需 `enrich --replay` 重放保存动作并核验原 actor Prompt，不重新采样。旧 sidecar 不会自动视为新 full-state；`saved` 自定义实验保留原样。若原自定义 Prompt/Response 来自 v2b，应恢复 `{goal_response}` 和上述当前状态指令。

## 历史 v2b 评分格式

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

## 历史 v2b 的数据与升级行为

`trace_gt` 仍保留原成功条件，daemon 继续核验其与 Parquet `result` 一致。`trace_goal_metadata` 增加 `scoring_version=alfworld-v2b-future-success-v1` 与固定 `scoring_response`，通过既有元数据链传递。trainer 校验版本后，仅替换 ALFWorld 实际评分目标；QA 仍评分原正确答案。

以下描述历史 v2b 实现，不是当前分支的默认行为：已有 Parquet、SQLite 和原 PDDL 文件无需重新生成。切换至该实现后需重启训练及 rollout 进程；旧 v2a rollout 不能直接混入 v2b。该版 Probe 允许复用保存的完整事实、规则及经核验的实际剩余预算重建新 Prompt，不重新采样、不覆盖原轨迹。旧版本仅按外层 sampling.max_steps 计算的预算可能高于环境实际上限，缺少真实预算依据时需要 `enrich --replay` 重放原动作确认，而非重新采样。已有 Probe 自定义 Prompt/Response 保留，需要在配置窗口选择新预测指令与固定 Response 才是在测试 v2b。

评分 Prompt 不读取未来奖励或最终 `won` 标签；这些信息仍只用于状态核验与原有终局信用回填。未来成功句的 LogP 是模型判断的代理信号，尚未校准为实际成功概率，也不保证每个有效准备动作都获得正信用。
