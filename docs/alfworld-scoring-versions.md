# ALFWorld 评分版本

最近归档实验为 **v2_e**，标签 `alfworld/v2_e`。v2_d/v2_e 均使用 v2_a 的 Parquet `result` 成功状态 Response、`pair_round_robin=false` 全组合；v2_e 相对 v2_d 仅增加 actor 原生最近历史窗口（默认 4 条）。当前工作分支在 v2_e 后，按用户要求改为截取原累计 `trace_action_history` 的可配置窗口，保留动作及执行后观察的旧日志格式；默认 4 条，`None` 为全部、`0` 为空，不跟随 actor 的 `history_length`。未另命名实验版本，既有标签保留原位置。K_PAIRS 默认 None，其它过滤及总额度不变。库默认 true 保留，可显式启用先 V 匹配后的池内严格取模轮转。此次仅更新 ALFWorld，QA/websearch 评分保持原样，新缓存契约为 `alfworld-post-action-history-v2`。

各版本及中间未命名改动见 [版本归档](../ALFWORLD_VERSIONS.md)，配对开关和额度见 [SLIC 配对](trace-slic.md)。原 v2_a/v2_b 标签保留原代码；用户实验口径 K_PAIRS=None 与旧标签默认 2 的差异须区分。历史代码已含旧轮转，不能记作无轮转实现。v2_d 已确认评分沿用 v2_a、无最近 4 步历史、`pair_round_robin=false`，其它过滤及 K 分组/总 cap 不变；该实验尚未固定精确提交或标签，v2_c 仍未定义。

| 项目 | 历史 v2_a | 历史 v2_b | 已归档 v2_e |
| --- | --- | --- | --- |
| `result` | 原 PDDL 成功条件 | 复用原值和哈希 | 复用原值和哈希，无需重转 |
| 目标位置 | 固定 Response | Prompt 的 requirements 区域 | 固定 Response |
| Response | 每题完整成功状态描述 | 统一未来成功句 | 同 v2_a，同题所有 turn 固定 |
| 评分历史 | 无累计历史 | 无累计历史 | actor 原生最近窗口，默认 4 条 |
| 剩余预算 | 未显式输入 | 显式输入 | 未显式输入 |
| 配对代码 | 先轮转再检查 V | 沿用旧轮转 | 实验 false 全组合，与 v2_d 相同；仍可显式开启 true |

## 当前分支评分格式

Prompt 包含任务、环境规则及谓词语义、完整初始事实、当前事实相对初始事实的新增/删除、当前观察，并按 `ALFWORLD_TRACE_HISTORY_LENGTH` 追加原动作日志窗口（默认最近 4 条，`None` 为全部），最后附加以下指令；不输入 v2b 的剩余动作预算：

```text
Based on the applicable environment rules, the initial state, and the current full environment state above, describe the current state in relation to the task goal. Desired goal conditions are not evidence that they currently hold.
```

窗口为 `trace_action_history[-N:]`（N>0）；`None` 使用完整列表，`0` 省略历史区段。每条保留动作、执行后 observation 和旧日志的其它字段，仍按原来的 JSON 格式渲染，例如：

```text
Recorded environment action history:
[{"action": "{action 12}", "environment_last_action": "{environment action 12}", "observation": "{observation after action 12}"}, {"action": "{action 13}", "environment_last_action": "{environment action 13}", "observation": "{observation after action 13}"}]
```

初始状态没有历史段；不足 N 条时保留全部已有记录。该评分窗口独立于 actor 的 `history_length`，包括 `history_length<=0` 的情况。格式/命令校验失败等记录按旧日志规则进入窗口，保留 `executed`、`external_error` 等已有字段，不推断动作成功。外部异常后的未知事实仍不可评分。actor Prompt、memory 写入及环境初始化均保持原样。

固定 Response 直接读取本题的 Parquet `result`，即确定性转换的完整 PDDL 成功状态陈述。同题全部状态使用同一 Response，仅平均其 tokens 的 teacher-forcing LogP；不将目标原文作为 v2b 的 Success conditions 区块提前放进 Prompt。共享前缀与原 reference fallback 继续使用相同输入和目标。

已有 GT Parquet 无需重新转换。更新后重启训练与 rollout 进程，使用新评分契约的 rollout/replay，避免混入旧信用。新 Probe 缓存保存完整动作日志、所选窗口和 `alfworld-post-action-history-v2` 版本；`probe.history_length` 修改后按缓存重建评分上下文。旧 v1 连续窗口可通过索引和重叠校验恢复历史，缺少所需记录时须 `enrich --replay` 重放保存动作并核验原 actor Prompt，不重新采样。旧 actor 原生快照和旧 sidecar 不会自动视为新 full-state；`saved` 自定义实验保留原样。若原自定义 Prompt/Response 来自 v2b，应恢复 `{goal_response}` 和上述当前状态指令。

已归档 v2_e 标签仍使用 actor 原生 `SimpleMemory.fetch(history_length)` 的动作前 observation＋action 窗口，默认 4 条；当前分支替换了这一历史区段，固定 GT 和 LogP → V → K-step 局部信用公式保持不变。之后 ALFWorld 终局回填另改为复用 QA 的同题正负范围动态缩放，保留末三个真实动作的回填位置及距离折扣，忽略固定 scale；该算法改动不修改评分 Prompt/Response，也不移动历史标签。

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
