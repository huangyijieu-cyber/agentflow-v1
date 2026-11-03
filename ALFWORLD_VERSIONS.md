# ALFWorld 实验版本归档

当前 `alfworld` 分支沿用 **v2a 的成功状态 Response**，评分 Prompt 新增与 actor 完全相同的原生最近历史窗口（默认 4 条），默认配对仍为先 V 匹配再排序轮转。固定 Response 为 Parquet `result` 中的完整成功状态描述；`TRACE_K_PAIRS_PER_GROUP=None`，同题允许跨 turn、无组内额度，保留全步默认 256 对上限。

`41f0552` 先恢复 v2a 评分、None 跨 turn 和全组合候选，`ad412f6` 随后应用先分配负例再检查 V 的轮转。此后已按用户纠正，将 `alfworld` 和 `websearch` 两分支的配对顺序都改为先 V 匹配再轮转。ALFWorld 的成功条件 Response 路径沿用 `3f7abe9`，保留 `ce3e972` 的独立测试夹具修复；历史 v2b 及已有标签保持原位置。

先匹配再轮转：每组 Chosen 按偏好分降序、同分 index 升序；每个 Chosen 先找出满足 `abs(V_chosen - V_negative) <= match_epsilon` 的兼容负例。`match_epsilon=0` 要求精确相等，正值使用真实数值邻域，不四舍五入或传递合并。兼容负例按偏好分升序、index 升序排列，并以其有序索引元组标识组内兼容池。同池 Chosen 按上述顺序从 0 局部编号 k，严格分配 `compatible[k % len(compatible)]`；每池游标独立，分配后即前进，后续质量过滤失败也不回退或补选。epsilon=0 时是各精确 V 池；epsilon>0 时相同兼容列表共享游标，不同列表即使重叠也独立轮转，不传递合并。每个 Chosen 最多提案一次，无兼容池才计 V 拒绝；信用差、长度、动作不同等筛选不变。`outcome_match_value=false` 的 outcome 路线仍跳过 V 匹配，在组内完整负例池使用一个游标。本次不修改该开关、阈值、后续优先级及数量上限。

`TRACE_PAIR_ROUND_ROBIN`（`data.trace.pair_round_robin`）默认 `true`，保持上述实现；设为 `false` 则枚举原分组的全部正负组合，同一个 Chosen 可匹配多个不同负例。两种模式使用相同 V 门槛、正负信用、质量过滤和额度。该开关与 `TRACE_K_PAIRS_PER_GROUP` 独立：None 仍跨 turn、无组内 cap；2 仍同 turn、每组最多 2 对。两任务分支均支持此开关，不新增版本名或移动历史标签。

本次仅在 `alfworld` 分支为 full-state 评分追加 actor 原生历史快照，QA / `websearch` 的评分不改。快照直接来自 `_build_prompt()` 中 `SimpleMemory.fetch(history_length)` 已用于 actor 的文本：每条是动作前 observation 与解析后的 action，保留原编号和换行。默认取最近 4 条实际存入 memory 的记录，自定义 history_length 则严格跟随原窗口；初始化或 history_length<=0 时为空。格式/命令校验失败、外部异常未写 memory 时沿用原窗口，模拟器正常返回但动作未生效的记录仍按原逻辑存入。该适配不改变 actor Prompt、memory 写入、执行步数或环境初始化，也不从累计 `trace_action_history` 截取动作后观察来替代。原 v2a 标签仍无此窗口，本次不新增版本名或移动标签。

## 用户此前确认的实验定义（本次评分窗口及轮转调整之前）

| 版本 | 评分变化 | 配对设置 |
| --- | --- | --- |
| v0 | 引入 TRACE 之前，按最终环境奖励构造偏好对 | 原实现 |
| v1a | 实际 PDDL 转固定 GT；完整环境状态和累计历史评分，计算过程信用 | 同题同原始 turn，最多 2 对/turn |
| v1b | 同 v1a 评分 | `None`，同题可跨 turn，无组内额度 |
| v2a | 移除累计评分历史；Response 仍是完整成功状态 | 同 v1b；全组合枚举，未应用排序轮转 |
| v2b | 成功条件进入 Prompt；Response 改为固定未来成功句，输入剩余预算 | 实验定义同 v1b；未应用排序轮转 |

各 TRACE 版本仍受全步默认 256 对限制及价值、信用、长度、动作不同等质量筛选。v1a/v1b 的候选也使用全组合枚举。v2a/v2b 是评分方式的区别，详细模板见 [评分版本说明](docs/alfworld-scoring-versions.md)。

## 已有历史标签

| 标签 | 原源码来源 | 标签内的实际行为 |
| --- | --- | --- |
| `alfworld/v0` | `8083f0d` | 原始奖励配对，未引入 TRACE |
| `alfworld/v1a` | `f57dd66` | 完整状态与累计历史，同 turn 默认 2 对，全组合枚举 |
| `alfworld/v1b` | `f57dd66` 加默认配置调整 | 同题跨 turn，默认 None，全组合枚举 |
| `alfworld/v2a` | `3f7abe9` | 无累计评分历史；旧快照默认 2 且含排序轮转 |
| `alfworld/v2b` | `ce3e972` | 未来成功句评分；旧快照默认 2 且含排序轮转 |

旧 v2a/v2b 快照与用户确认的实验配对定义存在上述差异，因此当前恢复版本应从 `alfworld` 分支获取，不能只检出旧 `alfworld/v2a` 标签。原 `alfworld-v2a` / `alfworld-v2b` 及旧 QA 标签也保留原位置。

## 使用当前 v2a 评分及排序轮转

在工作区无未提交改动时执行：

```bash
git fetch origin --prune --tags
git switch alfworld
git pull --ff-only origin alfworld
export TRACE_K_PAIRS_PER_GROUP=None
export TRACE_PAIR_ROUND_ROBIN=true   # 默认：先 V 匹配再轮转
# export TRACE_PAIR_ROUND_ROBIN=false  # 全部正负组合
```

显式设置环境变量可覆盖旧启动脚本或 shell 中残留的 `2`；同时检查启动脚本是否随后重新赋值。三份训练配置的默认值现为 None/null；底层 `TraceConfig()` 的库调用默认仍为 2，可显式传入 None 切换模式。

已有含 GT 的 Parquet 无需重新转换。更新后重启训练及 rollout 进程，使用新评分契约的 rollout/replay 数据，避免混入旧无窗口、累计历史或 v2b 的评分信用。新 Probe 缓存保存原生历史快照及版本；旧缓存即使有完整事实，缺少该快照时仍需 `enrich --replay` 重放已保存动作并核验原 actor Prompt，随后重评分，不重新采样。旧 sidecar 不能冒充新 full-state 契约；`saved` 自定义实验保留原样。检查原自定义 Prompt/Response 是否仍为 v2b 设置。GT/result、信用公式、轮转开关、额度及基础 GRPO/GiGPO 训练逻辑保持不变。
