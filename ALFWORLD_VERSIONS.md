# ALFWorld 实验版本归档

当前 `alfworld` 分支的正式训练默认使用 **v2a 评分＋先 V 匹配再排序轮转**：评分 Prompt 不含累计动作/观察历史，固定 Response 为 Parquet `result` 中的完整成功状态描述；`TRACE_K_PAIRS_PER_GROUP=None`，同题允许跨 turn、无组内额度，保留全步默认 256 对上限。

`41f0552` 先恢复 v2a 评分、None 跨 turn 和全组合候选，`ad412f6` 随后应用先分配负例再检查 V 的轮转。本次按用户纠正，将 `alfworld` 和 `websearch` 两分支的配对顺序都改为先 V 匹配再轮转。ALFWorld 评分仍来自 `3f7abe9`，保留 `ce3e972` 的独立测试夹具修复；历史 v2b 及已有标签保持原位置。

先匹配再轮转：每组 Chosen 按偏好分降序、同分 index 升序；每个 Chosen 先找出满足 `abs(V_chosen - V_negative) <= match_epsilon` 的兼容负例。`match_epsilon=0` 要求精确相等，正值使用真实数值邻域，不四舍五入或传递合并。兼容负例按偏好分升序、index 升序排列，并以其有序索引元组标识组内兼容池。同池 Chosen 按上述顺序从 0 局部编号 k，严格分配 `compatible[k % len(compatible)]`；每池游标独立，分配后即前进，后续质量过滤失败也不回退或补选。epsilon=0 时是各精确 V 池；epsilon>0 时相同兼容列表共享游标，不同列表即使重叠也独立轮转，不传递合并。每个 Chosen 最多提案一次，无兼容池才计 V 拒绝；信用差、长度、动作不同等筛选不变。`outcome_match_value=false` 的 outcome 路线仍跳过 V 匹配，在组内完整负例池使用一个游标。本次不修改该开关、阈值、后续优先级及数量上限。

`TRACE_PAIR_ROUND_ROBIN`（`data.trace.pair_round_robin`）默认 `true`，保持上述实现；设为 `false` 则枚举原分组的全部正负组合，同一个 Chosen 可匹配多个不同负例。两种模式使用相同 V 门槛、正负信用、质量过滤和额度。该开关与 `TRACE_K_PAIRS_PER_GROUP` 独立：None 仍跨 turn、无组内 cap；2 仍同 turn、每组最多 2 对。两任务分支均支持此开关，不新增版本名或移动历史标签。

## 用户此前确认的实验定义（本次应用排序轮转之前）

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

已有含 GT 的 Parquet 无需重新转换。切换后重启训练及 rollout 进程，并使用新一轮 rollout/replay 数据，避免混入 v2b 的评分上下文、信用或偏好对。Probe 复用保存的动作和完整事实重新构建 v2a 评分上下文；检查其自定义 Prompt/Response 是否仍为 v2b 设置，无需重新采样。策略输入、信用公式及基础 GRPO/GiGPO 训练流程保持本次恢复所用源码的行为。
