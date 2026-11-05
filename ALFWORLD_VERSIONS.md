# ALFWorld 实验版本归档

最近已归档实验为 **v2_e**，复现标签 `alfworld/v2_e`。用户已确认 v2_d、v2_e 均为 `pair_round_robin=false` 全组合，v2_e 相对 v2_d 仅增加 actor 原生最近历史窗口（默认 4 条）。功能逻辑基于 `7152e73`，标签记录提交由该功能代码、固化为 false 的三份 ALFWorld 训练配置及补充文档构成。

工作分支曾在 v2_e 后按用户要求，将评分历史改为原累计 `trace_action_history` 的最后 4 条，保留“动作＋执行后观察”的旧日志格式。固定 4 条，不跟随 actor 的 `history_length`；其余评分内容、Response、信用公式和配对配置不变。本次未另命名实验版本，也不移动既有 `alfworld/v2_e` 标签。新评分缓存契约为 `alfworld-post-action-recent-history-v1`。

随后工作分支将 ALFWorld 终局回填改为复用 QA 的同题正负局部信用范围动态缩放，替代训练/Probe 中固定系数 2。保留 K=3 最后三个真实动作的回填位置及 gamma 距离折扣，Prompt/Response、局部信用、配对和基础训练算法保持不变。`TRACE_TERMINAL_FILL_SCALE` 对 ALFWorld 忽略，关闭回填使用 `TRACE_TERMINAL_FILL_ENABLED=false`。该改动尚未另命名实验版本，以下历史标签不变。

工作分支曾进一步支持 `ALFWORLD_TRACE_HISTORY_LENGTH`，原动作后日志格式对应缓存版本 `alfworld-post-action-history-v2`。

当前按用户要求恢复正式 **v2_e（`ee5da31`）** 的 actor 原生历史格式，同时保留动态终局回填与独立评分窗口参数。`ALFWORLD_TRACE_HISTORY_LENGTH` 默认 `4`，`None`/`null` 为全部原生 memory，`0` 无历史，非负整数 N 为最近 N 条；每条仍是动作前 observation＋解析 action，并保留绝对编号。actor 自身的窗口不变。新缓存契约为 `alfworld-actor-memory-history-v2`，保存完整原生记录以便重评分。该恢复不另命名实验版本，不移动历史标签。参考模型更新是独立可选项，默认 `TRACE_REFERENCE_UPDATE_INTERVAL=0` 保留原固定参考；N 为每 N 个训练 step 更新，None 为每 step 更新。

## 版本汇总

| 版本 | 复现标签 | 主要变化 |
| --- | --- | --- |
| v0 | `alfworld/v0` | 无 TRACE；按最终环境 reward 构造偏好，使用 anchor 阈值 0.95 分簇及原取模轮转。 |
| v1_a | `alfworld/v1a` | 首次接入 TRACE：离线 PDDL 成功条件写入 Parquet `result`；完整环境状态＋累计历史评分，固定 GT 的平均 LogP → V → K 步信用＋固定系数 2 的终局回填；所有真实动作含末动作后的真实状态均参与；同题同 turn 最多 2 对。 |
| v1_b | `alfworld/v1b` | 评分同 v1_a；K_PAIRS=None，允许同题跨 turn、无组内 cap，保留全步 256 对；候选为全部正负组合。 |
| v2_a | `alfworld/v2a` | 移除累计评分历史，保留规则、完整事实和当前观察，Response 仍为成功状态描述；实际历史代码已含先轮转分配负例、再检查 V 的旧轮转。 |
| v2_b | `alfworld/v2b` | 成功条件移入 Prompt，输入剩余动作预算，Response 改为统一未来成功句；沿用旧轮转。 |
| v2_d | 未固定标签或精确源码基线 | 用户确认的窗口改动前实验：v2_a 成功状态 Response、无最近 4 步评分历史；`pair_round_robin=false`，全部正负组合，其它过滤、K 分组及总额度沿用原基线。 |
| v2_e | `alfworld/v2_e` | 相对 v2_d 仅在评分 Prompt 增加 actor 原生最近窗口，默认 4 条动作前 observation＋解析 action，保留绝对编号原文；`pair_round_robin=false` 全组合不变，旧缓存重放补齐，不重新采样。 |

用户此前确认 v2_a/v2_b 实验的 K_PAIRS 为 None，但已有旧标签的配置默认值实际为 2；旧标签均含旧轮转，不能当作无轮转快照。原标签、`alfworld-v2a` / `alfworld-v2b` 及 QA 标签保持原位置。v2_d 目前仅确认上述实验配置，不推断其精确提交或标签；v2_c 仍未定义。其它中间改动按提交记录，不补造版本名。评分模板见 [评分版本说明](docs/alfworld-scoring-versions.md)。

## 中间未命名改动

| 提交 | 改动 |
| --- | --- |
| `41f0552` | 回退 v2_a 评分，None 跨 turn，恢复全部正负组合。 |
| `ad412f6` | 应用先轮转后检查 V 的旧分配。 |
| `76949a6` | 改为先找 V 兼容池，再按负例最少使用次数分配。 |
| `fcca1b8` | 新增轮转开关，关闭后恢复全组合。 |
| `52549b5` | 将分配纠正为兼容池内严格 `pos[k] → neg[k % len(neg)]`，每池独立游标。 |

## v2_e 的输入与配对

原生历史直接来自 `_build_prompt()` 的 `SimpleMemory.fetch(history_length)` 已用于 actor 的文本，每条为动作前 observation 与解析后的 action，保留原编号和换行。默认最近 4 条实际存入 memory 的记录，自定义 history_length 则跟随实际窗口；初始化或 history_length<=0 时为空。未写 memory 的格式/命令校验失败、外部异常沿用原窗口；模拟器正常返回但动作未生效的记录仍按原逻辑写入。这里不是累计 `trace_action_history` 的末 4 条动作后观察。仅新增 ALFWorld 评分信息导出，不改 actor Prompt、memory 写入、环境初始化及 QA/websearch 评分。

`TRACE_PAIR_ROUND_ROBIN`（`data.trace.pair_round_robin`）在 v2_d/v2_e 实验均为 false，ALFWorld 三份训练配置默认也固定为 false，枚举组内全部正负组合。库 `TraceConfig()` 无配置调用仍默认 true；可显式开启 true，规则为：先 V 匹配，再按相同有序兼容负例列表分池，在各池中严格取模轮转；质量过滤失败游标仍前进、不补选。false 枚举原组全部正负组合。`TRACE_K_PAIRS_PER_GROUP` 独立控制分组和组 cap：当前训练默认 None，跨 turn、无组内 cap；设为 2 则同 turn、每组最多 2 对。两模式保留相同质量筛选及全步默认 256 对，具体规则见 [SLIC 配对](docs/trace-slic.md)。

## 使用与缓存

工作区无未提交改动时获取当前分支：

```bash
git fetch origin --prune --tags
git switch alfworld
git pull --ff-only origin alfworld
export TRACE_K_PAIRS_PER_GROUP=None
export TRACE_PAIR_ROUND_ROBIN=false  # v2_d / v2_e：全组合，仅比较历史窗口
```

固定复现 v2_e 可检出 `alfworld/v2_e`；v2_d 只记录已确认实验配置，不伪造精确源码或标签。注意启动脚本或 shell 的环境变量会覆盖代码默认配置。GT/result、信用公式、基础 GRPO/GiGPO 及配对额度不变，已有 GT Parquet 无需重新转换；更新后重启训练/rollout，使用新评分契约的数据，不混用旧信用。

当前分支的新 Probe 缓存保存完整原生 memory（`trace_full_actor_history`）、选中评分窗口和 `alfworld-actor-memory-history-v2` 版本。格式/命令校验失败或外部异常未写入 actor memory，因此不会新增原生历史条目；正常模拟器返回的 no-op 仍按原逻辑保留。旧正式 v2_e 的原生文本快照只在足以覆盖相同窗口时复用；需要其它窗口但缺完整原生记录时，使用 `enrich --replay` 重放保存动作并核验 actor Prompt，无需重新采样。动作后日志不能自动冒充原生历史；`saved` 自定义上下文保持原样。
