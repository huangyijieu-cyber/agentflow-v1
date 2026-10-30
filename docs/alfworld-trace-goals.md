# ALFWorld：实际 PDDL → 固定英文 GT → TRACE

转换与训练已分开：只转换训练 Parquet 选中的题，将固定英文 GT 写入 `result`，同时在该 Parquet 中保存规则和游戏映射。训练直接读取该行 `result`，不在 reset 或 turn 中转换，也不依赖单独的 SQLite 文件。

此实现使用完整环境状态进行参考模型评分（privileged/full-state）。策略模型继续接收原来的可见观察和有限历史。完整状态只进入评分上下文；固定 Response 描述成功条件，每道题的所有状态复用同一文本。

## 规则来源与转换

离线入口是 `agentflow/agentflow/trace/alfworld_goal_store.py prepare`。在服务器仓库根目录运行一次：

```bash
python agentflow/agentflow/trace/alfworld_goal_store.py prepare \
  --dataset /home/ma-user/work/code-rl/AgentFlow-distributed-GRPO/data/train/alfworld/train_all_alfworld_3305.parquet \
  --games-root /home/ma-user/work/code-rl/AgentFlow-distributed-GRPO/data/alfworld-master/json_data/json_2.1.1 \
  --alfworld-root /home/ma-user/work/code-rl/AgentFlow-distributed-GRPO/data/alfworld-master \
  --output /home/ma-user/work/code-rl/AgentFlow-distributed-GRPO/data/train/alfworld/train_all_alfworld_3305_with_gt.parquet \
  --expected-rows 3305 \
  --progress-every 1
```

激活原有 ALFWorld Python 环境，确保有 `pyarrow` 和 `PyYAML`。`--alfworld-root` 是包含 `alfworld` Python 包的官网源码目录，`--games-root` 是含 train 等 split 的 `json_2.1.1` 目录。选择阶段调用官方 `AlfredTWEnv` 的目录收集逻辑，不启动模拟器或模型；转换读取选中游戏内的 `pddl_domain`、`pddl_problem` 和同目录 `traj_data.json`。

**只处理输入 Parquet 的 3305 行。** 按 `question`（如 `train_all_alfworld_0`）中的零起始编号，匹配服务器官方环境的原始游戏目录顺序，不能用一开始为 1 的 `id`，也不能按路径排序后取前 3305 个。选定的游戏相对路径、原文件哈希和题号会一起保存，后续训练不重新按目录遍历顺序选题。缺失或超出目录范围、重复映射、不可解或不支持的选中游戏均明确报错。

输出保留原六列及行顺序，仅填充 `result`；追加四个字符串列：

| 列 | 内容 |
| --- | --- |
| `alfworld_goal_spec` | 预转换的完整 Goal、初始事实和动作规则 JSON |
| `alfworld_goal_spec_sha256` | 保存规则的完整性校验 |
| `alfworld_game_key` | 选中游戏在 games-root 下的相对路径 |
| `alfworld_goal_row` | 原始行号、question、题号和数据集/选择哈希 JSON |

完整规则较长，因此不放进 `result`；`result` 只保存用于 teacher-forcing 的固定英文 Response。`extra_info_ground_truth` 等其余原始列保持原值。可以继续用 `data/parquet_viewer.py` 查看新文件中的 `result`。

原始 Parquet 默认保留，输出已存在时拒绝覆盖；明确重建才使用 `--overwrite`。进度实时显示已处理/总题数、百分比、耗时及 ETA。只有最终出现 `[complete] ... Training Parquet saved successfully.` 才表示新训练文件发布成功；100% 后仍有保存和校验阶段。

内部先构建仅含所选题的临时 SQLite，成功发布 Parquet 后删除它，训练无需该中间文件。若已有完整 SQLite，可增加 `--reuse-store /path/to/complete.sqlite` 复用已转换规则，不再次调用 PDDL 转换器。若最终发布失败，新版本会保留完整中间文件并报告路径，供 `--reuse-store` 重试。旧版本此前报错时清理的临时文件不能据此恢复。

文件系统不支持硬链接（`Errno 95`）时，会自动使用加锁后的重命名发布，仍拒绝覆盖已存在的目标。输出目录需要支持创建文件和重命名。

运行时先按保存的相对路径选中游戏，再以实际 reset 返回的 `extra.gamefile` 核验游戏和 `traj_data.json` 的 SHA-256，并检查 `result` 与保存规则的固定 GT 完全一致。路径整体迁移后，内容一致仍可使用；缺失、陈旧或损坏明确报错，不回退实时转换。不要单独手改 `result`：需要同步完整规则及其哈希，通常应重新执行离线准备。

Probe 仍兼容独立 SQLite。需要时单独构建；这不是正式训练的必需步骤：

```bash
python agentflow/agentflow/trace/alfworld_goal_store.py build \
  --dataset /path/to/train_all_alfworld_3305.parquet \
  --games-root /path/to/alfworld-master/json_data/json_2.1.1 \
  --alfworld-root /path/to/alfworld-master \
  --expected-rows 3305 \
  --output /path/to/alfworld_trace_goals.sqlite
```

`build --all-games` 才表示显式转换整个游戏目录。`show --store ... --gamefile ...` 只导出保存内容；`alfworld.py convert` 仅用于单题转换调试。

转换保留 and/or/not、存在/全称量词、对象不同及共享容器/位置等绑定。英文使用显式变量（例如 O1、R1），避免将“存在一个目标物体”错误绑定到某个编号。`isToggled` 翻译为 toggle flag；当前 `isOn` 才表示开启。冷热标记按当前事实描述，不将被覆盖的热/冷状态当成永久历史。动作前提放在规则上下文，不添加到成功条件中。

完整状态以“初始事实 + 当前新增/删除事实”无损呈现，使用稳定短别名减少上下文长度。当前事实直接来自 TextWorld `facts`，不根据生成的动作猜测执行效果。失败动作不会自行修改事实；历史窗口缩短也不丢失持久事实。每个状态对照 `evaluate_goal(facts)` 和环境 `won`；不一致的状态保留供排查，但不用于评分/配对。

规则核验依据 ALFWorld 官方 [goal_library.py](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/alfworld/gen/goal_library.py)、[alfred.pddl](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/alfworld/data/alfred.pddl) 和 [TextWorld 环境包装器](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/alfworld/agents/environment/alfred_tw_env.py)。运行时仍以服务器实际加载的游戏为准。

## 正式训练接入

ALFWorld 环境、solver、两个 rollout 入口、daemon 和 trainer 均传递实际 GT、完整状态、逐行评分指令、来源哈希和有效性。训练 rollout 将完整 Parquet 行绑定给 ALFWorld 环境；环境读取该行 `result` 和已保存规则，并核验实际游戏。daemon 从原始样本 `result` 设置 `trace_gt`，同时核对环境记录的 GT，拒绝空值或不一致，不调用转换器。运行时只更新当前事实、渲染评分上下文和检查目标谓词。同一 uid 若混入不同游戏/Goal 会报错，避免跨题配对。重复动作在不同前状态下可作为不同决策；QA 的重复过滤规则保持原样。

沿用 QA 的数值配置；ALFWorld 的所有真实环境动作统一使用 TRACE 配对：

| 项目 | 默认值/行为 |
| --- | --- |
| epsilon / K / gamma | 0.1 / 3 / 0.8 |
| terminal fill | 开启，scale=2.0 |
| value match epsilon | 0.05 |
| 正/负信用阈值、信用差阈值 | 0.01 / 0.01 / 0.02 |
| length ratio / 每题每原始 turn 的 pair / 每步 pair | 1.8 / 2 / 256 |
| GT token 上限 | 256，超长判无效，不截断 |
| GRPO、SLIC loss、Replay | 沿用现有 QA 实现 |

首轮、中间轮和最后一轮都标为 `tool_plan`，仅在同题、同原始 turn id 内，按动作前 V 接近及严格正/负有效信用配对；每题每个 turn 默认独立最多 2 对。末动作不单独划入 `base` / outcome，它只能与其他轨迹中相同 turn id 的合格环境动作配对，不允许跨 turn。T 个动作对应 T+1 个真实状态，最后动作后的终止状态参与最后一步 `V_next - V`，但不新增模型回答或训练 turn。`trace_is_terminal` 只表示已完成的环境终止动作，用于截断信用窗口与终局回填；ALFWorld 的 `trace_is_final` 恒为 false。可信的步数预算耗尽属于终局失败；外部崩溃和无法核验的状态不作为失败奖励。

固定 Response 的平均 token LogP 为 ell，`V_t = log((-ell_0 + epsilon) / (-ell_t + epsilon))`。pre-state 与 post-state 统一用该轨迹初始 ell_0 作基线。每个动作的真实转移为 `delta_t = V_(t+1) - V_t`，局部信用是最多 K 个连续真实动作 delta 的 gamma 加权平均，遇到环境终止动作则纳入该动作的 delta 后结束。缺失任何所需状态、编号缺口或外部故障均阻断窗口，不编造末状态或回退到 outcome 配对。

窗口包含可信终止动作时，再加 `terminal_fill_scale * gamma ** (terminal_turn - start_turn) * outcome_advantage`。K=3 时最多给末三个真实动作回填，包含最后动作本身；末动作距离为 0，因此获得 `scale * outcome_advantage` 的回填。真实环境结果用于这项回填，不改变冻结参考模型的 LogP/V。QA 原来的最终答案路由及其信用函数保持不变。

## 服务器适配

使用服务器原有的 ALFWorld/TextWorld/fast_downward 环境和数据即可，不修改官网安装包。正式训练默认源码/数据布局仍为仓库根目录下的 `data/alfworld-master` 与其中的 `json_data`；若已有自定义路径，继续使用自己的环境路径配置。

在仓库根目录，使用原有训练启动方式，并确保：

```bash
export TASK=alfworld
export ALGORITHM=grpo+slic
export TRACE_ENABLED=true
export TRAIN_DATA_PATH=train/alfworld/train_all_alfworld_3305_with_gt.parquet
# TEST_DATA_PATH 设为自己实际存在的 ALFWorld 验证 Parquet。
```

训练只需要准备好的 Parquet，不需要设置 `ALFWORLD_TRACE_GOALS`。该配置保留用于旧 SQLite/Probe 兼容。每个 worker 需能访问相同内容的训练文件和原始游戏数据。第二次及后续训练继续使用 `..._with_gt.parquet`，无需重新转换。验证环境关闭 ALFWorld TRACE 的完整状态评分，继续按环境奖励验证，不要求为验证集额外转换 GT。

数据路径与 `BASE_DATA_DIR` 组合，由现有 `train-roma/config.yaml` 解析；使用自定义启动配置时，将 TASK、数据路径和 TRACE 配置传入两个 rollout 入口及训练进程。无需为 ALFWorld 改 QA 的 epsilon、K、gamma、配对阈值或 loss。参考模型保持冻结，且 tokenizer 必须与参考模型一致。

完整状态比策略可见上下文长。`MAX_PROMPT_LENGTH` 必须容纳规则、完整初始事实、状态变化和评分指令；超过上限会排除该状态并记录无效原因，不静默截断。先用 Probe 检查实际 token 数，再据服务器显存/序列上限设置。固定 GT 超过 256 tokens 时同样会明确标记无效，可检查转换文本及模型 tokenizer 后再决定是否提高与 QA 共用的上限。

窗口使用和旧缓存升级见 [alfworld-trace-probe.md](alfworld-trace-probe.md)。建议先跑固定四条轨迹，查看 GT、Goal/环境校验、local credit、terminal fill 和最终 effective credit，再开始训练。文本 LogP 是参考模型对目标描述的支持程度；它的成功率校准与信用质量需要实验检验。

## 本地验证记录

六类真实样例取自官方 `json_2.1.3_tw-pddl.zip`（release 0.4.2）和同路径的 `json_2.1.1_json.zip`（release 0.2.2）；均为 valid_unseen 中的原文件配对，解压校验 CRC，未重建元数据。六类转换和初始 Goal 计算均成功，初始 Goal 均为 false。

| 任务 | 样例目标 | 初始事实数 | 完整上下文字符数 |
| --- | --- | ---: | ---: |
| look_at_obj_in_light | look_at_obj_in_light-Pencil-None-DeskLamp-308 | 222 | 10376 |
| pick_and_place_simple | pick_and_place_simple-SoapBottle-None-Toilet-424 | 146 | 9513 |
| pick_clean_then_place_in_recep | pick_clean_then_place_in_recep-Bowl-None-Cabinet-10 | 526 | 16622 |
| pick_cool_then_place_in_recep | pick_cool_then_place_in_recep-Bread-None-CounterTop-10 | 470 | 15632 |
| pick_heat_then_place_in_recep | pick_heat_then_place_in_recep-Mug-None-CoffeeMachine-10 | 473 | 15809 |
| pick_two_obj_and_place | pick_two_obj_and_place-KeyChain-None-Safe-219 | 224 | 10965 |

这些是实际文件转换验证；动作执行与环境 won 的对照由运行时逐状态完成。本次离线文件还完成六类样例的实际 build、禁用转换器后的读取、独立新进程读取，以及文件字节不变检查。回归覆盖离线构建、Parquet 写入和读取、跨训练复用、来源变化拒绝、ALFWorld 环境/Probe 和既有 TRACE 行为。模型实际输出及部署环境性能仍需在服务器运行 Probe 检查。

新增 Parquet 流程另用上述六类真实样例运行了完整 `prepare` 命令：原六列（除 result）保持一致，六行 GT 非空，四列规则/映射写入成功，中间 SQLite 已删除。目录收集采用官网构造器与收集方法的独立测试替身（本地未安装 TextWorld）；新进程禁用 Arrow 导入与转换器后仍能读取六行并核验源文件。该验证不代表已在本地转换服务器的全部 3305 个游戏，也未执行模型训练。

统一动作配对改动不改变离线 GT/规则格式：已经准备好的 `..._with_gt.parquet` 可以直接复用，不需要重新转换。已有 Probe 缓存只要保留了全部 T+1 个真实状态，也可以直接重新评分，无需重新采样。

当前回归通过 184 项 ALFWorld 测试和 135 项 TRACE 回归，覆盖单动作终止、末动作真实 delta、末三动作回填、缺失末状态、相邻状态不一致和 QA 既有路由。模型评分及模拟器在链路测试中使用 CPU 替身，未在服务器执行完整模型训练。更新代码后应重启训练及 rollout 进程；旧训练 rollout 的 base/outcome 元数据会被明确拒绝，离线 Parquet 不受影响。

SLIC 工具/环境动作的配对现已恢复为同题、同原始 turn id：ALFWorld 所有动作及 QA 工具动作均按各 turn 独立扣取默认 2 对额度；baseline 也仅使用相同分组，空响应日志造成的压缩编号不能用作配对身份。QA analysis/final 的 outcome 分组、标签及配对规则保留；关闭 TRACE 后的这两个任务的 SLIC 动作配对也遵循同 turn 限制。K-step 信用仍沿真实轨迹计算，此限制只约束偏好对的两端，不改变信用窗口。
