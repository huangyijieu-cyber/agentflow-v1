# ALFWorld：实际 PDDL → 固定英文 GT → TRACE

此实现使用完整环境状态进行参考模型评分（privileged/full-state）。策略模型继续接收原来的可见观察和有限历史。完整状态只进入评分上下文；固定 Response 描述成功条件，每道题的所有状态复用同一文本。

## 规则来源与转换

入口是实际 reset 返回的 `extra.gamefile`，不通过 Parquet 的 id 或重新遍历目录猜游戏。转换器读取 `game.tw-pddl` 内的 `pddl_domain`、`pddl_problem`，以及同目录的 `traj_data.json`。

```bash
python agentflow/agentflow/trace/alfworld.py convert \
  --gamefile /absolute/path/to/trial/game.tw-pddl \
  --output outputs/alfworld_goal.json
```

此命令只使用 Python 标准库，不启动环境或模型。输出包含固定 `goal_response`、Goal AST、初始事实、动作规则、对象别名、原始来源与 SHA-256。实际 Goal 决定英文内容，任务类型和 `pddl_params` 用于一致性核验。支持六类常见任务；不支持的谓词、结构或额外变体会报错，不能按任务类型省略条件。

转换保留 and/or/not、存在/全称量词、对象不同及共享容器/位置等绑定。英文使用显式变量（例如 O1、R1），避免将“存在一个目标物体”错误绑定到某个编号。`isToggled` 翻译为 toggle flag；当前 `isOn` 才表示开启。冷热标记按当前事实描述，不将被覆盖的热/冷状态当成永久历史。动作前提放在规则上下文，不添加到成功条件中。

完整状态以“初始事实 + 当前新增/删除事实”无损呈现，使用稳定短别名减少上下文长度。当前事实直接来自 TextWorld `facts`，不根据生成的动作猜测执行效果。失败动作不会自行修改事实；历史窗口缩短也不丢失持久事实。每个状态对照 `evaluate_goal(facts)` 和环境 `won`；不一致的状态保留供排查，但不用于评分/配对。

规则核验依据 ALFWorld 官方 [goal_library.py](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/alfworld/gen/goal_library.py)、[alfred.pddl](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/alfworld/data/alfred.pddl) 和 [TextWorld 环境包装器](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/alfworld/agents/environment/alfred_tw_env.py)。运行时仍以服务器实际加载的游戏为准。

## 正式训练接入

ALFWorld 环境、solver、两个 rollout 入口、daemon 和 trainer 均传递实际 GT、完整状态、逐行评分指令、来源哈希和有效性。ALFWorld 使用环境生成的 GT，不读取数据表中空的答案字段。同一 uid 若混入不同游戏/Goal 会报错，避免跨题配对。重复动作在不同前状态下可作为不同决策；QA 的重复过滤规则保持原样。

沿用 QA 的算法配置与配对：

| 项目 | 默认值/行为 |
| --- | --- |
| epsilon / K / gamma | 0.1 / 3 / 0.8 |
| terminal fill | 开启，scale=2.0 |
| outcome match value | false |
| value match epsilon | 0.05 |
| 正/负信用阈值、信用差阈值 | 0.01 / 0.01 / 0.02 |
| length ratio / 每组 pair / 每步 pair | 1.8 / 2 / 256 |
| GT token 上限 | 256，超长判无效，不截断 |
| GRPO、SLIC loss、Replay | 沿用现有 QA 实现 |

真实终局环境动作走 `base` / outcome 路由；之前动作走 `tool_plan` / TRACE 路由。K=3 包括终局动作，因此最多向前两个工具决策回填。最后动作后的状态可以保存/评分以诊断，但不新增一次信用转移。可信的步数预算耗尽属于终局失败；外部崩溃和无法核验的状态不作为失败奖励。

固定 Response 的平均 token LogP 为 ell，`V_t = log((-ell_0 + epsilon) / (-ell_t + epsilon))`。局部信用和终局回填调用现有 QA 函数；不另建 ALFWorld 的信用公式。终局奖励不会改写参考模型的 LogP。

## 服务器适配

使用服务器原有的 ALFWorld/TextWorld/fast_downward 环境和数据即可，不修改官网安装包。正式训练默认源码/数据布局仍为仓库根目录下的 `data/alfworld-master` 与其中的 `json_data`；若已有自定义路径，继续使用自己的环境路径配置。

在仓库根目录，使用原有训练启动方式，并确保：

```bash
export TASK=alfworld
export ALGORITHM=grpo+slic
export TRACE_ENABLED=true
export TRAIN_DATA_PATH=train/alfworld/train_all_alfworld_3305.parquet
# TEST_DATA_PATH 设为自己实际存在的 ALFWorld 验证 Parquet。
```

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

这些是实际文件转换验证；动作执行与环境 won 的对照由运行时逐状态完成。自动回归包含 79 项 ALFWorld 测试和 122 项既有 TRACE 测试。模型实际输出及部署环境性能仍需在服务器运行 Probe 检查。
