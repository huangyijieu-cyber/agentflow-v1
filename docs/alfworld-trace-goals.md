# ALFWorld：实际 PDDL → 固定英文 GT → TRACE

转换与训练已分开：先离线批量转换，保存成一个 SQLite 文件；训练只读该文件，不在 reset 或 turn 中重新生成 GT。

此实现使用完整环境状态进行参考模型评分（privileged/full-state）。策略模型继续接收原来的可见观察和有限历史。完整状态只进入评分上下文；固定 Response 描述成功条件，每道题的所有状态复用同一文本。

## 规则来源与转换

离线入口是 `agentflow/agentflow/trace/alfworld_goal_store.py`，只使用 Python 标准库，不启动 ALFWorld、模型或训练器：

```bash
python agentflow/agentflow/trace/alfworld_goal_store.py build \
  --games-root /absolute/path/to/alfworld-master/json_data/json_2.1.1 \
  --output /absolute/path/to/artifacts/alfworld_trace_goals.sqlite
```

`--games-root` 指向实际游戏目录树，不是训练 Parquet。建议包含 train、valid_seen、valid_unseen 三个 split，覆盖训练和验证可能加载的所有游戏。脚本扫描 `game.tw-pddl`，读取内含的 `pddl_domain`、`pddl_problem` 及同目录 `traj_data.json`，将固定 `goal_response`、Goal AST、初始事实、动作规则、对象别名、来源哈希和转换版本一起保存。遵循官方 TextWorld 过滤条件，排除 movable、Sliced 及明确不可解游戏；其余不支持的谓词、结构或损坏来源会使构建报错，不静默遗漏。

输出是单个 SQLite 文件，供多个进程只读、按游戏索引取数据。转换成功后才发布完整文件；构建失败不会替换旧文件。目标已存在时默认拒绝覆盖；确实需要重建时显式增加 `--overwrite`。第二次及以后训练复用同一文件，不需要再次运行 build。

构建默认实时输出进度到终端，包括已处理/总文件数、百分比、已转换数、跳过数、耗时和预计剩余时间。默认每 25 个文件报告一次；首个、末个和间隔约 5 秒的更新也会显示。希望每个游戏都显示时增加 `--progress-every 1`；`--quiet` 仅保留最终 JSON 汇总。进度写入 stderr 并立即刷新，stdout 的 JSON 格式保持不变。

百分比按扫描到的全部游戏文件计算，包含官方规则排除的文件；`converted` 才是实际保存条目数。处理到 100% 后还有 `[saving]` 阶段，只有出现 `[complete] ... Goal store saved successfully.` 才表示目标文件已成功发布。预计剩余时间是按当前平均处理速度估算的。已启动的旧进程不会自动获得进度输出，新版本下次启动时生效。

可以导出某个游戏已经保存的英文 GT 和规则，不会重新转换：

```bash
python agentflow/agentflow/trace/alfworld_goal_store.py show \
  --store /absolute/path/to/artifacts/alfworld_trace_goals.sqlite \
  --gamefile /absolute/path/to/trial/game.tw-pddl \
  --output outputs/goal.json
```

原 `alfworld.py convert --gamefile ... --output ...` 仍可单独调试一个游戏的转换；它输出的单题 JSON 不是正式训练读取的批量存储文件。

运行时以实际 reset 返回的 `extra.gamefile` 为准，校验该游戏和 `traj_data.json` 的 SHA-256，再从保存文件查找已经转换好的结果，不根据 Parquet 的 id 或目录遍历序号查 GT。原始数据移动路径后，只要文件内容一致仍可匹配。缺失文件、条目、来源内容变化、版本或完整性校验失败均直接报错，不退回实时转换。

转换保留 and/or/not、存在/全称量词、对象不同及共享容器/位置等绑定。英文使用显式变量（例如 O1、R1），避免将“存在一个目标物体”错误绑定到某个编号。`isToggled` 翻译为 toggle flag；当前 `isOn` 才表示开启。冷热标记按当前事实描述，不将被覆盖的热/冷状态当成永久历史。动作前提放在规则上下文，不添加到成功条件中。

完整状态以“初始事实 + 当前新增/删除事实”无损呈现，使用稳定短别名减少上下文长度。当前事实直接来自 TextWorld `facts`，不根据生成的动作猜测执行效果。失败动作不会自行修改事实；历史窗口缩短也不丢失持久事实。每个状态对照 `evaluate_goal(facts)` 和环境 `won`；不一致的状态保留供排查，但不用于评分/配对。

规则核验依据 ALFWorld 官方 [goal_library.py](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/alfworld/gen/goal_library.py)、[alfred.pddl](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/alfworld/data/alfred.pddl) 和 [TextWorld 环境包装器](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/alfworld/agents/environment/alfred_tw_env.py)。运行时仍以服务器实际加载的游戏为准。

## 正式训练接入

ALFWorld 环境、solver、两个 rollout 入口、daemon 和 trainer 均传递实际 GT、完整状态、逐行评分指令、来源哈希和有效性。ALFWorld 环境读取离线文件中的固定 GT，不读取数据表中空的答案字段，也不调用转换器。运行时只更新当前事实、渲染评分上下文和检查目标谓词。同一 uid 若混入不同游戏/Goal 会报错，避免跨题配对。重复动作在不同前状态下可作为不同决策；QA 的重复过滤规则保持原样。

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
export ALFWORLD_TRACE_GOALS=/absolute/path/to/artifacts/alfworld_trace_goals.sqlite
export TRAIN_DATA_PATH=train/alfworld/train_all_alfworld_3305.parquet
# TEST_DATA_PATH 设为自己实际存在的 ALFWorld 验证 Parquet。
```

`ALFWORLD_TRACE_GOALS` 也可在 `train-roma/config.yaml` 的同名环境变量项配置，默认留空，启用 ALFWorld TRACE 时必须指定。每个 rollout worker 所在机器都需能只读访问该路径（共享磁盘或各节点同路径的相同文件）；复制保存文件与原始游戏数据即可，不需要安装 SQLite 服务或额外 Python 包。

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

这些是实际文件转换验证；动作执行与环境 won 的对照由运行时逐状态完成。本次离线文件还完成六类样例的实际 build、禁用转换器后的读取、独立新进程读取，以及文件字节不变检查。自动回归通过 113 项 ALFWorld 测试和 122 项既有 TRACE 测试，覆盖离线构建/只读加载、跨训练复用、来源变化拒绝、ALFWorld 环境/Probe 和既有 TRACE 行为。模型实际输出及部署环境性能仍需在服务器运行 Probe 检查。
