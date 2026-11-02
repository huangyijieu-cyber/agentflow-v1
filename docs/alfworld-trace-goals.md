# ALFWorld：实际 PDDL → 固定英文 GT → TRACE

转换与训练已分开：只转换训练 Parquet 选中的题，将固定英文 GT 写入 `result`，同时在该 Parquet 中保存规则和游戏映射。训练直接读取该行 `result`，不在 reset 或 turn 中转换，也不依赖单独的 SQLite 文件。

此实现使用完整环境状态进行参考模型评分（privileged/full-state）。策略模型继续接收原来的可见观察和有限历史。完整状态只进入评分上下文；固定 Response 描述成功条件，每道题的所有状态复用同一文本。

评分 Prompt 现不再附加累计动作/观察历史：保留任务、环境规则、事实解释、初始事实、当前新增/删除事实、当前观察和固定评分指令。当前事实已包含 Goal 所需的持久标记；历史记录仍保留在环境/轨迹中用于排查，策略最近四步历史保持原样。此缩减只改变 ALFWorld full-state 评分文本，不改变 GT Response、信用公式或配对规则。已有 GT Parquet 无需重新转换，重启训练/rollout 进程后使用新文本。已有 Probe 缓存的 `state_context` 不会自动更新；含完整事实时运行 `enrich --work-dir ...` 重建评分 sidecar 后再评分，无需重新采样。

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

已绑定准备好的 Parquet 行时，环境初始化直接使用该行已校验的游戏路径，通过 `SelectedGameAlfredTWEnv` 仅替换官方目录收集方法，跳过每个 worker 对整个 split 的重复扫描。日志出现 `[ALFWorld TRACE] Using saved Parquet game mapping; skipped full-split scan.` 表示生效。同一 worker 后续题目继续复用原有环境配置并切换游戏，不把第一题的单游戏列表当作题号索引。没有保存映射的验证或旧调用方式保留官方目录扫描；已有 Parquet 无需重新转换，worker 数量和 rollout/训练组装不变。

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

完整状态以“初始事实 + 当前新增/删除事实”无损呈现，使用稳定短别名减少上下文长度。当前事实直接来自 TextWorld `PddlEnv` 的原始 `_facts`，通过 `AlfWorldTraceFacts` 包装器导出为 `extra.trace_raw_facts`，不根据生成的动作猜测执行效果。公开的 `facts` 使用 `loc 7` 等可能随机化的显示名称，不能与保存的 PDDL 标识混用。包装器只增加评分信息，保留官方观察、命令和显示名称；已有 GT Parquet 无需重新转换。失败动作不会自行修改事实；历史窗口缩短也不丢失持久事实。每个状态对照 `evaluate_goal(facts)` 和环境 `won`；不一致的状态保留供排查，但不用于评分/配对。

TextWorld 动作执行后会把删除效果存为 `not_atlocation`、`not_holds`、`not_opened` 等显式否定事实，同时移除对应正事实。运行时先核验其基础谓词、参数数量和游戏对象，再投影到现有闭世界正事实集合：否定事实表示对应正事实缺席，状态渲染会保留初始事实的删除。未知谓词、未声明对象或同一事实同时为真和为假仍明确报错。该兼容不改变离线 Goal/GT 或已保存 Parquet。

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

## TRACE 评分性能

共享前缀与自定义四维 attention 的新评分路径、配置和分布式边界见 [TRACE 共享前缀 GT 评分](trace-shared-prefix.md)。以下 2026-11-02 日志记录的是新增该路径之前的独立序列批量评分，用于耗时对照。

2026-11-02 提供的 ALFWorld 首步日志中，64 条 rollout 产生 1884 个训练 turn，3768 个动作前/后状态候选去重为 984 个实际 GT 评分请求。整个 step 用时 691.82 秒，分布如下；不能把 rollout 后的全部等待都归给参考模型：

| 阶段 | 秒 |
| --- | ---: |
| rollout、组装和休眠（`timing_s/gen`） | 141.42 |
| actor 动作旧 LogP | 84.06 |
| 普通 reference 动作 LogP | 111.38 |
| TRACE 编码、GT 评分和 credit | 185.24 |
| 其中 GT reference 调用（已包含在上一项） | 112.98 |
| GRPO 更新 | 166.85 |
| SLIC 更新 | 0.002（本步没有合格配对） |

性能优化保留环境、rollout、冻结参考模型、GT 文本、teacher-forcing token 序列、V/credit 公式与同 turn 配对规则。对于启用 TRACE、两种 KL 均关闭的 GRPO/GiGPO/SLIC 组合，跳过损失不消费的普通动作 reference LogP；GT reference 评分继续执行。其他算法或需要 KL 时保留动作 reference 评分。编码先按最终输入文本复用相同状态，再使用 HF fast tokenizer 的批量 chat-template 接口；非 fast tokenizer 兼容原标量接口。最终仍按 prompt/GT token IDs 去重，原始行的无效原因、长度限制和映射保持不变。CPU 张量构造和评分结果回填采用批量数组操作，避免逐行创建 Torch 小张量；缓存仅存在于本次评分调用，不跨训练 step 积累。

批量分词可使用 fast tokenizer 的原生并行能力，并遵守部署环境的分词器并行设置。模型评分继续使用原有 Ray/FSDP/SP 分布式路径；不让 actor 与 reference 在同组 NPU 上额外并发，不改变 SP 或提高默认 microbatch。GT 虽短，模型仍需处理完整状态上下文；本地 CPU 微基准不等于服务器端的实际 step 加速。

日志中的 `trace/scoring_seconds` 只包含 GT reference 调用；`timing_s/trace` 还包含编码、张量准备、credit 和统计。新增 `trace/validation_seconds`、`trace/encoding_seconds`、`trace/scoring_prepare_seconds`、`trace/scoring_postprocess_seconds` 和 `trace/credit_seconds` 分别报告校验、编码、评分张量准备、结果回填以及 credit/统计耗时。`perf/action_ref_log_prob_skipped=1` 且 `timing_s/ref=0` 表示省掉了普通动作 reference，GT 评分仍由 `trace/scoring_seconds` 计时。

按上述单步数据，仅省掉普通动作 reference 就对应约 111 秒的原有工作量，但最终耗时需要用服务器上的后续 step 实测，不能承诺训练 step 与 rollout 一样短。本步选中 0 对是配对约束下的实际结果，性能改动不会放宽阈值或制造配对。

本地等价验证使用真实 `PreTrainedTokenizerFast`（Transformers 4.57.1 / Tokenizers 0.22.2，ByteLevel BPE 与 Jinja 模板），覆盖重复输入、Unicode、不同 GT/指令、缺失字段和长度边界。新旧 requests、行映射、计数和六个评分张量逐元素一致。3768 个候选、984 个唯一请求、约 1240-token 合成上下文的 CPU 基准中，编码中位耗时从 3.10 秒降至 0.424 秒，张量构造从 0.0357 秒降至 0.0188 秒；此结果仅说明本地 CPU 路径变化，未使用服务器的 Qwen tokenizer 或执行 NPU 训练。

后续审查发现，完整状态和来源等 `trace_*` 非张量字段原先会随 action old/ref LogP、GRPO 和 SLIC 的 RPC 发给 worker，进入模型计算前才被丢弃。现在仅在 trainer 侧创建轻量输入视图，提前排除这些 driver 专用字段；保留全部训练 tensor、其它非张量字段（包括多模态输入和组内 baseline）及 `meta_info`。原主 batch 仍完整保留用于 TRACE、配对和 replay。`perf/actor_rpc_trace_fields_omitted` 报告 GRPO RPC 排除的字段数。这减少确定存在的序列化和运输开销，但不能仅凭代码断言它解释了 GRPO 更新从一分钟变成三分钟。

GRPO 的 `dp_actor.py` forward/backward/optimizer 以及 actor 的默认 SP、microbatch、mini-batch 与适配前 `8083f0d` 相同；TRACE 的完整状态未加入策略 prompt，T+1 个评分状态也未变成额外 GRPO 行。所提供日志是第 1 步，真实训练行数约 1872、有效 token 总数约 115.8 万。比较更新速度应对齐这些数量、实际配置及稳态 step，不能将不同样本量或首次算子初始化的运行直接比较。当前向量化覆盖编码、张量构造和评分回填，credit 核心仍使用原公式与循环，未引入 IGPO 的全后缀奖励算法。

双机十六卡在 SP=2、全局 batch 不变时将数据并行组从 4 增至 8，有机会缩短模型评分和更新时间，但 CPU 侧串行处理不会自动减半，跨机 FSDP 通信也会增加成本。应先保持训练题数、rollout 数和全局 mini-batch 不变，确认节点间 HCCL 网络与带宽，再对比相同数据量的稳态时间。

## 零配对与阈值

提供的首步日志中，`candidates_sign=10`，`rejected_value_match=4`，`rejected_same_action=6`，而 `rejected_credit_gap=0`，最终 0 对。因此该轮降低 gap 不会恢复配对。当前默认先要求 chosen credit > 0.01、rejected credit < -0.01，进入这一步的差值本已严格大于 0.02；在正负阈值不变时，gap=0.005 不会改变它们的筛选结果。相同上下文的同一动作可能因 K-step 后续行为不同而得到不同 credit，但当前 response 相同，仍会被过滤。相同上下文中的不同动作、满足正负和其它约束时可以正常配对。

若实验意图仅为严格正负及至少 0.005 的信用差，可显式将 `TRACE_POSITIVE_THRESHOLD=0`、`TRACE_NEGATIVE_THRESHOLD=0` 和 `TRACE_GAP_EPSILON=0.005` 一起设置；这是候选门槛调整，代码不会自动应用，也不保证通过状态价值或相同动作过滤。保留同题同原始 turn、每 turn 最多两对。新增 `trace/config/{positive_threshold,negative_threshold,gap_epsilon,match_epsilon}` 指标打印实际采用的四个值。

正式入口 `train-roma/train_agent.py` 读取 `train-roma/config.yaml`，同名环境变量优先于 YAML 默认值，命令行 Hydra 覆盖最后生效。仅修改 `config_grpo+slic.yaml` 或 `igpo_gap_epsilon` 不保证这个入口的 TRACE gap 发生变化。策略最近 4 步的上下文相同，也不等于 GT 评分上下文相同，后者包含规则、完整状态和当前观察；此差别可能影响 V 匹配。评分文本已移除累计动作/观察历史，但当前观察仍可携带最近动作的反馈，因此相同完整事实也不保证评分 Prompt 完全一致。

## HCCL 端口故障定位

分布式启动脚本 `train-roma/run_distribute_train.sh` 在加载 Ascend 环境后、启动 Ray 前，默认设置 `HCCL_IF_BASE_PORT=20000`，并输出它及更高优先级的 `HCCL_HOST_SOCKET_PORT_RANGE`。按 32 个端口规划 `20000–20031`，避开已观察服务器的系统临时端口范围 `32768–60999`，防止普通连接的自动源端口分配使用默认 HCCL 的 `60000` 段。此配置适用于 QA 和 ALFWorld，不改变训练、rollout、SP 或 TRACE 算法。

所有节点使用相同配置，并确认该端口段没有其他固定服务使用、多节点间可以访问。脚本保留显式设置的 `HCCL_IF_BASE_PORT`；若使用自定义值，也应选择系统临时端口范围之外的可用端口段。`HCCL_HOST_SOCKET_PORT_RANGE` 的优先级更高，若平台或其他脚本设置了它，应删除该设置或把它配置到合适的端口段，避免覆盖本配置。只在 Ray 启动后的提交终端设置环境变量，不能保证已有 worker 继承；应在下一次正常启动训练及各节点 Ray 时生效，当前正在正常训练的进程无需中断。服务器若使用自己的 `start.sh`，将以下内容放在所有 Ascend 环境加载之后、`ray start` 之前即可：

```bash
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-20000}"
```

这是固定在启动脚本中的长期配置，不需要每次手动运行 `sysctl` 或监控工具。它针对端口范围重叠；此前 `60006` 绑定失败时的具体占用连接仍未被捕获，不能把预防措施视为对占用进程的确认。官方配置说明见 [HCCL_IF_BASE_PORT](https://www.hiascend.com/doc_center/source/en/canncommercial/850/commlib/hcclug/hcclug_000090.html)。

首次 actor `old_log_prob` 若报 `HcclGetRootInfo / Bind_Failed`，先核对成功 QA 的 TRACE、SP、模型和设备配置。两任务在这一阶段共用 actor 通信路径，GT 尚未评分，不能仅凭任务类型认定是 Goal 转换或参考模型导致。事后端口空闲也不能反推故障瞬间的占用者。

可在服务器另一个终端先启动一次只读捕获，再按原命令正常启动 ALFWorld：

```bash
python train-roma/diagnose_hccl_ports.py \
  --port-range 60000-60031 --duration 900 \
  --output /tmp/alfworld-hccl-port.jsonl
```

该工具仅持续读取 Linux `/proc`，记录端口的 LISTEN、ESTABLISHED、TIME_WAIT、可见占用进程和有限的 rank/HCCL 环境信息；不导入模型、不建立连接、不改端口或训练配置。报错后保留 JSONL，Ctrl+C 只停止捕获。现有输出拒绝覆盖。它只用于一次故障定位，不是后续每次训练的前置步骤；短于采样间隔的事件可能未被捕获，未观察到占用不等于端口始终空闲。

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

SLIC 工具/环境动作默认按同题、同原始 turn id 配对：ALFWorld 所有动作及 QA 工具动作均按各 turn 独立扣取默认 2 对额度；baseline 也使用相同分组，空响应日志造成的压缩编号不能用作配对身份。开启 TRACE 时，设置 `TRACE_K_PAIRS_PER_GROUP=None`（或 `null`）允许同题同类型跨 turn 配对，取消组内数量上限，仍保留全步默认 256 对上限及其它筛选条件；设为正整数则保留同 turn 分组和对应额度。QA analysis/final 的 outcome 分组和标签保留。关闭 TRACE 后的这两个任务的 SLIC 动作配对仍遵循原有同 turn 限制。K-step 信用始终沿真实轨迹计算，该开关不改变信用窗口。
