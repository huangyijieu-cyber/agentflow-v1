# SLIC turn 筛选消融实验

实验分支 `test_first`（基于 `cache`）新增 `data.slic_turn_scope`，用于在保持多步 rollout 的情况下，
只改变参与 SLIC 的 turn。`train-roma/config.yaml` 与 `config_grpo+slic.yaml`
默认开启本实验；未传入该字段的旧配置仍默认 `all`。
实现基于已核对的 GitHub `cache` 提交 `1ca2c83`。

## 生效范围

实验模式是 `analysis_first_tool_final`，适用于 `TASK=qa`、`OUTPUT_TYPE=final`。
当前训练数据只采集 `planner_logs`：

| Planner turn | 内容 | SLIC |
| --- | --- | --- |
| 0 | `analyze_query`，初始 analysis | 参与 |
| 1 | 第一次 `generate_next_step`，选择工具与 sub-goal | 参与 |
| 2 到倒数第二个 | 后续工具选择 | 排除 |
| 每条 rollout 的最后一个 | `generate_final_output` | 参与 |

这里的“首轮工具调用 turn”指 planner 的工具选择输出。
executor 的命令生成、实际工具执行、verifier 的判断不在 `planner_logs` 中。
不要把本实验模式用于额外包含 `base`、`direct` 的输出组合，
因为它们会改变 planner turn 的位置含义。

筛选发生在 SLIC 组建之前，因此被排除的 turn 不进入 baseline、候选 pair、
top-K 或 replay buffer。GRPO 仍使用完整 batch，`TOOL_STEPS` 仍是 5。
此开关仅影响包含 `slic` 的算法，其他 preference loss 保持原有行为。

final 的识别使用过滤前每条 rollout 的最大 turn，支持提前 STOP 导致的变长轨迹。
超长 prompt 被标记丢弃时，不会把它前面的工具 turn 提升为 final。
现有 `uid-turn` 配对规则保持不变：同一个问题、同一 turn 序号配对，
final 与非 final 不互配。不同长度轨迹的 final 序号不同，仍不会互配。
本实验仅缩小可配对范围，不新增跨序号 final 配对。

## 运行与对照

在训练服务器仓库根目录，用现有启动入口执行三组独立训练，
保持模型初始权重、训练数据、seed、采样温度、batch、学习率与评估设置相同。
`run_train.sh` 会照常启动 rollout 与训练进程；多机任务也可将相同环境变量
传给现有多机启动入口。

```bash
# A：GRPO，不使用 SLIC，保留 5 步工具预算
ALGORITHM=grpo SLIC_TURN_SCOPE=all bash train-roma/run_train.sh

# B：GRPO + SLIC，所有 turn，与原有逻辑相同
ALGORITHM=grpo+slic SLIC_TURN_SCOPE=all bash train-roma/run_train.sh

# C：GRPO + SLIC，仅 analysis、首轮工具选择与 final
ALGORITHM=grpo+slic SLIC_TURN_SCOPE=analysis_first_tool_final bash train-roma/run_train.sh
```

三组配置的 `TOOL_STEPS` 都保留为 5，而不是将 C 的工具预算改成 1。
已有的 `TOOL_STEPS=1` 结果可作为补充对照。
如已有 rollout 服务单独运行，也可只给训练入口增加参数：

```bash
python train-roma/train_agent.py ++data.slic_turn_scope=analysis_first_tool_final
```

恢复原有 SLIC 范围只需设置 `SLIC_TURN_SCOPE=all`。

## 观测与解释

除了原有验证得分与 `replay/n_new_pairs`、`replay/n_dpo_batch`，新增：

- `replay/slic_candidate_turns`：排除超长 prompt 后、应用 turn 筛选前的候选数。
- `replay/slic_filtered_turns`：因本次 turn 筛选而排除的数量。
- `replay/slic_pairs_analysis`、`replay/slic_pairs_first_tool`、
  `replay/slic_pairs_middle`、`replay/slic_pairs_final`：各阶段 top-K 选出的 pair 数，
  统计于每步 256 对上限与分区补齐之前；阶段按 chosen 一侧归类。

C 组的 `slic_pairs_middle` 应始终为 0。
该配置只决定 turn 是否有资格配对，不保证三类 turn 每步都有有效 pair；
同组需要同时存在正负 reward，且通过长度等原有过滤。

中间 turn 的上下文会受前面不同工具选择和结果影响，仅按 turn 序号匹配
可能带来偏好标签噪声。轨迹最终 reward 也不保证每个中间决策本身都更好。
但 `TOOL_STEPS=1` 同时改变轨迹长度、工具反馈、reward 分布和 SLIC 更新量，
所以它的收益尚不能单独证明“中间 turn 的配对有害”。

若 C 优于 B，说明限制配对范围有帮助；仍需关注 C 的 pair 数与实际 SLIC
更新量是否下降，因为 replay 采样量会随入队数量变化。
如需进一步区分“配对质量”与“减少 SLIC 更新量”的影响，应额外做更新预算匹配的对照。
analysis 与首轮工具选择也有各自的上下文差异，final 还依赖不同工具结果，
因此保留这三类 turn 也不能完全消除配对偏差。

## 本地验证

```bash
python3 -m unittest discover -s agentflow/tests -p 'test_slic_prompt_filter.py' -v
```

测试直接执行真实配对函数，使用 CPU tensor，无需加载分布式 worker。
它验证 turn 范围、原有 all 模式、变长轨迹、丢弃 final、baseline 隔离、
分区补齐、其他 loss 不受影响及配置到训练参数的传递。

本地验证：16 项测试通过；另将 `all` 模式与修改前提交的配对函数直接比较，
覆盖 36 种组合（3 种分组模式、3 种轨迹长度组合、有无丢弃标记、QA/非 QA、
8 分区补齐），所有 pair tensor、非 tensor 字段与元数据完全一致。
尚未运行 GPU/NPU 正式训练，此处验证不代表训练收益。
