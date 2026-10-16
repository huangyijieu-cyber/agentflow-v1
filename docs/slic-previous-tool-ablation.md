# SLIC：相同 turn、相同上一轮工具的配对实验

实验分支 `test_tool` 基于 GitHub `cache` 的 `1ca2c83`，独立于 `test_first` 的三类 turn 筛选实验。
这次保留所有 turn 的配对资格，只增加候选 pair 检查：

```text
原有配对条件
AND chosen.turn_index == rejected.turn_index
AND chosen.previous_tool_name == rejected.previous_tool_name
```

“相同 turn 数”按 turn 序号理解，例如 turn 3 只能和 turn 3 配对。
“上一条工具”是本轮生成之前最近一次 planner 工具选择，使用实际执行路径
解析出的规范工具名；本轮 response 选择的工具仍可以不同。

| 当前 planner turn | 上一轮工具来自哪里 | 筛选行为 |
| --- | --- | --- |
| 初始 analysis | 尚无工具选择 | 显式记录 `__no_previous_tool__`，按原规则配对 |
| 首次工具选择 | 尚无工具选择 | 同上；仍须同 turn 序号 |
| 第二次及之后的工具选择 | 上一次 `generate_next_step` 的工具选择 | 工具名相同才允许配对 |
| final | 最后一次工具选择 | 工具名相同才允许配对 |

例如两条轨迹此前分别为 `Web → Python` 和 `Wiki → Python`，
接下来一轮的 previous tool 都是 Python，因此仍允许配对。
本实验只限制最近一次工具选择，不要求整个工具历史一致，
也不要求工具 query、返回结果或当前上下文相同。
同工具只能减少一种配对偏差，不能保证状态完全可比。

## 保留的原有行为

- `REPLAY_COLLECTION` 的分组方式不变；`cross` 模式也额外要求相同 turn 序号。
- QA 的 final/非 final 隔离、超长 prompt 丢弃、长度比过滤、margin 分位阈值、
  hardness 排序和每组 top-K 均保留。
- 新条件在 top-K 之前检查，因此不会让不符合工具条件的 pair 占据 top-K 名额。
- baseline 和 group size 仍从原有组计算；上一轮工具不同的样本仍参与这些统计，
  不会按上一轮工具重分组。
- 中间 turn、analysis、final 都可能入选；不叠加 `test_first` 的 turn 范围限制。
- 完整 rollout 仍用于 GRPO，`TOOL_STEPS` 保持 5。

## 配置与运行

新增 `data.slic_match_previous_tool`，只对 QA 的 SLIC 配对生效。
未传入该字段的旧配置默认 `false`；本次修改的 `train-roma/config.yaml`
及 `config_grpo+slic.yaml` 默认 `true`。

在训练服务器仓库根目录，用原有入口独立运行对照组和实验组：

```bash
# 对照：原有全 turn SLIC 配对
ALGORITHM=grpo+slic SLIC_MATCH_PREVIOUS_TOOL=false bash train-roma/run_train.sh

# 实验：全 turn SLIC + 同序号、同上一轮工具限制
ALGORITHM=grpo+slic SLIC_MATCH_PREVIOUS_TOOL=true bash train-roma/run_train.sh
```

也可给训练入口传入 `++data.slic_match_previous_tool=true`。
两组均保持 5 步工具预算；模型初始权重、数据、seed、温度、batch 与评估设置保持一致。
纯 GRPO 和已有 `TOOL_STEPS=1` 结果可作为补充对照。

训练端与 rollout 端都需使用本次代码并重启：工具信息通过
`Planner log → Triplet.metadata → daemon non_tensor_batch → SLIC` 传递，
不从可能截断的 prompt 或 response 中反推。
每个 QA solve 会重置上一轮工具状态，提前 STOP 时 final 使用实际最后一次选择。

无上一轮工具使用显式边界标记；解析失败、工具无效或旧日志缺失信息则是 `None`。
两条未知信息不会被当作相同工具放行。
训练 batch 完全缺少 `previous_tool_name_list` 字段时会给出重启 rollout 的错误信息；
有字段但某行未知时，只跳过相关 pair，保留 GRPO 数据。

## 日志与结果解释

以下指标均以 `replay/slic_prev_tool_` 为前缀：

| 指标后缀 | 含义 |
| --- | --- |
| `candidate_pairs` | 通过原有 QA 阶段隔离、待接受新条件检查的正负候选对 |
| `turn_mismatch_pairs` | 因 turn 序号不同而拒绝 |
| `unknown_pairs` | 因任一侧上一轮工具信息未知而拒绝 |
| `mismatch_pairs` | 因上一轮工具名不同而拒绝 |
| `matched_pairs` | 通过新检查的候选对，之后仍需通过原有 margin/长度过滤和 top-K |
| `selected_pairs` | top-K 选中数，位于每步 256 对上限与分区补齐之前 |

入队和实际训练量仍看 `replay/n_new_pairs` 与 `replay/n_dpo_batch`。
如果 `unknown_pairs` 很多，应先检查 rollout 是否已更新、工具选择是否经常解析失败。
筛选会减少入队量，原有 replay 采样量又随入队速率变化；
实验收益可能同时来自配对质量变化和 SLIC 更新量下降。
若需区分这两个因素，应额外做更新预算匹配的对照。

## 本地验证

```bash
python3 -m unittest discover -s agentflow/tests -p 'test_slic*.py' -v
python3 -m unittest discover -s agentflow/tests -p 'test_vllm_native_token_ids.py' -v
```

测试执行真实 Solver、planner 日志、Triplet 编码、daemon batch 构建与配对函数，
用 CPU tensor 和模型调用替身覆盖元数据传递、规范工具名、提前 STOP、复用 planner、
未知工具、空 trace/超长 prompt 对齐、final、同序号约束、baseline 与 top-K 行为。
本地验证不包含 GPU/NPU 正式训练，也不代表训练收益。

本次验证结果：23 项 SLIC/元数据路径测试、8 项原生 token 记录测试全部通过。
另将关闭新开关的函数与原 `cache` 提交直接比较，覆盖 36 种组合
（3 种分组方式、3 种轨迹长度组合、有无丢弃标记、QA/非 QA、8 分区补齐），
输出 pair 的 tensor、非 tensor 字段及元数据完全一致。
