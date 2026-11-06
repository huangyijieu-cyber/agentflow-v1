# WebShop-SLIC v1

`webshop` 分支以 ALFWorld `5690b9c` 为迁移基线。WebShop 使用按题离线生成的候选商品 GT、当前真实页面状态和最近四条原生历史，复用已有 GTLogp、V、K-step 信用、动态终局回填、价值匹配配对及 SLIC loss。`credit.py` 的公式与配对函数保持不变。

## 固定 GT 与数据

模板版本为 `webshop-candidate-state-v1`。每题依据实际环境 goal 组装一次，在所有 rollout/turn 中保持完全一致：

```text
The current candidate product matches the product type required by the shopping instruction.
It satisfies the required attributes: ... .
Its price satisfies the task's price limit.
The selected options for this product are: ... .
```

无实质价格约束时省略价格句；`goal_options` 为空时省略规格句。当前原生生成器的无价格文案 sentinel 为恰好 `1000000`，原值仍保留供判奖。未知或损坏字段不当成无约束。字典规格按稳定键序输出，只有值的列表不猜测颜色/尺寸键。不加入购买完成陈述，不要求购买目标 ASIN，不从参考商品标题增加要求。

原训练集的 `question` 是整数环境索引（如 `6684`），不能改成需求文本。转换保留原 `id/question/source/extra_info`、schema、行数和行序，仅填充 `result` 并追加目标、模板、有效性和哈希字段。有效行的 `webshop_goal_spec.original_goal` 保存完整原始目标，`result` 是训练采样的权威评分 GT。

在实际训练服务器、原 WebShop 环境中，从仓库根目录运行以下命令：

```bash
python train-roma/prepare_webshop_gt.py \
  --input data/train/webshop_train_6810_webshop.parquet \
  --webshop-root data/WebShop-master \
  --goal-mode env-default \
  --expected-rows 6810
```

`env-default` 与当前 adapter 一致，`human_goals` 参数为 `None`。源码、商品文件、属性文件、索引必须来自实际训练环境；不要根据 6810 这个行数推断 goal 模式。当前环境工厂使用默认构造参数；若自行改动导出的 goal 模式、商品数量或数据文件，须同步修改环境构造参数，否则运行时会拒绝不匹配的 manifest。默认产物为：

```text
data/train/webshop_train_6810_webshop_with_gt.parquet
data/train/webshop_train_6810_webshop_with_gt.manifest.json.gz
data/train/webshop_train_6810_webshop_with_gt.report.json
```

现有独立候选模板转换脚本产生的同版本文件兼容。若已导出旧购买模板，使用原始空 `result` 的 Parquet 和旧 manifest 重新生成候选 GT，避免重新随机初始化目标和价格：

```bash
python train-roma/prepare_webshop_gt.py \
  --input data/train/webshop_train_6810_webshop.parquet \
  --manifest-in /path/to/existing.manifest.json.gz \
  --output data/train/webshop_train_6810_webshop_candidate_gt.parquet \
  --expected-rows 6810
```

所有输出防覆盖；默认遇无效行只写失败报告，不产出训练文件。`--allow-invalid` 仅供诊断，运行时拒绝无效行。修正输入后使用新的输出路径重试。`--manifest-in` 模式只需 Python 和 PyArrow，不初始化原生 WebShop。

## 冻结环境与启动

WebShop 的区间商品价格和目标上限在初始化时可能随机生成。manifest 保存完整有序目标清单、完整实际价格表以及源码/数据内容指纹；新导出的快照不声称还原旧训练 run。训练和验证均安装同一 manifest，再按原整数题号 reset。安装检查实际配置与资源内容，替换目标/价格、重建采样权重、清除旧 session。每题再次核验实际 goal、Parquet spec、固定 GT 和哈希；同一 uid 内不允许混合不同目标或 manifest。

两份训练配置 `train-roma/config.yaml`、`train-roma/config_grpo+slic.yaml` 默认：

| 配置 | 默认 |
|---|---|
| `TASK` / `ALGORITHM` / `OUTPUT_TYPE` | `webshop` / `grpo+slic` / `base` |
| `TRAIN_DATA_PATH` | `train/webshop_train_6810_webshop_with_gt.parquet` |
| `TEST_DATA_PATH` | `val/webshop_test_100_webshop.parquet` |
| `WEBSHOP_TRACE_MANIFEST` | `data/train/webshop_train_6810_webshop_with_gt.manifest.json.gz` |
| `WEBSHOP_ROOT` | 空值使用仓库 `data/WebShop-master` |
| `WEBSHOP_TRACE_HISTORY_LENGTH` | `4`；支持 `0`、`None`/`null` |
| `TRACE_REFERENCE_UPDATE_INTERVAL` | `None`：单 step 内冻结，完整更新后刷新供下一步使用 |
| `TRACE_K` / `TRACE_GAMMA` / `TRACE_EPSILON` | `3` / `0.8` / `0.1` |
| `TRACE_MATCH_EPSILON` | `0.05`；已有环境变量覆盖继续生效 |
| `TRACE_K_PAIRS_PER_GROUP` / `TRACE_PAIR_ROUND_ROBIN` | `None` / `false` |
| `TRACE_MAX_PAIRS_PER_STEP` | `256` |

如使用自定义数据文件，必须同时指定对应 manifest。例如以上第二个转换命令对应：

```bash
export TRAIN_DATA_PATH=train/webshop_train_6810_webshop_candidate_gt.parquet
export WEBSHOP_TRACE_MANIFEST="$PWD/data/train/webshop_train_6810_webshop_candidate_gt.manifest.json.gz"
export WEBSHOP_ROOT="$PWD/data/WebShop-master"
python train-roma/train_agent.py
```

其余模型、集群和 rollout 服务按已有部署方式启动。更新代码及配置后重启 Trainer 和所有 Rollout Workers，使用新的训练 run/replay；不要混入旧模板或旧评分上下文的偏好对。所有运行环境必须能读取相同内容的 Parquet、manifest 和原生商品资源。

验证不需要额外转换 GT；它关闭评分侧 TRACE、继续使用原 benchmark 奖励，但恢复同一冻结目标与价格。显式关闭 TRACE 或使用不含 `slic` 的算法，不要求转换后的数据或 manifest。

## 评分和训练边界

- 搜索页只列本页实际出现的商品，保持原顺序；商品页只使用当前商品，不读取未访问的目标商品。完整属性、标题、Features、Description、实际判奖价格属于评分侧额外信息。
- 任务条件与当前事实分开，多个商品的属性不能拼接为一个合格商品；可选规格不等于已选规格。当前商品、实际 `options`、终局成交快照从会话读取，不按动作文本猜测。
- 历史使用原生动作前 observation 和解析 action，默认最近四条，保留原序号。评分 prompt 独立于 actor prompt，终局观察剔除分数及匹配标签。
- 每个真实动作保留前后状态；T 个动作有 T+1 个评分状态，购买后的 S_T 不增加 actor 训练行。相邻动作的后状态和下一动作前状态必须一致。外部故障、缺失状态及轨迹缺口继续隔离信用窗口。
- 所有 WebShop 动作使用 `tool_plan` / TRACE 路由，不使用 QA final-answer 路由。动作身份和 TRACE 去重按原评分状态加实际动作处理；不同商品上的 `click[buy now]` 不因纯文本相同被删。
- `info.task_score` 的原始连续奖励用于同题终局优势及动态回填；`reward == 1.0` 才是完全成功。GRPO 仍接收原 adapter 的满分 1、其余 0。可靠预算耗尽记 0；外部异常及输出截断不当成购物失败。
- 回填仍限于最后三个真实动作，按距离折扣及同题正负局部信用范围缩放；缺少某个方向局部信用时该方向回填为零。固定 `terminal_fill_scale` 对此环境路径不生效。
- chosen/rejected 保留各自的原始 actor prompt/response token IDs。GT 只参与参考模型 teacher forcing，`trace_*` 字段在 actor RPC 前剔除。

## 验证

轻量 CPU 回归覆盖真实生产方法，原生环境和参考模型用可控 stub 替代：

```bash
python -m unittest discover -s agentflow/tests -p 'test_webshop_*.py'
python -m unittest discover -s agentflow/tests -p 'test_trace_*.py'
python -m unittest discover -s agentflow/tests -p 'test_alfworld_*.py'
```

这些检查不等于实际任务效果验证。本地没有完整商品数据、服务器训练集和分布式模型运行环境。正式训练前，用同一冻结参考模型检查固定真实轨迹：找到合适候选、选对/选错规格、买对/买错商品、返回列表和重复动作；观察 GTLogp、V、local credit、terminal fill 和最终 pair 数。购买前后商品与规格未变时局部增益可以接近零，购买决策主要由真实终局监督；仍须验证回填方向与候选信用是否合理。
