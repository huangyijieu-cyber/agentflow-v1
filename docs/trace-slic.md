# TRACE 状态价值与 SLIC 配对

本实现用冻结参考模型对 GT 的可预测性变化分配 Turn 信用，再按相近任务进度构造 SLIC 偏好对。默认对包含 `slic` 的算法启用；GRPO 的奖励、优势计算和现有 SLIC loss 代码保持原样。

## 状态与 GT 评分

每个可训练 Turn 都保存自己的评分上下文：从该次实际使用的 Prompt 中去掉当前决策的 Task、指令和输出格式要求，保留原有问题、上下文与工具信息，再附加统一回答指令：

```text
Answer the question directly and concisely.
```

评分上下文来自该 Turn 的原 Prompt，不通过 memory 另行重建。下一状态使用同一 Rollout 中下一个真实可训练 Turn 的评分上下文。因此 query analysis、工具规划、最终回答等类型都可计算价值与信用；相邻两个状态的角色和模板可能不同，去除决策指令只能减小模板影响，不能保证完全消除。

QA runner 自动追加的 `<answer>` 输出格式要求也只从评分上下文中移除，以便评分裸 GT；使用本次完整输入与原始问题的精确映射，保留用户问题自带的要求。真实生成 Prompt 与原生 Token IDs 保持原样。

对问题 $x$ 的第 $g$ 条 Rollout，在第 $t$ 个状态使用固定 GT $y^*$ 做独立 teacher forcing：

$$
\bar\ell_{g,t}=\frac{1}{|y^*|}\sum_{j=1}^{|y^*|}
\log\pi_{\mathrm{ref}}(y^*_j\mid S_{g,t},y^*_{<j}),
\qquad
V_{g,t}=\log\frac{-\bar\ell_{g,0}+\epsilon}{-\bar\ell_{g,t}+\epsilon}.
$$

参考模型冻结，评分温度固定为 1；仅对 GT Token 求平均，不将生成动作的 logP 当作 GT logP。GT 只进入独立评分请求，不写入 agent Prompt、工具输入或 Rollout 采样上下文。GT 有多个别名时，固定选择第一个非空答案，避免随状态选择最高概率别名。

空 GT、超过 `max_gt_tokens` 的 GT 和无法完整容纳的评分上下文跳过 TRACE 评分，不截断 GT 或状态来制造价值。缺失、非有限的评分不会被当作 0，也不会改用最终奖励补造信用。

## 相邻 Turn 信用

第一版默认 `k=1`：

$$
c_{g,t}=\delta_{g,t}=V_{g,t+1}-V_{g,t}
=\log\frac{-\bar\ell_{g,t}+\epsilon}{-\bar\ell_{g,t+1}+\epsilon}.
$$

配置保留 K-step 折扣平均形式，便于后续实验：

$$
c_{g,t}^{(K)}=
\frac{\sum_{u=t}^{h}\gamma^{u-t}(V_{g,u+1}-V_{g,u})}
{\sum_{u=t}^{h}\gamma^{u-t}},\qquad
h=\min(t+K-1,N_g-2).
$$

其中 $N_g$ 为实际记录的可训练状态数，$0\leq t<N_g-1$。只使用存在真实后继状态的增益；窗口不会跨 Rollout、缺失 Turn 或无效状态。`k=1` 时 `gamma` 不影响结果。

最后一个可训练 Turn 没有自然后继，其信用记录为 NaN，不进入偏好对。实现不会拼接已生成答案来虚构后状态，也不会把 terminal 价值设置成 0。这意味着典型 QA 轨迹的最后回答 Turn 仍由原 GRPO 训练，但没有 TRACE 偏好监督；这是当前基于“下一个真实 Turn”定义的边界限制。

## 偏好对与 Loss

候选对必须来自同一问题 `uid`、同一决策类型，允许不同 Turn 和不同 Rollout；使用价值匹配取代原先同 turn 的分组限制。类型由生成调用的角色决定，避免将分析、工具规划和回答动作相互比较。

$$
|V_i-V_j|\leq\epsilon_{\mathrm{match}},\quad
c_i^{(K)}>\delta_{\mathrm{pos}},\quad
c_j^{(K)}<-\delta_{\mathrm{neg}},\quad
c_i^{(K)}-c_j^{(K)}\geq\epsilon_{\mathrm{gap}}.
$$

Chosen 和 Rejected 必须严格正、负；零信用、中性动作不能充当 Rejected。没有有效负信用样本时直接跳过该组，不以成功/失败轨迹标签、最终 reward 或 Anchor 文本相似度兜底。工具执行失败、重复动作和无效 Response 按有效性标记过滤。保留响应长度比和采样数量限制，避免少数问题产生过多样本对。

SLIC loss 继续分别在两个动作各自的原始 Prompt 下计算策略 logP，执行现有的 per-token calibration 与 Chosen SFT，不把两个动作搬到公共 Prompt。GT 价值只用于选对，不替换 loss 中的策略 logP，也不向 loss 新增 reference ratio。传入 SFT 的 baseline 仍为旧策略的 per-token logP 均值，候选邻域改为同问题、同类型、相近价值的状态；不采用 GT logP 作为 baseline。

## 配置与推荐初值

全部参数位于 `data.trace`；基础声明在 `agentflow/verl/config.yaml`，`train-roma/config.yaml` 与 `train-roma/config_grpo+slic.yaml` 提供对应的 `TRACE_*` 环境变量。

| 参数 | 初值 | 含义 |
| --- | --- | --- |
| `enabled` | `true` | 仅算法包含 `slic` 时生效；关闭则恢复 legacy SLIC 配对 |
| `epsilon` | `1e-8` | NLL 比值的数值稳定项 |
| `k` | `1` | 信用窗口长度，先验证单步增益 |
| `gamma` | `0.9` | 多步信用折扣；K=1 时不生效 |
| `match_epsilon` | `0.1` | 动作前价值差上限；同初始状态下 NLL 比不超过约 `exp(0.1)=1.105` |
| `positive_threshold` | `0.01` | Chosen 信用须大于此值，约对应 GT 平均 NLL 下降超过 1% |
| `negative_threshold` | `0.01` | Rejected 信用须小于其负值，约对应 GT 平均 NLL 上升超过 1% |
| `gap_epsilon` | `0.02` | 正负信用差下限；配合严格正负阈值屏蔽接近零的差异 |
| `length_ratio_max` | `1.8` | 正负 Response 的最大长度比 |
| `k_pairs_per_group` | `2` | 每个候选组保留的最大对数 |
| `max_pairs_per_step` | `256` | 每训练 step 新增偏好对上限 |
| `max_gt_tokens` | `256` | GT Token 上限，超长时跳过而非截断 |
| `gt_alias_strategy` | `first` | 固定第一个非空 GT；当前只支持此策略 |
| `answer_instruction` | 见上文 | 统一的简短回答指令 |

这些数值是实验起点，不代表已调参最优。观察有效状态、正负信用样本和最终 pair 数量；若负样本稀少，应检查价值分布和阈值，不能放宽为非正样本或用失败轨迹代替。

从仓库根目录启动，环境变量可以覆盖 YAML 初值，命令行参数会在 YAML 参数之后追加并最终覆盖：

```bash
TRACE_MATCH_EPSILON=0.05 TRACE_POSITIVE_THRESHOLD=0.005 python train-roma/train_agent.py
python train-roma/train_agent.py data.trace.k=1 data.trace.enabled=false
```

`load_and_set_env_from_yaml` 先解析 `${env:VAR,default}` 并设置环境变量；`train_agent.py` 将 `${TRACE_*}` 展开为独立的 `key=value` 参数交给 Hydra。基础配置已声明 `data.trace.*`，这些参数不使用 `+` 前缀。布尔值请使用 `true/false`；回答指令作为单个参数传递，包含空格时无需修改训练脚本。

`TRACE_ENABLED=false` 或 `data.trace.enabled=false` 恢复原 SLIC 样本构造；不含 `slic` 的算法不会启动 TRACE 评分。

## 资源与解释限制

即使 KL loss 关闭，TRACE 仍需要冻结参考模型的 GT logP，因此可能额外加载一份参考模型权重并增加显存/内存占用。每个可训练状态还需额外的 teacher-forcing 前向计算；长上下文和长 GT 会增加耗时，应结合现有 reference micro-batch 配置控制开销。

价值提升衡量的是固定参考模型对 GT 的可预测性改善，并非动作的因果贡献证明。相近标量价值不保证状态拥有相同信息；角色模板变化、参考模型偏差和工具随机性仍可能影响信用。第一版应与 legacy SLIC 对照，重点检查不同类型的信用分布、可用偏好对比例以及最终任务质量。

## 本次验证

基线为远程 `cache` 的 `8083f0d7159ed2058b93e4460417c09fe2b11029`。

```bash
python3 -m unittest discover -s agentflow/tests -p 'test_trace_*.py' -v
python3 -m unittest discover -s agentflow/tests -p test_vllm_native_token_ids.py -v
git diff --check
```

45 项新增 CPU 测试和 8 项原生 Token 回归通过，覆盖状态链路、真实 Trainer 方法、GT 掩码、参考模型路由、LoRA adapter 隔离、跨 Turn 配对及关闭 TRACE 的回退。分布式模型前向在 CPU 测试中使用 stub，不代表已完成 GPU/NPU 训练验证。两份启动配置另通过默认值及环境变量/命令行覆盖共 4 组真实 Hydra/OmegaConf 解析验证。

现有 `test_slic_prompt_filter.py` 的 5 项旧测试因 fixture 缺少最新配对要求的 `anchor_list` 报错；在未修改的同一远程基线上复现了相同的 5 项错误。该旧 fixture 未在此次修改中重写。
