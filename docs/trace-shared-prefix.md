# TRACE 共享前缀 GT 评分

本实现迁移 IGPO 论文 Figure 3 的计算方式：相同前缀只前向一次，多个 GT 分支通过自定义四维 attention mask 相互隔离。它不是把独立序列合并为普通 batch，也不将 TRACE 换成 IGPO 的奖励公式。

参考代码固定到官方 [IGPO `64165e2` 的自定义注意力实现](https://github.com/GuoqingWang1/IGPO/blob/64165e2741ed8801f977948c8128080ce87b4101/scrl/llm_agent/vectorized_gt_logprob.py)。该仓库正式 generation 入口实际使用另一条普通批量评分路径；本次迁移的是论文中的共享计算机制。

## 完整状态如何共享

ALFWorld 每个 turn 的当前事实与观察会重写（当前 full-state Prompt 已移除累计动作/观察历史），不能假设后一轮完整文本是前一轮的追加。本实现直接使用现有评分器已经生成的 `prompt_ids` 和 `target_ids` 构建精确 token 前缀树，既共享同题的规则与初始事实，也共享任何确实一致的更长前缀。分包只影响计算组织，不改文本、token、GT 长度检查或评分目标。

每个物理节点保存 token、父节点和逻辑深度。对于 query 节点 q，只允许它关注自身及祖先节点；其它分支、未来状态和其它 GT 均不可见。RoPE 的 position ID 使用逻辑深度，而非节点在拼接数组中的物理位置。

每个 GT token 从其逻辑父节点的 logits 取概率。特别是 GT 的第一个 token 使用对应上下文最后一个节点的 logits，不能使用数组中前一个分支末尾的 logits。这一处理避免官方示意实现中直接使用 `gt_start - 1` 的对齐风险。

逐 token LogP 回填到原请求行后，继续使用原 `mean_gt_logps`、V、K-step credit、终局回填与同 turn 配对逻辑。完全相同的状态仍先使用原有去重。

## 配置

正式 `train-roma/config.yaml` 和 `config_grpo+slic.yaml` 默认请求共享评分。已有自定义配置若未设置新字段，仍采用原批量评分方式。对应环境变量为：

```bash
export TRACE_SHARED_PREFIX_SCORING=true
export TRACE_SHARED_PREFIX_MAX_TOKENS=8192
export TRACE_SHARED_PREFIX_MAX_MASK_MIB=256
export TRACE_SHARED_PREFIX_VERIFY=true
```

设 `TRACE_SHARED_PREFIX_SCORING=false` 恢复原 GT 批量评分。四个字段位于 `data.trace.shared_prefix_*`；不需要重新生成 GT Parquet，不需要改变 GRPO、rollout 或 actor 的 SP/microbatch。

token 数和 mask 预算共同限制每个树包的大小；超限不能截断输入或伪造评分。mask 预算仅覆盖四维 additive mask，不是整个模型峰值内存；eager attention 还需要与 head 数相关的中间张量。默认规划也限制树包的稠密 attention 规模，避免把几乎不共享前缀的长序列强行合并。

## Reference 与分布式边界

新入口仅用于冻结 reference 的 GT 评分，原 actor forward/backward、rollout、动作 LogP 和配对流程继续沿用原路径。树评分对现有 reference FSDP 模块临时选择支持任意 4D mask 的 eager attention，结束后恢复配置。首版支持纯文本 Qwen2/Qwen3 的标准 RoPE；不支持的模型、LoRA、fused forward 或其它不满足条件的配置会明确记录回退原因。

启动时可以保留 SP=2；但树评分绕开原 remove-padding/SP 序列切片，SP 组内各 rank 对完整树重复前向，只有原收集 rank 的结果回传。它不意味着树评分仍享有 SP=2 的序列并行收益。actor 训练的 SP 行为不变。

不同 rank 的树包数量可能不同。模型前向前统一协商计划与最大轮数，不足者执行不回填结果的 dummy 前向，确保 FSDP 参数通信顺序一致。规划或能力回退在所有相关 rank 上统一决策，不能让一个 rank 中途独自改走另一条通信路径。

首次共享评分默认与原评分路径作逐 GT token 数值对照，因此第一次会额外执行原路径，可能暂时增加首步耗时。数值不匹配时统一使用原结果，并在后续停止重复尝试该共享路径。后续成功验证的共享评分无需每步再执行原评分。BF16 与不同 attention backend 不承诺逐 bit 相同；有效性判断与最终返回精度应遵循原评分口径。

另一个原有性能因素是 reference 的 FSDP 包装会强制 CPU offload，即使 YAML 的 `ref.fsdp_config.param_offload=false`；因此之前 GT 评分的 113 秒也可能包含参数搬运。本次不改这项既有行为。共享评分减少前向次数时，也可能减少部分参数搬运。

## 验证与性能解读

新测试覆盖树路径、分支隔离、位置编码、首 GT token、不同 GT 长度、预算分包、配置路由和回退。真实小型 Qwen2/Qwen3 在 CPU 上逐 token 对照独立 teacher forcing；这不等于服务器 4B 模型、Ascend 或多机 HCCL 的完整训练验证。

本地随机小型 Qwen3 合成基准中，16 条请求共享 256-token 前缀，原始 token 总数 4288，树节点数 448，节省约 89.6%；逐 GT token LogP 最大差为 0。该微基准不能推导生产环境加速倍数。稠密 4D attention 仍与树包长度平方相关，实际速度还受共享程度、分包、SP 重复前向、reference 参数搬运和内存影响。

本地两进程 Gloo 测试也已通过：两端本地 pack 数不同，需要一端补 dummy 前向，仍能完成首批校验、后续免重复校验、跨 rank 一致回退，并返回一致的指标。它使用 CPU attention 模型模拟每次 FSDP 前向中的 collective，不代表真实 Ascend/HCCL/FSDP 已完成验证。可在服务器仓库中单独复现：

```bash
TRACE_TEST_PREFIX_DISTRIBUTED=1 python -m unittest discover \
  -s agentflow/tests -p 'test_trace_prefix_distributed.py'
```

服务器应观察下面的指标，并比较第 2 步以后的 `trace/scoring_seconds`。不能只看到开关为 true 就认定已经共享计算。

| 指标 | 含义 |
| --- | --- |
| `trace/prefix_requested` | trainer 是否请求新路径 |
| `trace/prefix_used` | 本批是否真正使用树评分结果 |
| `trace/prefix_saved_tokens` | 相对独立评分少算的 token 数，已排除 SP 复制计数 |
| `trace/prefix_original_tokens` / `tree_tokens` | 独立输入总 token 数与树节点总数 |
| `trace/prefix_forward_rounds` | 各 rank 对齐后的树前向轮数 |
| `trace/prefix_dummy_forwards` | 为保持通信顺序而增加、不回填结果的前向 |
| `trace/prefix_attention_cost_ratio` | 树包稠密 attention 格数与独立请求格数的比值 |
| `trace/prefix_validation_checked` / `passed` / `max_diff` | 是否执行数值校验、是否通过、最大逐 token 差值 |
| `trace/prefix_fallback_*` | 是否因模型、预算、无共享收益或数值校验等原因回退 |

所有 fallback 指标在各 rank 上统一。`prefix_used=0` 时不能把该批耗时解释为共享前缀评分的性能；如果 `fallback_planning=1`，应先检查输入长度与两个包预算，不能通过截断完整状态来强行启用。
