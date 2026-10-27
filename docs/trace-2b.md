# v2_b：IGPO 风格优势用于 SLIC 配对

v2_b 在 `v2_a@5fbe4cc` 上实现，分支为 `trace-2b`。它替换工具 Pair 的信用构造，保留 analysis／实际 final 的可信终局标签。SLIC Loss、旧策略 per-token LogP baseline、GRPO 奖励与优势算法、Replay 更新强度均保持原样。

实现与 CPU 集成回归已完成，版本见该分支的 Git 提交记录；尚未进行 GPU/NPU 训练收益验证。

当前默认 `data.trace.pair_score_mode=igpo`；设置为 `trace_fill` 可恢复 v2_a 的 K=3 与终局回填构造。设计依据和取舍见 [设计方案](trace-2b-design.md)。

## 信号与公式

沿用 v2_a 的评分上下文、冻结 reference、温度 1 和 GT Token 平均 LogP。GT 仅进入独立评分请求，不进入 agent 的生成上下文。状态价值继续用于配对前的进度匹配：

$$
V_{g,t}=\log\frac{-\bar\ell_{g,0}+\epsilon}{-\bar\ell_{g,t}+\epsilon},
\qquad |V_i-V_j|\leq0.05.
$$

工具即时信息增益改为：

$$
r^{IG}_{g,t}=\bar\ell_{g,t+1}-\bar\ell_{g,t}.
$$

按同题 uid，将有效工具转移的信息增益和可信 Rollout 的终局奖励分别做 z-normalization，标准差使用总体口径（ddof=0）：

$$
z^{IG}_{g,t}=\frac{r^{IG}_{g,t}-\mu^{IG}_x}{\sigma^{IG}_x+\epsilon_{norm}},
\qquad
z^O_g=\frac{R_g-\mu^O_x}{\sigma^O_x+\epsilon_{norm}}.
$$

IG 人口取当前 batch 的完整可信工具后缀，在按正负和 gap 选对前统计；同一真实转移只计一次。终局人口每条可信 Rollout 一票，包含没有工具的可信 Rollout，不按 Turn 复制奖励。重复副本不增加人口权重，真实 IG=0 仍是有效观测。

工具的最终优势为：

$$
A^{2b}_{g,t}=w_{IG}\sum_{\substack{u\in\mathrm{tool}\t\leq u<f}}
\gamma^{u-t}z^{IG}_{g,u}+w_O\gamma^{f-t}z^O_g.
$$

默认 `w_IG=w_O=1、gamma=1`。这是累计和，不除以窗口权重和；回传区间为当前工具到实际 final。折扣按原始 Turn 编号计算，final 占一个位置。analysis 不产生 IG、不进入工具 IG 统计，也不接收该优势。final 没有 GT 后继状态，不构造 `delta_final`。

IGPO 模式不再执行旧 K=3 信用和 `terminal_fill_scale=2` 回填，避免终局重复计权；`k`、旧 `gamma` 和 `terminal_fill_*` 仅在 `trace_fill` 模式生效。

参考 [IGPO §3.2–3.3](https://arxiv.org/html/2510.14967v2#S3)。本实现保留 frozen reference、analysis／final 的独立路由，并将优势用于 SLIC 选对；论文则使用策略评分并将回报用于策略优化。因此本版本是 IGPO-inspired，不是完整复现，也没有切换到仓库现有的 GiGPO。

## 配对与符号含义

| 类型 | 标签与分数 |
| --- | --- |
| analysis | 可信 verifier 成功／失败，沿用 v2_a |
| tool_plan | 有效 `A^(2b)` 严格正／负 |
| 实际 final_answer／direct／base | 可信 verifier 成功／失败，沿用 v2_a |

工具 Pair 仍要求同 uid、同类型、相近 V、合法且不同的动作、长度比符合上限。通过条件后按优势差降序排序，V 距离和旧策略平均 Token LogP 距离用于打破平局。没有合格正负两侧时跳过，不回退到旧信用或原始最终 reward。

**正、负优势表示组内相对累计收益，不等同于当前动作使 GT LogP 上升或下降。** 单步 IG 为正也可能是负优势；单步 IG 为负但未来收益高，也可能是正优势。默认不额外约束单步 raw IG 的符号。

可选 `igpo_absolute_ig_guard=cumulative` 会额外要求未归一化、折扣累计的 IG return 与 Chosen／Rejected 方向一致。该保守门槛不是论文规则，默认 `none`。

全局 cap 仍沿用 v2_a 的候选排序；工具优势差与 outcome reward 差尚未做跨来源尺度校准。如果 `rejected_global_cap>0`，需检查两种来源的选中占比。

## 有效性与故障边界

- 当前工具到 final 之间有外部故障、重复／无效工具、缺失状态或原始 Turn 缺口时，对应累计优势不可用；不填 0、不删项继续累计，不跨断点拼接。
- 故障之后的完整可信后缀可以保留。内部重试最终成功按最终状态处理，失败尝试不会使成功重试变成重复动作。
- verifier 结果不可信或实际 final 缺失时，不产生混合工具优势，也不退回 IG-only。
- 同题有效 IG 人口须来自至少 `igpo_min_rollouts` 条 Rollout；样本不足或 IG 标准差过小时跳过工具组，不退回 outcome-only。
- 终局奖励全部相同时，终局 z-score 为 0；若 IG 仍有差异，工具优势仍可用。
- final 后的附加输出不继承该终局标签，也不进入工具优势覆盖率的预期分母。

## 配置与切换

以下位于 `data.trace`，Python 默认值、基础 YAML 和两份启动 YAML 以及对应 `TRACE_*` 环境变量同步：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `pair_score_mode` | `igpo` | `igpo` 或 v2_a 的 `trace_fill` |
| `igpo_gamma` | `1.0` | 全未来回传折扣，与旧 TRACE gamma 独立 |
| `igpo_ig_weight` | `1.0` | 归一化工具信息增益的权重 |
| `igpo_outcome_weight` | `1.0` | 归一化终局分量权重 |
| `igpo_norm_epsilon` | `1e-6` | 标准化分母稳定项 |
| `igpo_ig_std_min` | `1e-3` | 工具组 IG 标准差下限 |
| `igpo_min_rollouts` | `2` | 工具组最少不同 Rollout 数 |
| `igpo_positive_threshold` | `0.1` | Chosen 优势严格大于该值 |
| `igpo_negative_threshold` | `0.1` | Rejected 优势严格小于其负值 |
| `igpo_gap_epsilon` | `0.5` | 工具优势差下限 |
| `igpo_absolute_ig_guard` | `none` | 可选 `cumulative` 的绝对累计 IG 方向门槛 |

状态匹配仍为 `epsilon=0.1、match_epsilon=0.05`。旧 `positive_threshold`、`negative_threshold`、`gap_epsilon` 仅供 `trace_fill`，不把它们的 `.01/.01/.02` 当成 IGPO 优势的同量纲阈值。

```bash
# 默认启动 v2_b
python train-roma/train_agent.py

# 对照：恢复 v2_a 的信用与配对分数
python train-roma/train_agent.py data.trace.pair_score_mode=trace_fill

# 消融：仍用 IGPO 累计优势，但采用 0.8 折扣
python train-roma/train_agent.py data.trace.igpo_gamma=0.8
```

阈值是项目实验初值，尚不能据此判断训练收益。gamma=1 的累计和可能与剩余 Turn 数相关；不通过额外除窗口长度隐式改回平均信用。

## 指标读取

IGPO 模式保留 GT 覆盖、外部故障和可信终局等通用状态指标，不输出旧 TRACE 信用覆盖或 local/fill 统计。新指标位于 `trace/igpo/`：

| 指标 | 含义与观察方法 |
| --- | --- |
| `raw_ig_{count,mean,p10,p50,p90}` | 真实单步 LogP 增量，保留零值；正均值不保证最终 reward 提升 |
| `normalized_ig_{count,mean,p10,p50,p90}` | 有效工具人口内的相对 IG；观察归一化后分布是否有差异 |
| `outcome_z_{count,mean,p10,p50,p90}` | 可信终局标准化分量，一条 Rollout 一票；同结果组为零 |
| `ig_return_*`、`terminal_return_*`、`advantage_*` | 已乘权重的工具累计项、终局项和总优势，分别有 count/mean/p10/p50/p90；检查终局项是否主导 |
| `advantage_coverage` | 有效优势工具数 / 实际 final 前预期工具数；无可信 final 的工具保留在分母中 |
| `groups_valid`、`groups_insufficient_rollouts`、`groups_insufficient_ig`、`groups_low_ig_std`、`groups_invalid_stats` | 工具同题组是否提供有效相对信号及跳过原因；分组指标不是 Pair 数量目标 |
| `returns_blocked`、`returns_blocked_external` | 全未来回传被阻断的工具数，以及其中受外部故障影响的工具数；全回传比 K=3 更严格 |
| `duplicate_rows`、`conflicting_turns` | 相同真实 Turn 的副本数及内容冲突数，防止重复人口加权或静默拼接 |
| `raw_ig_advantage_sign_flip_rate` | 非零 raw IG 与非零有效最终优势方向相反的比例，分母为 `sign_compared_turns`；该变化允许存在，不等同标签错误 |
| `advantage_remaining_turns_corr` | 有效优势与到 final 的原始 Turn 距离的 Pearson 相关，常量／样本不足时为 0；高相关提示长度偏差，需要分层抽查 |

选中工具 Pair 的分数差为 `trace/pair/igpo_advantage_gap_{p50,p90,min}`，不再使用旧 `credit_gap` 名称；状态 `value_gap` 和 `trace`／`outcome` 来源计数继续保留。

## 验证状态

179 项 TRACE CPU 回归和 8 项原生 Token IDs 回归通过，`git diff --check` 通过。覆盖奖励标准化与累计回传、零 IG、低方差与样本不足、终局去重、类型路由、故障隔离、严格配对及 v2_a 模式兼容等边界。

尚未进行 GPU/NPU 实际训练，没有训练收益结论；不能据 CPU 回归声称 reward 已改善。
