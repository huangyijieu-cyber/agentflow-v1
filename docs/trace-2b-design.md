# v2_b 方案：在 v2_a 上引入 IGPO 风格优势配对

日期：2026-10-27。状态：实现与 CPU 集成回归已完成，版本见 `trace-2b` 分支的 Git 提交记录；GPU/NPU 训练收益尚未验证。实现说明见 [v2_b 使用说明](trace-2b.md)。

## 版本与目标

- 原 v2 更名为 v2_a；算法实现对应 `6eb6215`。
- 基于当前 `5fbe4cc` 创建本地分支 `trace-2b`，该提交在 v2_a 上只精简了指标。
- 本轮只替换 SLIC 工具 Pair 的分数构造，保持原 GRPO 优势与更新、SLIC Loss 和 Replay 剂量不变，以便和 v2_a 对照。
- 本方案参考的是 Information Gain-based Policy Optimization（IGPO），不是仓库现有的 GiGPO。

参考：[IGPO §3.2–3.3](https://arxiv.org/html/2510.14967v2#S3)、[官方优势实现](https://github.com/GuoqingWang1/IGPO/blob/64165e2741ed8801f977948c8128080ce87b4101/verl/trainer/ppo/core_algos.py)、[官方启动设置](https://github.com/GuoqingWang1/IGPO/blob/64165e2741ed8801f977948c8128080ce87b4101/train.sh)。核心借鉴是 LogP 增量、IG 与终局奖励分别组内标准化、再计算折扣累计回报。论文使用该回报更新策略；本项目先将其用于 SLIC 选对，因此是 IGPO-inspired，而非完整复现。

## 1. 保留状态评分和匹配尺度

继续使用 v2_a 的实际 Turn 评分上下文、冻结参考模型、温度 1 和 GT Token 平均 LogP。不会额外将 GT 放入 Rollout 生成上下文。

状态匹配仍使用：

$$
V_{g,t}=\log\frac{-\bar\ell_{g,0}+\epsilon}{-\bar\ell_{g,t}+\epsilon},\qquad
|V_i-V_j|\leq0.05.
$$

其中 `epsilon=0.1`。V 只负责描述配对前状态，不再以其增量作为 v2_b 的工具配对分数。

与 IGPO 原文使用策略模型评分不同，本方案保留固定 reference，以减少与 v2_a 的额外差异；当前不重构评分视图或引入向量化 GT 前向。

## 2. 用原始 LogP 增量定义工具信息增益

仅对真实工具决策的相邻状态计算：

$$
r^{IG}_{g,t}=\bar\ell_{g,t+1}-\bar\ell_{g,t}.
$$

例如 GT 平均 LogP 从 -8 变为 -7.5，信息增益为 +0.5；从 -8 变为 -8.3，信息增益为 -0.3。它与 v2_a 的平滑 NLL 对数比增量不是同一种尺度。

analysis 不产生 IG reward、不进入工具 IG 的归一化人口，也不接收工具回传；final 不制造后继 GT 状态。

## 3. 同题分别归一化 IG 与终局奖励

在本次新采样 batch 中，以 uid 为组：

- IG 人口：有效工具转移，覆盖同题不同 Rollout 和 Turn，按真实转移去重；排除故障、无效与重复动作、后终局输出及分布式补齐副本，在正负和 gap 筛选之前统计。
- Outcome 人口：同题有可信终局的 Rollout，每条 Rollout 一票，不按 Turn 复制奖励。
- 不按原始 Turn 号分组，不采用 Anchor 文本簇，也不在 Replay 中重新统计不同策略时期的奖励。

定义：

$$
z^{IG}_{g,t}=\frac{r^{IG}_{g,t}-\mu^{IG}_{x}}{\sigma^{IG}_{x}+\epsilon_{norm}},\qquad
z^O_g=\frac{R_g-\mu^O_x}{\sigma^O_x+\epsilon_{norm}}.
$$

标准差使用总体标准差（ddof=0）。终局只使用实际 verifier 的可信结果，不把解析失败或外部中断当成失败奖励 0。

低方差处理：

- 同题有效工具样本须来自至少 2 条不同 Rollout；不足时跳过工具 Pair。
- IG 缺失、少于 2 个有效转移或标准差低于门槛时，不用终局优势替代工具信号。
- 可信终局奖励全部相同时，其归一化分量为 0；此时若 IG 有有效差异，工具仍可产生优势。
- 真实 IG=0 是有效观测，必须保留；不能把数值非零当作 Turn 存在或有效的判断条件。

## 4. 用全剩余轨迹累计优势替换 K=3 与旧回填

设 f 为实际 final 的原始 Turn 编号，工具优势为：

$$
A^{2b}_{g,t}=w_{IG}\sum_{\substack{u\in\mathrm{tool}\ t\leq u<f}}
\gamma_{IGPO}^{u-t}z^{IG}_{g,u}
+w_O\gamma_{IGPO}^{f-t}z^O_g.
$$

建议第一版 `w_IG=1、w_O=1、gamma_IGPO=1.0`。gamma=1 与 IGPO 论文及官方启动设置一致；可用 0.8 做折扣消融。

- 使用累计和，不除权重和，不再施加 K=3 截断。
- 终局结果只进入一次，不再额外加 v2_a 的 `terminal_fill_scale=2` 回填。
- `k`、旧 `gamma`、`terminal_fill_enabled` 和 `terminal_fill_scale` 仅供旧 trace 模式；IGPO 模式显式绕过它们，避免重复计权。
- 折扣距离按原始 Turn 编号计算；analysis 不接收优势，也不通过删行把不相邻状态拼成相邻。
- 只有当前工具到可信 final 的未来区间完整、有效时才产生该优势。缺失终局不填 0，也不静默退回 IG-only。

例如 `analysis → A → B → C → final`：

$$
A_A=z_A^{IG}+\gamma z_B^{IG}+\gamma^2z_C^{IG}+\gamma^3z^O,
\quad
A_B=z_B^{IG}+\gamma z_C^{IG}+\gamma^2z^O,
\quad
A_C=z_C^{IG}+\gamma z^O.
$$

## 5. 保留分类型路由，以新优势筛选工具 Pair

| 类型 | v2_b 标签 |
| --- | --- |
| analysis | 保留 v2_a 的可信终局成功／失败标签 |
| tool_plan | 使用 A^(2b) 的严格正、负优势 |
| 实际 final_answer／direct／base | 保留可信终局成功／失败标签 |

工具 Pair 要求同 uid、同类型，且：

$$
|V_i-V_j|\leq0.05,\qquad
A_i^{2b}>\tau_+,\qquad A_j^{2b}<-\tau_-,\qquad
A_i^{2b}-A_j^{2b}\geq m_A.
$$

通过条件后优先选择优势差大的 Pair；保留原来的两侧质量、长度比、不同动作和数量上限约束，无合格样本就跳过。outcome 路线仍按可信成功／失败分池，不改为对其终局相对优势求符号。

**关键语义变化：归一化后的负优势代表组内相对累计收益低，不等于绝对 GT LogP 下降。** 即使单步 IG 全为正，也可能出现负相对优势；单步 IG 为负但后续收益好，也可能产生正累计优势。这是本方案与 v2_a 的重要区别。

建议默认只要求最终优势严格正负，不再同时要求单步 raw IG 正负，以保留延迟收益。若仍需绝对贡献方向约束，可配置为额外保守消融，例如要求未归一化累计 IG return 的正负方向一致；这不是 IGPO 原始规则，不作为默认。

SLIC 两侧仍使用各自原始 Prompt。其旧策略 per-token LogP baseline 不替换为 IGPO 的均值或优势。

## 6. 外部故障隔离

复用 v2_a 的故障与 verifier 元数据：

- 含故障转移、缺失状态、重复／无效工具转移或原始 Turn 缺口的未来回传区间失效，不跨越、不当成 0，也不直接删掉无效奖励项后继续累计。
- 全未来回传会使故障前更多工具优势失效，这是比 K=3 更严格的结果；故障后的完整、可信后缀可独立保留。
- 内部重试最终成功且未形成独立失败 Turn 时，按已有最终成功状态保留。
- verifier 不可信时，不构造工具混合优势、analysis／final 奖励 Pair。

## 7. 建议配置初值

以下为项目实验初值，优势筛选阈值不是论文推荐值，不能与旧信用阈值视为同量纲：

```yaml
data:
  trace:
    pair_score_mode: igpo              # trace_fill 可恢复 v2_a
    igpo_gamma: 1.0
    igpo_ig_weight: 1.0
    igpo_outcome_weight: 1.0
    igpo_norm_epsilon: 1.0e-6
    igpo_ig_std_min: 1.0e-3
    igpo_min_rollouts: 2
    igpo_positive_threshold: 0.1
    igpo_negative_threshold: 0.1
    igpo_gap_epsilon: 0.5
    igpo_absolute_ig_guard: none
    match_epsilon: 0.05
```

IG 和 outcome 默认分别标准化；不启用 curriculum、二次优势归一化或动态权重。Python 配置、基础 YAML、两份启动 YAML 及 TRACE_* 环境变量一并同步。

## 8. 改动范围与验证

奖励标准化和回传独立实现于 `agentflow/agentflow/trace/igpo.py`；分数模式与阈值、GT 评分输入及 TRACE 信用集中于同包的 `credit.py`，指标位于 `metrics.py`，执行状态位于 `execution.py`。`trainer.py` 负责接入和 Pair 元数据，3 份 YAML 提供配置；复用 Solver、Daemon 和 Rollout 的既有元数据链路。该子包由原有 5 个平铺实现文件整理而来，业务逻辑保持不变。

必要指标：原始 IG、归一化 IG、归一化终局奖励和累计优势分布；有效组／Rollout 数；近零方差跳过；故障阻断；raw IG 与最终优势的符号差异；优势与剩余 Turn 数的关系；各来源 Pair 构成。全局 cap 仍沿用 v2_a，若触发，须检查不同来源分数尺度对选中占比的影响。

回归覆盖：手算标准化与累计和、零 IG 保留、零方差、单例、全失败但 IG 有差异、终局一票去重、analysis／final 路由、故障前后后缀、Turn 缺口、无终局、严格符号及差距优先；并检查 v2_a 模式行为不变。

本地验证已完成：179 项 TRACE CPU 回归和 8 项原生 Token IDs 回归通过，`git diff --check` 通过。尚未进行 GPU/NPU 实际训练，不能据这些工程验证声称 reward 改善。

建议对照：v2_a；v2_b 默认配置；v2_b 的 gamma=0.8。重点评估同预算验证 reward、完成率、有效 Pair 质量及回传的长度偏差，不以数量增加判定成功。

## 已确认并实施的范围

本轮只将 IGPO 优势用于 SLIC 选对，保持原 GRPO 更新不变；默认采用全剩余轨迹回传和相对优势正负，而非 v2_a 的 K=3 与绝对局部信用正负。同时替换主 GRPO 训练优势属于更大范围的优化器接入，不在本轮实施范围。
