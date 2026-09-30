# Subreward 与 GiGPO：Version 1 / Version 2 公式定义

本文记录约定的两个实验版本。Version 1 对应 2026-09-24 的奖励定义；Version 2 是当前定义。这里的奖励“反向传播”指沿时间向前折扣回传；参数梯度仍通过 policy loss 反向传播。本文不修改训练实现。

## 1. 符号与首次命中

- $q$：题目；$i$：该题的一条 rollout。
- $t=0,\ldots,T_i$：turn，$T_i$ 为最终回答 turn。
- $\mathcal F_q$：有效 subgoal 集合，$M_q=|\mathcal F_q|>0$。
- $H_{i,t}$：本轮首次命中的 subgoal 集合。
- $V_{i,t}$：本轮动作前已命中的 subgoal 集合。
- $R_i\in\{0,1\}$：最终答案奖励。

$$
V_{i,0}=\varnothing,\qquad H_{i,t}\cap V_{i,t}=\varnothing,\qquad
V_{i,t+1}=V_{i,t}\cup H_{i,t}.
$$

同一 subgoal 在一条 rollout 中只奖励一次；同一 turn 命中多个新 subgoal 时累加。最终回答 turn 不产生新的检索命中，即 $H_{i,T_i}=\varnothing$。

## 2. Version 1：归一化命中奖励，无时间回传

每个 subgoal 首次命中的原始奖励：

$$
r_{\mathrm{raw}}^{(1)}(f)=\frac{1}{M_q}.
$$

每个 turn 的原始奖励和用于 step 比较的奖励：

$$
s^{(1)}_{i,t,\mathrm{raw}}=\frac{|H_{i,t}|}{M_q},\qquad
\boxed{s^{(1)}_{i,t}=\frac{|H_{i,t}|}{M_q}+R_i}.
$$

最终奖励直接加到每个 turn；后续 subgoal 命中不回传给前面 turn。

按同题、相同动作前命中集合分组（只包含非最终回答 turn）：

$$
\boxed{\mathcal G_1(q,V)=\{(i,t):q_i=q,\ t<T_i,\ V_{i,t}=V\}}.
$$

同一条轨迹停留在同一状态的多个 turn 可进入同一组。

历史实现说明：9 月 24 日实际代码把首个 planner analysis 单独分组，只对工具 turn 使用上述集合分组。本文按约定公式描述 V1；复现历史代码时应保留这一差别。历史代码从数据读取每个 subgoal 的 `weight`；原始 pilot 构造脚本赋值为 $1/M_q$。

## 3. Version 2：单位命中奖励，0.5 折扣回传

$$
r_{\mathrm{raw}}^{(2)}(f)=1,\qquad
s^{(2)}_{i,t,\mathrm{raw}}=|H_{i,t}|.
$$

终点边界及递推：

$$
\boxed{s^{(2)}_{i,T_i}=R_i},\qquad
\boxed{s^{(2)}_{i,t}=|H_{i,t}|+0.5s^{(2)}_{i,t+1}\quad(t<T_i)}.
$$

等价展开式：

$$
s^{(2)}_{i,t}=\sum_{k=t}^{T_i-1}0.5^{k-t}|H_{i,k}|+0.5^{T_i-t}R_i.
$$

这里 $s^{(2)}$（公式中的 `subreward_t`）包含折扣后的最终奖励，不能再额外加一次最终奖励。

定义进入 anchor 后的相对轮数：

$$
a_{i,0}=0,\qquad
a_{i,t}=\begin{cases}
a_{i,t-1}+1,&V_{i,t}=V_{i,t-1},\\
0,&V_{i,t}\ne V_{i,t-1}.
\end{cases}
$$

$$
\boxed{\mathcal G_2(q,V,a)=\{(i,t):q_i=q,\ t<T_i,\ V_{i,t}=V,\ a_{i,t}=a\}}.
$$

planner analysis 使用 $(V,a)=(\varnothing,0)$，统一参与分组。命中发生后，下一 turn 的集合更新，相对轮数重置为 0。

## 4. Episode 与 step 优势

每道题有 $N_q$ 条有效 rollout，每条轨迹只计一次：

$$
A_i^{\mathrm{episode}}=\begin{cases}
R_i-\frac{1}{N_q}\sum_{j:q_j=q}R_j,&N_q>1,\\
0,&N_q=1.
\end{cases}
$$

对版本 $v\in\{1,2\}$，使用对应奖励和分组：

$$
A_{i,t}^{\mathrm{step},(v)}=\begin{cases}
s^{(v)}_{i,t}-\frac{1}{|\mathcal G_v(i,t)|}\sum_{(j,u)\in\mathcal G_v(i,t)}s^{(v)}_{j,u},
&t<T_i,\ |\mathcal G_v(i,t)|>1,\\
0,&\text{否则}.
\end{cases}
$$

两部分均只减均值，不除标准差，以 1:1 合并：

$$
\boxed{A_{i,t}^{(v)}=A_i^{\mathrm{episode}}+A_{i,t}^{\mathrm{step},(v)}}.
$$

最终回答 turn 只取 episode 优势。这里统一采用修正后的单轨迹 episode 优势为 0 的规则；9 月 24 日历史代码在单轨迹组上存在未归零的问题。

## 5. PPO 主损失

将 turn 优势广播给该 turn 内可训练 assistant token。令 $m_{i,t,k}$ 为 token mask，$c_{i,t,k}$ 为 token 上下文：

$$
\rho_{i,t,k}(\theta)=\frac{\pi_\theta(y_{i,t,k}\mid c_{i,t,k})}{\pi_{\mathrm{old}}(y_{i,t,k}\mid c_{i,t,k})}.
$$

按当前配置，clipping 区间为 $[0.8,1.3]$，每个 micro-batch 内使用 token-mean：

$$
L_{\mathrm{PPO}}^{(v)}=-\frac{\sum_{i,t,k}m_{i,t,k}\min\left[\rho_{i,t,k}A_{i,t}^{(v)},\operatorname{clip}(\rho_{i,t,k},0.8,1.3)A_{i,t}^{(v)}\right]}{\sum_{i,t,k}m_{i,t,k}}.
$$

之后按梯度累积系数缩放更新。当前配置关闭 KL loss 和 entropy 正则。此处为 PPO clipping 主项；底层安装的 VERL `vanilla` 实现可能还包含 dual-clip 等处理，以运行环境版本为准。

## 6. 示例

一条四轮轨迹：turn 2 命中一个新 subgoal，turn 3 最终答对，题目共 4 个 subgoal。

| turn | 事件 | V1 原始命中分 | V1 step 奖励 | V2 原始命中分 | V2 折扣回报 |
|---|---|---:|---:|---:|---:|
| 0 | analysis | 0 | 1 | 0 | 0.375 |
| 1 | 未命中 | 0 | 1 | 0 | 0.75 |
| 2 | 首次命中 | 0.25 | 1.25 | 1 | 1.5 |
| 3 | 最终答对 | 0 | 1 | 0 | 1 |

表中的回报尚未减组均值；最终回答 turn 不参加 step 优势计算。

## 7. 当前代码与指标对应

- `train-roma/rollout.py::build_turn_process_rewards`：按 turn 累加首次命中，得到 $s^{(2)}_{i,t,\mathrm{raw}}$。
- `agentflow/verl/daemon.py::_gigpo_return_to_go`：从末尾向前单次遍历，每轮计算 `future_return = immediate_reward + 0.5 * future_return`；最终回答轮的 `immediate_reward` 是最终奖励。与 V2 递推等价，没有为每个命中分别遍历所有之前的 turn。
- 公式中的 $s^{(2)}_{i,t}$ 写入 `step_reward_list`，供 GiGPO 组内中心化。
- 日志/metadata 中现有字段 `subreward` 仍是整条轨迹原始命中分数之和，不是公式中的递推回报；本文不重命名该字段。
- 当前首次命中匹配只读取搜索工具结果的前 2000 字符；9 月 24 日历史代码读取完整结果。两者都不额外验证 extraction。
- 首次命中公式要求数据中的 `count_each_subgoal_once` 为 true（代码默认值）；显式设置 false 会允许重复命中，不符合本文定义。

