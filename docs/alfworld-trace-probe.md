# ALFWorld 固定轨迹 TRACE 测试窗口

入口：train-roma/alfworld_trace_probe.py。

默认选 data/train/alfworld/train_all_alfworld_3305.parquet 第 0 行，即 train_all_alfworld_0；从 question 解析环境编号 0，而不是使用该行的 id=1。同一题采样 4 条轨迹，每条最多 50 个动作，策略 temperature=0.7、history_length=4。

首次采样保存真实动作、生成 Prompt/Response、可见状态、环境返回和最后动作后的状态。随后修改评分 Prompt/固定 Response 只重新评分这些状态；即使重启脚本，也不会重新生成已完成的轨迹。用新的 work-dir 才能建立另一组采样。

## 启动窗口

在平时能运行 ALFWorld 的 Python 环境、仓库根目录中执行。下面的模型名、服务地址和 tokenizer 路径须对应实际部署；采样与评分可使用不同服务。

```bash
python train-roma/alfworld_trace_probe.py \
  --policy-base-url http://127.0.0.1:19998/v1 \
  --policy-model Qwen3-30B-A3B-Instruct-2507 \
  --score-base-url http://127.0.0.1:19998/v1 \
  --score-model Qwen3-30B-A3B-Instruct-2507 \
  --tokenizer /path/to/scoring-model-tokenizer
```

打开 http://127.0.0.1:8765。程序不会在打开页面时自动调用模型。

1. 点击“首次采样 4 条轨迹”，完成后检查原始轨迹。
2. 在两个文本框中修改 Prompt 和固定 Response，点击“保存并重新评分”。
3. 逐轨迹检查每个 Turn 的 Mean LogP、V、delta、credit。
4. 展开明细，查看原动作 Prompt/Response、完整评分 Prompt/Response、chat 模板实际输入和动作后状态。
5. 选择历史评分作为对照，查看同一动作的 credit 变化；可导出 JSON、CSV 和离线 HTML 报告。

如果在远程服务器运行，可通过 SSH 转发本地 8765 到服务器的 127.0.0.1:8765；服务默认仅监听回环地址。

## 服务器适配清单

先在服务器仓库中切换到 trace 并拉取最新提交，再激活平时运行 ALFWorld 的 Python 环境。若服务器有未提交文件，先检查 git status，避免覆盖自己的工作。

```bash
git switch trace
git pull --ff-only origin trace
python -c "import pyarrow, transformers; print('probe dependencies OK')"
```

测试工具不需要启动训练器、Ray 或 VERL；页面使用 Python 标准库。读 Parquet 需要 pyarrow，评分需要与模型兼容的 transformers；本地 HF 评分还需要 torch。ALFWorld、TextWorld、fast_downward、numpy、PyYAML 等沿用已有 ALFWorld 运行环境，环境未安装时需要先按原项目方式安装并准备数据。无需为此升级已有 PyTorch/vLLM/NPU 组件。

需要按服务器实际情况配置以下项目：

| 项目 | 参数／配置 | 如何确定 |
| --- | --- | --- |
| 数据表 | --dataset | 指向 train_all_alfworld_3305.parquet 的绝对路径 |
| ALFWorld 源码 | --alfworld-root | 包含 alfworld/ Python 包的源码目录；已有默认布局可省略 |
| 环境数据 | --alfworld-data | 同时包含 json_2.1.1/ 和 logic/ 的目录；不是 Parquet 所在目录 |
| 动作生成服务 | --policy-base-url、--policy-model | base URL 以 /v1 结尾；模型名用该服务 /v1/models 返回的 id |
| 冻结评分服务 | --score-base-url、--score-model | 需要支持 /v1/completions echo 和 prompt token LogP；评分期间固定 checkpoint |
| 评分 tokenizer | --tokenizer | 评分模型对应的本地目录，不能误用动作模型的 tokenizer |
| 实验目录 | --work-dir | 持久磁盘路径；后续运行始终使用同一目录才能复用轨迹 |

采样与评分可以指向同一个支持所需协议的固定模型服务。若动作模型在训练中更新，评分端仍应使用固定参考模型；对比不同 Prompt 时保持评分模型和 tokenizer 不变。

假设模型服务已启动，完整启动模板如下。将 /path/to/...、POLICY_MODEL 和 SCORE_MODEL 替换为服务器真实值；两种模型相同时，两处模型名和地址可以相同。

```bash
python train-roma/alfworld_trace_probe.py serve \
  --dataset /path/to/data/train/alfworld/train_all_alfworld_3305.parquet \
  --alfworld-root /path/to/alfworld-master \
  --alfworld-data /path/to/alfworld-master/json_data \
  --policy-base-url http://127.0.0.1:19998/v1 \
  --policy-model POLICY_MODEL \
  --score-base-url http://127.0.0.1:19998/v1 \
  --score-model SCORE_MODEL \
  --tokenizer /path/to/frozen-score-model \
  --work-dir /path/to/persistent/alfworld_probe_01 \
  --port 8765
```

在自己的电脑另开终端，建立 SSH 转发：

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

然后在电脑浏览器打开 http://127.0.0.1:8765。若 SSH 使用非默认端口，增加 -p SSH_PORT。浏览器端访问窗口的 127.0.0.1 指自己的电脑；脚本参数里的模型 127.0.0.1 指运行脚本的服务器，模型在别的机器上时应修改 base URL。

vLLM 的 max_logprobs 必须允许至少 1；默认配置通常已经满足。若部署时禁用了 LogP、代理屏蔽了 /completions，或返回 HTTP 400/缺少 echoed token_logprobs，需调整该服务或让评分地址直连支持此协议的冻结模型服务。单纯 chat completion 的 logprobs=true 不等于给固定 Response 做 teacher forcing。采样服务若不接受 return_token_ids，在首次采样前将实验 config.json 的 sampling.return_token_ids 设为 false；本工具的重评分不依赖动作 token IDs。

scorer.max_context_tokens（默认 16384）应不超过评分服务的实际上下文上限，且包含状态、评分 Prompt 和固定 Response 的总 token 数；超长状态会显示无效原因，不静默截断。

NPU 服务器优先复用当前已经能运行的 vLLM/vLLM-Ascend 服务，HTTP 模式不要求测试脚本直接加载模型。backend=hf 的 auto 仅自动选 CUDA 或 CPU；若使用本地 HF NPU，需要在该 Python 环境注册 torch_npu 并设置 scorer.device，不应把 auto 当作 NPU 自动检测。

启动参数首次写入实验目录 config.json。采样完成后，后续使用相同 --work-dir 启动即可，修改窗口内的 Prompt/Response 后只点重新评分。score 模式不读取数据表、不加载 ALFWorld，也不会生成新动作。

缺少仓库默认 ALFWorld 目录布局时，另加：

```bash
--alfworld-root /path/to/alfworld-master \
--alfworld-data /path/to/alfworld-data/json_data
```

alfworld-root 内应包含 alfworld Python 包。alfworld-data 指向包含 json_2.1.1/ 与 logic/ 的目录，对应当前环境配置里的 ALFWORLD_DATA。其余环境依赖沿用原项目安装方式。

默认数据路径依次从当前目录、仓库目录和仓库父目录查找；可以用 --dataset 指定绝对路径。用 --row-index 改选数据行；已有完成缓存时仍复用缓存，需要新的 --work-dir 才会采新题。

## 纯命令行使用

第一次创建配置及采样：

```bash
python train-roma/alfworld_trace_probe.py sample --work-dir outputs/probe_case_01
```

第一次启动会在实验目录生成 config.json。可提前用窗口启动一次生成配置，再编辑其中 sampling 和 scorer 的服务及模型设置。也可以直接在命令行使用上面的服务、tokenizer 参数。

之后修改实验目录 config.json 中的 probe.prompt / probe.response，然后只重新评分：

```bash
python train-roma/alfworld_trace_probe.py score --work-dir outputs/probe_case_01
```

run 模式执行“有缓存就复用，否则首次采样，然后评分”：

```bash
python train-roma/alfworld_trace_probe.py run --work-dir outputs/probe_case_01
```

score 模式从不自动采样，也不导入 ALFWorld；缓存缺失或尚未完成时直接报错。采样中断后可再次使用 sample：重放并核对已保存动作来恢复环境，只为缺失后缀调用模型。若环境重放不一致，停止并保留缓存，不混入不同状态。

## 可修改的评分文本

评分输入与当前 TRACE 一样构造：

```text
user content = 保存的 trace_state_context + 两个换行 + 可编辑 Prompt
assistant target = 可编辑的固定 Response
```

user content 使用评分模型 tokenizer 的 chat template；只计算 assistant target 的 token LogP，温度固定为 1，不将状态、Prompt、chat 前缀或 EOS 的 LogP 混进均值。不自由生成评分 Response。

Prompt 支持 {goal}、{remaining_steps}、{turn_index} 占位符。Response 只支持 {goal}，确保同一题、不同 Turn 的评分目标一致；其他普通大括号保持原样，方便直接写 JSON 或自然语言。

默认 Prompt 允许预测 success/failure 两种句子，而默认固定评分 Response 是 The episode will end in success.。本测试按用户要求只评分这一固定目标，未自动添加双候选归一化或成功率校准；得到的是文本似然与其派生信用。

## TRACE 计算口径

调用现有 agentflow/agentflow/trace/credit.py 的 state_values 和 discounted_credits：

```text
ell_t   = 固定 Response 的平均 token LogP
V_t     = log((-ell_0 + epsilon) / (-ell_t + epsilon))
delta_t = V_(t+1) - V_t
credit_t = sum(gamma**j * delta_(t+j)) / sum(gamma**j)
```

默认 epsilon=0.1、K=3、gamma=0.8，窗口包含最多 K 个真实环境动作，尾部按实际权重归一化。T 个动作有 T+1 个状态，最后动作也使用真实执行后状态；末状态不是额外动作。这与 QA 把最终文字回答算作一个 Turn 的边界有所区别。

本诊断不加终局优势回填，不用实际成功奖励改写参考模型 LogP 或 V。环境结果仅独立展示，供检查信用是否符合行为。外部故障影响的窗口无效；缺状态、编号缺口或超长输入不截断、不填 0、不跨缺口拼接。无效数值在输出中为 null，并注明原因。

## 评分后端

默认 vllm 使用冻结评分服务 /v1/completions：把 chat 模板 token IDs 和固定 Response token IDs 拼成整数 token Prompt，设置 echo=true、logprobs=1、max_tokens=0，读取 Response 所处位置的 prompt-token LogP。它不是读取另一次自由生成回答的 LogP。

服务必须支持该 vLLM completion echo 协议。只有 chat/completions 的服务不足以做这项固定目标评分；缺失 echoed token LogP 会明确报错，不退回错误的生成评分。tokenizer 必须和评分服务实际模型一致。

也可用 --backend hf --score-model /path/to/frozen-model --tokenizer /path/to/tokenizer；它通过本地冻结 causal model 前向计算相同 token 条件概率。HF 需要 torch、transformers 和足够模型内存；默认自动选择 CUDA 或 CPU，其他设备可在 config.json 的 scorer.device 设置。

API key 从配置指定的环境变量读取，默认 OPENAI_API_KEY；配置和报告不写入实际密钥。采样默认请求 vLLM 原生 token IDs，其他兼容服务若拒绝该字段，可在 config.json 设置 sampling.return_token_ids=false。

## 保存位置

默认目录：outputs/alfworld_trace_probe/。

| 文件 | 内容 |
| --- | --- |
| config.json | 可编辑的采样、评分服务及探针配置 |
| trajectories.json | 首次固定轨迹与 T+1 个状态；后续评分不修改 |
| runs/<run-id>.json | 每次评分的完整文本、token LogP、V、delta、credit、无效原因及配置 |
| latest.json | 最新完整结果 |
| latest.csv | 每轨迹每动作的数值及两套 Prompt/Response |
| report.html | 可离线展开查看的最新报告 |

每次评分记录原轨迹 SHA-256、配置和评分指纹。页面可选任意历史记录比较同一动作的信用，便于调试评分文本，不对 Prompt 优劣自动下结论。

## 验证范围

标准库 unittest 覆盖采样一次后永久复用、断点恢复、修改 Prompt 重评分且原缓存字节不变、评分请求去重、真实后状态的末动作信用、TRACE 公式、故障与编号缺口、Response-only LogP、导出文本和 HTML 转义；HF 另有 CPU 张量 causal-shift 检查。

```bash
python -m unittest discover -s agentflow/tests -p 'test_alfworld_probe_*.py' -v
```

开发机器缺 ALFWorld 依赖、环境数据和运行中的模型服务，因此真实环境采样及实际部署 vLLM 的端到端冒烟仍需在目标运行环境执行。测试夹具仅用于测试，没有写成真实采样缓存。
