# 搜索缓存容量测试

在开发服务器运行，不需要 GPU。脚本不会修改 `search-service.env`、重启共享服务或使用现有数据库；所有 SQLite 和本地 HTTP 服务均为独立临时实例。只需要原服务已有的 Python、requests、bs4。

本地路径 `train-roma/test_search_cache_capacity.sh` 会加载开发服务器的服务 env 和现有 proxy 配置。Python 入口默认 `--mode local`，不会访问外网；真实外网测试需要明确指定 `--mode all` 或 `--mode upstream`。

当前服务已取消公共 worker/queue、派生HTTP接纳容量以及所有上游的全局RPM/并发/cooldown。
旧报告中的 worker、queue、HTTP 边界仍是旧实现的历史测量，不能当成当前上限，
也不要据此写回 `SEARCH_SERVICE_WORKERS`、`SEARCH_SERVICE_QUEUE_SIZE` 或
`SEARCH_{WIKI,BRAVE,WEB}_{RPM,CONCURRENCY}`；这些旧配置均不再生效。
当前有意义的容量观测是命中延迟、唯一 miss 吞吐、在途合并和真实外网负载；
全部工具的原始测量和共同负载结果仍保留，用于评审实际可承受的训练负载，
不再生成已经失效的服务端限速配置。

## 一次完整运行

测试时暂停使用同一代理/API账号的训练任务，否则结果包含它们占用的容量。真实测试会产生 Yibu API 调用，默认预算为最多 12000 次逻辑调用（HTTP跳转可能增加实际次数）。参数范围跑完可能需要几十分钟，默认新阶段调度预算两小时；已开始请求结束后收尾。

```bash
cd /home/ma-user/work/code-rl/new/idea
bash train-roma/test_search_cache_capacity.sh \
  --mode all \
  --urls-file /path/to/training-urls.txt
```

`training-urls.txt` 每行一个**实际训练会抓取的网页 URL**。也可使用含URL字段的 json/jsonl，按域名分别测试。不给URL文件时会尝试从query JSON/JSONL中提取URL；仍找不到则Web不测且报告明确标记缺项。不会用一个无关示例域名推断整个互联网的额度。默认最多3个非Wiki域名，更多域名使用 `--max-web-hosts` 调整。Wiki HTML已纳入Wiki组。

搜索 query 默认用一组实体名称，可替换成真实 InfoSeek query：

```bash
bash train-roma/test_search_cache_capacity.sh \
  --mode all \
  --queries-file /path/to/infoseek.jsonl \
  --query-key query \
  --urls-file /path/to/training-urls.txt
```

支持 txt/json/jsonl；外层 `query/question/problem` 自动读取。嵌套字段使用 `--query-key data.query`。InfoSeek的问题用于搜索；Wiki身份/全文API使用代表性文章标题，避免把复杂问题当成文章标题。Wiki extract探测按title而非pageid请求全文，报告范围不包含模型选页、embedding或摘要。

如果只先测本机缓存机制：

```bash
python3 train-roma/test_search_cache_capacity.py --mode local
```

## 测试范围

1. 历史报告的固定基线为16个worker、128额外槽位，144个任务可接纳、第145个拒绝，HTTP连接边界160；后来32-worker配置曾对应160个任务和176个HTTP连接。以上仅适用旧调度实现，当前没有这些固定边界；对应probe返回not_applicable，不再开连接验证旧上限。
2. 历史worker候选为8/16/32/64、256个客户端。当前只保留缓存工作负载的比较意义：warm命中、cold不同key、相同key在途合并，记录吞吐、p95、实际producer次数和资源信息；worker候选不再改变运行容量，兼容接收后只执行一组三类负载，不会按旧候选反复跑相同测试。重点验证命中不被慢miss拖住、相同key只启动一个producer。
3. loopback真实HTTP上游验证Wiki/Yibu/Web均不再创建共享限速桶，以及当前deadline/cancel行为。旧RPM/并发参数只用于兼容验证，不能推断外网额度。
4. 真实外网分别测试Wiki搜索API、页面身份API、完整正文API、Wiki HTML、Yibu和每个输入网页域名。通过生产 `SearchService.request` 出网，沿用proxy/UA/TLS/响应上限；绕过缓存、现有人工RPM/并发上限，禁用重试，防止旧设置或重试掩盖真实边界。
5. RPM逐级加压；如果只有客户端并发不足导致送不出目标速率，增加并发后重测。出现429/内部限流停止升级，记录最后稳定档和失败档。
6. 测并发突发：每档两轮，轮次之间静默65秒，测真实执行峰值。只设置并发64而实际仅重叠3个，不算测过64。
7. 按冷Wiki流程比例混合四类HTTP请求，并让已有稳定基线的Wiki/Yibu/Web桶同时通过同一代理测试120秒。独立单项的最低RPM仅作为首档种子，不是混合Wiki上限；首档通过后继续测混合Wiki的120/150/180/200等候选（完整模式不超过已测搜索API档）。升级时保持已验证的Yibu/Web负载不变，失败保留上一档通过组合。首档独立最高值如果不能同时运行，降低速率再验证，最多3轮。Web使用共同配置再验证，不把不同域名各自的配置直接拼成未经验证的全局值。report的coverage明确显示失败或缺测的工具；保留已确认的原始负载，不写回服务限速设置。

## 默认阶梯和通过标准

| 参数 | 默认 |
|---|---|
| 历史worker候选（当前不再生效） | 8,16,32,64 |
| 客户端并发 | 256 |
| 本地每种缓存测试窗口 | 3秒 |
| 本地模拟原文 | 16384字节 |
| 本地producer模拟延迟 | 0.05秒 |
| RPM阶梯 | 30,60,120,150,180,200,240,360,480,600 |
| HTTP并发阶梯 | 1,2,4,8,16,32,64 |
| 每档RPM窗口 | 60秒 |
| 最少样本 | 8 |
| 最低成功率 | 98% |
| 429/API内部限流 | 0次才通过 |
| 目标RPM达成 | ≥90% |
| 请求p95 | ≤20秒 |
| 原始请求总预算 | 30秒，内置连接/读等待仍遵循20/20或10/10 |
| burst轮数 | 2 |
| 临时5xx/网络故障的整档burst尝试 | 最多3次；间隔仍为65秒，原始HTTP重试仍为0；多次失败标记inconclusive |
| burst/确认前静默 | 65秒；429的Retry-After更长时按更长值 |
| 有稳定基线的桶共同确认 | 120秒，失败最多降档3轮 |

98%是测试程序的当前成功率判据，不是服务保证；输出会保留每档失败数量与错误码。延迟阈值按你的需求用 `--latency-limit` 调整。每次HTTP尝试计数看 `transport_metrics_final`，逻辑call和跳转请求数分开。

达到测试上限而未失败时，输出 `tested_ceiling_passed_upper_bound_not_found`：只证明至少支持该档，不声称这是物理上限。继续扩大范围：

```bash
bash train-roma/test_search_cache_capacity.sh --mode upstream \
  --rpm-steps 120,150,180,200,240,360,480,600,900,1200 \
  --concurrency-steps 1,2,4,8,16,32,64,128 \
  --request-budget 20000 \
  --urls-file /path/to/training-urls.txt
```

临近失败边界想精细定位，可把阶梯改成如 `--rpm-steps 150,160,170,180,190,200`。不能将HTTP503/407/超时一概认成Wiki官方额度不足；查看错误码和目标RPM是否真正达成。

## 报告

默认写到本机临时目录下 `agentflow-search-capacity/<UTC时间>-<PID>/`。终端打印完整路径，可用 `--output-dir` 指定一个**尚不存在**的目录。阶段结束持续保存，Ctrl+C也保留已完成记录。

- `report.json`：全部边界、每档数据、失败原因、独立值、共同出口确认值。
- `summary.md`：可阅读的结果表及共同验证的原始负载；所有旧worker/queue/上游限速配置项都不适用于当前实现。

没有额外乘0.8之类的折扣。所有原始探测仍保留，完整窗口确认负载写入`confirmed_measured_load`，
短窗口写入`smoke_measured_load`。`confirmed_network_settings`仅保留兼容字段、值为空，
不会再推荐任何已经取消的服务内限速配置。短于60秒的RPM或短于120秒的确认只属于冒烟。

RPM表的并发数是该档通过的最小测试cap，另列真实峰值，不声称它就是物理必需并发。所有工具的并发cap都只是压测客户端控制值，不再对应服务内限制。并发burst表才要求真正同时执行到该数值。`coverage` 中 all_requested_buckets_confirmed 为false时，结果只覆盖已通过的部分工具/域名。

旧报告的worker最大候选只证明当时本地模拟负载可支持它，不能当成当前混合搜索的生产上限。
旧的 `workers + queue + 16` HTTP容量公式也已移除。当前只为唯一未命中producer启动独立线程，
缓存命中直接返回、重复等待者不占上游执行线程；实际容量取决于系统线程/文件描述符、
CPU/SQLite/网络和上游能力，不存在可从worker/queue环境变量推导的固定任务上限。
系统资源不足不能当成API限流，也不能把历史边界值改名为新的实测上限。

暂停其他训练任务只能隔离我们可控的流量；公司代理上的其他用户仍可能影响结果。测试到的上限属于当前出口、账号、URL样本和时间窗口。Wiki/Yibu API的HTTP200错误JSON不会算成功；网页HTML按生产规则解析文本，不能保证识别所有验证码或业务错误页。脚本不会打印令牌、API key、proxy地址凭据或查询原文。

## 训练继续时的补测

加 `--background-training` 时，结果写入 `confirmed_additional_load`，不导出全局 `SEARCH_*` 配置：它表示背景训练存在时的额外负载，不能直接当成整个共享服务的总容量。

已有独立探测报告，无需重复整套1小时以上流程。将旧 `report.json` 留在服务器，更新主程序后执行：

```bash
bash train-roma/test_search_cache_capacity.sh \
  --mode confirm \
  --resume-report /原报告目录/report.json \
  --background-training \
  --wiki-confirm-steps 60,120,150,180,200
```

只读取旧报告中已经共同通过的Wiki/Yibu配置；重新确认基础档，再逐级提高混合Wiki负载，保持Yibu档不变。每档120秒＋档间65秒，全部5档约15–20分钟，提前失败则停止升级。不会重复独立API、burst、worker、队列或Web测试；旧数据不会冒充当前实测。

确认模式默认复用旧报告的探测并发cap，不代表已重新找到并发上限；全部cap仅控制压测负载，不对应当前服务限制。Wiki的搜索/身份/正文/HTML按1:10:10:3比例混合；HTML单项只能达到60RPM，不意味着整个Wiki桶也只能60RPM。背景训练流量会随时间变化，额外负载结果也会变化。
