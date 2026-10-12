# 搜索缓存容量测试

在开发服务器运行，不需要 GPU。脚本不会修改 `search-service.env`、重启共享服务或使用现有数据库；所有 SQLite 和本地 HTTP 服务均为独立临时实例。只需要原服务已有的 Python、requests、bs4。

本地路径 `train-roma/test_search_cache_capacity.sh` 会加载开发服务器的服务 env 和现有 proxy 配置。Python 入口默认 `--mode local`，不会访问外网；真实外网测试需要明确指定 `--mode all` 或 `--mode upstream`。

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

1. 用冻结的模拟producer验证默认16个worker、128额外槽位：144个任务可接纳，第145个拒绝；单独验证160个HTTP连接边界。
2. worker候选8/16/32/64，256个客户端，真实HTTP与SQLite：warm命中、cold不同key、相同key在途合并；记录吞吐、p95、排队、实际producer次数和资源信息。
3. loopback真实HTTP上游验证RPM间隔、并发桶以及当前deadline/cancel行为。这里验证实现遵守设定，不推断外网额度。
4. 真实外网分别测试Wiki搜索API、页面身份API、完整正文API、Wiki HTML、Yibu和每个输入网页域名。通过生产 `SearchService.request` 出网，沿用proxy/UA/TLS/响应上限；绕过缓存、现有人工RPM/并发上限，禁用重试，防止旧设置或重试掩盖真实边界。
5. RPM逐级加压；如果只有客户端并发不足导致送不出目标速率，增加并发后重测。出现429/内部限流停止升级，记录最后稳定档和失败档。
6. 测并发突发：每档两轮，轮次之间静默65秒，测真实执行峰值。只设置并发64而实际仅重叠3个，不算测过64。
7. 按冷Wiki流程比例混合四类HTTP请求，并让已有稳定基线的Wiki/Yibu/Web桶同时通过同一代理测试120秒。独立最高值如果不能同时运行，降低速率再验证，最多3轮。Web使用共同配置再验证，不把不同域名各自的配置直接拼成未经验证的全局值。失败或缺测的桶不导出对应设置，report的coverage明确显示缺项。

## 默认阶梯和通过标准

| 参数 | 默认 |
|---|---|
| worker候选 | 8,16,32,64 |
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
- `summary.md`：可阅读的结果表及通过共同负载验证的 `SEARCH_*` 数值。

没有额外乘0.8之类的折扣。最终配置候选取测试中通过的值；只有同时确认完成且窗口足够长，才输出 `confirmed_network_settings`。短于60秒的RPM或短于120秒的确认仅属于冒烟，不能导出持续负载的最终配置。需要把两份报告发回进行下一步配置评审。

RPM表的并发数是该档通过的最小测试cap，另列真实峰值，不声称它就是物理必需并发。最终 `SEARCH_*_CONCURRENCY` 是通过负载验证的配置cap；并发burst表才要求真正同时执行到该数值。`coverage` 中 all_requested_buckets_confirmed 为false时，结果只覆盖已通过的部分工具/域名。

worker最大候选只证明本地模拟负载可支持它，不能直接当成真实混合搜索的生产上限。队列长度没有独立的固定性能上限，额外排队不增加上游吞吐；报告验证当前容量与等待情况，不凭空测出“最大queue”。连接容量由 `workers + queue + 16` 推导，系统文件描述符/线程资源不足时报告inconclusive，不当成API限流。

暂停其他训练任务只能隔离我们可控的流量；公司代理上的其他用户仍可能影响结果。测试到的上限属于当前出口、账号、URL样本和时间窗口。Wiki/Yibu API的HTTP200错误JSON不会算成功；网页HTML按生产规则解析文本，不能保证识别所有验证码或业务错误页。脚本不会打印令牌、API key、proxy地址凭据或查询原文。
