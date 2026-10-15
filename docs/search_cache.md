# 统一搜索与共享缓存

`cache` 分支基于 `main`，集成共享搜索、缓存和一键启动功能，保留 `main` 的训练与奖励逻辑。
统一服务运行在开发服务器，配置、日志和 PID 默认放在
`/home/ma-user/work/code-rl/cache`；SQLite 数据库默认放在本地临时目录
`/var/tmp/agentflow-search-cache/search_cache.sqlite3`。训练机调用开发服务；缓存未命中时，开发服务沿用
EC2 → 个人主机 → Internet 的出口。模型选页、embedding、RAG 摘要和 reward 仍在训练机。

```text
训练任务 A / B / C
       ↓ HTTP（内网直连或 SSH 转发）
开发服务器：搜索服务 → 本地临时盘 SQLite 缓存
       ↓ 先查缓存 / 合并在途请求，唯一未命中请求独立执行
EC2 → 个人主机 → Wikipedia / Yibu / 网页
```

## 跟随训练自动启动（推荐）

使用 `cache` 分支代码，训练启动脚本会自动执行：

```text
本训练节点健康检查
→ 已就绪：直接复用
→ 未就绪：SSH 到开发服务器
→ 远端启动锁 + 健康检查，已有服务复用，否则启动一次
→ 本节点建立/复用 SSH 转发
→ 认证检查成功后启动 Ray / rollout / 训练
```

多个训练任务同时开始时，开发服务器的启动锁避免重复创建服务，本节点锁避免重复创建
同端口的 SSH 转发。服务独立于训练进程常驻，训练结束不会关闭它或清空缓存。
开发服务器重启后，下次训练会再次检查并启动服务；本方案不是 systemd 守护或运行中自动重启。

首次准备都可以在开发服务器完成：放好本分支代码、Python 依赖、服务令牌、Yibu 密钥，
以及平台提供的开发服务器 PEM 私钥和训练端本地配置。平台复制整个代码目录时，
同时带上这些本地文件；每个训练节点只运行原有训练启动命令即可。
不把 SSH 密码放入脚本，不自动覆盖或 pull 开发服务器的工作区。

训练端复制 `search-cache.env.example` 为 `search-cache.local.env`，填写共享令牌。
默认 SSH 目标为 `ma-user@7.150.11.99:31753`，默认远端仓库根目录
`SEARCH_CACHE_REMOTE_REPO_DIR=/home/ma-user/work/code-rl`；若实际源码不在这里，修改这个值。
缓存数据目录仍是 `/home/ma-user/work/code-rl/cache`，两种目录不能混淆。
示例默认 `SEARCH_CACHE_AUTO_START=1` 和 `SEARCH_CACHE_AUTO_TUNNEL=1`。
然后照常启动已有训练脚本即可，无须另起终端启动缓存服务或转发。

使用 PEM 私钥时，在开发服务器的仓库内准备 `pem/h50065774.pem`，并在
`train-roma/search-cache.local.env` 中设置：

```bash
SEARCH_CACHE_SSH_IDENTITY_FILE=pem/h50065774.pem
SEARCH_CACHE_SSH_AUTO_PREPARE=1
```

该相对路径对应复制后的训练服务器仓库根目录，不依赖训练目录的绝对路径。
平台必须复制 `pem/` 和本地 env 文件；它们被 Git 忽略，不会推送到仓库。
训练启动自动将 PEM 复制到当前训练用户的私有运行目录，设为 `600`，保持源文件不变；
随后通过 SSH `-i` 指定运行目录内的密钥。无需在训练机上手动 `chmod` 或登录。

自动准备模式会在首次 SSH 连接时记录 host key（`StrictHostKeyChecking=accept-new`），
后续服务器 host key 变化会拒绝连接，不会跳过校验。运行目录在同一训练节点上复用；
全新训练节点会进行首次记录。如果开发服务器已经有核验过的 known_hosts 文件，
可将其一起复制，并配置 `SEARCH_CACHE_SSH_KNOWN_HOSTS_FILE` 指向仓库内该文件；
此时始终采用严格校验（`StrictHostKeyChecking=yes`），仅接受预先记录的服务器密钥。

关闭 `SEARCH_CACHE_SSH_AUTO_PREPARE` 时，恢复原有行为：PEM 需已有合适权限，
SSH host key 需已确认。若 PEM 带有口令，仍需已解锁的 SSH agent；无人交互启动不能
输入私钥口令或账号密码。上述自动准备面向平台提供的无口令 PEM。

开发服务器准备好上述文件并配置共享令牌后，平台的一键命令保持原样，例如：

```bash
bash train-roma/run_distribute_train.sh
```

脚本自动加载配置、准备 SSH 文件、启动或复用开发服务、建立隧道，再启动训练。

直接内网 HTTP 访问时，设置对应 `SEARCH_CACHE_BASE_URL` 和
`SEARCH_CACHE_AUTO_TUNNEL=0`，同时在训练端设置 `SEARCH_SERVICE_HOST` 为开发服务器的
可达内网接口（或 `0.0.0.0`）。自动启动仍通过 SSH 执行，远端监听地址以训练端该配置为准。
如果由平台另外管理服务，设 `SEARCH_CACHE_AUTO_START=0` 禁用远端启动；已有服务仍可自动建隧道。
两项都设为 0 时仅做健康检查。
自动启动、SSH 和就绪等待都有截止时间，失败时训练不会继续使用直连搜索。

`train-roma/search_cache_bootstrap.py` 是训练端管理入口；
`train-roma/ensure_search_cache_service.sh` 是开发服务器的幂等启动入口。

## 开发服务器启动

SSH 地址是 `ssh://ma-user@7.150.11.99:31753`。`31753` 是 SSH 入口端口，不能当作 HTTP
服务端口。搜索服务默认监听开发服务器的 `127.0.0.1:8091`，可通过 SSH 转发使用；若训练机
能内网直连，也可以将监听地址设成对应内网 IP 或 `0.0.0.0`，并配置平台端口映射。

在开发服务器上的仓库根目录执行：

```bash
mkdir -p /home/ma-user/work/code-rl/cache
cp train-roma/search-service.env.example /home/ma-user/work/code-rl/cache/search-service.env
chmod 600 /home/ma-user/work/code-rl/cache/search-service.env
```

编辑该文件，填写随机共享令牌 `SEARCH_SERVICE_TOKEN` 和 Yibu 密钥
`YIBU_BRAVE_API_KEY`。训练机只需要共享服务令牌，不需要 Yibu 密钥。
安装服务的轻量依赖后启动：

```bash
python3 -m pip install requests beautifulsoup4
bash train-roma/run_search_cache.sh
```

脚本默认读取上述 env 文件，并 source 当前仓库的 `enable_search_proxy.sh` 设置现有出口。
若已在服务环境中配置好 `HTTP_PROXY` / `HTTPS_PROXY`，在 env 文件设置
`SEARCH_SERVICE_USE_PROXY=0`。服务端读取代理变量，训练端服务客户端禁用环境代理。

长期运行可以使用平台的后台进程管理器或：

```bash
nohup bash train-roma/run_search_cache.sh > /home/ma-user/work/code-rl/cache/service.log 2>&1 &
```

同一数据库只允许一个服务进程运行，数据库旁的锁文件保证缓存与请求合并状态的进程归属。
服务只需 CPU 和磁盘，不需启动训练 SDK、vLLM 或 embedding 服务。
若需要多进程/多副本，需先把请求合并状态迁移到共享协调组件。

### 数据库存放与升级

`SEARCH_CACHE_DIR` 继续保存 env、日志、PID 等服务文件。`SEARCH_CACHE_DB_PATH` 单独指定数据库，
默认 `/var/tmp/agentflow-search-cache/search_cache.sqlite3`；WAL、SHM 和数据库锁也放在其旁边。
该目录必须位于开发服务器本地文件系统，不能把 SQLite/WAL 直接放在 `fuse.s3fs` 上。
可用 `findmnt -T /var/tmp -o TARGET,FSTYPE` 确认挂载类型。

本地文件仍在时，服务重启可以复用缓存；容器重建或临时目录被清理后会重新积累。
不做 S3 快照，也不自动复制原先已经损坏的数据库。

升级已有服务时，在现有服务端 env 中更新以下项，不要用示例覆盖已有令牌和密钥。
已有 env 的显式值会覆盖代码默认值，单独同步 Python 代码不会改变这些值：

```bash
SEARCH_CACHE_DB_PATH=/var/tmp/agentflow-search-cache/search_cache.sqlite3
SEARCH_CACHE_MAX_TTL=604800
# SEARCH_BLOCKED_DOMAINS：复制 search-service.env.example 中完整的 325 项赋值。
```

旧的 `SEARCH_SERVICE_WORKERS`、`SEARCH_SERVICE_QUEUE_SIZE`，以及
`SEARCH_{WIKI,BRAVE,WEB}_{RPM,CONCURRENCY}` 均无需再设置；这些旧配置不再生效。

同步代码后必须重启开发服务器上的共享搜索服务。只重启训练会复用仍在运行的旧服务，
不会应用新路径或新有效期。启动日志会打印实际数据库路径；切换新数据库后缓存从零积累。

## 训练机接入

**方式一：SSH 转发。** 推荐使用前面的自动启动配置，每个节点会自动建立连接。
手动管理时，先设置 `SEARCH_CACHE_AUTO_START=0`、`SEARCH_CACHE_AUTO_TUNNEL=0`，再启动转发：

```bash
bash train-roma/open_search_cache_tunnel.sh > /tmp/search-cache-tunnel.log 2>&1 &
```

默认等价于从该训练节点转发本机 `127.0.0.1:8091` 到开发服务器的
`127.0.0.1:8091`，SSH 目标为 `ma-user@7.150.11.99:31753`。
脚本使用 `BatchMode=yes`，需要该密钥已获开发服务器账号授权；可以用
`SEARCH_CACHE_SSH_IDENTITY_FILE` 指定 PEM 文件。设置自动准备模式时处理密钥副本和首次
host key 记录；默认手动模式维持严格校验，不会将密码写进仓库。
目标、SSH 端口和本地端口分别由 `SEARCH_CACHE_SSH_TARGET`、`SEARCH_CACHE_SSH_PORT`、
`SEARCH_CACHE_LOCAL_PORT` 配置。

**方式二：内网直连。** 设置 `SEARCH_CACHE_BASE_URL` 为训练节点实际能够访问的 HTTP
地址，不需要 SSH 转发。SSH 地址本身无法确认 HTTP 端口是否已对训练机开放。

在开发服务器的仓库内复制训练端示例，填入与服务端相同的令牌，再让平台复制整个目录：

```bash
cp train-roma/search-cache.env.example train-roma/search-cache.local.env
chmod 600 train-roma/search-cache.local.env
```

或者在平台给每个训练节点注入以下环境变量：

```bash
export SEARCH_CACHE_ENABLED=1
export SEARCH_CACHE_BASE_URL=http://127.0.0.1:8091
export SEARCH_CACHE_TOKEN='与服务端相同的共享令牌'
```

`run_distribute_train.sh`、`run_train.sh`、`run_train_forever.sh` 检测到启用配置或本地 env
文件后，会在训练前加载并检查服务。直接启动 Python 时，先执行：

```bash
source train-roma/enable_search_cache.sh
```

平台/job 显式设置的 `SEARCH_CACHE_*` / `SEARCH_SERVICE_*` 变量优先于训练端 env 文件。
显式设置 `SEARCH_CACHE_ENABLED=0` 时，即使存在该文件也不会启动共享服务或隧道。

多节点时本地配置和 PEM 须随平台代码包进入每个节点；使用 localhost 时各节点自动建立转发。
VERL 的 Ray 入口会显式向 actor 传递客户端配置。直接调用其它已有 Ray 任务时，也应在
启动 Ray 前注入这些变量。

没有启用配置且不存在训练端 env 文件时，三个工具使用 `main` 原来的网络逻辑。
现有 `enable_search_proxy.sh` 设置的旧 `SEARCH_GATEWAY_BASE_URL` 不会自行开启共享缓存。
启用后的服务失败不会自动直连外网，也不会多层叠加重试。

## 各工具缓存边界

| 工具 | 缓存内容 | key 与有效期 |
| --- | --- | --- |
| Wiki 候选 | 完整、有序标题列表 | API 站点/语言、query、实际参数、版本；写入后 7 天，命中不续期 |
| Wiki 页面 | 页面身份、正文、URL | 站点/语言、标题身份或 pageid、参数、版本；写入后 7 天，命中不续期 |
| Web RAG | 当前 BeautifulSoup 规则解析出的网页文本 | 完整实际 URL、影响内容的 headers、解析版本与长度；写入后 7 天，命中不续期 |
| Yibu/Brave | 完整、有序上游 JSON，包括所有搜索结果和 snippets | endpoint、query、count、地区/语言/freshness、版本；包括 freshness 搜索均为写入后 7 天，命中不续期 |
| Base Generator / Python Coder | 不做全局结果缓存 | 非上述网络检索路径 |

成功响应中的真实零结果也缓存 7 天；429、超时、未知响应、5xx 不作为成功内容缓存。
Wiki 单页失败保留当前占位结果，但失败页面不入正常缓存；成功页面可以复用。
Wiki API 正文和 HTML 解析文本属于不同缓存类型，不能互相替代。

每条记录写入时设置 `expires = 写入时间 + 604800 秒`，之后命中不会修改 `expires`。
缓存读取允许并发，命中不再同步取得数据库写锁更新 `accessed`；LRU访问时间延后合并写回，
用于容量不足时近似淘汰最近没有使用的记录。即使频繁命中，到达原定有效期也必须重新访问上游。查询 `/metrics` 不延长有效期，容量上限仍可提前淘汰记录。
升级时已有健康记录保留当前的 `expires`，不会清库，也不会继续续期；不能追溯恢复旧记录此前的原始写入时间。

各工具统一使用 7 天，不再读取旧的 `SEARCH_CACHE_WIKI_TTL`、`SEARCH_CACHE_WEB_TTL`、
`SEARCH_CACHE_BRAVE_TTL`、`SEARCH_CACHE_BRAVE_FRESH_TTL`、`SEARCH_CACHE_EMPTY_TTL`。
保留总上限 `SEARCH_CACHE_MAX_TTL`，默认和示例均为 `604800`；设置得更低会缩短有效期。

缓存不改变候选排序、重复项、字段顺序和模型使用的文本长度。
从缓存读取时重新构造独立对象，避免 Wiki 原地加入摘要污染其他 query。
query 相关的选页、段落排序和摘要在训练机重新执行。
subreward 和已命中 subgoal 状态不共享；按原工具 observation 及当前 reward 规则逐轨迹计算。
返回的 `meta` 不拼入 observation，不能因为 reward 只看前 2000 字符就截断原始网页输入。

## 重复请求、限速与失败

显式 batch 按规范化后的完整请求 key 去重，再按原始位置还原结果：

```text
[A, B, A, C, B] → 获取 [A, B, C] → 返回 [结果A, 结果B, 结果A, 结果C, 结果B]
```

不同批次或训练任务同时获取相同原始数据时，通过 single-flight 共用一个获取任务。
先检查缓存和登记在途请求，只有唯一未命中的 producer 创建独立执行线程并访问上游；
命中直接返回，不经过公共 worker 队列；重复等待者只等待已有 Future，不占上游执行线程。
内部页面等缓存层也继续按各自 key 合并。每个等待者得到独立副本，失败由同一获取任务重试，
等待者不会分别访问上游。训练 rollout 按原逻辑逐步产生 query，不需要等待所有轨迹凑齐。

共享服务不再设置公共 worker 池、排队槽位或由两者推导的 HTTP 接纳连接上限。
旧的 32 worker、128 排队、160 未完成任务和 176 HTTP 连接数不再限制当前实现。
每个唯一未命中请求独立执行；真实同时运行数仍受到机器资源和外部服务能力影响。

Wiki API、Wiki HTML、Yibu和普通网页均不再使用服务内的全局RPM、并发桶或共享cooldown。
每次请求只管理自己的重试和退避，不会让其他请求统一等待。Yibu/普通网页的429响应中，
Retry-After超过30秒时直接返回上游429，不在服务内长时间睡眠；Wiki仍按自己的Retry-After
和剩余预算处理。其余可重试故障仍受最大重试次数和剩余预算限制。
取消人工限制不会取消Wikipedia、Yibu或网站自己的限流规则。

调用方等待期限和共享producer期限分开：每个新producer从创建时获得独立的
`SEARCH_SERVICE_REQUEST_TIMEOUT`（默认120秒）预算，不继承最早等待者仅剩的几秒。
每个调用方只等到自己的截止时间；其超时不会取消其他调用方共用的producer。
producer到达自身截止时间时会唤醒等待者返回超时；如果底层线程尚未真正退出，
仍保留该key的在途占位，后续相同请求不会启动重复上游调用。逾期结果不写入正常缓存。
这一占位不代表还能成功返回，可能表示正在退出的底层调用。

网络、重试及等待重复请求仍消耗各自对应预算。
服务故障、代理故障和上游错误可以通过 error code、upstream 和 request ID 区分。
过期成功结果刷新失败时不静默返回旧内容。

共享链路的失败保持结构化原因，不统一替换成工具 300 秒超时：

| 错误 | code | 行为 |
| --- | --- | --- |
| 代理连接/CONNECT 失败 | `upstream_proxy_error` | 保留可识别的 407/502/504 等状态与实际尝试次数；407 不重试 |
| TLS 失败 | `upstream_tls_error` | 不重试 |
| 上游单次 HTTP 超时 | `upstream_timeout` | 在服务总预算内有限重试 |
| 其他上游连接失败 | `upstream_network_error` | 在服务总预算内有限重试 |
| 上游 HTTP 非成功状态 | `upstream_http_error` | 保留原始 HTTP 状态，沿用可重试状态集合 |
| 服务总预算耗尽 | `deadline_exceeded` | 返回服务总预算耗尽错误 |

客户端将 `upstream`、`upstream_status`、`attempts`、`retryable` 和 request ID 传到工具层。
原始代理异常可能包含令牌和 URL，因此只提取状态码和安全的错误类型，不转发整段异常。
Wiki 候选页解析遇到共享网络失败时立即退出，不逐页重复失败链路；模型选页后执行的
Web RAG 也将共享链路故障向上抛出。单个网页 404/黑名单错误则保留在该页结果中，
允许继续处理其他页面，不把原因写成 `None`。失败内容不进入 embedding 或摘要。

Python客户端读取超时默认150秒，示例也显式设置 `SEARCH_CACHE_READ_TIMEOUT_SECONDS=150`。
启动shell对未设置值仍有600秒回退；平台预置env或实际local.env的显式值会覆盖Python默认值，
本次未改客户端超时配置，实际生效值应以训练进程env为准。
服务仍以 120 秒为单次逻辑请求预算，150 秒留给传输和响应开销，并小于外层工具的 300 秒。
客户端不重试共享服务请求，也不自动退回直连。

Executor 保留在 0.5 秒取消宽限内已经完成的结果/异常，不再清空后统一报超时；
真正超时后停止执行同一 command 的后续 block。当前同一 command 的全部 block
及 Wiki/Web 嵌套调用共用一个截止时间，不会每页、每个 block 重置 300 秒。

`agentflow/agentflow/tools/execution_context.py` 在各线程之间隔离执行状态，并将
同一工具的剩余时间传到搜索请求、Wiki 选页、embedding 和摘要阶段。

| 时间约束 | 当前行为 |
| --- | --- |
| Executor 总预算 | 默认 300 秒，同一 command 全部 block 共用 |
| 选页/embedding/摘要单阶段 | 最多 120 秒，同时不超过工具剩余时间；包含 SDK 调用和内部等待的墙钟时间 |
| 搜索客户端 | socket 超时不超过工具剩余时间，整个请求另有墙钟预算 |
| 共享服务 | `min(服务原有 120 秒预算, X-Search-Timeout-Seconds)`，不能由客户端延长 |
| 取消宽限 | 0.5 秒，已完成的具体错误仍保留 |

工具内部的 vLLM API 和 embedding SDK 使用本次调用独有的客户端，传入
`timeout=min(120, 工具剩余时间)`、`max_retries=0`；取消时关闭这些连接，
不会关闭其他 rollout 共用的原始模型客户端。工具内 vLLM 的 SDK 异常显式传出，
不再返回空字符串，也不再被 Tenacity 作为原始连接异常重试。
工具外的 Planner 等 vLLM 调用保持原来行为，没有改变其训练温度、token IDs 或 API 超时配置。

Wiki 选页的 API 异常不再变成空的选页列表；embedding/摘要的 API 异常也不再写成 `None`。
错误包含 `code`、`phase`、异常类型以及可取得的 HTTP 状态；阶段墙钟超时还包含实际预算。
典型阶段为 `wiki.select_pages`、`web.embedding`、`web.summary` 和 `search.fetch`。
API 返回 error envelope 的引擎也会被识别，避免把错误当成正文或空结果。
共享 API 故障停止后续页面；单页处理异常保留安全的类型信息，仍可处理其他页面。

超时会设置取消事件、关闭本次连接并禁止启动后续阶段，晚到的返回值不进入后续 RAG。
Python 不能强杀任意阻塞线程，因此不承诺内核卡死或不响应关闭的第三方调用立即退出；
调用方仍按墙钟预算返回错误。超过真正工具总预算时，仍会返回带阶段诊断的工具超时。

升级本次补丁必须同时同步新的 `execution_context.py`、Executor、Wiki/Web 工具、
搜索客户端、vLLM 引擎，以及共享服务的 `server.py`。无需新增训练配置项，
同步后重启共享搜索服务和训练进程；原先的 150 秒客户端配置仍适用。

服务端 `SEARCH_BLOCKED_DOMAINS` 是手工域名黑名单，代码默认留空；
`train-roma/search-service.env.example` 已填入用户提供的 325 个域名，按原始主机名保留，
不把 `www` 或学校院系子域名自动扩大为整个主域名。每项仍会同时拦截该主机及其子域名。
升级现有服务时，将示例中的完整 `SEARCH_BLOCKED_DOMAINS` 赋值复制到现有
`search-service.env`，保留原有令牌、密钥和其他配置；旧的空值不会自动加载示例清单。
逗号或空白分隔均可，示例按一行一个域名放在同一对单引号内。例如：

```bash
SEARCH_BLOCKED_DOMAINS='dead.example,another.example'
```

每项同时匹配域名自身及其子域名，忽略大小写、末尾的点，并统一国际化域名编码。
填写域名即可，不填写协议、端口或路径；`*.dead.example` 与 `dead.example` 等价。
域名匹配有边界，`notdead.example` 不会被 `dead.example` 拦截。

网页请求在启动唯一 producer、读取已有网页缓存之前检查黑名单；Wiki/Yibu 在读取其
API 缓存之前检查实际端点，所有上游 HTTP 请求及重定向目标也会检查。
命中后返回 HTTP 410，`error.code=domain_blocked`、`retryable=false`，消息为
`网站死亡：域名（域名黑名单）`；不访问被拦截的目标、不重试、不缓存错误。
重定向到黑名单时，初始允许域名的请求可能已经发送，但不会跟随到被拦截的目标。
普通搜索 query 中提及域名不会触发拦截，搜索结果也不按黑名单过滤；拦截的是实际请求目标。
不会因一次网络错误自动将网站加入黑名单。修改黑名单后需要重启共享服务。
`/metrics` 中的 `blocked_domain_requests` 统计域名拦截次数。

## 接口与指标

所有接口使用 `Authorization: Bearer <共享令牌>`，支持 gzip 响应。

| 接口 | 用途 |
| --- | --- |
| GET `/healthz` | 服务就绪检查 |
| GET `/metrics` | 缓存、合并、上游请求、失败和缓存容量统计 |
| POST `/v1/search/wikipedia` | query / max_pages / max_length / language → results |
| POST `/v1/search/brave` | query / count / country / search_lang / ui_lang / freshness → data |
| POST `/v1/fetch` | url / max_length → text |
| POST `/v1/batch` | requests 列表 → 相同长度、相同顺序的成功或失败项 |

客户端可通过 `SearchGatewayClient.from_env().batch(...)` 显式提交 batch。
现有工具的单请求调用也使用共享缓存和跨请求合并。

`upstream_errors/<code>` 按失败类型统计实际 HTTP attempt 的错误次数；
`requests_failed/<code>` 统计执行失败的逻辑请求原因，可区分代理、HTTP 超时和总预算耗尽。

`/metrics` 读取数据库失败时返回 HTTP 503 和 `cache_database_error`，详细异常写入服务日志，
避免未捕获异常造成空响应。`/healthz` 仅检查进程能否响应，不执行数据库完整性检查。

开发服务器向训练机发送结果仍消耗带宽，缓存命中仅避免再次访问个人主机出口。
保留当前截断规则并采用 gzip，不擅自减少候选数或缩短 RAG 输入。
实际收益应分别看 query、页面和 URL 命中率，以及真实上游请求数。

默认配置可在 `search-service.env.example` 修改：缓存有效载荷上限 2 GiB，
单次逻辑请求期限 120 秒，最大上游响应 20 MiB。服务期限、客户端连接超时及最大重试次数
保持原值；Python客户端默认读取超时为150秒；实际以进程env为准，启动shell未显式设置时仍会注入600秒，
本次未改该回退值。当前没有公共worker、queue、派生HTTP容量或上游共享限速配置。
SQLite 文件和 WAL 还会有索引/空闲页等开销，磁盘占用不等于有效载荷大小。
冷批次的大量不同query仍可能受真实上游限流、机器资源和逐请求重试影响；
缓存和合并只减少可复用请求，不能保证所有请求在预算内完成。

当前出口有自签代理证书，provider 默认沿用 `main` 的 TLS 设置。
有受信 CA 时可以设置 `SEARCH_SERVICE_VERIFY_TLS=1` 或 `SEARCH_SERVICE_CA_BUNDLE`。
训练客户端连接 HTTPS 共享服务时会校验证书。

## 离线验证

在仓库根目录运行：

```bash
python3 -m unittest discover -s agentflow/tests -p 'test_search_*' -v
bash -n train-roma/run_search_cache.sh train-roma/enable_search_cache.sh train-roma/open_search_cache_tunnel.sh
```

测试覆盖持久化、命中不续期、过期后重新获取、旧表结构兼容、并发合并、重复 batch 映射、
错误不缓存、指标接口数据库异常、共享限速已取消、黑名单及重定向拦截和三个工具的接入。
真实部署还需检查开发服务器端口可达性、代理出口、Yibu 账户额度，再用少量 rollout
检查 observation 和奖励，最后增加多任务并发。本地离线测试不代表已经完成远程部署。
