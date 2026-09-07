# V1 本地整合记录（2026-09-07）

基准仓库：https://github.com/huangyijieu-cyber/agentflow-v1 ，基准提交 `4d9baa2`。

以 Downloads/V1 中的文件为最新来源，同路径文件优先使用 V1；V1 缺失文件保留仓库版本。整理工作在本地独立副本进行，原 V1 未修改。

## 比对结果

基准仓库共 146 个跟踪文件，V1 提供其中 145 个，根目录 `.gitignore` 从仓库保留。140 个文件逐字节相同，5 个存在字节差异：

- `agentflow/agentflow/tools/search_gateway.py`：更新网关配置校验、超时、结构化错误、代理隔离和接口兼容处理。
- `train-roma/config.yaml`：默认 N_WORKERS 由 16 改为 32，启用工具及对应引擎中移除 Brave 项。
- `train-roma/serve_with_logs.sh`：更新设备配置提取命令中的转义。
- `agentflow/agentflow/tools/brave_search/tool.py`：仅末尾换行变化。
- `agentflow/agentflow/index.html`：仅 CRLF/LF 换行差异，Git 提交规范化为 LF。

新增源码脚本：`train-roma/enable_search_proxy.sh`、`train-roma/kill.sh`、`train-roma/train_control.sh`。

## 整理调整

- 代理脚本原有硬编码令牌和服务器地址改为读取 `PROXY_TOKEN`、`PROXY_HOST`；`PROXY_PORT` 默认为 18090，`SEARCH_GATEWAY_BASE_URL` 可覆盖默认地址。使用前需在本机环境中设置实际值。
- `.gitignore` 新增 PID 文件及 notebook 自动保存目录规则。
- 清除修改配置中的行尾空格。
- 数据、私钥、环境文件、训练输出、缓存及原仓库已忽略的服务器连接/分布式启动脚本保留在本地副本，按原仓库规则不上传。发布 ZIP 也只含 Git 跟踪文件。

## 验证

- 111 个 Python 源文件 AST 语法检查通过。
- 10 个 Shell 脚本 `bash -n` 检查通过；没有执行启动或停止训练脚本。
- YAML 解析通过，启用工具和对应引擎均为 4 项。
- 搜索网关使用模拟会话验证路径前缀、代理隔离、Wikipedia 列表兼容、fetch、Brave 参数、HTTP 错误、超时与非法配置处理，均通过。
- 待提交内容凭据特征检查未发现匹配；原硬编码代理令牌已确认不会进入提交。

未执行真实 GPU/NPU 分布式训练或外部搜索服务端到端测试。安装元数据引用的 agentflow/README.md 在 V1 和基准仓库中均不存在，此次未编造补齐；完整安装和运行仍需原项目环境、模型与数据。训练控制脚本沿用 V1 的服务器路径及进程清理逻辑，需在对应训练服务器使用。
