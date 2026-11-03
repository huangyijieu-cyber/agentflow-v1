# ALFWorld 实验版本归档

当前快照：**v0**。

本提交是从来源源码建立的实验归档快照，不冒充原始开发提交。`websearch` 分支保留原 `trace` 的完整开发历史。当前源码来源：`8083f0d7159ed2058b93e4460417c09fe2b11029`。

当前 `TRACE_K_PAIRS_PER_GROUP` 默认值：**不适用：保留原代码配置**。v1b 在相同源码上将两份训练 YAML 和包级 TRACE 配置默认值设为空；其它版本保留来源代码的默认值。

| 版本 | 可检出的标签 | 来源提交 | 行为 |
| --- | --- | --- | --- |
| v0 | `alfworld/v0` | `8083f0d7159ed2058b93e4460417c09fe2b11029` | 原始奖励配对，未引入 TRACE |
| v1a | `alfworld/v1a` | `f57dd66799591f764fa5163b1c703ab834042c58` | 完整状态与累计历史；同题同 turn，默认每 turn 2 对 |
| v1b | `alfworld/v1b` | `f57dd66799591f764fa5163b1c703ab834042c58` | 同 v1a 评分；同题跨 turn，无组内限额 |
| v2a | `alfworld/v2a` | `3f7abe9438c58d8d003984095ff1bc65af99d72b` | 移除累计评分历史；固定 Response 仍是成功条件 |
| v2b | `alfworld/v2b` | `ce3e97259c71ee71f5865273565a52706376e1a9` | 成功条件进入 Prompt；固定未来成功句，含剩余预算及测试修复 |

这些快照在 `alfworld` 分支按 v0 → v1a → v1b → v2a → v2b 顺序提交。每个标签固定指向对应快照；已有 `alfworld-v2a` / `alfworld-v2b` 和旧 QA 标签保留原位置。

## 切换示例

```bash
git fetch origin --prune --tags
git switch --detach alfworld/v1a
# 切换前清理此前 shell 中的覆盖值，才能使用各快照的默认配置。
unset TRACE_K_PAIRS_PER_GROUP
```

v1a 默认为 2，v1b 默认为 None。正整数表示同题同原始 turn 配对并按 turn 限额；None 表示同题同类型跨 turn，无组内限额，仍受全步默认 256 对限制。v2a/v2b 继续支持这两种设置。

v2a/v2b 是评分方式的区别：v2a 评分原成功条件陈述；v2b 将成功条件放入 Prompt 的要求区域，评分固定句 `The agent will successfully complete the task within the remaining action budget.`。v2b 来源包括 `5ad5aca` 的功能实现和 `ce3e972` 的测试夹具修复。

v1a 及以后训练复用已准备好的含 GT Parquet。v2a/v2b 无需重新转换，切换后重启训练及 rollout 进程；不同评分版本的旧 rollout 缓存不能混用。策略输入、算法和历史缺陷均以该版本来源源码为准，此归档不追补后续修复。
