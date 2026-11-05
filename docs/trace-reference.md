# TRACE 参考模型更新

ALFWorld 和 QA 共用 `data.trace.reference_update_interval`，仅在算法包含
`slic` 且 `data.trace.enabled=true` 时生效。支持 `grpo+slic`、`gigpo+slic`。

在 `train-roma/config.yaml` 或 `train-roma/config_grpo+slic.yaml` 中配置：

```yaml
env:
  TRACE_REFERENCE_UPDATE_INTERVAL: ${env:TRACE_REFERENCE_UPDATE_INTERVAL,0}
```

也可在原有启动命令前设置环境变量：

```bash
export TRACE_REFERENCE_UPDATE_INTERVAL=10    # 每完成 10 个训练 step 刷新
# export TRACE_REFERENCE_UPDATE_INTERVAL=None  # 每完成一个训练 step 刷新
# export TRACE_REFERENCE_UPDATE_INTERVAL=0     # 固定初始模型，默认值，保留原实验行为
```

直接使用 Hydra 参数时，写 `data.trace.reference_update_interval=null` 表示每步刷新。
`None`/`null` 字符串也会正规化为空值；负数、布尔值和小数不合法。

| 配置 | step 1 的参考 | step 2 的参考 | step 11 的参考 |
| --- | --- | --- | --- |
| `0` | 初始模型 | 初始模型 | 初始模型 |
| `10` | 初始模型 | 初始模型 | step 10 更新后的 actor |
| `None` / `null` / `1` | 初始模型 | step 1 更新后的 actor | step 10 更新后的 actor |

“step” 是完成整个训练更新的 step，不是 turn、rollout、微批次或单次优化器更新。
只有收集数据、因 stale 队列尚未满足条件提前返回时，不更新参考。
一个 step 内的 GT LogP、初始状态 V 基准、局部信用和终局回填全部使用同一个参考版本。
待 GRPO/GiGPO 和该步 SLIC 更新都完成后，才复制最终 actor 参数，供下一步使用。

此处 checkpoint 指 actor 的内存权重快照，不依赖 `trainer.save_freq`。
FSDP/FSDP2 的 actor 和 reference 同 rank 同进程时，直接复制已分片参数；不会经
driver 收集整模型，不会每步写磁盘，不复制优化器、不改 actor 梯度。参考模型继续
保持 eval 模式、评分温度 1。更新后清除共享前缀评分的验证状态，重新验证新权重。
不兼容的模型分片布局会明确报错；目前不支持 LoRA/ref_in_actor 和 Megatron 刷新。

刷新的是现有的独立 reference：TRACE GT 评分使用它；如果另行开启 KL reward 或
KL loss，它们也使用该参考模型。默认两种 KL 都关闭。SLIC loss、配对阈值、配对
方式、replay 采样与容量不变；已有 replay pair 保留其选入时的偏好标签，不重新评分。

## 保存和恢复

普通训练 checkpoint 开启时，会在 `global_step_N/trace_reference/` 一并保存参考模型
分片和 `trace_reference.json`，记录更新间隔与参考来源 step。例如 interval=10、
训练保存于 step 17 时，actor 是 step 17，参考仍是 step 10，恢复时分别加载。
该参考存储随 actor 的 checkpoint 保留策略清理。

恢复时必须使用相同 interval。旧 checkpoint 没有参考快照时，只有恰好处于更新边界
才能从恢复后的 actor 重建参考；否则报错，避免把较新的 actor 冒充旧参考。`0` 模式
沿用原恢复行为。远程 HDFS 恢复仍受原训练器的限制。

## 指标

- `trace/reference_step`：本 step 评分实际使用的权重来源，0 为初始模型。
- `trace/reference_step_next`：本 step 完成后参考来源，供下一 step 使用。
- `trace/reference_refreshed`：本 step 是否执行刷新。
- `timing_s/reference_refresh`：刷新阶段耗时，计入 step 总耗时。

在 `alfworld` 分支本次更新中，评分历史已恢复正式 v2_e（`ee5da31`）的原生 memory 格式，
同时保留独立 `ALFWORLD_TRACE_HISTORY_LENGTH`（默认 4、None 全部、0 不包含历史）和
同题局部信用范围内的动态终局回填。设置刷新间隔为 0 可隔离参考更新这一实验变量。
