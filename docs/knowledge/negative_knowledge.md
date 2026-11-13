# 负知识（append-only；每条 ≤120 字，严禁删改旧条目）

- 2026-11-13 m18417：TensorPhysics 编排层的 `CheckpointStore` 信封不含物理状态
  （payload 232 B、`restartable=False`、库内 0 个 `load` 消费者），**不能**断点续跑；
  真重启靠求解器自带 `IceDEM.snapshot()/from_snapshot()`（164733 B，逐比特相等）。
  另：`run()` 分块后状态回 `initialized` 可插采样，但 `dt` 必须是原生步整数倍。