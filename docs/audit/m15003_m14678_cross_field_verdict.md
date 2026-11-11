# m15003 · 交差核验（结榜审计）：m14678 GAP-CROSS-FIELD

**被审对象**: mission m14678 · TensorPhysics · GAP-CROSS-FIELD「跨库大规模场交换」· status=succeeded（2026-11-11 10:37:49Z）
**交付 commit**: `aba1e17` "GAP-CROSS-FIELD: in-memory cross-solver field exchange contract"，经 PR #390 merge（`fadfb66` Merge #390 (m14678) via merge-master）→ 现 main `5d4ee3b` 含该模块
**审计单**: m15003（board-audit）· 工作树 `/root/tp-agent/work/m15003`（= main，独立于被审工作树 `/root/tp-agent/work/m14678`）
**判级**: **L2**（机判同值；见 §5）· 落表 `tensor_maturity_verdicts` id 见 §6

---

## 1. 独立性与范围

- 独立性：本审计在**另一个干净检出**（m15003，main `5d4ee3b`）上重跑全部验收动作，不读被审工作树的 git 状态作为证据，只用其最终 commit 内容；被审 `result.json` 的自述只作为"待核验证"的声明。
- 未做（明确标 not_run）：
  - 未在集群 loong64 节点复跑（本机即 loongarch64，架构一致，见 `run_manifest.json` environment）。
  - 未做 ≥100 MiB 场的规模实验（无此交付，见 F3）。
  - 未验证 POSIX shm 在无亲缘关系**独立进程**（非 fork/非 subprocess 继承）下的实际行为——单测只在同进程与子进程内取证。标记 **not_run**。
  - 未验证 32 位平台 / 大端机上的 dtype 规范化路径（只有单测 `test_descriptor_rejects_non_contiguous_and_big_endian` 断言拒绝，属单测覆盖而非真机）。

## 2. 三件套逐项核验（①取证）

| 项 | 被审声明 | 本审计动作 | 结果 |
|---|---|---|---|
| run_artifacts | `artifacts/m14678/{results.json,run_manifest.json,fig_*.png}` + `runs/m14678/run.json` | 在 m15003 用 repro_cmd 重新生成；比对数字 | **真**。本轮重生成 `evidence.json` 记录：`shm_zero_copy` 4 MiB=46.07 ms、socket=23.66 ms、文件基线=2.04 ms；`file_api_calls=0`、`transfers/s=9323.7`；与原产物的 τ_w/拷贝台账/sha256 数字**逐位相同**（这正是"可复现"的判据）。注：`artifacts/`、`runs/` 未随 PR 进 main（合并门滤除工件），main 上只能靠 repro_cmd 重生成——已验证可行 |
| benchmark ref vs got | τ_w 解析 μU/H = 8.0000000000e-05 Pa vs 数值 7.9999999326e-05 Pa，rel 8.423e-09；剖面 rel L2 = 3.280e-09；回传 max\|Δ\|=0.0 | 独立复算 ref（μ=ρν=1e-3 Pa·s, U=1.6e-3 m/s, H=0.02 m ⇒ 1e-3·1.6e-3/0.02 = 8.000e-05 Pa ✓）；重跑得同值 | **真**。误差 8.4e-09 ≪ 3% 阈值；逐位一致 = 字节相等，非容差比较 |
| repro_cmd | `python tools/m14678_field_transport_benchmark.py --out artifacts/m14678` | 在本工作树原样执行 | **可执行，24.8 s 跑完**（4 阶段全绿，3 图重生成）。测试：`python -m pytest tests/test_field_exchange_transport.py -q` → **21 passed (2.13 s)**；`-k "field_exchange or volume_transfer or mapping"` → **81 passed, 1 skipped, 1 error**，该 ERROR 为 `tests/test_p27w3_integration_contracts.py` 缺 `artifacts/p27w3_contracts/peer_goldens/ice_bond_v1.json`（既有 checkout 缺失，与本改动无关；被审单已如实披露） |

代码真实性核查（防 MOCK/STUB）：`transport.py`(566 行)/`channels.py`(541 行)/`lbm.py`(239 行) 全为非占位实现；仅两处 `NotImplementedError` 出现在基类抽象方法（`FieldChannel.publish/receive` 与 `channels.py:108,111`），属正常抽象面。LBM 为自写 D2Q9 BGK + 半程反弹 + Guo 力项（`lbm.py:100-160`），可复现 Couette/Poiseuille 精确解——不是随机数或硬编码。
"零文件中介"取证：`no_file_audit` 用 `os.open/builtins.open/io.open` 三方 spy，500 次交换 0 次调用，`/dev/shm` 无新增条目——本审计重跑复现同值。

## 3. 判级（②）——L2，及为什么不是 L3/L4/L5

| 级 | 判据（平台 maturity_gate JUDGE_SCHEME） | 本案 |
|---|---|---|
| L1 | tests 非空 | 过（21 单测） |
| L2 | impl_files 真实现 + entry 可执行 + run_artifacts 存在非空 | **过**（§2 三项全真） |
| L3 | L2 + integration_callers（主求解器/平台侧调用方，grep 复核） | **不过**：全仓 grep，`field_exchange.transport/channels` 的调用方只有 `tests/test_field_exchange_transport.py` 与 `tools/m14678_*`；`src/tensorphysics/__init__.py` 未导出该模块；主耦合链 `fsi_ice/*`、`coupling_energy/three_field/*`、`fem_lbm_conservative_map/lattice.py:29` 只 import `field_exchange.mapping`（m14678 之前既有代码），**未使用传输层** |
| L4 | L3 + benchmark 数值对照 + repro_cmd | 基准证据本身达 L4 质量（完整算例 + 解析解误差 8.42e-09 + 可复现），但缺 L3 链 ⇒ 机判与实质都止于 L2 |
| L5 | L4 + external_anchor | 锚点齐（解析解/IEEE754/mmap/UDUNITS-2/MPI-4.0），但无科研工程闭环：无跨仓真实交换、无规模扩展实验 |

**级别=功能实质**：交付是一个**库内可用、可复现、有解析锚点的传输契约模块**（L2），未接入任何主求解器或平台调用链（L3 空）。报告写得再长也不改级别——本案报告质量确实高（诚实披露了哈希开销），但那是 L2 之上的证据密度，不是集成。

## 4. 审计发现（按严重度；详见 `artifacts/audit_m15003/verdict_evidence.json:audit_findings`）

- **F1（高，已量化，非阻断正确性）**：零拷贝通道在 ≥1 MiB **没有速度优势**。4 MiB：零拷贝 46.07 ms vs socket 23.66 ms vs 文件基线 2.04 ms。本审计实测本机 sha256 = **178.7 MB/s**；发/收各一次哈希 = 2×22.35 = 44.7 ms = 零拷贝延迟的 **102%** ⇒ 延迟全部来自完整性哈希，与"拷贝"无关（见 `artifacts/audit_m15003/fig_latency_attribution.png` 右图）。**被审报告 §5 已如实写出该现象**（"零拷贝买到的是零 payload 拷贝 + 零文件 I/O，不是低延迟"）——诚实申报，计入 credit；本审计的增量是把它量化成可核账（2×sha256 ≈ 100%）。
  可修方向（未在本单实施，属 m14678 后续）：接收端可选跳过重算（发布端 checksum + 帧头 CRCs/序号背书）、或换 xxhash/CRC32C、或分块增量哈希，使审计成本可控。
- **F2（中，方法论）**：传输矩阵**口径不一致**——socket 分支用后台发送线程做流式（pipelined）取逐帧 median，shm 两分支是同线程 `publish+receive` 往返；文件基线 `np.save/np.load` 未 `fsync`（page cache，4 MiB 时"4115 MB/s"不是持久化速度）。三者不可直接排名次，报告表格的排序结论需带此限定。
- **F3（中，范围/命名）**：**"跨库/大规模"名不副实**——两进程跑的是同一个 `field_exchange.lbm`；无第二种求解器库参与，无 TensorLBM/TensorFEM 真实跨仓交换；规模上限仅 **4 MiB** 场（524288 doubles），无 100 MiB 级实验。建议补一条跨仓（TensorLBM ↔ TensorPhysics）真实交换 + ≥100 MiB 规模案例才能撑起 GAP 名称。
- **F4（低，一致性）**：`src/tensorphysics/__init__.py:8` 从 `.fields` 导出 `FieldDescriptor/FieldExchangeError`，与 `field_exchange.transport` 的**同名异义**类并存（未 shadow，但同包两名，易误用）；且顶层未导出新模块 ⇒ 公众面未接入（这也是 L3 空的一个表征）。
- **F5（低，痕迹）**：m14678 `result.changed_files` 列了 `tools/m14678_smoke.py`，但 `aba1e17` 的 14 文件 diff 里没有它；`artifacts/`、`runs/` 未进 main。声明与 commit 有小偏差（不影响可复现性，repro_cmd 已重生成验证）。
- **F6（低，适用边界）**：单生产者/单消费者；环形槽被覆盖由 seq 校验抛 `FrameCorruptError`（有测），但**多生产者未实现、无背压**；`AnonymousShmChannel` 依赖 fork 继承（无亲缘须 `PosixShmChannel`/`SocketChannel`）。报告已声明这些限制。
- **F7（平台侧缺陷，非 m14678 责任）**：`maturity_gate` 的 grep 根 `/root/tensor-ecosystem/TensorPhysics` 停在 **2026-11-02**（`c4f0af5`），**整个 `field_exchange` 目录都不存在** ⇒ 即便将来真有 L3 集成调用方，L3/L4/L5 也会被误降到 L2。本次核验用 `maturity_gate.judge()` 预演三次（声明 5/4/3/2）确认：唯一的缺项恒为 `integration_callers`，声明 2 时 `why=[]` 直接通过。建议平台刷新 ECO 检出，否则 TensorPhysics 系审计系统性地卡在 L2。

## 5. 机判预演（`maturity_gate.judge` 只读调用）

```
judge(5, ev, 'TensorPhysics', 15003) -> (2, ['L5 缺: integration_callers,external_anchor(工件缺失)', 'L4 缺: integration_callers', 'L3 缺: integration_callers'])
judge(4, ev, 'TensorPhysics', 15003) -> (2, ['L4 缺: integration_callers', 'L3 缺: integration_callers'])
judge(3, ev, 'TensorPhysics', 15003) -> (2, ['L3 缺: integration_callers'])
judge(2, ev, 'TensorPhysics', 15003) -> (2, [])       # 声明 2 ⇒ 机判 2，无缺项
```
本 verdict **声明 level=2**，与机判一致（无虚标）。差异说明：L4 级"benchmark"键、`repro_cmd`、`external_anchor` 键仍如实填了真实指针，供后续集成补齐后一键升级参考。

## 6. 落表与产物

- verdict 落表：`tensor_maturity_verdicts`（repo=TensorPhysics, scope=`m14678 GAP-CROSS-FIELD`, mission_id=15003, level=2, verdict_by=`board-audit`），evidence 八键见 `artifacts/audit_m15003/verdict_evidence.json`。
- 本审计产物：
  - `scripts/audit_m15003_cross_field.py`（延迟归因测量，只测不改）
  - `artifacts/audit_m15003/evidence.json`（重跑复现 + 归因表 + findings）
  - `artifacts/audit_m15003/fig_latency_attribution.png`（左：四通道延迟 vs 载荷；右：2×sha256 占零拷贝延迟比）
  - `artifacts/m14678/*`（本工作树内重跑生成：results.json / run_manifest.json / 3 图）

## 7. 处置（disposition）

**Approve with notes**（confirm 消账于 L2），并建议（非阻断）：
1. m14678 后续单：接收端哈希可跳过 / 换快哈希（F1），修正传输矩阵口径（F2）；
2. 新开一条"跨仓真实交换 + ≥100 MiB"节点，才能把 GAP-CROSS-FIELD 从 L2 推到 L3/L4（F3）；
3. 平台刷新 `/root/tensor-ecosystem/TensorPhysics`（F7），否则该仓审计上限被人为锁在 L2。

不改变被审单的 succeeded 终态（本单只裁成熟度级别，不重做、不覆盖 honest 记录）。