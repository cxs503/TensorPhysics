# 科研闭环 E2E 首跑证据（2026-11-03）

R3 接线（PR #99，research_runner 终态回写 TensorPlatform 运行登记）的**实跑证据**：
一条真实科研任务从执行→独立验证→终态→平台回写→回读校验全链路贯通，
外加一条诚实失败（honest-FAIL）腿。驱动脚本 `scripts/research_loop_e2e_first_run.py`。

## 1. 任务设计（真实物理，无 mock）

- **执行后端** = R1 审计内核 `shared_core_resistance` @ SUBOFF-01 典型工况
  （L=4.356, B=0.508, Cb=0.542, V=3.0 m/s, ρ=998.2, ν=1.004e-6, S=4.417911491407555）。
- **独立验证器** = 三重独立判据（**不 import 被测内核**）：
  1. 闭式内联重推（Cf=0.075/(log10 Re−2)²；Cr=0.002(1+2Cb)Fn^2.5, Fn>0.05；G=9.81）位级对齐 rel 1e-12；
  2. Schoenherr-1932 历史独立相关线交叉核对（|Cf_ittc/Cf_schoenherr−1|<5%）；
  3. 冻结契约锚 SUBOFF-01 ct=0.003461889103158906（provenance=DERIVED, rel 1e-9）。
- **失败腿** = 故意错配的验证器（EXPERIMENTAL 带 0.0038 配 DERIVED 带的 1e-9 容差），
  验证失败分支真实注册 failed 而非被覆盖。

## 2. 实跑事件流（append-only，sqlite 证据库）

成功腿 `e2e-first-run-2026-11-03b-success`（max_attempts=3，一次过）：

```
 7    -       -> queued    task_created
 8    queued  -> claimed   worker_claimed
 9    claimed -> running   agent_started
10    running -> verifying agent_completed
11    verifying-> succeeded validation_passed
12    succeeded-> succeeded platform_registration registration=registered
```

诚实失败腿 `e2e-first-run-2026-11-03b-fail`（max_attempts=1）：

```
13    -       -> queued    task_created
14    queued  -> claimed   worker_claimed
15    claimed -> running   agent_started
16    running -> verifying agent_completed
17    verifying-> failed   task_failed
18    failed  -> failed    platform_registration registration=registered
```

## 3. TensorPlatform 账本回读（同客户端 GET，organization_id=1）

| run_id | status | validation_status |
|---|---|---|
| e2e-first-run-2026-11-03-success（首跑，脚本中途打印崩溃但注册已完成） | succeeded | passed |
| e2e-first-run-2026-11-03b-success | succeeded | passed |
| e2e-first-run-2026-11-03b-fail | failed | failed |

账本 manifest 含 `registered_by=tensorphysics.research_runner`、goal_id、agent
（=内核全名）、attempt/max_attempts、summary（含 ct=0.0034618891031589064，
与冻结 DERIVED 锚位级一致）、artifact_uris、validation 全文。

## 4. 结论（均已证实）

1. **科研循环状态机与平台证据面贯通**：VERIFYING→SUCCEEDED 的科学验收判定
   （independent validator）触发 TP 运行登记，登记是 reporting 非 gating
   （失败不阻断状态机，但如实入账）。
2. **诚实失败纪律实跑成立**：错配验证器产生 failed 任务，平台收到
   status=failed/validation_status=failed——没有被改写、没有被吞掉。
3. **幂等键生效**：Idempotency-Key=run_id，三次注册三次回读全一致。

## 5. 局限与发现

- 执行后端是确定性内核代理（非 LLM agent）——首跑目标是链路证据，LLM 后端
  接入是后续项（adapter 契约已就位：`AgentBackend` Protocol）。
- 发现（已证实）：TP `GET /api/v1/users/me` 对本账号返回 422，organization_id
  只能回退默认 1——登记链的 org 解析有一处待修（不影响本跑，admin 落 MIN org）。
- 单节点 TP（127.0.0.1:8010），未验证远端/多租户路径；sqlite WAL 证据库在
  运行目录本地。

## 6. 复现

```
TP_LOGIN_USER=... TP_LOGIN_PASSWORD=... \
E2E_EVIDENCE_DIR=<dir> RUN_TAG=<date-tag> \
python3 scripts/research_loop_e2e_first_run.py
```
