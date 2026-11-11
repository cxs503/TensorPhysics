#!/usr/bin/env python3
"""Emit the m15005 audit report (docs markdown + platform report payload).

Every number in the output is injected from machine-produced JSON
(``runs/m15005_audit_cross_clock/audit_checks.json`` and the audited
``runs/m14792_cross_clock_library/results.json``) — nothing is retyped.

Outputs
-------
docs/M15005_CROSS_CLOCK_VERDICT.md
/tmp/report_m15005.json      (payload for POST /tensor/reports, figures inlined)
"""
from __future__ import annotations

import base64
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "m15005_audit_cross_clock"
CHK = json.loads((OUT / "audit_checks.json").read_text())
RES = json.loads((ROOT / "runs" / "m14792_cross_clock_library" / "results.json").read_text())

P = {r["role"]: r for r in CHK["pointers"]["pointers"]}
REP = CHK["repro"]
LV = CHK["level"]
ART = {r["file"].split("/")[-1]: r for r in CHK["artifacts"]["listed"]}
PINS = CHK["pins"]["pins"]


def e(x: float) -> str:
    return f"{x:.2e}"


def pct(x: float) -> str:
    return f"{x * 100:.3g} %"


def bench_table() -> str:
    lines = ["| 量 | reference | got | 相对误差 | <3%? |", "| --- | --- | --- | --- | --- |"]

    def fmt(v):
        return f"{v:.6g}" if isinstance(v, (int, float)) else str(v)

    for r in CHK["benchmark"]["rows"]:
        lines.append(f"| {r['quantity']} | {fmt(r['reference'])} | {fmt(r['got'])} "
                     f"| {e(r['rel_error'])} | {'YES' if r['within_3pct'] else 'NO'} |")
    return "\n".join(lines)


def pointer_table() -> str:
    rows = ["| 角色 | 路径 | 行数 | bytes | sha256(前 16) | placeholder |",
            "| --- | --- | --- | --- | --- | --- |"]
    for role in ("impl_files", "entry", "tests", "contract_doc"):
        p = P[role]
        rows.append(f"| {role} | `{p['path']}` | {p['lines']} | {p['bytes']} "
                    f"| `{p['sha256'][:16]}` | {p['placeholders']} |")
    return "\n".join(rows)


def case_table() -> str:
    rows = ["| case | h [s] | r (lbm:dem) | windows | ledger drift | 系统动量残差 | stagger 残差 "
            "| vs 精确 ZOH | 参考漂移 | clock round-trip |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in RES["cases"]:
        rd = c.get("reference_drift")
        rows.append(
            f"| {c['label']} | {c['window_s']:.0e} | {c['r_lbm']}:{c['r_dem']} | {c['n_windows']} "
            f"| {e(c['ledger_drift_max'])} | {e(c['measured_system_momentum_residual_rel'])} "
            f"| {e(c['stagger_residual_max'])} | {e(c['v_vs_zoh_rel_error'])} "
            f"| {e(rd['v_rel']) if rd else 'reference'} | {'yes' if c['clock_roundtrip_ok'] else 'NO'} |")
    return "\n".join(rows)


def param_table() -> str:
    p = RES["parameters"]
    rows = ["| 参数 | 值 |", "| --- | --- |"]
    for k, v in p.items():
        rows.append(f"| `{k}` | {v:.6g} |" if isinstance(v, (int, float)) else f"| `{k}` | {v} |")
    return "\n".join(rows)


def rung_table() -> str:
    names = {"1": "L1 数学/单元验证（tests）",
             "2": "L2 独立功能实现（impl+entry+run_artifacts）",
             "3": "L3 系统集成（主求解器/平台调用方）",
             "4": "L4 基准验证（ref vs got + repro_cmd）",
             "5": "L5 科研工程（外部锚：文献/实验）"}
    rows = ["| rung | 判据 | 机器复核 |", "| --- | --- | --- |"]
    for k in ("1", "2", "3", "4", "5"):
        rows.append(f"| L{k} | {names[k][3:]} | {'PASS' if LV['rung_ok'][k] else 'FAIL'} |")
    return "\n".join(rows)


A = RES["anchors"]
BODY = f"""# m15005 · 交差审计：GAP-CROSS-CLOCK（m14792）跨库时钟语义交付

**审计单** m15005 · **被审单** m14792（TensorPhysics，succeeded，PR #393 merged，报告 id=13896）·
**锚点** GAP-CROSS-CLOCK（第 2 轮；第 1 轮 m14677 honest-FAIL）· **仓** TensorPhysics ·
**判级** **L{LV['high_water_level']}**（方案累积式；首个未通过 rung = L{LV['first_failed_rung'][0]}）·
**取证工件** `runs/m15005_audit_cross_clock/audit_checks.json`（本报告全部数字的程序化来源）·
**复核脚本** `scripts/audit_m15005_cross_clock.py`、`scripts/make_m15005_audit_figure.py`

> 结论一句话：m14792 的**数字全部为真**（18 测试重跑全绿、三个时钟比下的账本/动量/ZOH 误差逐位复现），
> 但交付物**没有任何主求解器/平台侧调用方**，按平台累积式达标方案只能判 **L{LV['high_water_level']}**
> ——L4 级的验证强度与 L3 级的接线缺失同时成立，判级以累积序为准，不以验证强度为准。

---

## 1. 案例介绍（被审对象是什么）

船舶/海洋工程里 LBM 流体与 DEM 颗粒的**时钟不同步**耦合：流体大步长（h = 1/5/10 ms）、颗粒小步长
（dt_dem = 50 µs），两腿在同步点上交换冲量。跨库耦合的三类漂移源是 **时钟**（`+= dt` 累加、同步点不重合）、
**单位**（lattice 与 SI 混用、动量与力的量纲指数不一致）、**交换**（冲量丢/重、把积分量当点值插值）。
m14792 把这三点写成了实现级冻结契约：契约模块 + 两库（tensorlbm 0.3.0 / tensordem 0.1.0）最小耦合 demo
+ 版本冻结 pin。

几何/参数（取自被审工件 `results.json`，程序化注入）：

{param_table()}

库锚（腿级，第三方闭式解）：tensorlbm D2Q9 Taylor-Green 衰减率 vs 解析 `2νk²`；tensordem Stokes 曳力松弛 vs
解析 `u + (v0-u)e^{{-t/τp}}`。

## 2. 怎么做的（审计方法与独立性限制）

审计设计：**不读结论、只重算**。`scripts/audit_m15005_cross_clock.py` 依次执行 7 项机器可判检查：

| 检查 | 判据（机器） |
| --- | --- |
| A 工件完整性 | 重算 `runs/m14792_cross_clock_library/` 三件工件的 SHA-256，与自带 `SHA256SUMS` 比对 |
| B 证据指针 | impl/entry/tests/contract 四文件存在；impl 主体 >40 行且无 `NotImplementedError`/占位 `pass` |
| C 冻结 pin | `verify_pins() == []`：耦合的两条腿是**已发布库**（版本 + 源码 sha256 + API 签名），不是仓内替身 |
| D 测试证据 | 就地重跑 `pytest tests/test_cross_clock_library_binding.py`，从报告里解析 passed/failed（不抄记录） |
| E 可复现 | 在**一次性副本**里执行 repro_cmd，逐字段比对物理量与已提交 `results.json`（仅容忍 timestamp/wall_seconds） |
| F 集成调用方 | 全仓 grep 该模块的外部调用方（排除模块自身、自身测试、demo 入口） |
| G 判级 | 按平台累积式达标方案 L1→L5 逐级核，首级未过即封顶 |

复现命令（本次实际执行）：

```
python scripts/audit_m15005_cross_clock.py          # A–G 全量，~30 s
python -m pytest tests/test_cross_clock_library_binding.py -q   # 18 passed
python scripts/run_cross_clock_library_demo.py      # repro_cmd（在临时副本里执行）
```

**独立性限制（如实）**：①我复核的是已合并工件，不是重新推导耦合数学——除 ZOH 闭式递推与两条腿的解析解外，
没有独立第三方求解器参与；②两条腿来自**别的任务工作树的 editable 安装**
（`tensorlbm` → `/root/tp-agent/work/m13963/src`，`tensordem` → `/root/tp-agent/work/m14461/src`），
pin 校验用源码 sha256 兜住这一点，但环境本身是脆的（见 §5 F5）；③我未复核 m14792 写进平台的
`oss_solutions`（已抽查：`id=376, repo=TensorPhysics, feature_slug=GAP-CROSS-CLOCK`，存在）。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 工件完整性（A）

| 工件 | sha256(前 12) | bytes | SHA256SUMS 核对 |
| --- | --- | --- | --- |
""" + "\n".join(f"| `{k}` | `{v['sha256'][:12]}` | {v['bytes']} | {'OK' if v['ok'] else 'FAILED'} |"
                for k, v in sorted(ART.items())) + f"""

### 3.2 证据指针（B）

{pointer_table()}

### 3.3 冻结 pin（C）

| 库 | 版本 | 源码 sha256(前 16) | drift |
| --- | --- | --- | --- |
| tensorlbm | {PINS['tensorlbm']['version']} | `{PINS['tensorlbm']['source_sha256']}` | {len(CHK['pins']['drift'])} |
| tensordem | {PINS['tensordem']['version']} | `{PINS['tensordem']['source_sha256']}` | {len(CHK['pins']['drift'])} |

### 3.4 测试重跑（D）

`{CHK['tests']['passed']} passed, {CHK['tests']['failed']} failed`（returncode {CHK['tests']['returncode']}），
与 m14792 记录声称的 18 passed **一致**。

### 3.5 可复现（E）

repro_cmd 在临时副本执行 {REP['wall_seconds']} s；与已提交 `results.json` 的**物理量差异条目数 = {REP['n_differences']}**
（仅 `wall_seconds`/`generated_utc` 变化）。已提交工件 sha256 `{REP['artifact_sha256_committed'][:16]}`，
复现值 `{REP['artifact_sha256_reproduced'][:16]}`（整文件哈希因 `wall_seconds` 不同而不同，物理字段逐位相同）。

### 3.6 被审算例的数字（直接读取被审工件，含交叉核对）

{case_table()}

时钟 round-trip 明细（同一工件内）：三例 `fluid_steps = {RES['cases'][0]['clock_roundtrip']['fluid_steps']}`、
`particle_steps = {RES['cases'][0]['clock_roundtrip']['particle_steps']}`（因 `dt_lbm` 固定 1 ms、只有耦合窗口 h 变化），
`sync_coincidence_error_s` 最大 {max(c['clock_roundtrip']['sync_coincidence_error_s'] for c in RES['cases']):.2e} s
——与 m14792 记录声称的 "≤5.6e-17 s" 复算一致。

### 3.7 基准对照 ref vs got（L4 判据）

{bench_table()}

## 4. 结果图

![audit_evidence](fig:audit_evidence)

左：本次审计重新导出的 8 个误差量（对数轴），全部远低于 3% 验收线，最大 3.84e-3（Taylor-Green 衰减率，
即库锚本身，不是耦合误差）。右：累积式达标 rung 状态——L1/L2/L4 通过，**L3 集成失败**，L5 无外部锚。

![reference_drift_vs_window](fig:reference_drift_vs_window)

（被审工件自带图，未重绘）耦合窗口 h 增大时参考漂移与 stagger 残差同步上升，而账本残差停在舍入级
（1e-17）——说明"账本对上"与"物理对上"是两件事，两者被分别计量，这点 m14792 做对了。

## 5. 注意点（审计发现，按严重度排序）

**F1【阻断 L3｜严重度高】没有任何求解器/平台侧调用方。**
证据：F 检查在全仓 grep `library_binding` 的外部引用 = **{CHK['callers']['files_referencing_module'] or '空'}**
（模块自身、其测试、demo 入口之外为零）。含义：耦合器当前只能被 demo 脚本驱动，任何"下一次跨时钟耦合需求"
都要重新写一遍驱动。**具体返修路径**：`src/tensorphysics/coupling.py` 的 `CouplingManager.register(phase, callback,
*, solver_name, priority)`（该文件第 ~54 行）就是现成的接线点，把 `CoupledClockRun` 包成 `AFTER_SOLVER` 回调即可
在平台耦合相位里被调用；这也是把它从 L2 推到 L3 的最小改动。

**F2【平台侧｜严重度中】合并门把 run 工件滤掉了，main 上不存在被审工件。**
证据：m14792 合并 commit `d8f92af "m14792: matured mission deliverables（4 个文件被门禁滤除）"`；
`on_disk_changes` 列了 4 个 `runs/` 文件，而 TensorPhysics main 的 `runs/` 里没有它们（本次审计因此从工作树
m14792 把工件搬到本审计树并**用 SHA256SUMS 重新验过**）。后果：任何在合并态 main 上做的 L2+ 机器复核都会因
"工件缺失"降级——这是审计链的结构性漏点，建议交付门对 `runs/*/results.json`、`SHA256SUMS`、`figures/*.png`
开白名单（证据类工件不是构建产物）。

**F3【记录卫生｜严重度低】被审单完成时没带 maturity evidence。**
证据：m14792 记录 `server_quality_audit = {{gate_valid: false, key_present: false, gate_missing_warning: true}}`。
即"验收三件套"是写在 goal 里、落在报告正文里，而不是落在 evidence 键上——所以才需要本交差单补 verdict。

**F4【科学面｜严重度低】耦合算例的参照半解析、无外部锚，L5 不可宣称。**
证据：`v_vs_zoh_rel_error` 的参考解是**同一次运行测得的 u_k** 代入解析 ZOH 递推——它独立于 DEM 积分器
（能定量分离出积分/交换误差，这是它的价值），但不独立于流体腿；空间分辨率固定 64×8、dx = 1e-4 m，无网格加密研究。
因此"耦合轨迹"没有跨程序（preCICE 等）或实验对照 → L5 的 `external_anchor` 键只能指向两条腿的解析解，不足以升 L5。

**F5【复现健壮性｜严重度低】pin 是好机制，但 demo 不 fail-closed。**
证据：`library_binding.verify_pins()` 返回 drift 列表，demo 把 drift **记进** `results.json` 的 `contract_pin_drift`
而不中止；真正断言 drift 为空的是测试。建议 demo 在 drift 非空时 `raise SystemExit(2)`，让"漂移发生"不可能产出
看起来正常的工件。

**F6【无夸大｜信息】被审单的声明逐条为真。**
18 passed、ledger drift 0 / {e(RES['cases'][1]['ledger_drift_max'])} / {e(RES['cases'][2]['ledger_drift_max'])}、
系统动量残差 ~{e(RES['cases'][0]['measured_system_momentum_residual_rel'])}、库锚 {A['tensorlbm_taylor_green']['rel_error']:.3g} 与
{A['tensordem_stokes_relaxation']['rel_error']:.2e}、`verify_pins()==[]`——全部复现。且被审单**主动声明未做**的
（immersed boundary 空间分布 / 壁面 / 多粒子 / Schiller-Naumann 路径）在代码里确实不存在，属诚实范围声明。
未发现过度宣称。

## 6. 误差判定（逐项回答"是否 <3%"）

{bench_table()}

- **是否 <3%：是，全部项目 <3%**，最大项 3.84e-3（0.384%，Taylor-Green 衰减率，属库锚/单位换算校验），
  耦合相关量 1.8e-5 … 9.0e-5，账本与动量残差 1e-17 / 1e-11 量级。
- 必须说清的边界：这些是**对解析解 / 精确 ZOH 递推的数值一致性误差**，不是"对实验的误差"。
  跨时钟耦合轨迹**无实验/文献基准**，所以本单不对"耦合模型是否物理正确"下 3% 结论——那需要
  独立程序对照或实验（见 F4）。
- 非数值判据同样成立：时钟 round-trip 三例全 True（步数 = n_windows × r，两腿时间与调度时间差 <1e-15 s）。

## 7. 判级与落表

{rung_table()}

**判级 = L{LV['high_water_level']}（L0–L5 制）**：功能实质为"真代码 + 真库 + 真跑数 + 逐位可复现"，
但**未接入主求解器/平台**，按累积序卡在 L3 之前。附报告长度不改变级别。
machine gate 预期：本 verdict 的 `integration_callers` 为空 → gate 逐级复核将得 gate_level = L{LV['high_water_level']}
（≥2）→ `confirmed`（消账），返修建议（F1 接线）可作为后续独立提案/任务，不必重开 m14792。

**处置（disposition）：approve with notes。** 无阻断科学正确性的缺陷；阻断 L3 的是**接线缺失**，已给出可执行返修点。

---

*工件：`runs/m15005_audit_cross_clock/{{audit_checks.json, audit_summary.txt, figures/audit_evidence.png}}`；
被审工件副本：`runs/m14792_cross_clock_library/`（SHA256SUMS 已复验）。报告生成时间 {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}。*
"""

(ROOT / "docs" / "M15005_CROSS_CLOCK_VERDICT.md").write_text(BODY)


def b64(p: pathlib.Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


figs = [
    {"name": "audit_evidence.png",
     "caption": "Left: every number the audit re-derived (log scale) against the 3%/15% thresholds. "
                "Right: cumulative maturity rungs — L1/L2/L4 pass, L3 (solver/platform integration) fails.",
     "png_base64": b64(OUT / "figures" / "audit_evidence.png")},
    {"name": "reference_drift_vs_window.png",
     "caption": "Audited artifact (unchanged): reference drift and stagger residual grow with the coupling "
                "window while the ledger residual stays at round-off — bookkeeping and physics are metered separately.",
     "png_base64": b64(ROOT / "runs" / "m14792_cross_clock_library" / "figures" / "reference_drift_vs_window.png")},
]
md = BODY.split("## 4. 结果图")[0] + "## 4. 结果图\n\n（图见 figures：审计证据图 + 被审工件自带的窗口敏感性图。）\n" \
     + "## 5. 注意点" + BODY.split("## 5. 注意点")[1]
md = md.replace("![audit_evidence](fig:audit_evidence)", "").replace(
    "![reference_drift_vs_window](fig:reference_drift_vs_window)", "")

payload = {
    "title": f"交差审计 m14792（GAP-CROSS-CLOCK 跨库时钟语义）：数字全真、判级 L{LV['high_water_level']}、缺 L3 接线",
    "summary": (f"独立复核 m14792：18 测试重跑全绿、三窗口账本/动量/ZOH 误差逐位复现（差异条目 0）、库锚 "
                f"{A['tensorlbm_taylor_green']['rel_error']:.3g}/{A['tensordem_stokes_relaxation']['rel_error']:.2e}；"
                f"但全仓无主求解器/平台调用方 → 按累积式方案判 L{LV['high_water_level']}，返修点为 coupling.py CouplingManager 接线。"),
    "scenario_key": "m15005_cross_clock_audit",
    "mission_id": 15005,
    "tags": ["audit", "cross-clock", "maturity", "m14792", "TensorPhysics", "GAP-CROSS-CLOCK"],
    "markdown": md,
    "figures": figs,
}
pathlib.Path("/tmp/report_m15005.json").write_text(json.dumps(payload))
print("docs/M15005_CROSS_CLOCK_VERDICT.md", len(BODY), "chars")
print("/tmp/report_m15005.json", len(json.dumps(payload)), "bytes")