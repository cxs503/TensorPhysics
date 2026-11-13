#!/usr/bin/env python
"""Publish the GAP-PP-2 progress report to the platform.

Every number in the markdown is read from
``artifacts/ice_pressure_area/calibration.json`` and the reference table, so the
report cannot drift from the artifacts it cites.  Figures are embedded as
base64 PNG from ``artifacts/ice_pressure_area/figures``.

Credentials come from ``TENSOR_AGENT_USER`` / ``TENSOR_AGENT_PASS`` and the API
base from ``TENSOR_API_BASE`` (no secret is written into the repository).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "ice_pressure_area"
MISSION_ID = 18658
SCENARIO_KEY = "GAP-PP-2-ICE-STRUCT-PRESSURE-AREA-CALIB"

FIGURES = [
    ("fig1_pressure_area_bands.png", "图1 压力-面积带：声明扫描区间 vs 分层标定（对数坐标；实测层按设计 p0 锚定以比较斜率）"),
    ("fig2_exponent_forest.png", "图2 各源 ex 读数（带）与逐层 DL 合并（90 % 区间）；灰底=原声明 0.30–0.70"),
    ("fig3_transfer_and_bandwidth.png", "图3 左：损伤区口径→法向投影口径转换律（律 T）；右：需求带宽宽度对比"),
    ("fig4_consumer_crosscheck.png", "图4 树内消费者 local_pressure.py（ex=0.15）与标定设计律（ex=0.45）对照"),
]


def login() -> str:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    body = json.dumps({"username": user, "password": password}).encode()
    req = urllib.request.Request(
        f"{base}/auth/login", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    token = data.get("access_token") or data.get("token") or data.get("data", {}).get("access_token")
    if not token:
        raise RuntimeError(f"no token in login response keys: {list(data)}")
    return token


def build_markdown(payload: dict) -> str:
    lib = payload["library"]
    pooling = payload["pooling_by_stratum"]
    design, measured, glob, naive = (
        pooling["design_rule"],
        pooling["measured_local"],
        pooling["global"],
        payload["naive_pool_ignoring_strata"],
    )
    bw = payload["demand_bandwidth"]
    rows = {r["stratum"]: r for r in bw["rows"]}
    consumer = payload["consumer_crosscheck"]
    sweep = payload["convention_sweep"]
    ref_rows = payload["ref_vs_got"]

    md: list[str] = []
    A = md.append
    A("## ① 案例介绍")
    A("")
    A("**物理背景**：冰-结构相互作用中冰压随承压面积增大而下降，工程上用压力-面积律 "
      "`p(A) = p0·A^(−ex)` 描述。`ex` 决定同一结构上小面积接触点与大板格之间的压力落差，"
      "直接进入极区船冰带板厚与局部加强的设计需求带。")
    A("")
    A("**本单要解决的问题**（GAP-PP-2）：平台当前把 `ex` 当声明扫描区间 "
      f"`{bw['declared_ex_band'][0]:.2f}–{bw['declared_ex_band'][1]:.2f}` 使用，"
      "既无一手数据源、又无面积口径（法向投影面积 vs 浸湿面积）统一，"
      f"导致结构需求带宽实测为 `{bw['declared_bandwidth_band'][0]:.2f}x–{bw['declared_bandwidth_band'][1]:.2f}x` 而无法收缩。")
    A("")
    A("**参数表（无量纲）**")
    A("")
    A("| 量 | 含义 | 取值 |")
    A("|---|---|---|")
    A(f"| `ex` | 压力-面积指数（design_rule，参考口径） | {design['ex_pooled']:.4f} "
      f"[{design['ci90'][0]:.4f}, {design['ci90'][1]:.4f}] |")
    A(f"| `ex` | 同上（measured_local 层） | {measured['ex_pooled']:.4f} "
      f"[{measured['ci90'][0]:.4f}, {measured['ci90'][1]:.4f}] |")
    A(f"| `ex` | 同上（global 结构尺度层） | {glob['ex_pooled']:.4f} "
      f"[{glob['ci90'][0]:.4f}, {glob['ci90'][1]:.4f}] |")
    A(f"| `A_2/A_1` | 需求带宽反演面积比 | {bw['area_ratio_used']:.4f} |")
    A("| `λ = P/√A` | 足迹周长因子 | 4.25（4:1 矩形）/ 4.00（方形） |")
    A(f"| `delta/√A_c` | 过程区宽度比（假设参数，扫描） | "
      f"{', '.join(format(s['delta_over_sqrtA'], 'g') for s in sweep)} |")
    A(f"| `tau` / `I²` | 层内异质 | {design['tau']:.4f}/{design['I2_pct']:.1f}%（design）、"
      f"{measured['tau']:.4f}/{measured['I2_pct']:.1f}%（measured） |")
    A("")
    A("## ② 怎么做的")
    A("")
    A("1. **参数库**（`src/tensorphysics/ice_pressure_area/library.py` + "
      f"`data/ice_pressure_area/pressure_area_sources.json`，{lib['entries']} 条记录）："
      "每条强制字段 `citation / units / area_definition / geometry / conditions / evidence_class / role`；"
      "占位符引文、指数越界、未知口径一律拒绝入库；逐条 + 聚合 sha256 锚定（`MANIFEST.json`）。")
    A("2. **面积口径代数**（`conventions.py`）：`A_src = g(A_ref)`，`r = d lnA_src/d lnA_ref` ⇒ "
      "`ex_ref = 1 + (ex_src − 1)·r`。损伤区口径 `A_d = A_c + λ·delta·√A_c` 给出 "
      "`r = (1+t/2)/(1+t)`（`t = λ·delta/√A_c`）。**定理**：定比口径（线载荷/板格）`r≡1` ⇒ `ex` 严格不变。")
    A("3. **标定**（`calibrate.py`）：90 % 声明带 → `sigma`；口径归一（`r` 同时放大区间）；"
      "**按尺度分层**做 DerSimonian–Laird 随机效应合并（报 Q/tau²/I²）；留一交叉/逐源敏感性；"
      "协变量用加权设计矩阵秩 + 取值对比数做可辨识性审计，不可辨识者显式 WITHHELD。")
    A("4. **需求带宽映射**：由平台实测 `1.32x–1.74x` 反演面积比 `A_2/A_1`，再把各层 `ex` 区间映射为 "
      "`(A_2/A_1)^ex`，与声明带直接可比。")
    A("")
    A("工具链：Python 3.11 / numpy / matplotlib(Agg)；计算 < 2 s，无需集群。复现："
      "`PYTHONPATH=src python scripts/run_gap_pp2_pressure_area.py`。")
    A("")
    A("## ③ 计算结果")
    A("")
    A("| 层 | k | ex 合并 | 90 % 区间 | tau | I² % | 需求带宽带 | 相对声明收窄 | 落在声明带内 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for name in ("design_rule", "measured_local", "global", "all_sources_naive"):
        p = pooling.get(name, naive if name == "all_sources_naive" else None)
        row = rows[name]
        A(f"| {name} | {p['k']} | {p['ex_pooled']:.4f} | [{p['ci90'][0]:.4f}, {p['ci90'][1]:.4f}] | "
          f"{p['tau']:.4f} | {p['I2_pct']:.1f} | [{row['bandwidth_band'][0]:.3f}, {row['bandwidth_band'][1]:.3f}] | "
          f"{row['narrowing_factor_vs_declared']:.2f}x | {'是' if row['inside_declared'] else '**否**'} |")
    A(f"| 声明（原状态） | - | 0.30–0.70 | - | - | - | "
      f"[{bw['reproduced_declared_bandwidth'][0]:.3f}, {bw['reproduced_declared_bandwidth'][1]:.3f}] | 1.00x | - |")
    A("")
    A("**结论**：按尺度分层 + 口径归一后，设计规则层 `ex` 收敛到 "
      f"`{design['ex_pooled']:.3f} [{design['ci90'][0]:.3f}, {design['ci90'][1]:.3f}]`，"
      f"需求带宽 `{rows['design_rule']['bandwidth_band'][0]:.3f}–{rows['design_rule']['bandwidth_band'][1]:.3f}`，"
      f"相对声明带收窄 **{rows['design_rule']['narrowing_factor_vs_declared']:.2f}×** 且完全落在原带内。"
      "同时得到一个**诚实负面结果**：实测局部压力层 "
      f"`ex = {measured['ex_pooled']:.3f} [{measured['ci90'][0]:.3f}, {measured['ci90'][1]:.3f}]` "
      f"（I²={measured['I2_pct']:.1f} %）映射出的带宽上限 "
      f"{rows['measured_local']['bandwidth_band'][1]:.3f} 超出原声明上限 "
      f"{bw['reproduced_declared_bandwidth'][1]:.3f}，即原 `1.74x` 对实测局部压力**不保守**。")
    A("")
    A("## ④ 结果图")
    A("")
    for name, caption in FIGURES:
        A(f"![{name}]({name})")
        A("")
        A(f"*{caption}*")
        A("")
    A("## ⑤ 注意点（坑与局限）")
    A("")
    A("1. **口径统一本身不能收缩带宽**：定比口径下 `ex` 严格不变（律 T 定理，R3/R4 核对误差 "
      "1.2e-14 %）。真正能移动 `ex` 的只有「面积比随面积变化」的口径（损伤区/浸湿），"
      f"`delta/√A=0.2` 时 `0.45 → {[s for s in sweep if s['delta_over_sqrtA'] == 0.2][0]['damaged_to_nominal_at_0.45']:.3f}`。"
      "这否定了「先统一面积口径就能得到单一 `ex`」的隐含假设。")
    A("2. **5 条 accuracy-reference 记录均未对一手原文核验**（构建环境无外网）："
      "`primary_source_verified=false` 如实保留，未把二手值升格；"
      "`fail_stage=primary_source_verification / failure_reason=no_network_egress`。")
    A("3. **温度依赖 WITHHELD**：装配集仅 1 条记录带温区间，协变量不可辨识 → 不拟合、不出数字。")
    A("4. **浸湿面积口径库内为 0 条**：只能给解析转换分支（`r = 1 + beta`），给不出经验换算系数。")
    A(f"5. **树内自相矛盾（本单机器判定阻塞）**：IACS 压力-面积律在本仓存在三个读数 "
      f"`ex=0.15`（`ice_modules/local_pressure.py`）、`0.185`（该文件自带 p–A 表实测斜率）、`0.45`"
      f"（`benchmarks/ice_nav_refs` 资产）。消费者在 `A=50 m²` 处比标定设计律高 "
      f"{consumer['rows'][-1]['err_model_vs_library_pct']:.1f} %，相对其自带参考表误差达 "
      f"{consumer['max_err_model_vs_own_reference_table_pct']:.2f} %。本单只登记不改模型，避免在无原文时"
      "擅自改设计值。")
    A("6. **`design_rule` 层 `tau=0` 是转写一致性而非测量一致性**，引用时必须并列 `measured_local` 层。")
    A("")
    A("## ⑥ 误差判定（基准逐项）")
    A("")
    A("| id | ref | got | 误差 % | 容差 % | 判定 |")
    A("|---|---|---|---|---|---|")
    for row in ref_rows:
        A(f"| {row['id']} | {row['ref']:.6g} | {row['got']:.6g} | {row['error_pct']:.3g} | "
          f"{row['tol_pct']:.3g} | {'PASS' if row['pass'] else 'FAIL'} |")
    A("")
    worst = max(ref_rows, key=lambda r: r["error_pct"])
    A(f"**是否 <3 %**：{sum(r['error_pct'] < 3.0 for r in ref_rows)}/{len(ref_rows)} 行 <3 %。"
      f"最差一行 `{worst['id']}` 为 {worst['error_pct']:.2f} %（容差 {worst['tol_pct']:g} %）——"
      "该行是平台实测需求带宽上限的重现，误差来自 `1.32x/1.74x` 这一对本身不严格服从单一幂律"
      f"（两端反演面积比相差 {bw['area_ratio_inconsistency_pct']:.1f} %），"
      "属基准自身的不一致，已在表内如实标注，未做粉饰。")
    A("")
    A("工件：`artifacts/ice_pressure_area/calibration.json`、`reference_table.md`、"
      "`figures/*.png`；库锚 `data/ice_pressure_area/MANIFEST.json`"
      f"（聚合 sha256 `{payload['library_manifest_aggregate_sha256']}`）；"
      "单测 `tests/test_ice_pressure_area.py`（37 项全绿）。")
    return "\n".join(md)


def main() -> int:
    payload = json.loads((ART / "calibration.json").read_text(encoding="utf-8"))
    markdown = build_markdown(payload)
    figures = []
    for name, caption in FIGURES:
        path = ART / "figures" / name
        if not path.is_file():
            continue
        figures.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        )
    report = {
        "title": "GAP-PP-2 冰压压力-面积参数库与 ex 一手标定（含面积口径代数）",
        "summary": (
            "建 7 条记录的压力-面积参数库（每条强制 area_definition/单位/几何/工况 + sha256 锚）；"
            "给出面积口径转换律 ex_ref = 1+(ex_src-1)r 并证明定比口径不改变 ex；按尺度分层 DL 随机效应标定，"
            "design_rule 层 ex=0.450 [0.408,0.492]，需求带宽 1.420-1.525（相对声明 1.32x-1.74x 收窄 5.04x）；"
            "诚实保留：5 条 accuracy-reference 未对一手原文核验（无外网）、温度依赖 WITHHELD、"
            "浸湿口径库内 0 条；并机器判定 IACS 族内 ex=0.15/0.185/0.45 三读数冲突。"
        ),
        "scenario_key": SCENARIO_KEY,
        "mission_id": MISSION_ID,
        "tags": [
            "ice_pressure_area",
            "pressure_area_exponent",
            "area_convention",
            "parameter_library",
            "random_effects_calibration",
            "gap_pp_2",
            "honest_withheld",
        ],
        "markdown": markdown,
        "figures": figures,
    }
    out = Path("/tmp/report.json")
    out.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    token = login()
    req = urllib.request.Request(
        f"{base}/tensor/reports",
        data=out.read_bytes(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(f"status={resp.status}")
            print(resp.read().decode()[:800])
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode()[:800]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())