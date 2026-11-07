#!/usr/bin/env python3
"""Publish the IN16-K8 report to the TensorPlatform report API.

Reads credentials from the environment (``TENSOR_API_BASE``, ``TENSOR_AGENT_USER``,
``TENSOR_AGENT_PASS``); nothing is written into the payload or the repository.
Figures are downscaled and palette-quantised so the whole multipart JSON stays
well below the 250 kB payload limit of ``POST /tensor/reports``.

Usage:  python3 scripts/publish_m5044_report.py [--dry-run]
"""
from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import sys

import requests
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "m5044" / "REPORT.md"
ART = ROOT / "artifacts" / "m5044" / "clock_semantics.json"
MISSION_ID = 5044
NODE = "IN16-K8"

CAPTIONS = {
    "convergence": "图 1 交换间隔 h 与耦合解相对误差（log-log）。虚线=声明误差带 C*omega_max*h（C=1），点线=1 阶参考斜率；实测阶 1.000。",
    "interpolation": "图 2 交换区间内部重构误差（h=12 ms）：因果 ZOH 一阶，非因果线性插值二阶；阴影=声明带宽。",
    "coupled": "图 3 两自由度耦合响应（比率 4/3、h=12 ms）：解析解 vs 多速率解，下图为误差轨迹与声明带宽（无累积漂移）。",
    "causality": "图 4 时钟格：主步/子步/LCM 交换时刻；红叉为交换后才产生的子步样点，被因果门拒绝（CausalityViolation）。",
}


def encode_figure(path: pathlib.Path, width: int = 720) -> str:
    """Downscale + palette-quantise a PNG and return base64 (keeps payload small)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.width > width:
            ratio = width / float(img.width)
            img = img.resize((width, max(1, int(round(img.height * ratio)))), Image.LANCZOS)
        img = img.convert("P", palette=Image.ADAPTIVE, colors=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    data = json.loads(ART.read_text())
    crit = data["criterion"]
    markdown = REPORT.read_text()

    figures = []
    for name, rel in data["figures"].items():
        path = ROOT / rel
        b64 = encode_figure(path)
        figures.append(
            {
                "name": f"{name}.png",
                "caption": CAPTIONS.get(name, name),
                "png_base64": b64,
            }
        )
    payload = {
        "mission_id": MISSION_ID,
        "title": f"{NODE}: 耦合时钟语义实现级成熟化（有理比率 + GCD 节拍 + 因果门，M12）",
        "summary": (
            "将 M12 时钟语义从文档级落到 src：任意有理比率的 GCD 原子格时钟驱动器、"
            "LCM 交换节拍（机器验证为最粗公共周期）、未来数据不可用的因果负例；"
            f"多速率两积分器对拍对标解析模态解，实测收敛阶 {crit['multirate_order']:.4f}，"
            f"声明误差带带宽比最坏 {crit['band_worst_ratio_refinement']:.4f}（C=1），"
            f"pytest {crit['pytest_cases']} 用例全通过；v1 契约与指纹 pin 未改动。"
        ),
        "scenario_key": "tensorphysics.in16_k8.coupling_clock_rational",
        "tags": [
            "TensorPhysics",
            "M12",
            "coupling-clock",
            "multirate",
            "gcd-cadence",
            "causality",
            "IN16-K8",
        ],
        "markdown": markdown,
        "figures": figures,
    }

    body = json.dumps(payload)
    print(f"payload bytes: {len(body.encode('utf-8'))} (figures: {len(figures)})")
    if dry_run:
        pathlib.Path("/tmp/report_m5044.json").write_text(body)
        print("dry run: wrote /tmp/report_m5044.json")
        return 0

    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    login = requests.post(
        f"{base}/auth/login", json={"username": user, "password": password}, timeout=30
    )
    login.raise_for_status()
    token = login.json().get("access_token") or login.json().get("token")
    if not token:
        print("login response without token:", json.dumps(login.json())[:300])
        return 2
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(f"{base}/tensor/reports", headers=headers, data=body, timeout=120)
    print("POST /tensor/reports ->", resp.status_code)
    print(resp.text[:400])
    if resp.status_code not in (200, 201):
        return 1
    check = requests.get(
        f"{base}/tensor/reports", headers=headers, params={"mission_id": MISSION_ID}, timeout=60
    )
    print("verify GET ->", check.status_code)
    try:
        rows = check.json()
        items = rows.get("items", rows) if isinstance(rows, dict) else rows
        print("reports for mission:", len(items))
    except Exception:  # pragma: no cover - reporting only
        print(check.text[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())