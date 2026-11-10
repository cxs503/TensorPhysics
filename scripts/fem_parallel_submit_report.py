"""Submit the FEM parallel-decomposition report to the platform report service.

Builds ``/tmp/report.json`` from ``artifacts/fem_parallel/report.json`` (numbers)
and ``artifacts/fem_parallel/figures/*.png`` (base64), then POSTs it to
``$TENSOR_API_BASE/tensor/reports`` with a fresh agent token.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts", "fem_parallel")

CAPTIONS = {
    "strong_scaling.png": "强扩展：S(P)、E(P)（仅求解 / 组装+setup+求解两种口径，含验收线 E=0.6）",
    "weak_scaling.png": "弱扩展：T(P)/T(1) 随 P 变化，理想值 1，验收线 1.5",
    "communication.png": "通信开销：T_comm/T_solve 随 P 变化，及 T_solve、T_comm 绝对值",
    "iterations.png": "CG / PCG-Jacobi / PCG-ASM 迭代数对比",
    "accuracy.png": "Q1 制造解相对 L2 误差随 h 收敛（参考斜率 h²）",
    "load_balance.png": "各秩本地组装/本地矩阵向量积耗时（负载均衡判据数据源）",
    "fields.png": "并行 FE 解场、解析解场与逐点误差云图（制造解算例）",
}


def main() -> int:
    with open(os.path.join(ART, "report.json")) as fh:
        rep = json.load(fh)
    doc = os.path.join(HERE, rep["doc"])
    with open(doc) as fh:
        markdown = fh.read()

    figures = []
    for name in rep.get("figures", []):
        path = os.path.join(ART, "figures", name)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            figures.append({
                "name": name,
                "caption": CAPTIONS.get(name, name),
                "png_base64": base64.b64encode(fh.read()).decode(),
            })

    checks = rep["checks"]
    verdict = ", ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in checks.items())
    payload = {
        "title": "TensorPhysics: MPI 域分解并行 FEM 基础设施与强/弱扩展验证 (p284w3-n04)",
        "summary": (
            f"TensorPhysics 新增 fem_parallel 模块（MPI 语义传输层 + owned-row 并行组装 + "
            f"halo 交换 + 并行 PCG）；N={rep['n_dofs_strong']:,} DOF 强/弱扩展与通信开销量化；"
            f"判据：{verdict}；overall={'PASS' if rep['all_pass'] else 'NOT ALL PASS'}"
        ),
        "scenario_key": "p284w3-n04-fem-mpi-parallel",
        "mission_id": 9417,
        "tags": ["FEM", "MPI", "domain-decomposition", "PCG", "strong-scaling",
                 "weak-scaling", "communication-overhead", "TensorPhysics"],
        "markdown": markdown,
        "figures": figures,
    }
    out = "/tmp/report.json"
    with open(out, "w") as fh:
        json.dump(payload, fh)
    print("wrote", out, "with", len(figures), "figures")

    base = os.environ.get("TENSOR_API_BASE")
    user = os.environ.get("TENSOR_AGENT_USER")
    pwd = os.environ.get("TENSOR_AGENT_PASS")
    if not base:
        print("TENSOR_API_BASE not set; report JSON prepared only")
        return 0
    token = os.environ.get("TOKEN")
    if not token and user and pwd:
        res = subprocess.run(
            ["curl", "-s", "-X", "POST", f"{base}/auth/login",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"username": user, "password": pwd})],
            capture_output=True, text=True,
        )
        try:
            token = json.loads(res.stdout).get("access_token") or json.loads(res.stdout).get("token")
        except Exception:
            print("login failed:", res.stdout[:300])
    if not token:
        print("no token; report JSON prepared only")
        return 0
    res = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{base}/tensor/reports",
         "-H", f"Authorization: Bearer {token}",
         "-H", "Content-Type: application/json",
         "-d", f"@{out}"],
        capture_output=True, text=True,
    )
    print(res.stdout[:500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
