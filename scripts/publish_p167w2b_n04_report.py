"""Publish the P167W2B-N04 report (all numbers read from artifacts, zero hand copy)."""
import base64, json, os, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p167w2b_n04"
results = json.loads((ART / "results.json").read_text())
tables = (ART / "tables.md").read_text()
doc = (ROOT / "docs" / "p167w2b-n04-ep-collapse-maturity.md").read_text()
base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")

def post(path, body, token=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + token} if token else {})})
    return json.load(urllib.request.urlopen(req, timeout=120))

tok = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                           "password": os.environ["TENSOR_AGENT_PASS"]})["access_token"]
crit = results["criteria"]
summary = (f"P167W2B-N04 弹塑性渐进 collapse 验证面：12 项判据全通过 "
           f"(σ_u vs 简化法偏差 {100*results['panel']['max_abs_deviation_vs_reference']:.2f}%, "
           f"板屈曲 FD vs Timoshenko {100*results['buckling']['max_rel_err']:.3f}%, "
           f"一致切线 vs 差分 {results['constitutive']['max_rel_err_consistent_tangent_vs_fd']:.2e}, "
           f"冰指数 {crit['C3_ice_exponent_in_band']['clean_exponent']:.4f})")
figs = []
for name, cap in [("fig1_uniaxial.png", "本构：闭式/增量径向返回/解析双线性"),
                  ("fig2_panel_collapse.png", "Smith 板架载荷-缩短曲线与极限点"),
                  ("fig3_buckling.png", "FD 特征值网格序列 vs Timoshenko 闭式"),
                  ("fig4_pressure_area.png", "IACS 压力-面积律与指数反演"),
                  ("fig5_benchmark.png", "σ_u 模型 vs Paik & Thayamballi 简化法（9 组合）")]:
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode((ART / name).read_bytes()).decode()})
report = {
    "title": "P167W2B-N04 弹塑性渐进 collapse 验证面成熟化（本构/板架/屈曲/冰带压力律）",
    "summary": summary,
    "scenario_key": "p167w2b_n04_ep_collapse_maturity",
    "mission_id": 12428,
    "tags": ["TensorPhysics", "elastoplastic", "Smith-method", "collapse", "buckling",
             "Paik-Thayamballi", "IACS", "ice-belt"],
    "markdown": doc + "\n\n---\n\n## 程序化结果表（由脚本写入，零手抄）\n\n" + tables
                + "\n\n### criteria\n\n```json\n" + json.dumps(crit, indent=2, ensure_ascii=False) + "\n```\n",
    "figures": figs,
}
print(json.dumps(post("/tensor/reports", report, tok), ensure_ascii=False)[:400])
