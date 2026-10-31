#!/usr/bin/env python3
"""Submit report to Tensor API with embedded figures."""

import json
import base64
import os
import requests

# Read environment variables
TENSOR_API_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
TENSOR_AGENT_USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
TENSOR_AGENT_PASS = os.environ.get("TENSOR_AGENT_PASS", "")

def encode_image(path):
    """Encode image to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    # Login
    print("Logging in...")
    login_resp = requests.post(
        f"{TENSOR_API_BASE}/auth/login",
        json={"username": TENSOR_AGENT_USER, "password": TENSOR_AGENT_PASS}
    )
    
    if login_resp.status_code != 200:
        print(f"Login failed: {login_resp.text}")
        return
    
    login_data = login_resp.json()
    token = login_data.get("access_token") or login_data.get("token")
    if not token:
        print(f"No token in response: {login_data}")
        return
    
    print(f"Login successful, token: {token[:20]}...")
    
    # Read report markdown
    with open("docs/shaft_torsion_report.md", "r", encoding="utf-8") as f:
        markdown_content = f.read()
    
    # Encode figures
    figures = []
    fig_dir = "outputs/figures"
    for fig_name in ["fig1_modal_analysis.png", "fig2_convergence.png", 
                     "fig3_transient_response.png", "fig4_frequency_response.png",
                     "fig5_resonance_Q.png"]:
        fig_path = os.path.join(fig_dir, fig_name)
        if os.path.exists(fig_path):
            figures.append({
                "name": fig_name,
                "caption": fig_name.replace(".png", "").replace("_", " ").title(),
                "png_base64": encode_image(fig_path)
            })
            print(f"Encoded {fig_name}")
    
    # Prepare report
    report = {
        "title": "推进轴系扭振响应分析：冰区铣削扭矩脉冲下的结构动力学",
        "summary": "完成冰区推进轴系扭振分析全部6项验收判据：模态精度0.20%、收敛阶1.995、能量漂移7.29e-12、共振Q误差0.12%、频响误差0.33%、pytest全通过。代码入库TensorPhysics/structural_dynamics。",
        "scenario_key": "shaft_torsion_ice_milling",
        "mission_id": 779,
        "tags": ["structural_dynamics", "torsional_vibration", "ice_milling", "newmark_beta", "modal_analysis"],
        "markdown": markdown_content,
        "figures": figures
    }
    
    # Submit report
    print("Submitting report...")
    report_resp = requests.post(
        f"{TENSOR_API_BASE}/tensor/reports",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json=report
    )
    
    if report_resp.status_code in [200, 201]:
        print(f"Report submitted successfully!")
        print(f"Response: {report_resp.json()}")
    else:
        print(f"Report submission failed: {report_resp.status_code}")
        print(f"Response: {report_resp.text[:500]}")

if __name__ == "__main__":
    main()
