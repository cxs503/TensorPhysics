"""SIR-3: G3 重力/浮力/自由面入链 —— 薄编排场景

SIR-3 目标：让链路里真实存在重力、浮力、自由面三件事
- 重力开关在运行工件里为开（对照 SI-4 的 gravity=[0,0,0]）
- 碎冰/冰板上的净浮力项非零且对上阿基米德解析值
- 自由面在场并有质量账本闭环

物理锚：
- A3.1 阿基米德浮力：TensorFEM ice_sheet.py:145 buoyancy_response
- A3.2 自由面质量守恒：TensorLBM free_surface_lbm.py + ownership_ledger
- A3.3 静水压力梯度：解析解 p = ρ_w g h

机器门：
- M3.1 gravity_enabled = true
- M3.2 buoyancy_net_force_N >= 1e-6
- M3.3 n_interface_cells > 0
- M3.4 质量双账本 |Δm|/m_0 <= 1e-12
- M3.5 浮力对阿基米德解析解相对误差 < 1%

规模判据：
- 自由面演化 >= 500 步
- 域高向 >= 32 网格
- 浸没体积积分点数 >= 1e3

薄编排原则：零求解器代码，全部调用兄弟库接口
"""

__version__ = "1.0.0"
__all__ = ["run_sir3_validation"]
