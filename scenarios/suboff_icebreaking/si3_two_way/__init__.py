"""SI-3: LBM×FEM 双向耦合模块。

本模块实现 SUBOFF 破冰场景的第三阶：双向流固耦合。
移植自 wedge S4 范式，包含：
- immersed_boundary.py: 浸入边界模块（结构→流体反馈）
- conservation_ledger.py: 守恒账本（动量/能量双账本）
- coupling_loop.py: 双向耦合主循环
"""

from .immersed_boundary import (
    compute_immersed_boundary_from_plate_deflection,
    apply_immersed_boundary_to_lbm_snapshot,
    save_immersed_boundary_data,
    load_immersed_boundary_data,
)

from .conservation_ledger import (
    compute_momentum_balance,
    compute_energy_balance,
    compute_conservation_ledger,
    save_conservation_ledger,
    load_conservation_ledger,
    aggregate_conservation_ledgers,
)

from .coupling_loop import run_two_way_coupling

__all__ = [
    # immersed_boundary
    "compute_immersed_boundary_from_plate_deflection",
    "apply_immersed_boundary_to_lbm_snapshot",
    "save_immersed_boundary_data",
    "load_immersed_boundary_data",
    # conservation_ledger
    "compute_momentum_balance",
    "compute_energy_balance",
    "compute_conservation_ledger",
    "save_conservation_ledger",
    "load_conservation_ledger",
    "aggregate_conservation_ledgers",
    # coupling_loop
    "run_two_way_coupling",
]
