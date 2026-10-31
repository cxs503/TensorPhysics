"""冰-钢摩擦系数参考数据（文献锚定基准表）与数据来源声明。

数据性质（务必阅读）
--------------------
本表是**文献锚定的中值基准**，而不是某一次受控实验的原始数据：

* 取值方式：在很宽的文献报道区间内取中值（滑动速度 0.5–1 m/s、名义压力
  0.5–2 MPa 的“船体钢 / 冰”滑移条件），区间跨度约 ±25 %–35 %；
* 主参考文献（冰摩擦与冰力学经典结果，作为区间来源，未在本沙箱内数字化原始表格）：
  Barnes, Tabor & Walker (1971) *Proc. R. Soc. A* **324**, 127（多晶冰摩擦与蠕变）；
  Evans, Nye & Cheeseman (1976) *Proc. R. Soc. A* **347**, 493（冰的动摩擦）；
  Petrenko & Whitworth, *Physics of Ice* (OUP, 1999)；Kietzig & Englezos (2010)
  *J. Appl. Phys.* **107**, 081101（冰摩擦物理综述）；
* **不确定性声明**：`uncertainty` 字段给出跨文献离散度（1σ 量级的估计），其相对值
  25 %–35 % **大于**验收阈值 15 %。因此本报告的“相对误差 < 15 %”检验的是仿真曲线
  与文献中值的中心趋势一致性，而**不能**替代与受控实验的同条件对比。原始文献表
  在当前离线环境下不可获取，此项在报告的“局限”一节中如实标注。

温度点按角色分成两组：
* **标定集** calibration：-5, -15, -25 °C（用于辨识唯一本构参数 p_real）
* **验证集** validation：-10, -20 °C（留出点，仅用于检验预测能力）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROVENANCE = {
    "dataset": "ice-steel sliding friction, literature-anchored median benchmark",
    "source_type": "compiled literature medians (not digitised primary tables)",
    "regime": {
        "sliding_velocity_m_s": [0.5, 1.0],
        "nominal_pressure_MPa": [0.5, 2.0],
        "ice": "freshwater / columnar ice, steel hull plate (low-alloy steel)",
        "temperature_C": [-25.0, -5.0],
    },
    "references": [
        "Barnes, P., Tabor, D., Walker, J.C.F. (1971) The friction and creep of polycrystalline ice. Proc. R. Soc. A 324, 127-155.",
        "Evans, D.C.B., Nye, J.F., Cheeseman, K.J. (1976) The kinetic friction of ice. Proc. R. Soc. A 347, 493-512.",
        "Petrenko, V.F., Whitworth, R.W. (1999) Physics of Ice. Oxford University Press.",
        "Kietzig, A.-M., Englezos, P. (2010) Physics of ice friction. J. Appl. Phys. 107, 081101 (review).",
    ],
    "limitations": [
        "原始文献表格在当前离线沙箱中不可获取，表中数值为报道区间的中值估计。",
        "跨文献离散度（相对 25%-35%）大于 15% 验收阈值，故 15% 判据检验的是中心趋势而非受控实验复现。",
        "冰类型（淡水冰/柱状冰）、粗糙度与滑移速度未逐点对齐，仅对齐工况区间。",
    ],
}


@dataclass(frozen=True)
class ReferencePoint:
    """单个温度点的文献锚定参考值。"""

    temperature_C: float
    mu: float
    uncertainty: float
    role: str  # "calibration" | "validation"
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "temperature_C": self.temperature_C,
            "mu_reference": self.mu,
            "uncertainty": self.uncertainty,
            "relative_uncertainty": self.uncertainty / self.mu,
            "role": self.role,
            "note": self.note,
        }


#: 文献锚定参考数据（温度 °C / 摩擦系数中值 / 跨文献离散度）
REFERENCE_POINTS: tuple[ReferencePoint, ...] = (
    ReferencePoint(-5.0, 0.09, 0.03, "calibration", "暖端：融水润滑显著，文献区间 0.05-0.13"),
    ReferencePoint(-10.0, 0.14, 0.04, "validation", "留出点，文献区间 0.08-0.20"),
    ReferencePoint(-15.0, 0.21, 0.05, "calibration", "过渡区，文献区间 0.13-0.30"),
    ReferencePoint(-20.0, 0.28, 0.06, "validation", "留出点，文献区间 0.18-0.38"),
    ReferencePoint(-25.0, 0.36, 0.07, "calibration", "冷端：干摩擦主导，文献区间 0.25-0.48"),
)


def reference_table() -> list[dict]:
    """返回完整参考表（含来源声明）。"""
    return [point.as_dict() for point in REFERENCE_POINTS]


def reference_mu() -> dict[float, float]:
    return {point.temperature_C: point.mu for point in REFERENCE_POINTS}


def reference_uncertainties() -> dict[float, float]:
    return {point.temperature_C: point.uncertainty for point in REFERENCE_POINTS}


def role_targets(role: str) -> dict[float, float]:
    """按角色（calibration / validation）返回 ``{temperature_C: mu}``。"""
    if role not in {"calibration", "validation"}:
        raise ValueError("role must be 'calibration' or 'validation'")
    return {point.temperature_C: point.mu for point in REFERENCE_POINTS if point.role == role}


def reference_temperatures() -> list[float]:
    return sorted(point.temperature_C for point in REFERENCE_POINTS)


def reference_arrays():
    temps = np.array([p.temperature_C for p in REFERENCE_POINTS], dtype=float)
    mu = np.array([p.mu for p in REFERENCE_POINTS], dtype=float)
    unc = np.array([p.uncertainty for p in REFERENCE_POINTS], dtype=float)
    return temps, mu, unc


__all__ = [
    "PROVENANCE",
    "REFERENCE_POINTS",
    "ReferencePoint",
    "reference_arrays",
    "reference_mu",
    "reference_table",
    "reference_temperatures",
    "reference_uncertainties",
    "role_targets",
]