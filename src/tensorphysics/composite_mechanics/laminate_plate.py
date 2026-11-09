# -*- coding: utf-8 -*-
"""层合复合材料船体板的 CLT / FSDT 弯曲与屈曲验证面。

物理背景
--------
船体板（甲板板、舷侧板、上层建筑、冰带加强板）越来越多采用纤维增强层合板。
层合板与均质各向同性板的本质差别有两点：

1. **面内刚度与弯曲刚度解耦**（``A/B/D`` 三块矩阵，``B`` = 拉-弯耦合）；
2. **弯曲-扭转耦合**：铺层非正交各向（angle-ply）时 ``D16, D26 != 0``，弯曲刚度
   张量出现 ``w,xx * w,xy`` 型交叉项，破坏 Navier 双三角级数的模态解耦，矩形板
   因此出现二维边界层与「弯矩-扭矩」耦合变形。

本模块提供三个**互相独立**的裁决面（三重裁决）与负控（退化极限）：

``navier_clt_deflection``
    CLT（Kirchhoff）四边简支矩形板 Navier 双三角级数**精确解**（闭式逐模态）。
    适用：对称正交各向层合板（``D16 = D26 = 0``，如 ``[0/90]s``、``[0/90]2s``）。

``levy_deflection``
    CLT Levy 单三角级数解：``y = 0, b`` 简支，``x = 0, a`` 可为简支(SS)或固支(CC)。
    沿 ``x`` 解四阶常微分方程（每阶 n 一个 4x4 线性系统）。SS-SS 必须与 Navier
    逐位一致（实现自检），CC-CC 给出与 Navier 不同的、可独立交叉验证的解。

``ritz_deflection`` / ``ritz_clamped_x_deflection``
    一般对称层合板（含 ``D16, D26 != 0``）的 Rayleigh-Ritz 双三角级数能量解，
    以及 ``x`` 边固支时用「固支-固支梁函数 x 正弦」基 + Gauss 数值积分的独立
    能量解（用于交叉验证 Levy 的 CC 分支）。

``fsdt_navier_deflection``
    一阶剪切变形理论（Reissner-Mindlin，``k = 5/6``）Navier 精确解，逐模态求解
    ``3x3`` 系统（``W, Phi_x, Phi_y``）。``h/a -> 0`` 时收敛到 CLT，误差
    ``O((h/a)^2)``；``h/a > 1/10`` 时剪切修正不可忽略。

屈曲
----
``buckling_uniaxial`` / ``buckling_biaxial``
    正交各向对称层合板闭式解（任务书闭式的量纲正确形式）。
``buckling_eigen_uniaxial`` / ``buckling_eigen_shear``
    一般各向异性层合板（含 ``D16, D26``）广义特征值屈曲：``K w = N G w``；
    纯剪用广义 pencil 取最小正实特征值。

负控（退化极限）
----------------
* 各向同性单层（``E1 = E2``、``G12 = E/(2(1+nu))``）-> 经典板理论：中心挠度系数
  ``0.00406``、单轴屈曲系数 ``k = 4.0``、纯剪屈曲系数 ``k_s ~ 9.34``；
* ``h/a -> 0`` 时 FSDT -> CLT（``O((h/a)^2)``）；
* 对称**正交各向**铺层（cross-ply）``D16 = D26 = 0``。

已知纠正（honest correction，详见报告）
---------------------------------------
任务书里「对称铺层 ``D16 = D26 = 0``」与「``[0/90]`` 非对称铺层产生
``D16, D26 != 0``」两句都不严格成立：

* 对称 ``[+-45]s`` 这类对称**角铺层**：``B = 0`` 但 ``D16 != 0``（弯曲-扭转耦合
  依然存在，这也是平衡对称铺层抗剪切屈曲能力强的原因之一）；
* ``[0/90]``（两层非对称 cross-ply）：``Qbar16 = Qbar26 = 0``，故 ``D16 = D26 = 0``，
  只有 ``B11, B22 != 0``（拉-弯耦合）。
* 正确表述：``D16 = D26 = 0``  <=>  铺层序列为正交各向（orthotropic）；
  ``B = 0``  <=>  铺层关于中面对称。本模块用 ``is_orthotropic()`` /
  ``is_symmetric()`` 两个独立判据分别记录这两件事。

单位约定
--------
长度 m、模量 Pa、刚度 ``A``: Pa·m、``B``: Pa·m^2、``D``: Pa·m^3；
面内力 ``N``: N/m（正值为压）；分布载荷 ``q0``: Pa。

参考文献（公式来源）
--------------------
* Reddy, J.N. (2004) *Mechanics of Laminated Composite Plates and Shells*, 2nd ed.,
  CRC Press —— CLT 的 ``Qbar`` 变换与 ``A/B/D``；FSDT 平衡方程与正交各向层合板
  Navier 解（本模块逐模态 3x3 系统即该书 FSDT 一节的对角化形式）。
* Whitney, J.M. (1987) *Structural Analysis of Laminated Anisotropic Plates* ——
  各向异性板 Ritz / 双三角级数解及其对 ``M_n = 0`` 自然边界的讨论。
* Timoshenko & Woinowsky-Krieger (1959) *Theory of Plates and Shells* ——
  各向同性板 ``0.00406`` 挠度系数、单轴屈曲 ``k = 4``、纯剪屈曲 ``k_s ~ 9.34``。
* Jones, R.M. (1999) *Mechanics of Composite Materials*, 2nd ed. —— ``A/B/D`` 与
  ``D* = D - B A^-1 B``（面内不动边界下的等效弯曲刚度）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Lamina",
    "Laminate",
    "BendingResult",
    "BucklingResult",
    "parse_layup",
    "validate_plate_geometry",
    "abd_matrices",
    "navier_clt_deflection",
    "levy_deflection",
    "ritz_deflection",
    "ritz_clamped_x_deflection",
    "fsdt_navier_deflection",
    "buckling_uniaxial",
    "buckling_biaxial",
    "buckling_eigen_uniaxial",
    "buckling_eigen_shear",
    "clamped_beam_functions",
    "isotropic_plate_rigidity",
    "T300_5208",
]


# ---------------------------------------------------------------------------
# 材料层
# ---------------------------------------------------------------------------
def _finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} 必须有限，得到 {x!r}")
    return v


@dataclass
class Lamina:
    """正交各向异性单层：面内 ``E1, E2, G12, nu12``，横向剪切 ``G13, G23``。

    ``G13`` / ``G23`` 缺省取 ``G12``（横向各向同性近似）。
    """

    E1: float
    E2: float
    G12: float
    nu12: float
    G13: Optional[float] = None
    G23: Optional[float] = None
    name: str = "lamina"

    def __post_init__(self) -> None:
        self.E1 = _finite(self.E1, "E1")
        self.E2 = _finite(self.E2, "E2")
        self.G12 = _finite(self.G12, "G12")
        self.nu12 = _finite(self.nu12, "nu12")
        for nm in ("E1", "E2", "G12"):
            if getattr(self, nm) <= 0.0:
                raise ValueError(f"{nm} 必须 > 0，得到 {getattr(self, nm)!r}")
        self.G13 = self.G12 if self.G13 is None else _finite(self.G13, "G13")
        self.G23 = self.G12 if self.G23 is None else _finite(self.G23, "G23")
        if self.G13 <= 0 or self.G23 <= 0:
            raise ValueError("G13 / G23 必须 > 0")
        if abs(self.nu12 * self.nu21) >= 1.0:
            raise ValueError(
                "单层刚度不正定：nu12*nu21 = {:.4f}（须 < 1），检查 E2/E1 与 nu12".format(
                    self.nu12 * self.nu21
                )
            )

    # -- 基本常数 ---------------------------------------------------------
    @property
    def nu21(self) -> float:
        return self.nu12 * self.E2 / self.E1

    @property
    def delta(self) -> float:
        return 1.0 - self.nu12 * self.nu21

    @property
    def Q11(self) -> float:
        return self.E1 / self.delta

    @property
    def Q22(self) -> float:
        return self.E2 / self.delta

    @property
    def Q12(self) -> float:
        return self.nu12 * self.E2 / self.delta

    @property
    def Q66(self) -> float:
        return self.G12

    @classmethod
    def isotropic(cls, E: float, nu: float, name: str = "isotropic") -> "Lamina":
        """各向同性等价单层（``E1 = E2 = E``，``G12 = G13 = G23 = E/(2(1+nu))``）。"""
        G = E / (2.0 * (1.0 + nu))
        return cls(E1=E, E2=E, G12=G, nu12=nu, G13=G, G23=G, name=name)

    def is_isotropic(self, rtol: float = 1e-9) -> bool:
        G = self.E1 / (2.0 * (1.0 + self.nu12))
        return (
            abs(self.E1 - self.E2) <= rtol * self.E1
            and abs(self.G12 - G) <= rtol * G
            and abs(self.nu12 - self.nu21) <= rtol
        )

    # -- 变换刚度 ---------------------------------------------------------
    def qbar(self, theta_deg: float) -> np.ndarray:
        """``theta`` 度铺层的折减刚度 ``Qbar``（3x3，序 ``[11, 22, 12]``）。"""
        th = math.radians(float(theta_deg))
        c, s = math.cos(th), math.sin(th)
        c2, s2 = c * c, s * s
        c4, s4 = c2 * c2, s2 * s2
        Q11, Q22, Q12, Q66 = self.Q11, self.Q22, self.Q12, self.Q66
        Qb11 = Q11 * c4 + 2.0 * (Q12 + 2.0 * Q66) * s2 * c2 + Q22 * s4
        Qb22 = Q11 * s4 + 2.0 * (Q12 + 2.0 * Q66) * s2 * c2 + Q22 * c4
        Qb12 = (Q11 + Q22 - 4.0 * Q66) * s2 * c2 + Q12 * (c4 + s4)
        Qb66 = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * s2 * c2 + Q66 * (c4 + s4)
        Qb16 = (Q11 - Q12 - 2.0 * Q66) * c2 * c * s - (Q22 - Q12 - 2.0 * Q66) * c * s2 * s
        Qb26 = (Q11 - Q12 - 2.0 * Q66) * c * s2 * s - (Q22 - Q12 - 2.0 * Q66) * c2 * c * s
        return np.array(
            [[Qb11, Qb12, Qb16], [Qb12, Qb22, Qb26], [Qb16, Qb26, Qb66]], dtype=float
        )

    def qbar_shear(self, theta_deg: float) -> Tuple[float, float, float]:
        """横向剪切变换刚度 ``(Qbar44, Qbar55, Qbar45)``。"""
        th = math.radians(float(theta_deg))
        c, s = math.cos(th), math.sin(th)
        Q44 = self.G13 * s * s + self.G23 * c * c
        Q55 = self.G13 * c * c + self.G23 * s * s
        Q45 = (self.G23 - self.G13) * s * c
        return Q44, Q55, Q45


def isotropic_plate_rigidity(E: float, nu: float, h: float) -> float:
    """均质各向同性板的弯曲刚度 ``D = E h^3 / (12 (1 - nu^2))`` [N·m]。"""
    return _finite(E, "E") * _finite(h, "h") ** 3 / (12.0 * (1.0 - _finite(nu, "nu") ** 2))


#: T300/5208 石墨-环氧典型单层参数（Reddy 2004 书中的标准算例材料）
def T300_5208() -> Lamina:
    """T300/5208 石墨-环氧：``E1 = 181 GPa``, ``E2 = 10.3 GPa``, ``G12 = 7.17 GPa``,
    ``nu12 = 0.28``（Reddy 教材常用值）。"""
    return Lamina(
        E1=181.0e9,
        E2=10.3e9,
        G12=7.17e9,
        nu12=0.28,
        G13=7.17e9,
        G23=7.17e9,
        name="T300/5208",
    )


# ---------------------------------------------------------------------------
# 铺层串解析
# ---------------------------------------------------------------------------
def parse_layup(spec: str, ply_thickness: float) -> List[Tuple[float, float]]:
    """把铺层串解析为 ``[(angle_deg, thickness), ...]``（自下而上）。

    支持 ``"0/90/0"``、``"[0/90]s"``、``"[0/90]2s"``、``"[±45]s"``、``"[45/-45]T"``。
    后缀 ``s``/``S`` 表示关于中面对称（镜像），方括号组尾部 ``2`` 表示重复 2 次。
    """
    t = _finite(ply_thickness, "ply_thickness")
    if t <= 0:
        raise ValueError("ply_thickness 必须 > 0")
    s = str(spec).strip().replace(" ", "")
    sym = False
    repeat = 1
    if s.endswith(("s", "S")):
        sym = True
        s = s[:-1]
    elif s.endswith(("t", "T")):
        s = s[:-1]
    m = re.match(r"^\[([^\]]+)\](\d)?$", s)
    if m:
        s = m.group(1)
        if m.group(2):
            repeat = int(m.group(2))
    tokens = [tok for tok in s.split("/") if tok]
    if not tokens:
        raise ValueError(f"无法解析铺层串 {spec!r}")
    angles: List[float] = []
    for tok in tokens:
        if tok[0] in "±" or tok.startswith("+-"):
            base = float(tok.replace("±", "").replace("+-", ""))
            angles.extend([base, -base])
        else:
            angles.append(float(tok))
    seq: List[float] = []
    for _ in range(repeat):
        seq.extend(angles)
    if sym:
        # 关于中面镜像：ply k 与 ply n+1-k 的铺层角相同（角度不取负）
        seq = seq + list(reversed(seq))
    return [(a, t) for a in seq]


def abd_matrices(
    plies: Sequence[Tuple[float, float]],
    lamina: Lamina,
    shear_correction: float = 5.0 / 6.0,
) -> Dict[str, object]:
    """由铺层序列积分得到 ``A/B/D`` 与横向剪切刚度 ``A44, A55, A45``。"""
    if not plies:
        raise ValueError("铺层序列为空")
    h = sum(t for _, t in plies)
    if h <= 0:
        raise ValueError("层合板总厚必须 > 0")
    z = [-0.5 * h]
    for _, t in plies:
        if t <= 0:
            raise ValueError(f"单层厚度必须 > 0，得到 {t!r}")
        z.append(z[-1] + t)
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    A44 = A55 = A45 = 0.0
    k = float(shear_correction)
    for (th, _t), z0, z1 in zip(plies, z[:-1], z[1:]):
        Qb = lamina.qbar(th)
        A += Qb * (z1 - z0)
        B += 0.5 * Qb * (z1 * z1 - z0 * z0)
        D += (Qb / 3.0) * (z1**3 - z0**3)
        q44, q55, q45 = lamina.qbar_shear(th)
        A44 += k * q44 * (z1 - z0)
        A55 += k * q55 * (z1 - z0)
        A45 += k * q45 * (z1 - z0)
    return {
        "A": A,
        "B": B,
        "D": D,
        "z": z,
        "thickness": h,
        "A44": A44,
        "A55": A55,
        "A45": A45,
    }


class Laminate:
    """层合板：铺层序列 + 单层材料 -> ``A/B/D``、横向剪切刚度、等效弯曲刚度。"""

    def __init__(
        self,
        plies: Sequence[Tuple[float, float]],
        lamina: Lamina,
        shear_correction: float = 5.0 / 6.0,
        name: str = "",
    ) -> None:
        self.plies = [(float(a), float(t)) for a, t in plies]
        self.lamina = lamina
        self.shear_correction = float(shear_correction)
        self.name = name or "laminate"
        res = abd_matrices(self.plies, lamina, self.shear_correction)
        self.A = res["A"]  # type: ignore[assignment]
        self.B = res["B"]  # type: ignore[assignment]
        self.D = res["D"]  # type: ignore[assignment]
        self.z = list(res["z"])  # type: ignore[arg-type]
        self.thickness = float(res["thickness"])  # type: ignore[arg-type]
        self.A44 = float(res["A44"])  # type: ignore[arg-type]
        self.A55 = float(res["A55"])  # type: ignore[arg-type]
        self.A45 = float(res["A45"])  # type: ignore[arg-type]

    @classmethod
    def from_spec(
        cls, spec: str, ply_thickness: float, lamina: Lamina, **kwargs
    ) -> "Laminate":
        kwargs.setdefault("name", spec)
        return cls(parse_layup(spec, ply_thickness), lamina, **kwargs)

    # -- 刚度分量 ---------------------------------------------------------
    @property
    def D11(self) -> float:
        return float(self.D[0, 0])

    @property
    def D22(self) -> float:
        return float(self.D[1, 1])

    @property
    def D12(self) -> float:
        return float(self.D[0, 1])

    @property
    def D66(self) -> float:
        return float(self.D[2, 2])

    @property
    def D16(self) -> float:
        return float(self.D[0, 2])

    @property
    def D26(self) -> float:
        return float(self.D[1, 2])

    @property
    def H(self) -> float:
        """扭转-弯曲组合刚度 ``H = D12 + 2 D66``。"""
        return self.D12 + 2.0 * self.D66

    def is_symmetric(self, rtol: float = 1e-6) -> bool:
        """``B = 0``（无拉-弯耦合）。"""
        scale = float(np.max(np.abs(self.A))) * self.thickness
        if scale <= 0:
            return True
        return float(np.max(np.abs(self.B))) <= rtol * scale

    def is_orthotropic(self, rtol: float = 1e-6) -> bool:
        """``D16 = D26 = 0``（弯曲-扭转解耦）。"""
        scale = max(abs(self.D11), abs(self.D22))
        if scale <= 0:
            return False
        return max(abs(self.D16), abs(self.D26)) <= rtol * scale

    def coupling_ratios(self) -> Dict[str, float]:
        return {
            "D16": self.D16,
            "D26": self.D26,
            "abs_D16_over_D11": abs(self.D16) / self.D11,
            "abs_D26_over_D22": abs(self.D26) / self.D22,
            "sign_D16": float(np.sign(self.D16)),
            "sign_D26": float(np.sign(self.D26)),
        }

    def d_star(self) -> np.ndarray:
        """面内不动（immovable edges）等效弯曲刚度 ``D* = D - B A^-1 B``。"""
        try:
            Ainv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError as exc:  # pragma: no cover
            raise ValueError(f"A 矩阵奇异，无法求 D*：{exc}") from exc
        return self.D - self.B @ Ainv @ self.B

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "thickness": self.thickness,
            "n_plies": len(self.plies),
            "angles": [a for a, _ in self.plies],
            "A11": float(self.A[0, 0]),
            "A22": float(self.A[1, 1]),
            "A66": float(self.A[2, 2]),
            "A16": float(self.A[0, 2]),
            "B_max": float(np.max(np.abs(self.B))),
            "D11": self.D11,
            "D22": self.D22,
            "D12": self.D12,
            "D66": self.D66,
            "D16": self.D16,
            "D26": self.D26,
            "A44": self.A44,
            "A55": self.A55,
            "is_symmetric": self.is_symmetric(),
            "is_orthotropic": self.is_orthotropic(),
        }


# ---------------------------------------------------------------------------
# 几何闸门（先验几何，再求解）
# ---------------------------------------------------------------------------
def validate_plate_geometry(a: float, b: float, h: float) -> Dict[str, object]:
    """矩形板几何闸门：尺寸正定、长宽比合理、厚跨比给出 CLT 适用性标志。"""
    a = _finite(a, "a")
    b = _finite(b, "b")
    h = _finite(h, "h")
    if a <= 0 or b <= 0 or h <= 0:
        raise ValueError(f"板几何必须为正：a={a}, b={b}, h={h}")
    aspect = a / b
    if not (0.02 <= aspect <= 50.0):
        raise ValueError(f"长宽比 a/b = {aspect:.3f} 超出 [0.02, 50] 可靠区间")
    h_over_a = h / a
    return {
        "a": a,
        "b": b,
        "h": h,
        "aspect_ratio": aspect,
        "h_over_a": h_over_a,
        "thin_plate_ok": h_over_a <= 0.1,
        "shear_dominated": h_over_a > 0.2,
    }


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
def _series_field(coeffs: np.ndarray, a: float, b: float, X, Y) -> np.ndarray:
    """在网格 ``(X, Y)`` 上求双三角级数场（矩阵乘法）。"""
    X = np.atleast_1d(np.asarray(X, dtype=float))
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    if coeffs.size <= 1:
        raise ValueError("该解答没有双三角级数系数，请使用 BendingResult.meta['field']")
    M, N = coeffs.shape[0] - 1, coeffs.shape[1] - 1
    m = np.arange(1, M + 1)
    n = np.arange(1, N + 1)
    Sx = np.sin(np.outer(m * math.pi / a, X))  # (M, nx)
    Sy = np.sin(np.outer(n * math.pi / b, Y))  # (N, ny)
    return Sx.T @ coeffs[1:, 1:] @ Sy


@dataclass
class BendingResult:
    """弯曲解答：中心挠度 + 级数系数（可复算场）+ 元数据。"""

    method: str
    a: float
    b: float
    q0: float
    w_max: float
    coeffs: np.ndarray
    x_max: float
    y_max: float
    meta: Dict[str, object] = field(default_factory=dict)

    def field(self, X, Y) -> np.ndarray:
        if self.coeffs.size <= 1 and "field" in self.meta:
            return self.meta["field"](X, Y)  # type: ignore[operator]
        return _series_field(self.coeffs, self.a, self.b, X, Y)

    def to_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "a": self.a,
            "b": self.b,
            "q0": self.q0,
            "w_max": self.w_max,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "meta": {k: v for k, v in self.meta.items() if k != "field"},
        }


@dataclass
class BucklingResult:
    """屈曲解答：临界面内载荷 + 模态 + 候选表。"""

    method: str
    load_type: str
    n_cr: float
    m: int
    n: int
    table: List[Dict[str, float]] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "load_type": self.load_type,
            "n_cr": self.n_cr,
            "mode": [self.m, self.n],
            "table": self.table,
            "meta": self.meta,
        }


def _require_symmetric(lam: Laminate, tol: float = 1e-6) -> None:
    if not lam.is_symmetric(tol):
        raise ValueError(
            "弯曲/屈曲解析面要求对称铺层（B = 0）；非对称铺层请用 Laminate.d_star() "
            "做面内不动等效，或改用全耦合求解（本模块未实现）"
        )


def _require_orthotropic(lam: Laminate, tol: float = 1e-6) -> None:
    if not lam.is_orthotropic(tol):
        raise ValueError(
            "该方法要求正交各向铺层（D16 = D26 = 0），本例 |D16|/D11 = {:.3e}；"
            "请改用 ritz_deflection / buckling_eigen_*".format(abs(lam.D16) / lam.D11)
        )


# ---------------------------------------------------------------------------
# 1) CLT Navier 精确解
# ---------------------------------------------------------------------------
def navier_clt_deflection(
    lam: Laminate,
    a: float,
    b: float,
    q0: float = 1.0,
    m_max: int = 99,
    n_max: int = 99,
) -> BendingResult:
    """四边简支对称正交各向层合板在均布载荷 ``q0`` 下的 CLT 精确解。

    ``q_mn = 16 q0/(pi^2 m n)``，
    ``W_mn = q_mn / [pi^4 (D11 m^4/a^4 + 2H m^2n^2/(a^2b^2) + D22 n^4/b^4)]``，仅奇模态。
    """
    _require_symmetric(lam)
    _require_orthotropic(lam)
    geo = validate_plate_geometry(a, b, lam.thickness)
    q0 = _finite(q0, "q0")
    coeffs = np.zeros((m_max + 1, n_max + 1))
    D11, H, D22 = lam.D11, lam.H, lam.D22
    for m in range(1, m_max + 1, 2):
        for n in range(1, n_max + 1, 2):
            al = m * math.pi / a
            be = n * math.pi / b
            Dmn = D11 * al**4 + 2.0 * H * al * al * be * be + D22 * be**4
            coeffs[m, n] = (16.0 * q0 / (math.pi**2 * m * n)) / Dmn
    w_center = float(_series_field(coeffs, a, b, [0.5 * a], [0.5 * b])[0, 0])
    return BendingResult(
        method="CLT-Navier",
        a=a,
        b=b,
        q0=q0,
        w_max=w_center,
        coeffs=coeffs,
        x_max=0.5 * a,
        y_max=0.5 * b,
        meta={
            "geometry": geo,
            "D11": D11,
            "D22": D22,
            "D12": lam.D12,
            "H": H,
            "D66": lam.D66,
            "m_max": m_max,
            "n_max": n_max,
            "formula": (
                "W_mn = 16 q0/(pi^2 m n) / "
                "(pi^4 (D11 m^4/a^4 + 2H m^2n^2/(a^2b^2) + D22 n^4/b^4))"
            ),
        },
    )


# ---------------------------------------------------------------------------
# 2) CLT Levy 单三角级数解
# ---------------------------------------------------------------------------
def levy_deflection(
    lam: Laminate,
    a: float,
    b: float,
    q0: float = 1.0,
    x_edges: str = "SS",
    n_terms: int = 61,
) -> BendingResult:
    """Levy 解：``y = 0, b`` 简支，``x = 0, a`` 简支(``"SS"``)或固支(``"CC"``)。

    每阶 ``n``（奇）解四阶常微分方程
    ``D11 Y'''' - 2H lam_n^2 Y'' + D22 lam_n^4 Y = q_n``，
    ``lam_n = n pi / b``，``q_n = 4 q0/(n pi)``；用 4 个特征根 + 特解构造，
    边界条件消元（``SS``: ``Y = 0, D11 Y'' - D12 lam_n^2 Y = 0``；
    ``CC``: ``Y = Y' = 0``）。
    """
    _require_symmetric(lam)
    _require_orthotropic(lam)
    geo = validate_plate_geometry(a, b, lam.thickness)
    q0 = _finite(q0, "q0")
    key = str(x_edges).upper()
    if key not in ("SS", "CC"):
        raise ValueError("x_edges 只支持 'SS' 或 'CC'")
    D11, H, D22, D12 = lam.D11, lam.H, lam.D22, lam.D12
    x_sample = np.linspace(0.0, a, 401)

    def Y_profile(n: int) -> np.ndarray:
        lam_n = n * math.pi / b
        qn = 4.0 * q0 / (n * math.pi)
        roots = np.roots([D11, 0.0, -2.0 * H * lam_n**2, 0.0, D22 * lam_n**4])
        yp = qn / (D22 * lam_n**4)
        Bmat = np.zeros((4, 4), dtype=complex)
        rhs = np.zeros(4, dtype=complex)
        for row, xv in enumerate((0.0, 0.0, a, a)):
            e = np.exp(roots * xv)
            if row in (0, 2):  # Y = 0
                Bmat[row] = e
                rhs[row] = -yp
            elif key == "SS":  # D11 Y'' - D12 lam_n^2 Y = 0
                Bmat[row] = D11 * (roots**2) * e - D12 * lam_n**2 * e
                rhs[row] = D12 * lam_n**2 * yp
            else:  # Y' = 0
                Bmat[row] = roots * e
                rhs[row] = 0.0
        C = np.linalg.solve(Bmat, rhs)
        return np.real(np.exp(np.outer(x_sample, roots)) @ C + yp)

    table = {n: Y_profile(n) for n in range(1, n_terms + 1, 2)}

    def field(X, Y) -> np.ndarray:
        X = np.atleast_1d(np.asarray(X, dtype=float))
        Y = np.atleast_1d(np.asarray(Y, dtype=float))
        out = np.zeros((X.size, Y.size))
        for n, Yv in table.items():
            out += np.outer(np.interp(X, x_sample, Yv), np.sin(n * math.pi * Y / b))
        return out

    fld = field([0.5 * a], [0.5 * b])
    return BendingResult(
        method=f"CLT-Levy({key})",
        a=a,
        b=b,
        q0=q0,
        w_max=float(fld[0, 0]),
        coeffs=np.zeros((1, 1)),
        x_max=0.5 * a,
        y_max=0.5 * b,
        meta={
            "geometry": geo,
            "x_edges": key,
            "n_terms": n_terms,
            "D11": D11,
            "H": H,
            "D22": D22,
            "field": field,
        },
    )

# ---------------------------------------------------------------------------
# 3) Rayleigh-Ritz（一般对称层合板，含 D16, D26）
# ---------------------------------------------------------------------------
def _sin_cos_table(M: int, length: float) -> np.ndarray:
    """精确表 ``S[m, k] = int_0^L sin(m pi x/L) cos(k pi x/L) dx``，``m,k = 1..M``。"""
    S = np.zeros((M + 1, M + 1))
    for m in range(1, M + 1):
        for k in range(1, M + 1):
            term1 = (1.0 - math.cos((m + k) * math.pi)) / (2.0 * (m + k))
            term2 = 0.0 if m == k else (1.0 - math.cos((m - k) * math.pi)) / (2.0 * (m - k))
            S[m, k] = (term1 + term2) * length / math.pi
    return S


def _ritz_bending_matrix(lam: Laminate, a: float, b: float, M: int, N: int) -> np.ndarray:
    """CLT 弯曲刚度矩阵 ``K``（双三角级数基），含 ``D16, D26`` 交叉项。

    基 ``phi_mn = sin(m pi x/a) sin(n pi y/b)``；``W^T K W = int D_ijkl w_,ij w_,kl dA``。
    交叉项积分 ``int phi_xx phi_xy = -a_m^2 a_p b_q Sx[m,p] Sy[n,q]``（二阶导带负号），
    矩阵统一取对称部分 ``sym(A) = (A + A^T)/2``（二次型只依赖对称部分）。
    """
    alpha = np.array([0.0] + [m * math.pi / a for m in range(1, M + 1)])
    beta = np.array([0.0] + [n * math.pi / b for n in range(1, N + 1)])
    Sx = _sin_cos_table(M, a)
    Sy = _sin_cos_table(N, b)
    ab4 = 0.25 * a * b
    K = np.zeros((M * N, M * N))
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            i = (m - 1) * N + (n - 1)
            K[i, i] += (
                lam.D11 * alpha[m] ** 4
                + 2.0 * lam.D12 * alpha[m] ** 2 * beta[n] ** 2
                + lam.D22 * beta[n] ** 4
                + 4.0 * lam.D66 * alpha[m] ** 2 * beta[n] ** 2
            ) * ab4
    if lam.D16 == 0.0 and lam.D26 == 0.0:
        return K
    # 交叉项：按 m 分块向量化（(p,q) 展平为列索引 j = (p-1)*N + (q-1)）
    q_of_j = np.arange(M * N) % N
    colscale = Sy[1:, 1:][:, q_of_j]      # (N, M*N)  ->  Sy[n,q(j)]
    bb = beta[1:]                          # (N,)
    cross = np.zeros((M * N, M * N))
    for m in range(1, M + 1):
        # v[j] = alpha_p beta_q Sx[m,p]
        v = ((alpha[1:][:, None] * bb[None, :]) * Sx[m, 1:][:, None]).ravel()
        rows = slice((m - 1) * N, m * N)
        block = np.zeros((N, M * N))
        if lam.D16 != 0.0:
            block -= 4.0 * lam.D16 * alpha[m] ** 2 * colscale * v[None, :]
        if lam.D26 != 0.0:
            block -= 4.0 * lam.D26 * (bb**2)[:, None] * colscale * v[None, :]
        cross[rows, :] = block
    return K + 0.5 * (cross + cross.T)


def _ritz_load_vector(a: float, b: float, q0: float, M: int, N: int) -> np.ndarray:
    F = np.zeros(M * N)
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            if m % 2 == 1 and n % 2 == 1:
                F[(m - 1) * N + (n - 1)] = 4.0 * q0 * a * b / (math.pi**2 * m * n)
    return F


def ritz_deflection(
    lam: Laminate,
    a: float,
    b: float,
    q0: float = 1.0,
    m_terms: int = 12,
    n_terms: int = 12,
) -> BendingResult:
    """一般对称层合板（允许 ``D16, D26 != 0``）四边简支 Rayleigh-Ritz 解。

    试函数 ``w = sum W_mn sin(m pi x/a) sin(n pi y/b)``（满足几何边界 ``w = 0``），
    由总势能驻值 ``K W = F`` 求解。``D16 = D26 = 0`` 时 ``K`` 对角，结果与 Navier
    闭式逐位一致（实现自检）。
    """
    _require_symmetric(lam)
    geo = validate_plate_geometry(a, b, lam.thickness)
    q0 = _finite(q0, "q0")
    M, N = int(m_terms), int(n_terms)
    if M < 1 or N < 1:
        raise ValueError("m_terms / n_terms 必须 >= 1")
    K = _ritz_bending_matrix(lam, a, b, M, N)
    F = _ritz_load_vector(a, b, q0, M, N)
    W = np.linalg.solve(K, F)
    coeffs = np.zeros((M + 1, N + 1))
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            coeffs[m, n] = W[(m - 1) * N + (n - 1)]
    w_center = float(_series_field(coeffs, a, b, [0.5 * a], [0.5 * b])[0, 0])
    return BendingResult(
        method="CLT-Ritz",
        a=a,
        b=b,
        q0=q0,
        w_max=w_center,
        coeffs=coeffs,
        x_max=0.5 * a,
        y_max=0.5 * b,
        meta={
            "geometry": geo,
            "m_terms": M,
            "n_terms": N,
            "D16": lam.D16,
            "D26": lam.D26,
            "strain_energy_density": 0.5 * float(W @ K @ W),
            "K_condition": float(np.linalg.cond(K)),
            "n_modes": M * N,
        },
    )


# ---------------------------------------------------------------------------
# 4) 独立能量解：x 边固支（固支-固支梁函数 x 正弦 + Gauss 积分）
# ---------------------------------------------------------------------------
def clamped_beam_functions(n_modes: int = 6) -> List[Tuple[float, float]]:
    """固支-固支梁特征值 ``mu`` 与系数 ``sigma``：``cosh mu cos mu = 1``。

    ``X(xi) = cosh(mu xi) - cos(mu xi) - sigma (sinh(mu xi) - sin(mu xi))``，
    ``xi = x/L``，``sigma = (cosh mu - cos mu)/(sinh mu - sin mu)``。
    """
    mus: List[float] = []
    grid = np.linspace(0.5, (n_modes + 1.2) * math.pi, 200001)
    f = np.cosh(grid) * np.cos(grid) - 1.0
    for i in range(len(grid) - 1):
        if f[i] * f[i + 1] < 0:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if (math.cosh(lo) * math.cos(lo) - 1.0) * (math.cosh(mid) * math.cos(mid) - 1.0) <= 0:
                    hi = mid
                else:
                    lo = mid
            mus.append(0.5 * (lo + hi))
            if len(mus) >= n_modes:
                break
    out = []
    for mu in mus:
        sig = (math.cosh(mu) - math.cos(mu)) / (math.sinh(mu) - math.sin(mu))
        out.append((mu, sig))
    return out


def _beam_values(mu: float, sigma: float, xi: np.ndarray, length: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ch, sh = np.cosh(mu * xi), np.sinh(mu * xi)
    c, s = np.cos(mu * xi), np.sin(mu * xi)
    X = ch - c - sigma * (sh - s)
    X1 = (mu / length) * (sh + s - sigma * (ch - c))
    X2 = (mu / length) ** 2 * (ch + c - sigma * (sh + s))
    return X, X1, X2


def ritz_clamped_x_deflection(
    lam: Laminate,
    a: float,
    b: float,
    q0: float = 1.0,
    m_terms: int = 6,
    n_terms: int = 12,
    n_gauss: int = 60,
) -> BendingResult:
    """``x = 0, a`` 固支、``y = 0, b`` 简支的正交各向对称层合板独立能量解。

    基函数 ``X_p(x) sin(n pi y/b)``（``X_p`` 为固支-固支梁函数），刚度矩阵用
    Gauss-Legendre 数值积分生成 —— 与 Levy 的解析 ODE 求解路径完全独立，
    用于交叉验证 Levy 的 CC 分支。
    """
    _require_symmetric(lam)
    _require_orthotropic(lam)
    validate_plate_geometry(a, b, lam.thickness)
    q0 = _finite(q0, "q0")
    P, N = int(m_terms), int(n_terms)
    gx, wx = np.polynomial.legendre.leggauss(n_gauss)
    xg = 0.5 * a * (gx + 1.0)
    wgx = 0.5 * a * wx
    yg = 0.5 * b * (gx + 1.0)
    wgy = 0.5 * b * wx
    beams = clamped_beam_functions(P)
    if len(beams) < P:
        raise RuntimeError("固支梁特征根不足")
    A0 = np.zeros((P, n_gauss))
    A1 = np.zeros((P, n_gauss))
    A2 = np.zeros((P, n_gauss))
    for p, (mu, sig) in enumerate(beams):
        X, X1, X2 = _beam_values(mu, sig, xg / a, a)
        A0[p], A1[p], A2[p] = X, X1, X2
    n_arr = np.arange(1, N + 1)
    lam_y = n_arr * math.pi / b
    Y0 = np.sin(np.outer(lam_y, yg))  # (N, gy)
    Y1 = np.cos(np.outer(lam_y, yg)) * lam_y[:, None]
    Y2 = -np.sin(np.outer(lam_y, yg)) * lam_y[:, None] ** 2

    def ip(A, B, w):
        return (A * w) @ B.T

    Mx0 = ip(A0, A0, wgx)
    Mx1 = ip(A1, A1, wgx)
    Mx2 = ip(A2, A2, wgx)
    Mx2m = ip(A2, A0, wgx)
    Ny0 = ip(Y0, Y0, wgy)
    Ny1 = ip(Y1, Y1, wgy)
    Ny2 = ip(Y2, Y2, wgy)
    Ny2m = ip(Y0, Y2, wgy)
    K = (
        lam.D11 * np.kron(Mx2, Ny0)
        + lam.D22 * np.kron(Mx0, Ny2)
        + 2.0 * lam.D12 * np.kron(0.5 * (Mx2m + Mx2m.T), Ny2m)
        + 4.0 * lam.D66 * np.kron(Mx1, Ny1)
    )
    F = np.zeros(P * N)
    intX = A0 @ wgx
    for p, (mu, sig) in enumerate(beams):
        for n in range(1, N + 1):
            if n % 2 == 1:
                F[p * N + (n - 1)] = q0 * intX[p] * (2.0 * b / (n * math.pi))
    Wv = np.linalg.solve(K, F)
    Wmat = Wv.reshape(P, N)
    xc = np.array([0.5 * a])
    yc = np.array([0.5 * b])
    Xc = np.array([_beam_values(mu, sig, xc / a, a)[0][0] for mu, sig in beams])
    Yc = np.sin(lam_y * 0.5 * b)
    w_center = float(Xc @ Wmat @ Yc)
    return BendingResult(
        method="CLT-Ritz(clamped-x)",
        a=a,
        b=b,
        q0=q0,
        w_max=w_center,
        coeffs=np.zeros((1, 1)),
        x_max=0.5 * a,
        y_max=0.5 * b,
        meta={
            "m_terms": P,
            "n_terms": N,
            "n_gauss": n_gauss,
            "mu": [mu for mu, _ in beams],
            "K_condition": float(np.linalg.cond(K)),
            "D11": lam.D11,
            "H": lam.H,
            "D22": lam.D22,
        },
    )


# ---------------------------------------------------------------------------
# 5) FSDT Navier 精确解（正交各向对称层合板）
# ---------------------------------------------------------------------------
def fsdt_navier_deflection(
    lam: Laminate,
    a: float,
    b: float,
    q0: float = 1.0,
    m_max: int = 39,
    n_max: int = 39,
) -> BendingResult:
    """一阶剪切变形理论（``k = 5/6``）四边简支正交各向层合板 Navier 解。

    逐模态解 ``3x3`` 系统（``W, Phi_x, Phi_y``）：

    .. code::

        [ s1       s3       kA55 al ] [X]   [0]
        [ s3       s2       kA44 be ] [Y] = [0]
        [ kA55 al  kA44 be  s0      ] [W]   [q_mn]

    ``s1 = D11 al^2 + D66 be^2 + kA55``，``s2 = D66 al^2 + D22 be^2 + kA44``，
    ``s3 = (D12 + D66) al be``，``s0 = kA55 al^2 + kA44 be^2``
    （``al = m pi/a``，``be = n pi/b``）。``h/a -> 0`` 时 ``W -> W_CLT``。
    """
    _require_symmetric(lam)
    _require_orthotropic(lam)
    geo = validate_plate_geometry(a, b, lam.thickness)
    q0 = _finite(q0, "q0")
    coeffs = np.zeros((m_max + 1, n_max + 1))
    A44, A55 = lam.A44, lam.A55
    D11, D22, D12, D66 = lam.D11, lam.D22, lam.D12, lam.D66
    for m in range(1, m_max + 1, 2):
        for n in range(1, n_max + 1, 2):
            al = m * math.pi / a
            be = n * math.pi / b
            s1 = D11 * al * al + D66 * be * be + A55
            s2 = D66 * al * al + D22 * be * be + A44
            s3 = (D12 + D66) * al * be
            s0 = A55 * al * al + A44 * be * be
            qmn = 16.0 * q0 / (math.pi**2 * m * n)
            Kmat = np.array(
                [[s1, s3, A55 * al], [s3, s2, A44 * be], [A55 * al, A44 * be, s0]]
            )
            sol = np.linalg.solve(Kmat, np.array([0.0, 0.0, qmn]))
            coeffs[m, n] = sol[2]
    w_center = float(_series_field(coeffs, a, b, [0.5 * a], [0.5 * b])[0, 0])
    return BendingResult(
        method="FSDT-Navier",
        a=a,
        b=b,
        q0=q0,
        w_max=w_center,
        coeffs=coeffs,
        x_max=0.5 * a,
        y_max=0.5 * b,
        meta={
            "geometry": geo,
            "A44": A44,
            "A55": A55,
            "shear_correction": lam.shear_correction,
            "h_over_a": lam.thickness / a,
            "D11": D11,
            "D22": D22,
            "m_max": m_max,
            "n_max": n_max,
        },
    )


# ---------------------------------------------------------------------------
# 6) 屈曲：闭式（正交各向对称）与广义特征值（一般各向异性）
# ---------------------------------------------------------------------------
def buckling_uniaxial(
    lam: Laminate, a: float, b: float, m_max: int = 9, n_max: int = 9
) -> BucklingResult:
    """四边简支正交各向对称层合板沿 ``x`` 单轴压缩的临界线载 ``N_x,cr`` [N/m]。

    ``N_x(m,n) = (D11 al^4 + 2H al^2 be^2 + D22 be^4)/al^2``，
    ``al = m pi/a``，``be = n pi/b``，``H = D12 + 2D66``；取最小值。
    """
    _require_symmetric(lam)
    _require_orthotropic(lam)
    validate_plate_geometry(a, b, lam.thickness)
    D11, H, D22 = lam.D11, lam.H, lam.D22
    table: List[Dict[str, float]] = []
    best = (float("inf"), 1, 1)
    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            al = m * math.pi / a
            be = n * math.pi / b
            N = (D11 * al**4 + 2.0 * H * al * al * be * be + D22 * be**4) / (al * al)
            table.append({"m": m, "n": n, "N_x_cr": N})
            if N < best[0]:
                best = (N, m, n)
    return BucklingResult(
        method="CLT-closed-form",
        load_type="uniaxial_Nx",
        n_cr=float(best[0]),
        m=best[1],
        n=best[2],
        table=table,
        meta={"D11": D11, "H": H, "D22": D22, "a": a, "b": b, "h": lam.thickness},
    )


def buckling_biaxial(
    lam: Laminate, a: float, b: float, m_max: int = 9, n_max: int = 9
) -> BucklingResult:
    """四边简支正交各向对称层合板双轴等压（``N_x = N_y = N``）临界线载 [N/m]。

    ``N(m,n) = (D11 al^4 + 2H al^2 be^2 + D22 be^4)/(al^2 + be^2)``
    （任务书闭式的量纲正确形式，``al = m pi/a``，``be = n pi/b``）。
    """
    _require_symmetric(lam)
    _require_orthotropic(lam)
    validate_plate_geometry(a, b, lam.thickness)
    D11, H, D22 = lam.D11, lam.H, lam.D22
    table: List[Dict[str, float]] = []
    best = (float("inf"), 1, 1)
    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            al = m * math.pi / a
            be = n * math.pi / b
            N = (D11 * al**4 + 2.0 * H * al * al * be * be + D22 * be**4) / (
                al * al + be * be
            )
            table.append({"m": m, "n": n, "N_biax_cr": N})
            if N < best[0]:
                best = (N, m, n)
    return BucklingResult(
        method="CLT-closed-form",
        load_type="biaxial_equal",
        n_cr=float(best[0]),
        m=best[1],
        n=best[2],
        table=table,
        meta={"D11": D11, "H": H, "D22": D22, "a": a, "b": b, "h": lam.thickness},
    )


def _geo_uniaxial(lam: Laminate, a: float, b: float, M: int, N: int) -> np.ndarray:
    """几何刚度矩阵（``x`` 单轴压缩），``Gx_ij = int w,x_i w,x_j dA``。"""
    ab4 = 0.25 * a * b
    G = np.zeros((M * N, M * N))
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            i = (m - 1) * N + (n - 1)
            G[i, i] = (m * math.pi / a) ** 2 * ab4
    return G


def _geo_shear(lam: Laminate, a: float, b: float, M: int, N: int) -> np.ndarray:
    """几何刚度矩阵（纯剪），``Gxy_ij = int (w,x_i w,y_j + w,x_j w,y_i) dA``。

    纯剪载荷势 ``V = N_xy int w,x w,y dA`` 的 Hessian 即 ``Gxy``（``W^T G W = 2V``）。
    按 m 分块向量化，避免 ``(MN)^2`` 的 Python 循环。
    """
    alpha = np.array([0.0] + [m * math.pi / a for m in range(1, M + 1)])
    beta = np.array([0.0] + [n * math.pi / b for n in range(1, N + 1)])
    Sx = _sin_cos_table(M, a)  # Sx[k, j] = int sin(k) cos(j) dx
    Sy = _sin_cos_table(N, b)
    q_of_j = np.arange(M * N) % N          # 列索引 j = (p-1)*N + (q-1) -> q
    sy_nq = Sy[1:, 1:][:, q_of_j]          # [n, j] = Sy[n, q]
    sy_qn = Sy[1:, 1:].T[:, q_of_j]        # [n, j] = Sy[q, n]
    bb = beta[1:]
    G = np.zeros((M * N, M * N))
    for m in range(1, M + 1):
        rows = slice((m - 1) * N, m * N)
        # term1: al_m be_q Sx[p,m] Sy[n,q]
        v1 = (Sx[1:, m][:, None] * bb[None, :]).ravel()      # [j] = Sx[p,m] be_q
        block = alpha[m] * sy_nq * v1[None, :]
        # term2: be_n al_p Sx[m,p] Sy[q,n]
        v2 = np.repeat(Sx[m, 1:] * alpha[1:], N)  # [j=(p,q)] = al_p Sx[m,p]（与 q 无关）
        block += bb[:, None] * sy_qn * v2[None, :]
        G[rows, :] = block
    return 0.5 * (G + G.T)


def buckling_eigen_uniaxial(
    lam: Laminate, a: float, b: float, m_terms: int = 8, n_terms: int = 8
) -> BucklingResult:
    """一般各向异性对称层合板（含 ``D16, D26``）单轴压缩临界载荷（广义特征值）。

    解 ``K W = N Gx W``（``K`` 含 ``D16, D26`` 交叉项，``Gx`` 单轴几何刚度），
    取最小正特征值。``D16 = D26 = 0`` 时与 ``buckling_uniaxial`` 闭式一致。
    """
    _require_symmetric(lam)
    validate_plate_geometry(a, b, lam.thickness)
    M, N = int(m_terms), int(n_terms)
    K = _ritz_bending_matrix(lam, a, b, M, N)
    G = _geo_uniaxial(lam, a, b, M, N)
    # G 为对角正定 -> 白化后作标准对称特征值问题（纯 numpy，无 scipy 依赖）
    d = np.sqrt(np.diag(G))
    Kt = K / np.outer(d, d)
    vals, vecs = np.linalg.eigh(0.5 * (Kt + Kt.T))
    k = int(np.argmin(vals))
    mat = np.abs(vecs[:, k]).reshape(M, N)
    m = int(np.argmax(np.max(mat, axis=1))) + 1
    n = int(np.argmax(np.max(mat, axis=0))) + 1
    table = [
        {"mode_index": i + 1, "N_x_cr": float(vals[i])}
        for i in range(min(8, vals.size))
    ]
    return BucklingResult(
        method="CLT-eigen(uniaxial)",
        load_type="uniaxial_Nx",
        n_cr=float(vals[k]),
        m=m,
        n=n,
        table=table,
        meta={"m_terms": M, "n_terms": N, "D16": lam.D16, "D26": lam.D26},
    )


def buckling_eigen_shear(
    lam: Laminate, a: float, b: float, m_terms: int = 8, n_terms: int = 8
) -> BucklingResult:
    """一般对称层合板纯剪屈曲临界剪流 ``N_xy,cr`` [N/m]（广义特征值）。

    解 ``K W = N_xy Gxy W``；``Gxy`` 不定号，用 ``scipy.linalg.eig`` 取最小正实
    特征值（纯剪屈曲是保守系统但载荷算子是双线性的，不能用 ``eigh``）。
    """
    _require_symmetric(lam)
    validate_plate_geometry(a, b, lam.thickness)
    M, N = int(m_terms), int(n_terms)
    K = _ritz_bending_matrix(lam, a, b, M, N)
    G = _geo_shear(lam, a, b, M, N)
    # K 为 SPD：Cholesky 白化 K w = N G w  ->  I x = (1/N) C x，
    # C = L^{-1} G L^{-T} 为**实对称**矩阵（G 不定号但对称），用 eigh 求
    # 最大正特征值 mu_max  ->  N_cr = 1/mu_max（比直接对不定 pencil 做非对称 eig
    # 快且稳；两者数学等价）。
    L = np.linalg.cholesky(K)
    Linv = np.linalg.inv(L)
    C = Linv @ G @ Linv.T
    C = 0.5 * (C + C.T)
    mu, vecs = np.linalg.eigh(C)
    if mu[-1] <= 0.0:
        raise RuntimeError("纯剪屈曲未找到正特征值，增大 m_terms/n_terms")
    n_cr = 1.0 / float(mu[-1])
    x = Linv.T @ vecs[:, -1]
    mat = np.abs(x).reshape(M, N)
    m = int(np.argmax(np.max(mat, axis=1))) + 1
    n = int(np.argmax(np.max(mat, axis=0))) + 1
    pos = sorted(1.0 / float(v) for v in mu if v > 0.0)
    table = [{"mode_index": i + 1, "N_xy_cr": v} for i, v in enumerate(pos[:8])]
    return BucklingResult(
        method="CLT-eigen(shear)",
        load_type="pure_shear_Nxy",
        n_cr=float(n_cr),
        m=m,
        n=n,
        table=table,
        meta={"m_terms": M, "n_terms": N, "D16": lam.D16, "D26": lam.D26},
    )
