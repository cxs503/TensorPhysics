"""复合层力学领域模块 (composite_mechanics)。

当前提供：
- :mod:`tensorphysics.composite_mechanics.snow_ice_composite`
  雪-冰复合层 (snow / snow-ice / ice) 弯曲力学：复合弯曲刚度 D_composite、
  弹性加权中性轴 z_na、弯曲强度折减系数 η_snow，以及 D_composite 注入
  Winkler 弹性地基板后的中心挠度。
- :mod:`tensorphysics.composite_mechanics.laminate_plate`
  层合复合材料船体板的 CLT / FSDT 弯曲与屈曲验证面：``A/B/D`` 积分、
  Navier 双三角级数精确解、Levy 单三角级数解、Rayleigh-Ritz 能量解
  （含 ``D16/D26`` 弯-扭耦合）、FSDT 剪切修正、单轴/双轴/纯剪屈曲闭式与
  广义特征值解，以及各向同性/薄板退化负控。
"""

from .snow_ice_composite import (  # noqa: F401
    DEFAULT_MATERIALS,
    IceLayer,
    LayerStack,
    SALORANTA_2000,
    SALORANTA_2000_CITATION,
    default_layers,
    equivalent_homogeneous_modulus,
    snow_ice_composite,
    snow_on_ice_fixed_total,
    winkler_plate_center_deflection,
    winkler_plate_deflection_profile,
    winkler_plate_field,
)
from .laminate_plate import (  # noqa: F401
    BendingResult,
    BucklingResult,
    Lamina,
    Laminate,
    T300_5208,
    abd_matrices,
    buckling_biaxial,
    buckling_eigen_shear,
    buckling_eigen_uniaxial,
    buckling_uniaxial,
    clamped_beam_functions,
    fsdt_navier_deflection,
    isotropic_plate_rigidity,
    levy_deflection,
    navier_clt_deflection,
    parse_layup,
    ritz_clamped_x_deflection,
    ritz_deflection,
    validate_plate_geometry,
)

__all__ = [
    "DEFAULT_MATERIALS",
    "IceLayer",
    "LayerStack",
    "SALORANTA_2000",
    "SALORANTA_2000_CITATION",
    "default_layers",
    "equivalent_homogeneous_modulus",
    "snow_ice_composite",
    "snow_on_ice_fixed_total",
    "winkler_plate_center_deflection",
    "winkler_plate_deflection_profile",
    "winkler_plate_field",
    # 层合板 CLT/FSDT 验证面
    "BendingResult",
    "BucklingResult",
    "Lamina",
    "Laminate",
    "T300_5208",
    "abd_matrices",
    "buckling_biaxial",
    "buckling_eigen_shear",
    "buckling_eigen_uniaxial",
    "buckling_uniaxial",
    "clamped_beam_functions",
    "fsdt_navier_deflection",
    "isotropic_plate_rigidity",
    "levy_deflection",
    "navier_clt_deflection",
    "parse_layup",
    "ritz_clamped_x_deflection",
    "ritz_deflection",
    "validate_plate_geometry",
]