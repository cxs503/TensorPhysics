"""复合层力学领域模块 (composite_mechanics)。

当前提供：
- :mod:`tensorphysics.composite_mechanics.snow_ice_composite`
  雪-冰复合层 (snow / snow-ice / ice) 弯曲力学：复合弯曲刚度 D_composite、
  弹性加权中性轴 z_na、弯曲强度折减系数 η_snow，以及 D_composite 注入
  Winkler 弹性地基板后的中心挠度。
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
]