"""ice_anchor — 破冰物理锚库（Ice-Breaking Physics Anchor Library）。

本模块为 TensorPlatform 冰区航行学科提供 ≥12 个可点验的物理锚点，
覆盖：
  A. 平整冰阻力经验公式（Lindqvist 1989, Keinonen, Huang 等）
  B. 冰压痕/压溃强度典型值
  C. 破冰弯曲断裂厚度-线载荷关系
  D. 模型试验公开数据集

每个锚点包含：
  - 公式或数据（闭式表达或数值表）
  - 文献出处（DOI/ISBN/可检索引用）
  - 适用域（冰厚/航速/船型范围）
  - 映射到 SIR 七腿输出量

版本: 1.0.0 (m598)
"""

from .anchors import (
    ANCHORS,
    ANCHOR_REGISTRY,
    get_anchor,
    list_anchors,
    lindqvist_1989,
    keinonen_1983,
    huang_2014,
    enkvist_1972,
    riska_1997,
    ice_crushing_strength,
    ice_flexural_strength,
    ice_indentation,
    bending_fracture_load,
    model_test_acecrc,
    model_test_hamburg,
    model_test_arctech,
    ice_thermal_properties,
    ice_density_salinity,
)
from .mapping import ANCHOR_SIR_MAPPING, load_mapping_yaml

__version__ = "1.0.0"
__all__ = [
    "ANCHORS",
    "ANCHOR_REGISTRY",
    "get_anchor",
    "list_anchors",
    "ANCHOR_SIR_MAPPING",
    "load_mapping_yaml",
    # individual anchors
    "lindqvist_1989",
    "keinonen_1983",
    "huang_2014",
    "enkvist_1972",
    "riska_1997",
    "ice_crushing_strength",
    "ice_flexural_strength",
    "ice_indentation",
    "bending_fracture_load",
    "model_test_acecrc",
    "model_test_hamburg",
    "model_test_arctech",
    "ice_thermal_properties",
    "ice_density_salinity",
]
