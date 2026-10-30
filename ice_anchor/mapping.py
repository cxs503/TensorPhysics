"""锚点 → SIR 七腿映射加载器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_YAML_PATH = Path(__file__).parent / "anchor_mapping.yaml"


def load_mapping_yaml(path: Optional[Path] = None) -> Dict[str, Any]:
    """加载锚点映射 YAML。"""
    p = path or _YAML_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 模块级缓存
_MAPPING_CACHE: Optional[Dict[str, Any]] = None


def get_mapping() -> Dict[str, Any]:
    """获取映射数据（带缓存）。"""
    global _MAPPING_CACHE
    if _MAPPING_CACHE is None:
        _MAPPING_CACHE = load_mapping_yaml()
    return _MAPPING_CACHE


def get_anchor_to_sir(anchor_id: str) -> List[Dict[str, str]]:
    """获取指定锚点映射到的 SIR 腿列表。"""
    m = get_mapping()
    a2s = m.get("anchor_to_sir", {})
    entry = a2s.get(anchor_id, {})
    return entry.get("sir_legs", [])


def get_coverage_matrix() -> tuple:
    """返回覆盖矩阵 (headers, rows)。"""
    m = get_mapping()
    cm = m.get("coverage_matrix", {})
    return cm.get("headers", []), cm.get("rows", [])


# 预加载
ANCHOR_SIR_MAPPING = get_mapping()
