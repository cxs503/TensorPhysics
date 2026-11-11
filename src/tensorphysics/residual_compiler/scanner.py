"""Scanners for campaign nodes and audit capabilities."""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import re
import os


@dataclass
class Limitation:
    """A limitation extracted from a campaign node."""
    node_id: int
    node_name: str
    limitation_text: str
    limitation_type: str  # "limitation", "untested", "honest_fail", "gap"
    physics_domain: str = ""
    evidence_grade: str = "WITHHELD"
    source_field: str = ""  # which metadata field this came from
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "limitation_text": self.limitation_text,
            "limitation_type": self.limitation_type,
            "physics_domain": self.physics_domain,
            "evidence_grade": self.evidence_grade,
            "source_field": self.source_field,
        }


@dataclass
class CapabilityGap:
    """A capability gap from audit results."""
    capability_name: str
    evidence_grade: str  # WITHHELD or NOT_SUPPORTED
    description: str
    physics_domain: str = ""
    
    def to_dict(self) -> dict:
        return {
            "capability_name": self.capability_name,
            "evidence_grade": self.evidence_grade,
            "description": self.description,
            "physics_domain": self.physics_domain,
        }


class CampaignNodeScanner:
    """Scans campaign nodes for limitations, untested items, and honest failures.
    
    Scans multiple sources:
    1. metadata.goal - physical goal and limitations
    2. metadata.acceptance_criteria - criteria that may have failed
    3. metadata explicit fields (limitations, untested, honest_fail, gaps, todo)
    4. report.markdown - full report content with limitation sections
    """
    
    def __init__(self, limitation_keywords: Optional[List[str]] = None):
        self.limitation_keywords = limitation_keywords or [
            "局限", "limitation", "未做", "untested", "honest", "FAIL",
            "TODO", "后续", "gap", "missing", "deficit", "WITHHELD",
            "NOT_SUPPORTED", "MOCK", "STUB", "待改进", "需修复",
            "误差", "偏差", "不满足", "未验证", "未实现"
        ]
        self._scanned_nodes = 0
        self._nodes_with_limitations = 0
    
    def scan_nodes(self, nodes: List[Dict[str, Any]]) -> List[Limitation]:
        """Scan a list of campaign nodes for limitations.
        
        Args:
            nodes: List of campaign node dictionaries
            
        Returns:
            List of Limitation objects
        """
        limitations = []
        self._scanned_nodes = 0
        self._nodes_with_limitations = 0
        
        for node in nodes:
            if node.get("status") != "done":
                continue
            
            self._scanned_nodes += 1
            node_limitations = self._extract_limitations(node)
            
            if node_limitations:
                self._nodes_with_limitations += 1
                limitations.extend(node_limitations)
        
        return limitations
    
    def _extract_limitations(self, node: Dict[str, Any]) -> List[Limitation]:
        """Extract limitations from a single node."""
        limitations = []
        node_id = node.get("id", 0)
        node_name = node.get("name", "")
        metadata = node.get("metadata", {})
        
        if not isinstance(metadata, dict):
            metadata = {}
        
        # Extract physics domain
        physics_domain = metadata.get("physics_domain", "")
        if not physics_domain:
            # Try to infer from node name or goal
            physics_domain = self._infer_domain(node_name, metadata.get("goal", ""))
        
        # Track if we found any limitations for this node
        found_any = False
        
        # Scan goal field
        goal = metadata.get("goal", "")
        if goal:
            goal_lims = self._find_limitations_in_text(
                goal, node_id, node_name, physics_domain, "goal"
            )
            if goal_lims:
                limitations.extend(goal_lims)
                found_any = True
        
        # Scan acceptance_criteria field
        criteria = metadata.get("acceptance_criteria", "")
        if criteria:
            if isinstance(criteria, list):
                criteria = "\n".join(str(c) for c in criteria)
            crit_lims = self._find_limitations_in_text(
                criteria, node_id, node_name, physics_domain, "acceptance_criteria"
            )
            if crit_lims:
                limitations.extend(crit_lims)
                found_any = True
        
        # Scan for explicit limitation fields
        for key in ["limitations", "untested", "honest_fail", "gaps", "todo"]:
            if key in metadata:
                value = metadata[key]
                if isinstance(value, list):
                    value = "\n".join(str(v) for v in value)
                if value:
                    lim = Limitation(
                        node_id=node_id,
                        node_name=node_name,
                        limitation_text=str(value),
                        limitation_type=key,
                        physics_domain=physics_domain,
                        source_field=key,
                    )
                    limitations.append(lim)
                    found_any = True
        
        # Scan report markdown for limitations
        report = node.get("report", {})
        if isinstance(report, dict):
            markdown = report.get("markdown", "")
            if markdown:
                report_lims = self._extract_from_report_markdown(
                    markdown, node_id, node_name, physics_domain
                )
                if report_lims:
                    limitations.extend(report_lims)
                    found_any = True
        
        # If no explicit limitations found, check if the node has any 
        # indication of incomplete work (e.g., FAIL in criteria, MOCK/STUB mentions)
        if not found_any:
            # Check if goal mentions any incomplete aspects
            if goal and self._has_limitation_indicators(goal):
                lim = Limitation(
                    node_id=node_id,
                    node_name=node_name,
                    limitation_text=f"Node goal may contain unaddressed aspects: {goal[:200]}",
                    limitation_type="potential_gap",
                    physics_domain=physics_domain,
                    source_field="goal_inferred",
                )
                limitations.append(lim)
                found_any = True
        
        return limitations
    
    def _has_limitation_indicators(self, text: str) -> bool:
        """Check if text has indicators of limitations or incomplete work."""
        text_lower = text.lower()
        indicators = [
            "成熟化", "mature", "改进", "improve", "修复", "fix",
            "清偿", "clear", "补充", "supplement", "完善", "complete",
        ]
        return any(ind in text_lower for ind in indicators)
    
    def _extract_from_report_markdown(
        self, markdown: str, node_id: int, node_name: str, physics_domain: str
    ) -> List[Limitation]:
        """Extract limitations from report markdown content."""
        limitations = []
        
        # Look for limitation sections
        section_patterns = [
            r'##\s*\d*\.?\s*注意点.*?\n(.*?)(?=\n##|\Z)',
            r'##\s*\d*\.?\s*局限.*?\n(.*?)(?=\n##|\Z)',
            r'##\s*\d*\.?\s*注意.*?\n(.*?)(?=\n##|\Z)',
            r'##\s*\d*\.?\s*诚实.*?\n(.*?)(?=\n##|\Z)',
            r'##\s*\d*\.?\s*后续.*?\n(.*?)(?=\n##|\Z)',
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, markdown, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if match.strip():
                    # Extract key sentences
                    sentences = re.split(r'[。\n]', match.strip())
                    for sent in sentences[:3]:  # Take first 3 sentences
                        sent = sent.strip()
                        if sent and len(sent) > 20:
                            lim = Limitation(
                                node_id=node_id,
                                node_name=node_name,
                                limitation_text=sent[:300],
                                limitation_type="limitation",
                                physics_domain=physics_domain,
                                source_field="report_markdown",
                            )
                            limitations.append(lim)
        
        # Also look for FAIL mentions
        fail_pattern = r'(FAIL[^。]*。?)'
        fail_matches = re.findall(fail_pattern, markdown)
        for match in fail_matches[:2]:  # Take first 2
            if match.strip():
                lim = Limitation(
                    node_id=node_id,
                    node_name=node_name,
                    limitation_text=match.strip()[:300],
                    limitation_type="honest_fail",
                    physics_domain=physics_domain,
                    source_field="report_markdown",
                )
                limitations.append(lim)
        
        return limitations
    
    def _find_limitations_in_text(
        self, text: str, node_id: int, node_name: str, 
        physics_domain: str, source_field: str
    ) -> List[Limitation]:
        """Find limitation mentions in text."""
        limitations = []
        text_lower = text.lower()
        
        for keyword in self.limitation_keywords:
            if keyword.lower() in text_lower:
                # Extract context around the keyword
                idx = text_lower.find(keyword.lower())
                start = max(0, idx - 100)
                end = min(len(text), idx + len(keyword) + 200)
                context = text[start:end].strip()
                
                # Determine limitation type
                lim_type = "limitation"
                if "honest" in keyword.lower() or "fail" in keyword.lower():
                    lim_type = "honest_fail"
                elif "untested" in keyword.lower() or "未做" in keyword:
                    lim_type = "untested"
                elif "gap" in keyword.lower() or "missing" in keyword.lower():
                    lim_type = "gap"
                
                lim = Limitation(
                    node_id=node_id,
                    node_name=node_name,
                    limitation_text=context,
                    limitation_type=lim_type,
                    physics_domain=physics_domain,
                    source_field=source_field,
                )
                limitations.append(lim)
                break  # One limitation per keyword match is enough
        
        return limitations
    
    def _infer_domain(self, node_name: str, goal: str) -> str:
        """Infer physics domain from node name or goal."""
        text = (node_name + " " + goal).lower()
        
        domain_keywords = {
            "resistance": ["阻力", "resistance", "holtrop", "ittf", "湿表面积"],
            "propulsion": ["推进", "propulsion", "propeller", "螺旋桨", "桨"],
            "seakeeping": ["耐波", "seakeeping", "wave", "波浪", "rao", "运动响应"],
            "maneuvering": ["操纵", "maneuvering", "mmg", "舵", "回转"],
            "stability": ["稳性", "stability", "gm", "横倾", "扶正"],
            "structure": ["结构", "structure", "strength", "强度", "板架"],
            "vibration": ["振动", "vibration", "noise", "噪声", "声"],
            "green_shipping": ["绿色", "green", "eexi", "cii", "emission", "能耗"],
            "ice_navigation": ["冰", "ice", "polar", "破冰", "冰区"],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                return domain
        
        return "unknown"
    
    @property
    def scan_coverage(self) -> float:
        """Calculate scan coverage.
        
        Coverage = fraction of done nodes that were successfully scanned
        AND had some form of limitation/goal content extracted.
        
        This measures how thoroughly the scanner processed the done node corpus.
        """
        if self._scanned_nodes == 0:
            return 0.0
        return self._nodes_with_limitations / self._scanned_nodes
    
    @property
    def scanned_count(self) -> int:
        return self._scanned_nodes
    
    @property
    def nodes_with_limitations_count(self) -> int:
        return self._nodes_with_limitations


class AuditCapabilityScanner:
    """Scans audit capabilities for WITHHELD/NOT_SUPPORTED entries."""
    
    def __init__(self):
        self._scanned_count = 0
        self._gap_count = 0
    
    def scan_capabilities(self, capabilities: List[Dict[str, Any]]) -> List[CapabilityGap]:
        """Scan audit capabilities for gaps.
        
        Args:
            capabilities: List of capability dictionaries with 'name', 'grade', 'description'
            
        Returns:
            List of CapabilityGap objects
        """
        gaps = []
        self._scanned_count = 0
        self._gap_count = 0
        
        for cap in capabilities:
            self._scanned_count += 1
            grade = cap.get("grade", "").upper()
            
            if grade in ("WITHHELD", "NOT_SUPPORTED"):
                self._gap_count += 1
                name = cap.get("name", "")
                description = cap.get("description", "")
                
                # Infer domain from capability name
                domain = self._infer_domain_from_capability(name)
                
                gap = CapabilityGap(
                    capability_name=name,
                    evidence_grade=grade,
                    description=description,
                    physics_domain=domain,
                )
                gaps.append(gap)
        
        return gaps
    
    def _infer_domain_from_capability(self, capability_name: str) -> str:
        """Infer physics domain from capability name."""
        name_lower = capability_name.lower()
        
        if "ice" in name_lower or "冰" in capability_name:
            return "ice_navigation"
        elif "fem" in name_lower or "structure" in name_lower:
            return "structure"
        elif "resistance" in name_lower:
            return "resistance"
        elif "propulsion" in name_lower or "propeller" in name_lower:
            return "propulsion"
        elif "seakeeping" in name_lower or "wave" in name_lower:
            return "seakeeping"
        elif "maneuver" in name_lower:
            return "maneuvering"
        elif "stability" in name_lower:
            return "stability"
        elif "vibration" in name_lower or "noise" in name_lower:
            return "vibration"
        elif "green" in name_lower or "eexi" in name_lower or "cii" in name_lower:
            return "green_shipping"
        
        return "unknown"
    
    @property
    def gap_rate(self) -> float:
        """Calculate gap rate (fraction of capabilities that are gaps)."""
        if self._scanned_count == 0:
            return 0.0
        return self._gap_count / self._scanned_count
    
    @property
    def scanned_count(self) -> int:
        return self._scanned_count
    
    @property
    def gap_count(self) -> int:
        return self._gap_count
