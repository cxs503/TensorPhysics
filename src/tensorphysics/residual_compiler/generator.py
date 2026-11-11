"""Proposal generator: creates candidate debt proposals from limitations and gaps."""
from typing import List, Dict, Any, Optional
from .schema import ProposalSchema
from .scanner import Limitation, CapabilityGap
from .config import CompilerConfig
import hashlib
import re


class ProposalGenerator:
    """Generates candidate proposal drafts from limitations and capability gaps."""
    
    def __init__(self, config: Optional[CompilerConfig] = None):
        self.config = config or CompilerConfig()
        self._generated_count = 0
        self._source_count = 0
    
    def generate_from_limitations(
        self, limitations: List[Limitation]
    ) -> List[ProposalSchema]:
        """Generate proposals from campaign node limitations.
        
        Args:
            limitations: List of Limitation objects
            
        Returns:
            List of ProposalSchema objects
        """
        proposals = []
        self._source_count = len(limitations)
        self._generated_count = 0
        
        for lim in limitations:
            proposal = self._create_proposal_from_limitation(lim)
            if proposal and proposal.validate():
                proposals.append(proposal)
                self._generated_count += 1
        
        return proposals
    
    def generate_from_gaps(
        self, gaps: List[CapabilityGap]
    ) -> List[ProposalSchema]:
        """Generate proposals from audit capability gaps.
        
        Args:
            gaps: List of CapabilityGap objects
            
        Returns:
            List of ProposalSchema objects
        """
        proposals = []
        self._source_count += len(gaps)
        
        for gap in gaps:
            proposal = self._create_proposal_from_gap(gap)
            if proposal and proposal.validate():
                proposals.append(proposal)
                self._generated_count += 1
        
        return proposals
    
    def _create_proposal_from_limitation(
        self, lim: Limitation
    ) -> Optional[ProposalSchema]:
        """Create a proposal from a limitation."""
        # Generate proposal ID
        proposal_id = self._generate_id(f"lim_{lim.node_id}_{lim.limitation_type}")
        
        # Generate title from limitation text
        title = self._generate_title(lim.limitation_text, lim.node_name)
        
        # Generate physical goal
        physical_goal = self._generate_physical_goal(lim)
        
        # Generate implementation path
        impl_path = self._generate_implementation_path(lim)
        
        # Generate machine criteria
        machine_criteria = self._generate_machine_criteria(lim)
        
        # Calculate priority score
        priority = self._calculate_priority(lim)
        
        proposal = ProposalSchema(
            proposal_id=proposal_id,
            title=title,
            physics_domain=lim.physics_domain,
            physical_goal=physical_goal,
            implementation_path=impl_path,
            machine_criteria=machine_criteria,
            source_node_id=str(lim.node_id),
            priority_score=priority,
            source_type="limitation",
            evidence_grade=lim.evidence_grade,
            notes=f"Extracted from node {lim.node_id} ({lim.node_name[:50]}...)",
        )
        
        return proposal if proposal.validate() else None
    
    def _create_proposal_from_gap(
        self, gap: CapabilityGap
    ) -> Optional[ProposalSchema]:
        """Create a proposal from a capability gap."""
        # Generate proposal ID
        proposal_id = self._generate_id(f"gap_{gap.capability_name}")
        
        # Generate title
        title = f"实现 {gap.capability_name} 能力"
        
        # Generate physical goal from description
        physical_goal = self._generate_goal_from_description(gap.description, gap.capability_name)
        
        # Generate implementation path
        impl_path = self._generate_impl_from_description(gap.description, gap.capability_name)
        
        # Generate machine criteria
        machine_criteria = self._generate_criteria_from_description(gap.description)
        
        # Calculate priority score
        priority = self._calculate_priority_from_gap(gap)
        
        proposal = ProposalSchema(
            proposal_id=proposal_id,
            title=title,
            physics_domain=gap.physics_domain,
            physical_goal=physical_goal,
            implementation_path=impl_path,
            machine_criteria=machine_criteria,
            source_node_id="audit_capability",
            priority_score=priority,
            source_type="capability_gap",
            source_capability=gap.capability_name,
            evidence_grade=gap.evidence_grade,
            notes=f"From audit capability: {gap.capability_name}",
        )
        
        return proposal if proposal.validate() else None
    
    def _generate_id(self, seed: str) -> str:
        """Generate a unique proposal ID."""
        hash_input = f"{seed}_{id(self)}".encode()
        return f"prop_{hashlib.md5(hash_input).hexdigest()[:12]}"
    
    def _generate_title(self, limitation_text: str, node_name: str) -> str:
        """Generate a proposal title from limitation text."""
        # Clean up the text
        text = limitation_text.strip()
        if len(text) > 100:
            text = text[:100] + "..."
        
        # Try to extract a meaningful title
        # Look for key phrases
        for keyword in ["局限", "limitation", "未做", "FAIL", "gap"]:
            if keyword in text:
                idx = text.find(keyword)
                start = max(0, idx - 30)
                end = min(len(text), idx + 70)
                title = text[start:end].strip()
                if title:
                    return f"解决: {title}"
        
        # Fallback to node name
        return f"改进: {node_name[:80]}"
    
    def _generate_physical_goal(self, lim: Limitation) -> str:
        """Generate physical goal from limitation."""
        text = lim.limitation_text
        
        # Try to extract physical goal from context
        if "物理目标" in text:
            idx = text.find("物理目标")
            end = min(len(text), idx + 300)
            return text[idx:end].strip()
        
        # Generate from limitation type
        if lim.limitation_type == "honest_fail":
            return f"修复 {lim.node_name} 中的失败判据，使误差满足验收标准"
        elif lim.limitation_type == "untested":
            return f"补充 {lim.node_name} 中未测试的功能验证"
        elif lim.limitation_type == "gap":
            return f"填补 {lim.node_name} 中的能力缺口"
        else:
            return f"改进 {lim.node_name} 中的局限性"
    
    def _generate_implementation_path(self, lim: Limitation) -> str:
        """Generate implementation path from limitation."""
        text = lim.limitation_text
        
        # Try to extract implementation path from context
        if "实现路径" in text:
            idx = text.find("实现路径")
            end = min(len(text), idx + 400)
            return text[idx:end].strip()
        
        # Generate generic implementation path
        return f"(1) 分析 {lim.node_name} 的局限性根因; (2) 设计修复方案; (3) 实现并验证; (4) 更新验收判据"
    
    def _generate_machine_criteria(self, lim: Limitation) -> str:
        """Generate machine-verifiable criteria from limitation."""
        # Generate criteria based on limitation type
        if lim.limitation_type == "honest_fail":
            return "误差 < 3% 或满足节点原始判据; pytest 通过"
        elif lim.limitation_type == "untested":
            return "新增测试用例覆盖未测试功能; pytest ≥ 5 例"
        elif lim.limitation_type == "gap":
            return "能力从 WITHHELD 升级为 REAL; 有解析锚或实验对标"
        else:
            return "局限性消除; 验收判据全部 PASS"
    
    def _generate_goal_from_description(
        self, description: str, capability_name: str
    ) -> str:
        """Generate physical goal from capability description."""
        if description:
            # Extract first sentence or key phrase
            sentences = re.split(r'[。.;；]', description)
            if sentences:
                return sentences[0].strip()
        
        return f"实现 {capability_name} 能力的真实算法，从 WITHHELD/NOT_SUPPORTED 升级为 REAL"
    
    def _generate_impl_from_description(
        self, description: str, capability_name: str
    ) -> str:
        """Generate implementation path from capability description."""
        return f"(1) 分析 {capability_name} 的物理模型; (2) 实现真实算法替代 MOCK/STUB; (3) 建立解析锚或实验对标; (4) 验证误差 < 15%"
    
    def _generate_criteria_from_description(self, description: str) -> str:
        """Generate machine criteria from capability description."""
        # Try to extract criteria from description
        if "误差" in description:
            # Extract error threshold
            match = re.search(r'误差\s*[≤<]\s*(\d+)%', description)
            if match:
                threshold = match.group(1)
                return f"与参考值误差 ≤ {threshold}%; pytest 通过"
        
        return "能力证据等级从 WITHHELD/NOT_SUPPORTED 升级为 REAL; 有可复现的验证用例"
    
    def _calculate_priority(self, lim: Limitation) -> float:
        """Calculate priority score for a limitation-based proposal."""
        weights = self.config.priority_weights
        domain_crit = self.config.domain_criticality.get(lim.physics_domain, 0.5)
        
        # Evidence grade score
        grade_score = {"WITHHELD": 0.7, "MOCK": 0.8, "STUB": 0.9, "NOT_SUPPORTED": 1.0}.get(
            lim.evidence_grade, 0.5
        )
        
        # Gap severity based on limitation type
        severity_score = {
            "honest_fail": 0.9,
            "gap": 0.8,
            "untested": 0.7,
            "limitation": 0.6,
        }.get(lim.limitation_type, 0.5)
        
        # Dependency count (placeholder - would need actual dependency data)
        dep_score = 0.5
        
        priority = (
            weights.get("evidence_grade", 0.3) * grade_score +
            weights.get("domain_criticality", 0.3) * domain_crit +
            weights.get("gap_severity", 0.2) * severity_score +
            weights.get("dependency_count", 0.2) * dep_score
        )
        
        return round(priority, 3)
    
    def _calculate_priority_from_gap(self, gap: CapabilityGap) -> float:
        """Calculate priority score for a capability gap-based proposal."""
        weights = self.config.priority_weights
        domain_crit = self.config.domain_criticality.get(gap.physics_domain, 0.5)
        
        # Evidence grade score
        grade_score = {"WITHHELD": 0.7, "NOT_SUPPORTED": 1.0}.get(gap.evidence_grade, 0.5)
        
        # Gap severity (all capability gaps are high severity)
        severity_score = 0.8
        
        # Dependency count (placeholder)
        dep_score = 0.5
        
        priority = (
            weights.get("evidence_grade", 0.3) * grade_score +
            weights.get("domain_criticality", 0.3) * domain_crit +
            weights.get("gap_severity", 0.2) * severity_score +
            weights.get("dependency_count", 0.2) * dep_score
        )
        
        return round(priority, 3)
    
    @property
    def generation_rate(self) -> float:
        """Calculate generation rate (fraction of sources that generated proposals)."""
        if self._source_count == 0:
            return 0.0
        return self._generated_count / self._source_count
    
    @property
    def generated_count(self) -> int:
        return self._generated_count
    
    @property
    def source_count(self) -> int:
        return self._source_count
