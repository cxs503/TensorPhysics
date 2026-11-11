"""Tests for the residual-driven topic compiler.

Covers:
- Campaign node scanning
- Audit capability scanning
- Proposal generation
- Triple-retrieval deduplication
- Full pipeline integration
- Schema validation
- Configuration handling
"""
import pytest
import json
import os
import tempfile
from typing import List, Dict, Any

from tensorphysics.residual_compiler import (
    ResidualCompiler,
    ProposalSchema,
    PROPOSAL_REQUIRED_FIELDS,
    CampaignNodeScanner,
    AuditCapabilityScanner,
    ProposalGenerator,
    TripleRetrievalDedup,
)
from tensorphysics.residual_compiler.config import CompilerConfig
from tensorphysics.residual_compiler.scanner import Limitation, CapabilityGap


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_campaign_nodes() -> List[Dict[str, Any]]:
    """Sample campaign nodes for testing."""
    return [
        {
            "id": 1001,
            "name": "Holtrop-Mennen 阻力成熟化",
            "status": "done",
            "metadata": {
                "goal": "【物理目标】实现 Holtrop-Mennen 阻力计算。【局限】高频外推未验证",
                "physics_domain": "resistance",
                "acceptance_criteria": "误差 < 5%",
            },
        },
        {
            "id": 1002,
            "name": "螺旋桨 BEMT 验证",
            "status": "done",
            "metadata": {
                "goal": "【物理目标】BEMT 推进效率验证。honest FAIL: K_T 偏差 +41%",
                "physics_domain": "propulsion",
            },
        },
        {
            "id": 1003,
            "name": "耐波性 RAO 计算",
            "status": "done",
            "metadata": {
                "goal": "【物理目标】频域 RAO 计算。未做：非线性效应",
                "physics_domain": "seakeeping",
            },
        },
        {
            "id": 1004,
            "name": "冰阻力 DEM 验证",
            "status": "done",
            "metadata": {
                "goal": "【物理目标】DEM 冰阻力计算。gap: 缺少碎冰场验证",
                "physics_domain": "ice_navigation",
            },
        },
        {
            "id": 1005,
            "name": "稳性计算",
            "status": "running",  # Not done, should be skipped
            "metadata": {
                "goal": "GM 计算",
                "physics_domain": "stability",
            },
        },
    ]


@pytest.fixture
def sample_audit_capabilities() -> List[Dict[str, Any]]:
    """Sample audit capabilities for testing."""
    return [
        {
            "name": "fem.simp.density_filter",
            "grade": "WITHHELD",
            "description": "密度滤波对棋盘格的抑制（无滤波/有滤波柔度比）≥5 倍",
        },
        {
            "name": "fem.simp.min_compliance",
            "grade": "REAL",
            "description": "p=1 一维最小柔度解与解析解误差 ≤2%",
        },
        {
            "name": "ice.resistance.dem_straight_ahead",
            "grade": "WITHHELD",
            "description": "DEM 碎冰场直航冰阻力与 Lindqvist 解析值误差 ≤15%",
        },
        {
            "name": "node.p350w1_n01.undelivered",
            "grade": "NOT_SUPPORTED",
            "description": "P350W1-N01 未交付",
        },
    ]


@pytest.fixture
def sample_node_corpus() -> List[str]:
    """Sample node corpus for dedup testing."""
    return [
        "Holtrop-Mennen 阻力计算成熟化",
        "螺旋桨 BEMT 推进效率验证",
        "频域耐波性 RAO 计算",
        "冰阻力 DEM 直航验证",
        "稳性 GM 计算",
    ]


@pytest.fixture
def sample_proposal_pool() -> List[str]:
    """Sample proposal pool for dedup testing."""
    return [
        "改进 Holtrop 高频外推精度",
        "修复 BEMT K_T 偏差",
        "补充耐波性非线性效应验证",
    ]


@pytest.fixture
def compiler_config() -> CompilerConfig:
    """Default compiler configuration."""
    return CompilerConfig()


# ============================================================================
# Schema Tests
# ============================================================================

class TestProposalSchema:
    """Tests for ProposalSchema."""
    
    def test_schema_has_8_required_fields(self):
        """Test that schema defines 8 required fields."""
        assert len(PROPOSAL_REQUIRED_FIELDS) == 8
    
    def test_valid_proposal(self):
        """Test creating a valid proposal."""
        proposal = ProposalSchema(
            proposal_id="test_001",
            title="Test Proposal",
            physics_domain="resistance",
            physical_goal="Test goal",
            implementation_path="Test path",
            machine_criteria="Test criteria",
            source_node_id="1001",
            priority_score=0.8,
        )
        assert proposal.validate() is True
    
    def test_invalid_proposal_missing_field(self):
        """Test that proposal with missing field is invalid."""
        proposal = ProposalSchema(
            proposal_id="test_002",
            title="",  # Empty title
            physics_domain="resistance",
            physical_goal="Test goal",
            implementation_path="Test path",
            machine_criteria="Test criteria",
            source_node_id="1001",
            priority_score=0.8,
        )
        assert proposal.validate() is False
    
    def test_proposal_to_dict(self):
        """Test converting proposal to dictionary."""
        proposal = ProposalSchema(
            proposal_id="test_003",
            title="Test",
            physics_domain="resistance",
            physical_goal="Goal",
            implementation_path="Path",
            machine_criteria="Criteria",
            source_node_id="1001",
            priority_score=0.75,
        )
        d = proposal.to_dict()
        assert d["proposal_id"] == "test_003"
        assert d["priority_score"] == 0.75
    
    def test_proposal_from_dict(self):
        """Test creating proposal from dictionary."""
        data = {
            "proposal_id": "test_004",
            "title": "Test",
            "physics_domain": "propulsion",
            "physical_goal": "Goal",
            "implementation_path": "Path",
            "machine_criteria": "Criteria",
            "source_node_id": "1002",
            "priority_score": 0.9,
        }
        proposal = ProposalSchema.from_dict(data)
        assert proposal.proposal_id == "test_004"
        assert proposal.validate() is True


# ============================================================================
# Scanner Tests
# ============================================================================

class TestCampaignNodeScanner:
    """Tests for CampaignNodeScanner."""
    
    def test_scan_done_nodes(self, sample_campaign_nodes):
        """Test scanning done nodes."""
        scanner = CampaignNodeScanner()
        limitations = scanner.scan_nodes(sample_campaign_nodes)
        
        # Should find limitations in done nodes
        assert len(limitations) > 0
        assert scanner.scanned_count == 4  # 4 done nodes
    
    def test_skip_non_done_nodes(self, sample_campaign_nodes):
        """Test that non-done nodes are skipped."""
        scanner = CampaignNodeScanner()
        scanner.scan_nodes(sample_campaign_nodes)
        
        # Should only scan done nodes
        assert scanner.scanned_count == 4
    
    def test_extract_limitations_from_goal(self, sample_campaign_nodes):
        """Test extracting limitations from goal field."""
        scanner = CampaignNodeScanner()
        limitations = scanner.scan_nodes(sample_campaign_nodes)
        
        # Should find limitations with keywords
        lim_types = [lim.limitation_type for lim in limitations]
        assert any(t in ["limitation", "honest_fail", "untested", "gap"] for t in lim_types)
    
    def test_scan_coverage_calculation(self, sample_campaign_nodes):
        """Test scan coverage calculation."""
        scanner = CampaignNodeScanner()
        scanner.scan_nodes(sample_campaign_nodes)
        
        coverage = scanner.scan_coverage
        assert 0.0 <= coverage <= 1.0
    
    def test_infer_physics_domain(self):
        """Test physics domain inference."""
        scanner = CampaignNodeScanner()
        
        node = {
            "id": 9999,
            "name": "冰阻力计算",
            "status": "done",
            "metadata": {"goal": "计算冰区航行阻力"},
        }
        limitations = scanner.scan_nodes([node])
        
        if limitations:
            assert limitations[0].physics_domain == "ice_navigation"


class TestAuditCapabilityScanner:
    """Tests for AuditCapabilityScanner."""
    
    def test_scan_withheld_capabilities(self, sample_audit_capabilities):
        """Test scanning WITHHELD capabilities."""
        scanner = AuditCapabilityScanner()
        gaps = scanner.scan_capabilities(sample_audit_capabilities)
        
        # Should find WITHHELD and NOT_SUPPORTED
        assert len(gaps) >= 2
        grades = [g.evidence_grade for g in gaps]
        assert "WITHHELD" in grades or "NOT_SUPPORTED" in grades
    
    def test_skip_real_capabilities(self, sample_audit_capabilities):
        """Test that REAL capabilities are skipped."""
        scanner = AuditCapabilityScanner()
        gaps = scanner.scan_capabilities(sample_audit_capabilities)
        
        # Should not include REAL capabilities
        grades = [g.evidence_grade for g in gaps]
        assert "REAL" not in grades
    
    def test_gap_rate_calculation(self, sample_audit_capabilities):
        """Test gap rate calculation."""
        scanner = AuditCapabilityScanner()
        scanner.scan_capabilities(sample_audit_capabilities)
        
        gap_rate = scanner.gap_rate
        assert 0.0 <= gap_rate <= 1.0


# ============================================================================
# Generator Tests
# ============================================================================

class TestProposalGenerator:
    """Tests for ProposalGenerator."""
    
    def test_generate_from_limitations(self, sample_campaign_nodes):
        """Test generating proposals from limitations."""
        scanner = CampaignNodeScanner()
        limitations = scanner.scan_nodes(sample_campaign_nodes)
        
        generator = ProposalGenerator()
        proposals = generator.generate_from_limitations(limitations)
        
        assert len(proposals) > 0
        assert all(p.validate() for p in proposals)
    
    def test_generate_from_gaps(self, sample_audit_capabilities):
        """Test generating proposals from capability gaps."""
        scanner = AuditCapabilityScanner()
        gaps = scanner.scan_capabilities(sample_audit_capabilities)
        
        generator = ProposalGenerator()
        proposals = generator.generate_from_gaps(gaps)
        
        assert len(proposals) > 0
        assert all(p.validate() for p in proposals)
    
    def test_generation_rate(self, sample_campaign_nodes, sample_audit_capabilities):
        """Test generation rate calculation."""
        scanner = CampaignNodeScanner()
        limitations = scanner.scan_nodes(sample_campaign_nodes)
        
        generator = ProposalGenerator()
        generator.generate_from_limitations(limitations)
        
        rate = generator.generation_rate
        assert 0.0 <= rate <= 1.0
    
    def test_priority_score_range(self, sample_campaign_nodes):
        """Test that priority scores are in valid range."""
        scanner = CampaignNodeScanner()
        limitations = scanner.scan_nodes(sample_campaign_nodes)
        
        generator = ProposalGenerator()
        proposals = generator.generate_from_limitations(limitations)
        
        for p in proposals:
            assert 0.0 <= p.priority_score <= 1.0


# ============================================================================
# Dedup Tests
# ============================================================================

class TestTripleRetrievalDedup:
    """Tests for TripleRetrievalDedup."""
    
    def test_dedup_removes_duplicates(
        self, sample_campaign_nodes, sample_node_corpus, sample_proposal_pool
    ):
        """Test that dedup removes duplicate proposals."""
        scanner = CampaignNodeScanner()
        limitations = scanner.scan_nodes(sample_campaign_nodes)
        
        generator = ProposalGenerator()
        proposals = generator.generate_from_limitations(limitations)
        
        dedup = TripleRetrievalDedup(
            node_corpus=sample_node_corpus,
            proposal_pool=sample_proposal_pool,
        )
        unique = dedup.deduplicate(proposals)
        
        # Should have fewer or equal proposals after dedup
        assert len(unique) <= len(proposals)
    
    def test_dedup_rate_calculation(
        self, sample_campaign_nodes, sample_node_corpus
    ):
        """Test dedup rate calculation."""
        scanner = CampaignNodeScanner()
        limitations = scanner.scan_nodes(sample_campaign_nodes)
        
        generator = ProposalGenerator()
        proposals = generator.generate_from_limitations(limitations)
        
        dedup = TripleRetrievalDedup(node_corpus=sample_node_corpus)
        dedup.deduplicate(proposals)
        
        rate = dedup.dedup_rate
        assert 0.0 <= rate <= 1.0
    
    def test_empty_corpus_no_crash(self, sample_campaign_nodes):
        """Test that empty corpus doesn't crash."""
        scanner = CampaignNodeScanner()
        limitations = scanner.scan_nodes(sample_campaign_nodes)
        
        generator = ProposalGenerator()
        proposals = generator.generate_from_limitations(limitations)
        
        dedup = TripleRetrievalDedup(node_corpus=[], proposal_pool=[])
        unique = dedup.deduplicate(proposals)
        
        # Should return all proposals when no corpus
        assert len(unique) == len(proposals)


# ============================================================================
# Integration Tests
# ============================================================================

class TestResidualCompiler:
    """Integration tests for the full compiler pipeline."""
    
    def test_full_pipeline(
        self,
        sample_campaign_nodes,
        sample_audit_capabilities,
        sample_node_corpus,
        sample_proposal_pool,
    ):
        """Test the full compilation pipeline."""
        compiler = ResidualCompiler()
        report = compiler.compile(
            campaign_nodes=sample_campaign_nodes,
            audit_capabilities=sample_audit_capabilities,
            node_corpus=sample_node_corpus,
            proposal_pool=sample_proposal_pool,
        )
        
        # Check report structure
        assert report.timestamp
        assert report.total_nodes_scanned > 0
        assert report.final_proposal_count >= 0
        assert len(report.proposals) == report.final_proposal_count
    
    def test_all_proposals_valid(
        self,
        sample_campaign_nodes,
        sample_audit_capabilities,
    ):
        """Test that all output proposals are valid."""
        compiler = ResidualCompiler()
        report = compiler.compile(
            campaign_nodes=sample_campaign_nodes,
            audit_capabilities=sample_audit_capabilities,
        )
        
        for prop_data in report.proposals:
            proposal = ProposalSchema.from_dict(prop_data)
            assert proposal.validate() is True
    
    def test_proposals_sorted_by_priority(
        self,
        sample_campaign_nodes,
        sample_audit_capabilities,
    ):
        """Test that proposals are sorted by priority."""
        compiler = ResidualCompiler()
        report = compiler.compile(
            campaign_nodes=sample_campaign_nodes,
            audit_capabilities=sample_audit_capabilities,
        )
        
        if len(report.proposals) > 1:
            priorities = [p["priority_score"] for p in report.proposals]
            assert priorities == sorted(priorities, reverse=True)
    
    def test_save_and_load_report(
        self,
        sample_campaign_nodes,
        sample_audit_capabilities,
    ):
        """Test saving and loading report."""
        compiler = ResidualCompiler()
        compiler.compile(
            campaign_nodes=sample_campaign_nodes,
            audit_capabilities=sample_audit_capabilities,
        )
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            compiler.save_report(temp_path)
            
            with open(temp_path, "r") as f:
                loaded = json.load(f)
            
            assert "timestamp" in loaded
            assert "proposals" in loaded
            assert "criteria_met" in loaded
        finally:
            os.unlink(temp_path)
    
    def test_config_compatibility(self):
        """Test configuration compatibility with proposal_seeding.json."""
        config = CompilerConfig()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            loaded = CompilerConfig.from_file(temp_path)
            
            assert loaded.scan_coverage_threshold == config.scan_coverage_threshold
            assert loaded.generation_rate_threshold == config.generation_rate_threshold
        finally:
            os.unlink(temp_path)
    
    def test_criteria_met_reporting(
        self,
        sample_campaign_nodes,
        sample_audit_capabilities,
    ):
        """Test that criteria_met is properly reported."""
        compiler = ResidualCompiler()
        report = compiler.compile(
            campaign_nodes=sample_campaign_nodes,
            audit_capabilities=sample_audit_capabilities,
        )
        
        assert "scan_coverage_ge_95pct" in report.criteria_met
        assert "generation_rate_ge_80pct" in report.criteria_met
        assert "dedup_recall_ge_80pct" in report.criteria_met
        assert "dedup_precision_ge_85pct" in report.criteria_met
        assert "all_proposals_valid" in report.criteria_met
        assert "proposals_have_8_fields" in report.criteria_met


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_input(self):
        """Test with empty input."""
        compiler = ResidualCompiler()
        report = compiler.compile(
            campaign_nodes=[],
            audit_capabilities=[],
        )
        
        assert report.total_nodes_scanned == 0
        assert report.final_proposal_count == 0
    
    def test_malformed_node_metadata(self):
        """Test with malformed node metadata."""
        nodes = [
            {
                "id": 9999,
                "name": "Test",
                "status": "done",
                "metadata": "not a dict",  # Malformed
            },
        ]
        
        scanner = CampaignNodeScanner()
        limitations = scanner.scan_nodes(nodes)
        
        # Should handle gracefully
        assert isinstance(limitations, list)
    
    def test_missing_metadata_fields(self):
        """Test with missing metadata fields."""
        nodes = [
            {
                "id": 9999,
                "name": "Test",
                "status": "done",
                "metadata": {},  # Empty metadata
            },
        ]
        
        scanner = CampaignNodeScanner()
        limitations = scanner.scan_nodes(nodes)
        
        # Should handle gracefully
        assert isinstance(limitations, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
