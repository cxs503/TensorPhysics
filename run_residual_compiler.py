#!/usr/bin/env python3
"""Run the residual compiler against actual campaign data and produce artifacts."""
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tensorphysics.residual_compiler import ResidualCompiler
from tensorphysics.residual_compiler.config import CompilerConfig


def load_campaign_nodes():
    """Load campaign nodes from work/api data."""
    path = os.path.join(os.path.dirname(__file__), "work/api/nodes_top100plans.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_audit_capabilities():
    """Load audit capabilities (simulated from audit_capabilities tool output)."""
    # These are the actual capabilities from the audit_capabilities tool
    return [
        {
            "name": "fem.simp.density_filter",
            "grade": "WITHHELD",
            "description": "密度滤波对棋盘格的抑制（无滤波/有滤波柔度比）≥5 倍（节点判据 (e) 声明阈）",
        },
        {
            "name": "fem.simp.min_compliance",
            "grade": "REAL",
            "description": "p=1 一维最小柔度解与解析解 C*=F²L/(E₀V*) 误差 ≤2%",
        },
        {
            "name": "fem.simp.sensitivity_consistency",
            "grade": "WITHHELD",
            "description": "解析灵敏度与有限差分一致（绝对误差 ≤1e-4，节点判据 (f) 声明阈）",
        },
        {
            "name": "ice.maneuver.energy_ledger",
            "grade": "REAL",
            "description": "转弯工况动能+冰阻力做功+推力做功的能量账本残差 <10%",
        },
        {
            "name": "ice.maneuver.turning_radius",
            "grade": "WITHHELD",
            "description": "δ=20°/C_ice=0.5 稳态转弯半径与经验公式误差 ≤20%",
        },
        {
            "name": "ice.resistance.dem_straight_ahead",
            "grade": "WITHHELD",
            "description": "DEM 碎冰场直航冰阻力 R_ice 与 Lindqvist 解析值误差 ≤15%（V=3m/s）",
        },
        {
            "name": "node.p350w1_n01.undelivered",
            "grade": "NOT_SUPPORTED",
            "description": "P350W1-N01（ship-performance-platform）未交付：infra x5 agent_lost",
        },
        {
            "name": "node.p350w1_n04.undelivered",
            "grade": "NOT_SUPPORTED",
            "description": "P350W1-N04（TensorPhysics）未交付：infra x5 agent_lost",
        },
        {
            "name": "node.p350w1_n07.undelivered",
            "grade": "WITHHELD",
            "description": "P350W1-N07（TensorPlatform）采集时仍在执行（待验）",
        },
        {
            "name": "platform.evidence_grade.engine",
            "grade": "REAL",
            "description": "证据等级引擎对 30 例带标注对抗夹具的判定与标注一致（一致率 30/30）",
        },
        {
            "name": "structure.ice.plate_beam_elastic",
            "grade": "REAL",
            "description": "板-梁组合结构弹性刚度与解析（板弯曲+梁弯曲并联）误差 ≤5%",
        },
        {
            "name": "structure.ice.plate_beam_ultimate",
            "grade": "REAL",
            "description": "渐进 collapse 极限承载力与刚塑性屈服线解析解误差 ≤5%",
        },
    ]


def build_node_corpus(nodes):
    """Build node name corpus from campaign nodes."""
    return [n.get("name", "") for n in nodes if n.get("name")]


def build_proposal_pool():
    """Build proposal pool from existing proposals (if any)."""
    # In a real system, this would load from a proposal database
    # For now, return empty list
    return []


def main():
    """Run the residual compiler."""
    print("=" * 60)
    print("Residual-Driven Topic Compiler (p474w1-n03)")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading campaign nodes...")
    nodes = load_campaign_nodes()
    print(f"  Loaded {len(nodes)} campaign nodes")
    
    print("\n[2/5] Loading audit capabilities...")
    capabilities = load_audit_capabilities()
    print(f"  Loaded {len(capabilities)} capabilities")
    
    # Build corpora for dedup
    print("\n[3/5] Building dedup corpora...")
    node_corpus = build_node_corpus(nodes)
    proposal_pool = build_proposal_pool()
    print(f"  Node corpus: {len(node_corpus)} entries")
    print(f"  Proposal pool: {len(proposal_pool)} entries")
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "proposal_seeding.json")
    if os.path.exists(config_path):
        config = CompilerConfig.from_file(config_path)
        print(f"\n  Loaded config from {config_path}")
    else:
        config = CompilerConfig()
        config.save(config_path)
        print(f"\n  Created default config at {config_path}")
    
    # Run compiler
    print("\n[4/5] Running compilation pipeline...")
    compiler = ResidualCompiler(config)
    report = compiler.compile(
        campaign_nodes=nodes,
        audit_capabilities=capabilities,
        node_corpus=node_corpus,
        proposal_pool=proposal_pool,
    )
    
    # Print summary
    print("\n[5/5] Compilation complete!")
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Nodes scanned:              {report.total_nodes_scanned}")
    print(f"  Nodes with limitations:     {report.nodes_with_limitations}")
    print(f"  Scan coverage:              {report.scan_coverage:.1%}")
    print(f"  Capabilities scanned:       {report.total_capabilities_scanned}")
    print(f"  Capability gaps found:      {report.capability_gaps_found}")
    print(f"  Limitations extracted:      {report.total_limitations_extracted}")
    print(f"  Gaps extracted:             {report.total_gaps_extracted}")
    print(f"  Proposals generated:        {report.total_proposals_generated}")
    print(f"  Generation rate:            {report.generation_rate:.1%}")
    print(f"  Duplicates removed:         {report.duplicates_removed}")
    print(f"  Final proposals:            {report.final_proposal_count}")
    print(f"  Dedup recall (est.):        {report.dedup_recall:.1%}")
    print(f"  Dedup precision (est.):     {report.dedup_precision:.1%}")
    
    print(f"\n{'='*60}")
    print("CRITERIA CHECK")
    print(f"{'='*60}")
    for criterion, met in report.criteria_met.items():
        status = "✓ PASS" if met else "✗ FAIL"
        print(f"  {criterion}: {status}")
    
    # Save artifacts
    output_dir = os.path.join(os.path.dirname(__file__), "artifacts", "residual_compiler")
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "compiler_report.json")
    compiler.save_report(report_path)
    print(f"\n  Report saved to: {report_path}")
    
    # Print top proposals
    print(f"\n{'='*60}")
    print("TOP 10 CANDIDATE PROPOSALS (by priority)")
    print(f"{'='*60}")
    for i, prop in enumerate(report.proposals[:10], 1):
        print(f"\n  #{i} [Priority: {prop['priority_score']:.3f}]")
        print(f"  Title: {prop['title'][:80]}")
        print(f"  Domain: {prop['physics_domain']}")
        print(f"  Source: {prop['source_type']} (node: {prop['source_node_id']})")
        print(f"  Goal: {prop['physical_goal'][:100]}...")
    
    return report


if __name__ == "__main__":
    main()
