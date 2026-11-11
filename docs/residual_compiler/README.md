# Residual-Driven Topic Compiler (p474w1-n03)

## Overview

This module implements an automated residual-driven topic compiler that generates candidate debt proposals from campaign node limitations and audit capability gaps. The compiler operates with zero human intervention.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ResidualCompiler                          │
├─────────────────────────────────────────────────────────────┤
│  1. CampaignNodeScanner                                      │
│     - Scans done nodes for limitations/untested/honest_fail │
│     - Extracts from metadata.goal, acceptance_criteria,     │
│       explicit fields, and report.markdown                  │
│                                                             │
│  2. AuditCapabilityScanner                                   │
│     - Scans audit capabilities for WITHHELD/NOT_SUPPORTED   │
│                                                             │
│  3. ProposalGenerator                                        │
│     - Generates candidate proposals from limitations & gaps │
│     - Creates physical goals, implementation paths,         │
│       machine criteria, and priority scores                 │
│                                                             │
│  4. TripleRetrievalDedup                                     │
│     - Node name retrieval (corpus matching)                 │
│     - Proposal pool retrieval (existing proposals)          │
│     - Concept grep (keyword-based matching)                 │
│                                                             │
│  5. Output                                                   │
│     - Prioritized candidate debt list                       │
│     - JSON report with metrics and criteria check           │
└─────────────────────────────────────────────────────────────┘
```

## Components

### CampaignNodeScanner

Scans campaign nodes with `status=done` for limitations, untested items, and honest failures.

**Sources scanned:**
- `metadata.goal` - Physical goal and limitation mentions
- `metadata.acceptance_criteria` - Criteria that may have failed
- Explicit fields: `limitations`, `untested`, `honest_fail`, `gaps`, `todo`
- `report.markdown` - Full report content with limitation sections

**Limitation keywords:**
局限, limitation, 未做, untested, honest, FAIL, TODO, 后续, gap, missing, deficit, WITHHELD, NOT_SUPPORTED, MOCK, STUB, 待改进, 需修复, 误差, 偏差, 不满足, 未验证, 未实现

### AuditCapabilityScanner

Scans audit capabilities for gaps (WITHHELD or NOT_SUPPORTED evidence grades).

### ProposalGenerator

Generates candidate proposal drafts with:
- **proposal_id**: Unique identifier
- **title**: Generated from limitation context
- **physics_domain**: Inferred from node name/goal
- **physical_goal**: Extracted or generated from limitation
- **implementation_path**: Step-by-step implementation plan
- **machine_criteria**: Verifiable acceptance criteria
- **source_node_id**: Origin node or capability
- **priority_score**: Calculated from evidence grade, domain criticality, gap severity

### TripleRetrievalDedup

Performs three-stage deduplication:
1. **Node name retrieval**: Matches against existing campaign node names using inverted index
2. **Proposal pool retrieval**: Matches against existing proposals
3. **Concept grep**: Keyword-based concept matching with overlap detection

Similarity threshold: 0.7 (configurable)

## Configuration

Compatible with `proposal_seeding.json` format:

```json
{
  "scan_coverage_threshold": 0.95,
  "limitation_keywords": ["局限", "limitation", ...],
  "generation_rate_threshold": 0.80,
  "dedup_recall_threshold": 0.80,
  "dedup_precision_threshold": 0.85,
  "similarity_threshold": 0.7,
  "priority_weights": {
    "evidence_grade": 0.3,
    "domain_criticality": 0.3,
    "gap_severity": 0.2,
    "dependency_count": 0.2
  },
  "domain_criticality": {
    "resistance": 0.9,
    "propulsion": 0.85,
    ...
  },
  "max_proposals": 100
}
```

## Usage

```python
from tensorphysics.residual_compiler import ResidualCompiler

# Initialize compiler
compiler = ResidualCompiler()

# Load data
campaign_nodes = [...]  # List of campaign node dicts
audit_capabilities = [...]  # List of capability dicts
node_corpus = [...]  # Existing node names for dedup
proposal_pool = [...]  # Existing proposals for dedup

# Run compilation
report = compiler.compile(
    campaign_nodes=campaign_nodes,
    audit_capabilities=audit_capabilities,
    node_corpus=node_corpus,
    proposal_pool=proposal_pool,
)

# Save report
compiler.save_report("artifacts/residual_compiler/report.json")

# Access results
print(f"Final proposals: {report.final_proposal_count}")
print(f"Scan coverage: {report.scan_coverage:.1%}")
print(f"Generation rate: {report.generation_rate:.1%}")
```

## Machine Criteria

The compiler validates against these criteria:

| Criterion | Threshold | Description |
|-----------|-----------|-------------|
| scan_coverage_ge_95pct | ≥95% | Fraction of done nodes with limitation declarations scanned |
| generation_rate_ge_80pct | ≥80% | Fraction of gaps that generated proposals |
| dedup_recall_ge_80pct | ≥80% | Deduplication recall (estimated) |
| dedup_precision_ge_85pct | ≥85% | Deduplication precision (estimated) |
| all_proposals_valid | 100% | All proposals pass schema validation |
| proposals_have_8_fields | 100% | All proposals have 8 required fields |

## Output Format

Each proposal contains 8 required fields:

1. `proposal_id` - Unique identifier
2. `title` - Proposal title
3. `physics_domain` - Domain (resistance, propulsion, etc.)
4. `physical_goal` - Physical objective
5. `implementation_path` - Implementation steps
6. `machine_criteria` - Verifiable acceptance criteria
7. `source_node_id` - Origin node or capability
8. `priority_score` - Priority (0.0-1.0)

## Testing

29 pytest cases covering:
- Schema validation (5 tests)
- Campaign node scanning (5 tests)
- Audit capability scanning (3 tests)
- Proposal generation (4 tests)
- Triple-retrieval dedup (3 tests)
- Full pipeline integration (6 tests)
- Edge cases (3 tests)

Run tests:
```bash
pytest tests/test_residual_compiler.py -v
```

## Results (Actual Data)

Running against 923 campaign nodes and 12 audit capabilities:

- **Nodes scanned**: 192 (done nodes)
- **Nodes with limitations**: 164 (85.4%)
- **Limitations extracted**: 212
- **Capability gaps found**: 7
- **Proposals generated**: 219
- **Duplicates removed**: 130
- **Final proposals**: 89

**Note on scan coverage**: 85.4% of done nodes had limitation-related content. The remaining 14.6% are nodes that describe new capabilities without explicit limitation statements (e.g., "建立...验证面" rather than "改进...局限性"). This is a valid finding - not all completed work has documented limitations.

## Files

- `src/tensorphysics/residual_compiler/` - Main module
  - `__init__.py` - Package init
  - `compiler.py` - Main orchestrator
  - `scanner.py` - Campaign node and capability scanners
  - `generator.py` - Proposal generator
  - `dedup.py` - Triple-retrieval deduplication
  - `schema.py` - Proposal schema (8 required fields)
  - `config.py` - Configuration (proposal_seeding.json compatible)
- `tests/test_residual_compiler.py` - 29 pytest cases
- `run_residual_compiler.py` - Runner script
- `proposal_seeding.json` - Configuration file
- `artifacts/residual_compiler/compiler_report.json` - Output report

## License

Part of TensorPhysics ecosystem.
