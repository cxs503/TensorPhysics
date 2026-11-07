# P16 全战役集成验收门 (P16-G1) — 验收报告

- schema: `p16_campaign_gate_verdict.v1`
- 生成时间 (UTC): `2026-11-07T07:03:48.049352+00:00`
- 战役计划: `P16·求解器成熟化与船舶新域拓展(Plan15提案播种)` (plan_id 源 http://127.0.0.1:8010/api/v1/tensor/campaigns/plans/33, status `active`)
- 物理域: `['LBM', 'FEM', 'DEM', 'ship-performance']`
- 耦合要求: `{'cross_solver_coupling': 'LBM-wind→ship-platform, FEM-vibration→BEM-radiation, DEM-clump→ice-field'}`

## 1. 门裁决总览

| 判据 | 内容 | 裁决 | 说明 |
|---|---|---|---|
| (a) | N1-N8 全部节点 acceptance_criteria 通过 | ⚪ WITHHELD | 0/8 nodes PASS, 0 FAIL, 8 WITHHELD; 2 delivery record(s) accepted by the platform |
| (b) | 跨域接口契约验证（格式/类型/量纲一致） | ❌ FAIL | 2 contract checks run (0 PASS / 2 FAIL); missing producer payloads: ['wind_coefficients.json', 'wet_modal_handoff.json'] |
| (c) | P15 回归基线 11 节点 |Δ| < 1% | ❌ FAIL | 0/11 P15 nodes PASS; measured probes: 3/11; max |delta| = 1.351e-02 |
| (d) | 战役级验收报告生成（数字可回溯 artifacts JSON） | ✅ PASS | gate_verdict.json + docs/P16_CAMPAIGN_INTEGRATION_GATE.md emitted with sha256 provenance of every input artifact |

**总裁决 (overall_status): `FAIL`**

阻塞项 (blocking):

- `a:WITHHELD:N1-N8 全部节点 acceptance_criteria 通过`
- `b:FAIL:跨域接口契约验证（格式/类型/量纲一致）`
- `c:FAIL:P15 回归基线 11 节点 |Δ| < 1%`

## 2. 判据 (a) N1-N8 节点验收

| 节点 | 仓库 | mission | 节点状态 | 交付 | 裁决 | 通过/总判据 |
|---|---|---|---|---|---|---|
| P16-N1 | TensorLBM | 5184 | in-flight | no | ⚪ WITHHELD | 0/8 |
| P16-N2 | TensorFEM | 5197 | in-flight | no | ⚪ WITHHELD | 0/8 |
| P16-N3 | TensorDEM | 5186 | in-flight | no | ⚪ WITHHELD | 0/7 |
| P16-N4 | ship-performance-platform | 5187 | done | yes | ⚪ WITHHELD | 0/6 |
| P16-N5 | ship-performance-platform | 5188 | done | yes | ⚪ WITHHELD | 0/6 |
| P16-N6 | ship-performance-platform | 5189 | in-flight | no | ⚪ WITHHELD | 0/6 |
| P16-N7 | TensorLBM | 5190 | in-flight | no | ⚪ WITHHELD | 0/7 |
| P16-N8 | TensorFEM | 5191 | in-flight | no | ⚪ WITHHELD | 0/6 |

## 3. 判据 (b) 跨域接口契约

| 契约 | 生产者 → 消费者 | 检查数 | 失败数 | 裁决 |
|---|---|---|---|---|
| `wind_endpoint_v1` | ship-platform (P16-N7 integration half) → TensorPhysics gate | 2 | 2 | ❌ FAIL |
| `wet_modal_endpoint_v1` | ship-platform (P16-N8 integration half) → TensorPhysics gate | 2 | 1 | ❌ FAIL |

### `wind_endpoint_v1` — ship-platform wind-resistance endpoint accepts the LBM coefficient table

| check | 裁决 | 数值/说明 |
|---|---|---|
| `E1_wind_endpoint_present` | ❌ FAIL | /api/v1/calculations/wind_resistance present=False (scanned 443 OpenAPI paths; wind-related paths found: ['/api/v1/polaris/operational-window']) |
| `E2_wind_endpoint_fields` | ❌ FAIL | accepts ('beta_deg', 'U_ref_m_s', 'rho_air_kg_m3', 'A_front_m2', 'A_lateral_m2', 'z_ref_m'); missing ['beta_deg', 'U_ref_m_s', 'rho_air_kg_m3', 'A_front_m2', 'A_lateral_m2', 'z_ref_m'] |

### `wet_modal_endpoint_v1` — ship-platform modal endpoint is able to carry wet-modal / added-mass data

| check | 裁决 | 数值/说明 |
|---|---|---|
| `E3_modal_endpoint_present` | ✅ PASS | /api/v1/fem/modal present=True |
| `E4_modal_endpoint_wet_fields` | ❌ FAIL | observed modal schema properties ['E', 'boundary_type', 'length', 'method', 'modal_mass', 'modal_stiffness', 'mode_shapes', 'n_modes', 'natural_frequencies', 'nu', 'nx', 'ny', 'rho', 'thickness', 'width']; missing wet-modal fields ['fluid_rho', 'sound_speed', 'added_mass', 'added_mass_matrix'] (endpoint currently exposes dry modes only) |

## 4. 判据 (c) P15 回归基线（|Δ| < 1%）

| P15 节点 | 仓库 | P15 pass rate | 本次 pass rate | Δ | 用例数保留 | 裁决 |
|---|---|---|---|---|---|---|
| IN14-D1 | TensorDEM | 1.0000 | — | — | None | ❌ FAIL |
| IN14-D2 | TensorDEM | 1.0000 | — | — | None | ❌ FAIL |
| IN14-D3 | TensorDEM | 1.0000 | — | — | None | ❌ FAIL |
| IN14-D4 | TensorFEM | 1.0000 | — | — | None | ⚪ WITHHELD |
| IN14-D5 | TensorFEM | 1.0000 | 1.0000 | +0.00e+00 | True | ⚪ WITHHELD |
| IN14-D6 | TensorFEM | 1.0000 | 1.0000 | +0.00e+00 | False | ❌ FAIL |
| IN14-D7 | TensorLBM | 1.0000 | — | — | None | ❌ FAIL |
| IN14-D8 | TensorLBM | 1.0000 | — | — | None | ❌ FAIL |
| IN14-D9 | TensorLBM | 1.0000 | — | — | None | ❌ FAIL |
| IN14-D10 | TensorFVM | 1.0000 | — | — | None | ⚪ WITHHELD |
| IN14-D11 | TensorBEM | 1.0000 | 0.9865 | -1.35e-02 | True | ❌ FAIL |

## 5. 审计链（输入 artifact sha256）

| artifact | 路径 | sha256 |
|---|---|---|
| campaign | `artifacts/p16_gate/campaign_nodes.json` | `8a46ba98196f6679…` |
| node_evidence | `artifacts/p16_gate/node_evidence.json` | `9276e407eab55ea4…` |
| p15_baseline | `artifacts/p16_gate/p15_baseline.json` | `4fa49691abc1b356…` |
| probe_results | `artifacts/p16_gate/probe_results.json` | `d25ed25534daf7bb…` |
| ship_facts | `artifacts/p16_gate/ship_platform_facts.json` | `cc10c13edb60b6a3…` |

## 6. 局限

- Node acceptance is only as good as the delivered p16_node_acceptance.v1 artifact; absent artifacts are reported WITHHELD, never inferred from PR merge state.
- Regression deltas cover the probe metrics recorded by the platform for P15; probes that cannot execute in the gate environment stay WITHHELD.
- Cross-domain payload validators check format/type/dimension consistency, not the physical accuracy of the delivered coefficients (that is N7/N8 acceptance).
