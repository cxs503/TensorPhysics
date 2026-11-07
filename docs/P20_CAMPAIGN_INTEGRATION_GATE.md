# P20 全战役集成验收门 (P20-G1) — 验收报告

- schema: `p20_campaign_gate_verdict.v1`
- 生成时间 (UTC): `2026-11-07T08:36:31.334350+00:00`
- 战役计划: `P20 船舶总体性能集成战役` (plan_id=27, status=`done`)
- 节点数: 27

## 1. 门裁决总览

| 判据 | 内容 | 裁决 | 说明 |
|---|---|---|---|
| (a) | DAG 无环验证 + 全节点依赖收敛 | ✅ PASS | all 27 nodes acyclic and dependencies satisfied; 27/27 in 'done' state |
| (b) | 跨仓接口契约验证（FSI/静水力/耐波） | ✅ PASS | 3 contract checks run (3 PASS / 0 FAIL); missing payloads: none |
| (c) | 成熟化单汇总（≥2条 N07/N08 报告完整） | ✅ PASS | 5/5 maturity reports; N07=PASS, N08=PASS; overall TRL = 7.0 |
| (d) | 战役总结报告六要素完整 | ✅ PASS | 6/6 elements present; missing: none |

**总裁决 (overall_status): `PASS`**

## 2. DAG 收敛详情

- 无环: 是
- 依赖收敛: 是
- 已完成节点: 27/27
- 拓扑序: `['P20-N01', 'P20-N02', 'P20-N03', 'P20-N04', 'P20-N05', 'P20-N07', 'P20-N08', 'P20-N06', 'P20-N12', 'P20-N13', 'P20-N14', 'P20-N15', 'P20-N16', 'P20-N17', 'P20-N09', 'P20-N10', 'P20-N18', 'P20-N19', 'P20-N20', 'P20-N11', 'P20-N21', 'P20-N22', 'P20-N23', 'P20-N24', 'P20-N25', 'P20-N26', 'P20-N27']`
- 阻塞节点: `[]`

## 3. 跨仓接口契约详情

### fsi_handoff_v1: TensorLBM surface forces → TensorFEM Neumann BC
- 裁决: `PASS` (6 checks, 0 failed)
- 生产者: TensorLBM (P20 FSI node)
- 消费者: TensorFEM boundary condition loader

### hydrostatic_handoff_v1: TensorCAD hull geometry → ship-platform hydrostatics
- 裁决: `PASS` (6 checks, 0 failed)
- 生产者: TensorCAD (P20 geometry node)
- 消费者: ship-platform hydrostatic calculator

### seakeeping_handoff_v1: TensorBEM potential flow → ship-platform slamming / motion
- 裁决: `PASS` (5 checks, 0 failed)
- 生产者: TensorBEM (P20 seakeeping node)
- 消费者: ship-platform slamming analyzer

## 4. 成熟化单汇总

- 已交付: 5/5
- 验证通过: 5
- 整体 TRL: 7.0

| 节点 | TRL | 交付 | 验证 | 测试通过率 |
|---|---|---|---|---|
| P20-N01 | 8 | ✅ | passed | 100% |
| P20-N02 | 7 | ✅ | passed | 100% |
| P20-N03 | 7 | ✅ | passed | 100% |
| P20-N07 | 7 | ✅ | passed | 100% |
| P20-N08 | 6 | ✅ | passed | 100% |

## 5. 战役总结报告六要素

- 已包含: ['case_introduction', 'methodology', 'results', 'figures', 'notes', 'error_analysis']
- 缺失: []

## 6. 局限性与说明

- DAG convergence depends on node status being accurately reported; stale status data may produce false PASS/FAIL.
- Contract validators check format/type/dimension consistency, not the physical accuracy of the delivered data.
- Maturity assessment requires delivered artifacts; absent artifacts are reported as not delivered, never inferred.
- Report element check is structural (key presence), not semantic (content quality).

## 7. 溯源 (Provenance)

- `campaign`: present=True, sha256=`f97e60ef3d4a3dec...`
- `fsi`: present=True, sha256=`3cbdc44a1601aef6...`
- `hydrostatic`: present=True, sha256=`74412d4592a5eec2...`
- `seakeeping`: present=True, sha256=`222c7edda9115c0d...`
- `report`: present=True, sha256=`4a0a669810cafc69...`
