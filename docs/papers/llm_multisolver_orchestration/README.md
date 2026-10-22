# LLM Multi-Solver Orchestration — 同步论文（骨架阶段）

英文 arXiv 预印本 + 国际期刊投稿（候选 CPC / Ocean Engineering / AI4Science）。
**署名与投稿去向由 owner 终定。** 分支 `paper/agent-orchestration`。

## 写作纪律

- **零手抄**：正文一切实证数字只能来自平台导出工件（SQL→JSON 导出、
  run manifest、gate 报告），落稿前以 `\pending{...}` 占位。
- 系统描述只写已提交实现（TensorPlatform `feat/tensor-missions`、
  TensorPhysics 阶梯），不写计划态。
- 术语与平台对齐：mission / claim token / dispatch envelope / role /
  gate / stage promotion / append-only event stream。

## 状态板

| 章节 | 状态 | 数据依赖 |
|---|---|---|
| abstract / intro / related / system / method | 骨架成稿（系统描述部分定稿级） | — |
| 5 case study（wedge S0–S4） | S0 已写；S1–S4 结果占位 | P3 战役工件 |
| 6 ablation（7 模型 bake-off + 舰队遥测） | 设计成稿，数字占位 | m19–m25 出表 + 战役事件流导出 |
| 7 discussion / 8 conclusion | 骨架成稿 | 终局结果 |
| references.bib | 9 条起步；2025–2026 LLM4Science 近作投稿前补 | — |

## 数据导出（待跑）

```
# bake-off 表：m19–m25（同 goal 同预算，仅模型不同）
SELECT id, model, status, result->'progress'->>'tool_calls' AS tools,
       result->'evidence'->>'finished_status' AS stop,
       created_at, finished_at
FROM tensor_dev_missions WHERE id BETWEEN 19 AND 25 ORDER BY id;

# 舰队遥测：每 mission 事件形状（kind 计数、nudge/abort 次数）
# dispatch 包络拒绝分类计数：parent session 事件 subtype=dispatch_sub_mission
```
