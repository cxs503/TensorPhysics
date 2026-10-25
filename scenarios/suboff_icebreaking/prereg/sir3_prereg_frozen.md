# SIR-3 预注册冻结记录

- **冻结时间**: 2026-10-25T14:30:00Z
- **MD5**: 47c72ea0ec404a7f3cb31ad8ba13891d
- **状态**: 已冻结，append-only

## 冻结内容

sir3_prereg.md 已完整，包含：
- 科学目标（重力/浮力/自由面入链）
- 物理锚（A3.1 阿基米德浮力, A3.2 自由面质量守恒, A3.3 静水压力梯度）
- 机器门（M3.1-M3.5）
- 规模判据（500步/32网格/1e3积分点）
- 内容符合判据（3个语义门）
- 交付物清单
- 停机条件

## 执行计划

1. 探索 TensorFEM ice_sheet.py buoyancy_response 接口
2. 探索 TensorLBM free_surface_lbm.py 接口
3. 创建薄编排场景 scenarios/suboff_icebreaking/sir3_buoyancy_freesurface/
4. 运行验证，生成 validation_report.json + gates.json
5. 写测试验证判据
