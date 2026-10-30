# 船舶总体性能平台审计报告

> 审计时间: 2025-07-11  
> 审计分支: agent/m7  
> 审计方法: audit_capabilities() + 6 端点 API 探测 + 源码定位

---

## 一、能力矩阵表（学科 × 证据等级 × 状态）

### 1.1 学科能力覆盖

| 学科 | 对应能力 | 证据等级 | API 端点测试结果 | 状态 |
|------|---------|---------|-----------------|------|
| **阻力** | `calc.resistance` | REAL | GET → 405 Method Not Allowed（需 POST） | ⚠️ 算法真实但 GET 不支持 |
| **推进** | `propeller.open_water` | REAL | 未直接测试（不在 6 端点中） | ✅ 正常 |
| **耐波** | 无独立能力注册 | — | GET calculations/seakeeping → 405 | ⚠️ 端点存在但未纳入能力清单 |
| **操纵** | 无 | NOT_SUPPORTED | — | ❌ 缺失 |
| **稳性** | `calc.stability` | REAL | GET → 405 Method Not Allowed（需 POST） | ⚠️ 算法真实但 GET 不支持 |
| **结构** | 无 | NOT_SUPPORTED | — | ❌ 缺失 |
| **振动噪声** | 无 | NOT_SUPPORTED | — | ❌ 缺失 |
| **绿色航行** | 无 | NOT_SUPPORTED | — | ❌ 缺失 |
| **冰区航行** | 无（仓库有 benchmark 场景） | NOT_SUPPORTED | — | ❌ 平台能力缺失 |

### 1.2 平台基础设施能力（21 项全量）

| # | 能力名称 | 证据等级 | 所属类别 | 状态 |
|---|---------|---------|---------|------|
| 1 | `calc.resistance` | REAL | 阻力计算 | ✅ |
| 2 | `calc.stability` | REAL | 稳性计算 | ✅ |
| 3 | `propeller.open_water` | REAL | 推进计算 | ✅ |
| 4 | `geometry.generate` | REAL | 几何生成 | ✅ |
| 5 | `geometry.hydro_curves` | REAL | 几何水动力曲线 | ✅ |
| 6 | `geometry.list_models` | REAL | 几何模型列表 | ⚠️ API 路由错误（见缺陷 #5） |
| 7 | `geometry.verify_official` | REAL | 几何验证 | ✅ |
| 8 | `shiphull.generate` | REAL | 船体生成 | ✅ |
| 9 | `stdmodels.lookup` | REAL | 标模查询 | ✅ |
| 10 | `dev.gate_5415` | REAL | 开发工具 | ✅ |
| 11 | `dev.git` | REAL | 开发工具 | ✅ |
| 12 | `dev.logs` | REAL | 开发工具 | ✅ |
| 13 | `dev.nginx_audit` | REAL | 开发工具 | ✅ |
| 14 | `dev.probe` | REAL | 开发工具 | ✅ |
| 15 | `dev.restart` | REAL | 开发工具 | ✅ |
| 16 | `dev.services` | REAL | 开发工具 | ✅ |
| 17 | `gov.stats` | REAL | 治理统计 | ✅ |
| 18 | `knowledge.docs` | REAL | 知识文档 | ✅ |
| 19 | `ai.predict` | **NOT_SUPPORTED** | AI 预测 | ❌ 不支持 |
| 20 | `methods.computation` | **NOT_SUPPORTED** | 计算方法 | ❌ 不支持 |
| 21 | `misc.stubs` | **STUB** | 杂项桩 | ❌ 空壳 |

**统计汇总:**
- REAL: 18 / 21 (85.7%)
- NOT_SUPPORTED: 2 / 21 (9.5%)
- STUB: 1 / 21 (4.8%)
- MOCK: 0 / 21 (0%)
- WITHHELD: 0 / 21 (0%)

### 1.3 API 端点探测结果

| 端点 | 方法 | HTTP 状态 | 关键发现 |
|------|------|----------|---------|
| `calculations/resistance` | GET | 405 | 仅支持 POST，GET 被拒 |
| `calculations/stability` | GET | 405 | 仅支持 POST，GET 被拒 |
| `calculations/seakeeping` | GET | 405 | 端点存在但未注册为能力 |
| `standard-models/platforms` | GET | 200 | 返回 8 个平台完整几何参数 |
| `geometry/list-models` | GET | 404 | **路由 Bug**: 将 "list-models" 当作船名查询 |
| `tensorlbm/benchmarks` | GET | 200 | 11 个 benchmark 全部 `never_run` |

### 1.4 标模库数据质量（standard-models/platforms 返回）

返回 8 个平台模型，数据完整性检查：

| 模型 | 类型 | Lpp | B | T | Cb | 数据完整性 |
|------|------|-----|---|---|-----|-----------|
| MV Blue Marlin | semi_submersible | 216 | 42 | 10 | 0.70 | ✅ 完整 |
| Typical FPSO | offshore_fpso | 300 | 60 | 22 | 0.85 | ✅ 完整 |
| Cylindrical FPSO | offshore_fpso | 80 | 80 | 25 | 0.785 | ✅ 完整 |
| Semi-submersible Platform | offshore_platform | 100 | 80 | 20 | 0.25 | ✅ 完整 |
| Spar Platform | offshore_platform | 40 | 40 | 200 | 0.785 | ✅ 完整 |
| TLP | offshore_platform | 80 | 80 | 35 | 0.30 | ✅ 完整 |
| Jack-up Platform | offshore_platform | 70 | 60 | 6 | 0.60 | ✅ 完整 |
| FLNG | offshore_fpso | 488 | 74 | 26 | 0.82 | ✅ 完整 |

> 标模数据质量良好，所有模型均包含 Lpp/B/T/Cb/Cm/Cp/Cw/KG/DWT/displacement/design_speed 等完整参数。

### 1.5 TensorLBM Benchmark 状态

11 个 benchmark 全部为 `never_run` 状态：

`accuracy`, `all_models`, `bluff_body_matrix`, `dam_break`, `dam_break_3d`, `dam_break_quant`, `fs_2d`, `marine`, `mlups`, `multiphase`, `propeller`

> **结论**: TensorLBM 集成框架已搭建但从未执行过任何 benchmark，无法验证计算精度。

### 1.6 源码可追溯性

| 预期文件路径 | 实际状态 |
|-------------|---------|
| `backend/api/calculations.py` | ❌ 工作树中不存在 |
| `backend/platform_harness/capabilities.json` | ❌ 工作树中不存在 |
| `backend/platform_harness/guard.py` | ❌ 工作树中不存在 |

> **结论**: 预期的后端源码目录结构 (`backend/`) 在当前工作树中不存在。API 服务运行在 :8001 端口但源码不在本仓库中，可能是独立部署的服务。工作树实际包含 `src/tensorphysics/`（研究运行时框架）和 `scenarios/`（冰区航行场景脚本），但不含船舶性能计算的后端实现代码。这导致无法对 REAL 能力进行源码级精度验证。

---

## 二、缺陷清单

| # | 能力/端点 | 证据等级 | 缺陷描述 | 修复方向 |
|---|---------|---------|---------|---------|
| 1 | `ai.predict` | NOT_SUPPORTED | AI 预测能力完全缺失 | 集成机器学习阻力/推进预测模型，或明确标注为路线图项 |
| 2 | `methods.computation` | NOT_SUPPORTED | 计算方法能力缺失 | 实现计算方法选择/调度接口，支持 Holtrop/ITTC 等方法切换 |
| 3 | `misc.stubs` | STUB | 杂项桩为空壳实现 | 移除或实现具体功能 |
| 4 | `calc.resistance` (GET) | REAL | GET 方法返回 405，仅支持 POST | 增加 GET 方法返回计算参数 schema 或历史结果列表 |
| 5 | `calc.stability` (GET) | REAL | GET 方法返回 405，仅支持 POST | 同上，增加 GET 查询接口 |
| 6 | `geometry/list-models` | REAL | **路由 Bug**: GET /geometry/list-models 返回 404，系统将 "list-models" 当作船名查询 | 修复路由优先级，将 `list-models` 作为保留路径优先匹配 |
| 7 | `calculations/seakeeping` | 未注册 | 耐波计算端点存在（返回 405）但未注册为平台能力 | 将 seakeeping 纳入 capabilities 清单并标注证据等级 |
| 8 | TensorLLM benchmarks | — | 11 个 benchmark 全部 `never_run` | 执行至少 `marine` 和 `propeller` benchmark 以验证精度 |
| 9 | 操纵性 | NOT_SUPPORTED | 9 大学科中操纵性完全缺失 | 规划操纵性计算能力（MMG/Abkowitz 模型） |
| 10 | 结构强度 | NOT_SUPPORTED | 结构学科完全缺失 | 规划结构强度评估能力 |
| 11 | 振动噪声 | NOT_SUPPORTED | 振动噪声学科完全缺失 | 规划振动噪声分析能力 |
| 12 | 绿色航行 | NOT_SUPPORTED | 绿色航行（EEDI/CII）缺失 | 规划能效指标计算能力 |
| 13 | 冰区航行 | NOT_SUPPORTED | 平台能力缺失（仓库有场景脚本但无平台能力） | 将冰区场景脚本集成为平台能力 |
| 14 | 后端源码 | — | `backend/` 目录不存在于工作树 | 确认后端代码仓库位置，或将其纳入本工作树 |

---

## 三、改进优先级

### P0 — 阻断性缺陷（必须立即修复）

| 优先级 | 缺陷 # | 描述 | 理由 |
|--------|--------|------|------|
| **P0** | #6 | `geometry/list-models` 路由 Bug | 用户无法通过 API 获取模型列表，基础功能不可用 |
| **P0** | #14 | 后端源码不在工作树 | 无法进行源码级审计和精度验证，阻断改进工作 |

### P1 — 高优先级（本迭代应修复）

| 优先级 | 缺陷 # | 描述 | 理由 |
|--------|--------|------|------|
| **P1** | #7 | seakeeping 未注册为能力 | 耐波是 9 大核心学科之一，端点存在却不被追踪 |
| **P1** | #8 | TensorLBM benchmarks 从未执行 | 无法验证 LBM 计算精度，marine/propeller benchmark 应优先执行 |
| **P1** | #4, #5 | resistance/stability GET 返回 405 | API 设计应支持 GET 查询（参数 schema 或历史结果） |
| **P1** | #2 | `methods.computation` NOT_SUPPORTED | 计算方法选择是阻力/推进计算的基础设施 |
| **P1** | #3 | `misc.stubs` STUB | 空壳能力影响平台可信度，应移除或实现 |

### P2 — 中长期规划（路线图）

| 优先级 | 缺陷 # | 描述 | 理由 |
|--------|--------|------|------|
| **P2** | #9 | 操纵性能力缺失 | 核心学科覆盖缺口 |
| **P2** | #10 | 结构强度能力缺失 | 核心学科覆盖缺口 |
| **P2** | #11 | 振动噪声能力缺失 | 核心学科覆盖缺口 |
| **P2** | #12 | 绿色航行（EEDI/CII）缺失 | 行业合规需求 |
| **P2** | #13 | 冰区航行平台能力缺失 | 仓库已有场景脚本，可集成为正式能力 |
| **P2** | #1 | `ai.predict` NOT_SUPPORTED | AI 预测为增强能力，非阻断项 |

---

## 四、总体评估

### 4.1 优势
- **核心计算能力真实**: 阻力、稳性、推进（敞水）3 项核心计算能力均为 REAL，非 MOCK/STUB
- **几何/标模基础设施完善**: geometry.generate/hydro_curves/verify_official + stdmodels.lookup 均为 REAL，标模库数据完整（8 个平台模型参数齐全）
- **开发运维工具齐全**: 7 个 dev.* 能力均为 REAL
- **无 MOCK 能力**: 21 项能力中没有硬编码/随机返回的 MOCK 项

### 4.2 风险
- **学科覆盖率不足**: 9 大船舶性能学科仅覆盖阻力、推进、稳性 3 个（33%），操纵/结构/振动噪声/绿色航行/冰区航行均缺失
- **TensorLBM 精度未验证**: 11 个 benchmark 全部从未执行，LBM 计算精度无法确认
- **源码不可追溯**: 后端实现代码不在工作树中，REAL 能力无法进行源码级验证
- **API 路由缺陷**: geometry/list-models 路由 Bug 影响基础可用性
- **耐波能力未注册**: 端点存在但不被能力系统追踪，存在管理盲区

### 4.3 建议行动项
1. **立即**: 修复 geometry/list-models 路由 Bug（P0）
2. **立即**: 确认后端代码仓库位置并纳入审计范围（P0）
3. **短期**: 将 seakeeping 注册为能力；执行 TensorLBM marine/propeller benchmark（P1）
4. **短期**: 为 resistance/stability 增加 GET 查询接口（P1）
5. **中期**: 规划操纵性、结构、绿色航行能力建设（P2）
6. **长期**: 集成冰区航行场景脚本为正式平台能力；规划 AI 预测能力（P2）
