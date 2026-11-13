# 物性数据库 v1（MPDB v1）— 统一 schema · 查询面 · 迁移映射

锚点：`FEAT-ASSET-PROPS-DB`（mission 17459，仓 TensorPhysics）。
资产形式普查结论「物性数据当前为空/散装」在本单被落成一件可查询资产：
**3087 行 / 19 种物质 / 7 个物质类 / 27 个文献源 / 20 个独立锚点全过**。

- 包：`src/tensorphysics/material_properties/`
- 数据：`src/tensorphysics/material_properties/data/mpdb_v1.{jsonl,substances,anchors,meta,schema}.json*`
- 一键复现：`python scripts/build_material_properties_db.py`（`--check` 校验已提交工件，`--validate` 只跑 schema+锚点）
- 工件：`artifacts/mpdb_v1/`（曲线图、锚点审计图、数字表、存量盘点 csv/md/json）
- 测试：`tests/test_material_properties_db.py`（32 项，全绿）

---

## 1 验收标准对照

| # | 验收标准 | 落点 | 证据 |
|---|---|---|---|
| ① | 统一 schema：物质×温度/压力×物性×出处×适用域 | `schema.py`（`make_record`/`validate_record`/`to_json_schema`） | 3087 行 `validate_records(...) == []`；`mpdb_v1.schema.json` 可被 jsonschema 验证 |
| ② | ≥5 类物质 100+ 行带 provenance（ITTC/标准手册原表） | 7 类：liquid / gas / ice / sediment / metal / composite / cementitious；3087 行 100% 带 source_id+locator | `mpdb_v1.meta.json`：`rows_with_provenance = 3087`；`FORMULA_PRIMARY 2135 / TABLE_PRIMARY 303 / DERIVED 649` |
| ③ | 查询 API + CLI + 单位约定（SI 基） | `query.MaterialDB.get/query/table/domain`；`cli.py`（list/describe/props/query/table/validate/build/inventory） | `pytest -k cli`；`python -m ...cli query --substance seawater --property density --T-C 15 --salinity 35` |
| ④ | 盘点存量：ship 仓散装 JSON 迁移映射表（哪些并入/哪些保留） | `inventory.py` + `artifacts/mpdb_v1/property_carriers.{csv,md,json}` | 504 个载体：MIGRATE 393 / REFERENCE 2 / KEEP_DOMAIN 12 / KEEP_GENERATED 97；inline 常量位点 1278 |
| ⑤ | solver 接入样例 ≥2 仓 | `adapters.py`：LBM 缩放、FVM/DG 空气状态、FEM 冰板材料、船舶阻力水环境（4 个接入面 / 3 个求解器族 + 船舶平台） | `pytest -k "lbm or fvm or fem_ice or ship_resistance"` |

## 2 Schema（一行 = 物质×条件×物性×出处×适用域）

```json
{
  "schema_version": "mpdb-1.0.0",
  "id": "seawater.density.T288.15K.p101325Pa.S35",
  "substance": "seawater", "substance_class": "liquid",
  "property": "density", "value": 1025.97, "unit": "kg/m^3",
  "conditions": {"T_K": 288.15, "p_Pa": 101325.0, "salinity_g_per_kg": 35.0},
  "domain": {"T_K": [273.15, 453.15], "p_Pa": [101325, 101325], "salinity_g_per_kg": [0, 160]},
  "provenance": {"source_id": "SRC-SHARQAWY-DENSITY",
                 "citation": "Sharqawy, Lienhard & Zubair (2010) DWT 16:354-380, Table 4 ...",
                 "locator": "Table 4, eq. (2)-(5)", "method": "correlation",
                 "evidence_class": "FORMULA_PRIMARY", "access": "offline_no_pdf",
                 "formula_id": "sharqawy2010_density"},
  "uncertainty": {"rel": 0.0001, "basis": "stated 0.01% for 0-40 degC, S <= 42 g/kg"},
  "variant": null, "derived_from": null, "notes": null
}
```

三条硬约束（`schema.validate_record` 强制）：

1. **单位**:`unit` 必须等于 `PROPERTY_UNITS[property][0]`（SI 基）。degC/deg/g·kg⁻¹ 只作为**访问单位**出现在 CLI/查询入参，永不进 payload；
2. **出处**:`source_id` 必须在 `SOURCE_REGISTRY` 中解析、且 `locator`（表/式/页）非空；`FORMULA_PRIMARY` 必须给 `formula_id`，`DERIVED` 必须给 `derived_from`；
3. **适用域**:`domain` 非空且为闭区间；查询面按 `domain` 与「已采样区间」双重守卫，越界**拒绝返回**而不是静默外插。

`variant` 处理同一物质的不同状态（一年冰/多年冰、初始模量/有效模量、水平/垂直抗压），避免 id 冲突与静默取值。

## 3 覆盖与出处（27 个源，全部带页码/表号/式号）

| 物质 | 类 | 行数 | 物性 | 主源 |
|---|---|---|---|---|
| water | liquid | 1208 | density, μ, ν, cp, k, a, L_f, T_f | Kell 1975 / IAPWS R7-97；Kestin+1978；CRC 97th p.6-4；IAPWS R15-11；Bilaniuk & Wong 1993 |
| seawater | liquid | 1625 | density, μ, ν, cp, T_f | Sharqawy+2010 Table 3/4/5；Millero & Leung 1976 |
| air | gas | 121 | density, μ, ν, k, cp, Pr, a | ISO 2533；Sutherland 1893（White 2006 Table 1-2）；CRC p.6-190 |
| ice_fresh | ice | 13 | density(T), E, ν, α, k, cp, L_f | Hobbs 1974 Table 2.1/2.3-2.6/6.1 |
| ice_sea | ice | 33 | bulk density, E(init/eff), ν, σ_t/σ_c/σ_s, σ_f(T,S), v_b(T,S), friction, S | Timco & Weeks 2010 Table 2-6, eq.(8)；Frankenstein & Garner 1967 eq.(1) |
| sand/silt/clay/gravel | sediment | 26 | ρ_s, ρ_dry, γ′, φ, c, repose, D50 | Soulsby 1997 Table 2-3/2-4；DNV-RP-C212 Table 3-1 |
| steel S235/S355/AH36/EH36 | metal | 28 | ρ, E, G, ν, ReH, Rm, α, η | EN 10025-2 / EN 1993-1-1 Table 3.1；IACS UR S6/W11 |
| alu 5083-O/H116, 6082-T6 | metal | 18 | ρ, E, ν, f_o, f_u, ρ_haz | EN 1999-1-1 Table 3.2/3.2b/3.2c |
| GFRP laminate/sandwich | composite | 9 | ρ, E, ν, σ_t, η | Lloyd's Register Pt3 Ch4 Sec3；ship 仓 plate_materials 资产 |
| concrete C30 | cementitious | 6 | ρ, E, ν, f_ck, f_ctm, α_t | EN 1992-1-1 Table 3.1 |

**「零手抄」的落实方式**：流体/冰强度等可复算量走 `correlations.py` 中的**公式**（`FORMULA_PRIMARY`，2135 行），表值只在 `tables.py` 声明一次并附定位（`TABLE_PRIMARY`，303 行），比值量走 `DERIVED`（649 行，带 `derived_from` 引用分子/分母行 id）。任何行都能由 `repro_cmd` 重建；`build --check` 会比对已提交 JSONL 的逐行数值与出处，漂移即失败。

## 4 精度：20 个独立锚点（全部通过）

图：`artifacts/mpdb_v1/mpdb_v1_anchor_audit.png`；表：`mpdb_v1_numbers.json` / `mpdb_v1.anchors.json`。

| 锚点 | 计算 | 参考 | 相对误差 | 允许带 |
|---|---|---|---|---|
| 淡水 ρ(4 °C) | 999.972 | 999.97 kg/m³ | 0.000% | 0.05% |
| 淡水 ρ(20 °C) | 998.204 | 998.21 kg/m³ | 0.000% | 0.05% |
| 淡水 ν(15 °C, ITTC) | 1.13699e-6 | 1.1390e-6 m²/s | 0.18% | 1% |
| 海水 ν(15 °C, 35 g/kg, ITTC) | 1.18899e-6 | 1.1883e-6 m²/s | **0.06%** | 2% |
| 海水 ρ(25 °C, 35 g/kg) | 1023.343 | 1023.34 kg/m³ | 0.000% | 0.1% |
| 海水 T_f(35 g/kg) | 271.228 | 271.25 K | 0.008% | 0.2% |
| 淡水 μ(20 °C) | 1.00175e-3 | 1.0016e-3 Pa·s | 0.014% | 1% |
| 淡水 μ(0 °C) | 1.75306e-3 | 1.7914e-3 Pa·s | 2.14% | 3% |
| 淡水 a(20 °C) | 1482.359 | 1482.3 m/s | 0.004% | 0.2% |
| 空气 ρ(15 °C) | 1.22498 | 1.2255 kg/m³ | 0.04% | 0.2% |
| 空气 μ(15 °C) | 1.78930e-5 | 1.7894e-5 Pa·s | 0.006% | 2% |
| 空气 k(20 °C) | 0.025714 | 0.02587 W/(m·K) | 0.60% | 2% |
| 空气 a(15 °C) | 340.297 | 340.29 m/s | 0.002% | 0.5% |
| 纯冰 ρ(0 °C) | 916.8 | 916.8 kg/m³ | 0.000% | 0.2% |
| 一年冰 σ_f(−10 °C, 5 g/kg) | 0.667 MPa | 0.70 MPa（表值典型，带 0.36-2.6） | 4.75% | 30%（文献散布） |
| 钢 S355 E / ReH | 2.10e11 / 3.55e8 | EN 1993/10025 | 0.000% | 0.1% |
| 铝 5083 E | 7.00e10 | EN 1999-1-1 | 0.000% | 0.1% |
| 石英砂 ρ_s | 2650 | Soulsby Table 2-3 | 0.000% | 0.1% |
| 混凝土 C30 E | 3.30e10 | EN 1992-1-1 | 0.000% | 0.1% |

**关键量是否 <3%**：20/20 锚点通过；除两条例外（冰弯曲强度取文献散布带 ±30%；0 °C 淡水黏度 Vogel 拟合 2.14%，其源公式在 0 °C 边界即宣称 3% 精度），**全部锚点误差 < 1%**。

## 5 查询面（API / CLI）

```python
from tensorphysics.material_properties import MaterialDB
db = MaterialDB.load()
db.get("seawater", "density", T_C=15, salinity=35)          # 1025.97 kg/m^3 + provenance
db.get("seawater", "kinematic_viscosity", T_C=15, salinity=35)
db.get("ice_sea", "youngs_modulus", variant="effective")     # 多状态必须显式 variant
db.query("water", ["density", "dynamic_viscosity"], T_C=20)  # 多物性同一条件
```

行为契约（均被测试覆盖）：

- **不静默外插**：`db.get("water","density",T_C=150)` → `OutOfDomainError`（可 `clamp=True` 饱和到最近采样态，并在 `interpolated` 里标注 `clamped to ...`）；
- **不静默歧义**：`ice_sea.youngs_modulus` 不指定 variant → `AmbiguousValueError` 列出 `initial/effective`；
- **缺条件报错**：`seawater.density` 不给 salinity → `MissingConditionError`（列出采样区间）；单值轴（p=1 atm）自动采用并记录；压力缺省取标准大气并显式记录；
- **插值透明**：0.5 K 网格上线性插值，`interpolated={"T_K":"linear"}`；0.5 K 格内插值误差 <1e-6（曲率项 O(ΔT²)）。

CLI：

```bash
python -m tensorphysics.material_properties.cli list --class liquid
python -m tensorphysics.material_properties.cli describe --substance seawater
python -m tensorphysics.material_properties.cli query --substance seawater \
    --property density --property kinematic_viscosity --T-C 15 --salinity 35 --json
python -m tensorphysics.material_properties.cli table --substance air --property density --csv
python -m tensorphysics.material_properties.cli validate          # schema + 锚点 + 行/公式一致性
python -m tensorphysics.material_properties.cli build --check      # 已提交工件与重建是否一致
python -m tensorphysics.material_properties.cli inventory --repo <ship 仓> --out artifacts/mpdb_v1
```

## 6 存量盘点与迁移映射（④）

扫描对象：`ship-performance-platform`（HEAD `3ab633e`，本地私有仓），只读扫描，规则全部在 `inventory.py` 中显式声明，可复现。
产物：`artifacts/mpdb_v1/property_carriers.csv`（逐文件）、`.md`（可读表）、`.json`（含 summary）。

| bucket | 数量 | 含义 / 动作 |
|---|---|---|
| `MIGRATE_TO_MPDB` | 393（33 高置信 + 360 中置信） | 载荷本身是物性目录或成块内联常量 → 换 MPDB 查询 |
| `REFERENCE_MPDB` | 2 | 案例/试验数据把物性当输入 → 保留文件，字面量改由 MPDB 解析 |
| `KEEP_DOMAIN` | 12 | 无机型无关物性载荷的域数据 → 保留 |
| `KEEP_GENERATED` | 97 | artifacts/docs/备份 → 永不作物性源 |

内联常量为 1278 处（`RHO_SEA = 1025.0`、`RHO_AIR_15C = 1.225`、`NU_*`、`E_*` 等）。高置信并入清单（首批，直接对应 MPDB 物质）：

| ship 仓文件 | 并入 MPDB | 处理 |
|---|---|---|
| `backend/data/aluminum_materials.yaml` | `alu_5083_O / H116 / 6082_T6` | 目录整体退役，改查询 |
| `backend/api/config/haz_alloys.yaml` | 同上 + `steel_AH36` | 与上面重复的第二份 HAZ 表，一并退役（两处 E/ρ/f_o 已出现不一致：2660 vs 2650） |
| `backend/modules/cabin_noise/data/plate_materials.json` | `steel_* / alu_* / gfrp_sandwich` | 保留 η_struct 语义，物性改查 MPDB |
| `backend/modules/waste_heat_orc/data/orc_fluids.json` | —（**保留**） | 有机工质不在 v1 范围 → v2 候选，已在 `KEEP_DOMAIN` |
| `backend/api/air_resistance.py` 等 425 个模块的内联常量 | 按内容映射 water/seawater/air/ice/steel | 分批替换：先改高频 water/seawater/air（覆盖 240/239/82 个文件） |

**与普查「254 处」的对账（如实说明）**：本单在 ship 仓当前 HEAD 上用 5 种可复现口径分别计数——全 JSON 304；含 JSONL/YAML/CSV 的数据载体 356；剔除 artifacts/docs/备份后 109；含物性键的数据文件 53；含内联物性常量的 Python 模块 91。**没有任何单一规则复现出 254**，故本单不沿用该数字，改为给出上表口径明确、逐文件可核的盘点表（504 个载体 / 1278 个内联位点）。差异来源：仓已前进（HEAD 3ab633e）、普查口径未随工件固化。此差异记为本单的生态摩擦项。

## 7 solver 接入样例（⑤）

`adapters.py` 四个接入面，全部返回普通 dict（可 JSON 化）并携带 provenance：

```python
# 1) TensorLBM：真实水/海水 -> 格子参数（Re 精确保持，BGK tau 稳定下限守卫）
from tensorphysics.material_properties import lbm_scaling
p = lbm_scaling(substance="seawater", T_C=15, salinity=35, L_phys=0.1, U_phys=0.02, n_lattice=256)
#   -> rho0, nu_phys, Re, nu_lat, tau, omega, dx, dt, Ma
# 2) TensorFVM/DG：空气状态（rho, mu, nu, k, cp, Pr, a）
from tensorphysics.material_properties import fvm_air_state
# 3) TensorFEM：海冰板材料（E 取 initial/effective，σ_f 由 (T,S)->v_b 现算）
from tensorphysics.material_properties import fem_ice_material, plate_rigidity
# 4) ship-performance：阻力/自航水环境 + ITTC-57
from tensorphysics.material_properties import ship_resistance_env, ittc57_cf
env = ship_resistance_env(T_C=15, salinity=35, L=200.0, V=8.0)   # -> rho, nu, Re, Cf
```

运行台账：`provenance_ledger({...})` 把一次求解消费的所有物性行（source_id/locator/row_ids/不确定度）摊平成一串记录，可直接写进求解器的 run ledger，实现「求解结果可回溯到物性表的哪一行」。

代表性的接入自检（测试内断言）：`tau = 0.523`（>0.51 稳定）、`Ma = 0.087`、`Re` 精确保持；空气 `Pr = 0.712`、`a = 343.2 m/s`；海冰 `v_b(−10 °C, 5 g/kg) = 0.0273`、`σ_f = 0.667 MPa`；`Re = 1.34e9 @200 m/8 kn`、`Cf_ITTC57 = 1.42e-3`。

## 8 注意点 / 局限（诚实申报）

1. **离线取证**：构建沙箱无外网（HTTPS 出站被拦），27 个文献源均**未能重开 PDF**，`provenance.access = "offline_no_pdf"` 已逐行标注。补偿手段：能写成公式的一律写成公式（2135 行可复算），并用 20 个独立锚点把公式与「文献表值」对齐；表值行（303 行）只有表号/页码，机器无法重验，这是 v1 的已知弱项，v2 应在有网环境回填 `access="pdf_opened"` 与逐行截图哈希。
2. **单点误差已记录**：0 °C 淡水黏度偏差 2.14%（Vogel 拟合边界）；海冰弯曲强度取 Timco–Weeks 拟合（散布 ±30%），工程上应做区间分析而不是取单值。
3. **压力轴稀疏**：水/海水在 1 atm 单点；空气采样 0.5/1/2 bar。深水（>10 MPa）、高温高压不在 v1 覆盖，查询会明确拒绝外插。
4. **单元格**：0.5 K 网格 + 线性插值是精度与体积的折中（0.5 K 内误差 <1e-6），若需要 IAPWS-IF97 级精度应改调库而不是插表。
5. **映射表是静态扫描**：`MIGRATE_TO_MPDB` 的 360 条中置信项来自内联常量正则，存在假阳性（如 `cavitation_inception.py` 未匹配到 MPDB 物质，已标注 "manual triage"），迁移时应逐条人工确认。
6. **未做**（明确边界）：不改 ship 仓代码（本单只出映射表与接入样例，实迁移归 ship 仓战役）；不加 ORC 工质、不加温度相关的金属 E(T)、不加冰强度应变率效应。

## 9 复现命令

```bash
cd <worktree>
PYTHONPATH=src python scripts/build_material_properties_db.py            # 重建 3087 行
PYTHONPATH=src python scripts/build_material_properties_db.py --check    # 工件一致性
PYTHONPATH=src python scripts/build_material_properties_db.py --validate # schema + 锚点
PYTHONPATH=src python -m pytest tests/test_material_properties_db.py -q  # 32 项
PYTHONPATH=src python -m tensorphysics.material_properties.cli inventory \
    --repo /root/ship-performance-platform-incoming/ship-performance-platform --out artifacts/mpdb_v1
```