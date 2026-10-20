# suboff_icebreaking —— 完整 SUBOFF 破冰战役

owner 指令（2026-10-21）：围绕完整的 SUBOFF 破冰，把 TensorLBM / TensorFVM 的
流体功能、TensorFEM 的有限元功能、TensorDEM 的离散元功能都开发到位。

目标场景：**全附体 SUBOFF 潜艇**（sail + 尾舵 + 轴向附体）在冰区近水面航行，
顶升破断连续冰板，碎冰在流体载荷下演化——流体（LBM+FVM 双路）、结构（FEM
冰板弹性+断裂）、离散（DEM 碎冰接触+湿耦合）三物理双向耦合的完整模拟。

## 库分工

| 库 | 角色 | 战役内能力目标 |
|---|---|---|
| TensorLBM | 流体（格子 Boltzmann） | SUBOFF 全附体近水面/冰面下外流；自由面；冰下流道 |
| TensorFVM | 流体（有限体积 body-fitted/RANS） | SUBOFF 贴体外流；与 LBM 跨库互证；可交换流体腿 |
| TensorFEM | 结构（有限元） | 连续冰板：弹性地基板 + cohesive 断裂；破冰载荷学 |
| TensorDEM | 离散（离散元） | 碎冰块：接触/恢复/浮沉/湿耦合；碎冰场演化 |
| TensorPhysics | 场景编排 | 适配器+耦合账本+阶梯判据（wedge 范式继承） |

## 阶梯（每阶 prereg 先冻结后跑；判据全机器裁定）

### Phase 0 —— 能力审计（SI-0，四腿并行，纯读）
每库产出 `scenarios/suboff_icebreaking/audits/SI-<X>-<lib>.md`：
现有模块清单对照战役需求 → 差距清单（按阻断性排序）→ 每差距提议解析锚
（可机器裁定）→ 工作量分级。**禁止审计腿写功能代码。**

### Phase 1 —— 单库能力到位（SI-A/B/C/D）
- **SI-A TensorLBM 流体腿**：全附体近水面流场。锚：①已 verified 的 sail 线
  阻力基线复现；②冰面（顶盖滑移/无滑移）下通道流解析解；③自由面既有锚。
- **SI-B TensorFVM 流体腿**：SUBOFF 贴体网格+外流。锚：①阻力量级 vs
  ITTC-1957 参考带；②与 SI-A 同工况跨库互证门（约定口径与容差）。
- **SI-C TensorFEM 冰板腿**：①弹性地基板静载 Hetényi 级数解；②cohesive
  断裂能量账本机器精度（wedge S1 双账本范式）；③板-水附加质量量级锚。
- **SI-D TensorDEM 碎冰腿**：①单块浮沉（浮力=重力+衰减到静平衡）；
  ②两块碰撞恢复系数标定/复现；③湿耦合阻力锚（calibrated_wet_coupling 线）。

### Phase 2 —— 耦合阶梯（依赖有序）
- **SI-1 LBM×FEM 单向**：流载→冰板响应；节点力账本 max|对账差| 机器精度门。
- **SI-2 FEM→DEM 断裂生成**：cohesive 破坏单元→DEM 颗粒；质量/断裂能量
  守恒门（生成前后账本差 ≤ 机器精度级）。
- **SI-3 DEM×流体双向**：wedge S4 闭环交换技术的 SUBOFF 版（主/子时钟比率
  与交换节拍显式；冲量双账本）。
- **SI-4 全链演示**：SUBOFF 顶破冰板完整演化；论文级平台报告（物理表述+
  云图+曲线+审计+局限披露）。
- **SI-5 FVM 对照腿**：同一冰况下 FVM 替换 LBM 流体腿的可交换性（适配器
  接口契约门）。

## 纪律（全部沿用既有范式）

- prereg append-only，判据先冻结后跑；判决数字零手抄（值-文件-键三元组
  gates.json，派发器复算未过不推）。
- 双账本对账（力/冲量/断裂能量）机器精度级；跨跑逐位门禁用（#268 DG0 教训）。
- GPU 纪律：GPU5 唯一（CUDA_VISIBLE_DEVICES=5），GPU5_BUSY 协议排队；
  B 线算力在飞时本战役计算腿先走 CPU 锚。GPU0-4 他人服务禁碰。
- 长计算 nohup 分离 + DONE.json 完成清单（launch-harvest 分离）。
- merge 留 owner；合格自动推送（commit+门绿亲验+审批过+记录净）。
- 时区：所有 prereg/报告时间戳 UTC Z 后缀（dispatch 日志 CST 标 Z 教训）。

## 与既有线的关系

- **wedge_icebreaking**：技术先导线。S0-S3 已闭环；**S4（闭环双向交换）是
  SI-3 的直接前置**，随本战役重启（中继恢复后重派）。
- **suboff_sail**：SUBOFF 几何与 LBM 流体基础的 verified 来源（SI-A 复用）。
- **B 线 benchmark closure**：互不阻断（GPU 排队协调）。
