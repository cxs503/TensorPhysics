# Mission 8 补报：SDAA 加速卡点亮 e2e 验证

**任务编号**: m8（补报单 m99）  
**仓库**: TensorPhysics  
**分支**: agent/m8  
**执行时间**: 2026-10（原单）；补报 2026-10  
**报告性质**: honest-FAIL 补报——原单成功交付但缺论文级报告  

---

## 1. 目的

验证 SDAA（思腾合力加速卡）在 TensorPhysics 运行环境中能否被 PyTorch 生态正确识别并完成基本张量计算。这是硬件点亮（light-up）级别的端到端验证，目标仅为确认：

1. `torch_sdaa` 扩展可正常导入；
2. 加速卡设备可被枚举并返回有效设备名；
3. 基本矩阵乘法可在 SDAA 设备上执行并同步返回。

**本验证不涉及任何业务代码修改，唯一交付物为探针输出文件 `sdaa_probe.txt`。**

---

## 2. 方法

### 2.1 探针命令

在工作区根目录 `/root/tp-agent/work/m8` 执行：

```python
import torch_sdaa, torch, time
print("card", torch.sdaa.device_count(), torch.sdaa.get_device_name(0))
x = torch.randn(1000, 1000).to("sdaa")
t = time.time()
y = x @ x.T
torch.sdaa.synchronize()
print("matmul_ms", round((time.time() - t) * 1000, 1))
```

### 2.2 验证维度

| 维度 | 检查内容 | 通过标准 |
|------|----------|----------|
| 驱动/扩展 | `import torch_sdaa` 无异常 | 零 ImportError |
| 设备枚举 | `torch.sdaa.device_count() >= 1` | 至少 1 卡可见 |
| 设备标识 | `get_device_name(0)` 返回非空字符串 | 有效设备路径 |
| 计算正确性 | 1000×1000 matmul 完成并同步 | 无运行时异常 |
| 性能基线 | matmul 耗时记录 | 数值有限、可复现 |

### 2.3 证据链

- 原始输出持久化为 `sdaa_probe.txt`（未跟踪文件，未入 git）
- 输出经程序化提取（零手抄），数字直接注入本报告

---

## 3. 关键结果

### 3.1 探针输出（程序化提取）

| 指标 | 值 | 来源 |
|------|-----|------|
| 加速卡数量 | **1** | `torch.sdaa.device_count()` |
| 设备名称 | **/dev/tcaicard5** | `torch.sdaa.get_device_name(0)` |
| 矩阵乘法耗时 | **12.9 ms** | `time.time()` 差值，1000×1000 float32 matmul + synchronize |

### 3.2 原始输出

```
card 1 /dev/tcaicard5
matmul_ms 12.9
```

### 3.3 判定

| 检查项 | 结果 |
|--------|------|
| torch_sdaa 导入 | ✅ PASS（有 pynvml FutureWarning，不影响功能） |
| 设备枚举 ≥ 1 | ✅ PASS（card_count = 1） |
| 设备名有效 | ✅ PASS（/dev/tcaicard5） |
| matmul 执行+同步 | ✅ PASS（12.9 ms，有限值） |
| **总体** | **✅ PASS — 硬件点亮成功** |

---

## 4. 审计与局限

### 4.1 本验证证明了什么

- SDAA 加速卡驱动已安装且对 `torch_sdaa` 扩展可见
- PyTorch 张量可迁移至 SDAA 设备并执行基本线性代数运算
- 设备同步机制正常工作

### 4.2 本验证未证明什么（honest-FAIL 披露）

1. **非科学验证**：1000×1000 随机矩阵乘法是硬件烟雾测试，不代表任何科学计算工作负载的精度或性能
2. **无对照基线**：未与 CPU/CUDA 同规模 matmul 做对比，无法判断 12.9 ms 是否合理
3. **无精度验证**：未检查 SDAA 计算结果与 CPU 参考值的数值一致性（element-wise 误差）
4. **无业务集成**：未将 SDAA 接入 TensorPhysics 的任何求解器路径（TensorDEM/TensorLBM/TensorFEM/TensorFVM）
5. **无持久化**：探针输出为未跟踪文件，未入 git，无 checksum 验证
6. **单卡单点**：仅验证 1 张卡（/dev/tcaicard5），未测试多卡/分布式场景
7. **环境警告**：存在 `pynvml` FutureWarning，提示依赖链可能需要更新

### 4.3 后续建议

| 优先级 | 行动 | 目的 |
|--------|------|------|
| P1 | 将 `sdaa_probe.txt` 纳入 git 或 artifact store | 证据持久化 |
| P2 | 增加 CPU 对照 matmul benchmark | 建立性能基线 |
| P3 | 增加数值精度校验（SDAA vs CPU atol/rtol） | 计算正确性 |
| P4 | 在 TensorDEM/TensorLBM 求解器中试跑 SDAA 后端 | 业务集成验证 |
| P5 | 多卡/分布式通信测试 | 扩展性验证 |

---

## 5. 结论

Mission 8 的 SDAA 加速卡点亮 e2e 验证在硬件层面 **PASS**：1 张 SDAA 卡（/dev/tcaicard5）被正确识别，1000×1000 矩阵乘法在 12.9 ms 内完成并同步。但此验证仅为硬件烟雾测试，不构成科学计算精度或业务集成的证据。原单声明"e2e 验证完成"在硬件点亮意义上成立，在科学计算端到端意义上不成立——此为 honest-FAIL 披露。

补报单（m99）不修改任何业务代码，仅补充本论文级报告并注册至 TensorPlatform。
