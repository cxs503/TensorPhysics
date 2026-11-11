# J2 弹塑性梁/板验证面（beam_hinge / plate_limit）

节点 p387w2-n06。模块位于 `src/tensorphysics/plasticity/`，验收harness `verification_hinge.run_all_criteria(fast=False)`，完整论文级报告见 `docs/reports/p387w2_n06/REPORT.md`，证据 `artifacts/p387w2_n06/evidence.json`。

## 能力

| 模块 | 能力 | 独立参考 |
|---|---|---|
| `beam_hinge.FiberSection` | 矩形/I 形/圆纤维截面，含闭式 A/I/W_el/Z | 闭式积分 |
| `beam_hinge.section_response` | 单轴 J2 径向返回映射（子增量），M/切线/耗散 | 矩形理想塑性闭式 M(κ) |
| `beam_hinge.moment_curvature` / `unload_from` | M-κ 路径与卸载回弹残余曲率 | 闭式 + 弹性卸载恒等式 |
| `beam_hinge.solve_beam` | 位移法梁有限元 + 分布式纤维塑性 + 铰定位 | 极限分析机构（0.585786L / 0.3L） |
| `plate_limit` | 矩形四角铰 / 条带中线铰 / 圆板锥形机构上限解 | 24 / 8 / 6 × M_p 归一化 |
| `verification_hinge` | (a)-(e) 判据 + 负控的统一裁决 | — |

## 运行

```bash
python -m pytest tests/test_j2_beam_plate_hinge.py -q      # 17 用例，约 11 s
python generate_p387w2_n06_figures.py                      # 落盘 evidence.json + 5 图
python publish_p387w2_n06_report.py --publish              # 生成并发布报告
```

## 判据结果（机器注入）

| 判据 | 模型 | 参考 | 误差 | 门限 | 判定 |
|---|---|---|---|---|---|
| `a_elastic_EI_rectangle` | 1.4e+13 | 1.4e+13 | 1.395e-16 | 0.02 | PASS |
| `a_yield_M_rectangle` | 1.56667e+08 | 1.56667e+08 | 1.902e-16 | 0.03 | PASS |
| `a_plastic_M_rectangle` | 2.34787e+08 | 2.35e+08 | 9.057e-04 | 0.05 | PASS |
| `a_elastic_EI_ibeam` | 2.26516e+13 | 2.26516e+13 | 0.000e+00 | 0.02 | PASS |
| `a_yield_M_ibeam` | 1.68988e+08 | 1.68988e+08 | 0.000e+00 | 0.03 | PASS |
| `a_plastic_M_ibeam` | 1.93475e+08 | 1.93522e+08 | 2.441e-04 | 0.05 | PASS |
| `a_elastic_EI_circle` | 1.64935e+13 | 1.64934e+13 | 1.118e-05 | 0.02 | PASS |
| `a_yield_M_circle` | 1.84571e+08 | 1.84569e+08 | 1.118e-05 | 0.03 | PASS |
| `a_plastic_M_circle` | 3.12947e+08 | 3.13333e+08 | 1.233e-03 | 0.05 | PASS |
| `b_hinge_propped_cantilever` | 0.586294 | 0.585786 | 5.072e-04 | 0.01 | PASS |
| `b_converged_propped_cantilever` | 1 | 1 | 0.000e+00 | 0 | PASS |
| `b_hinge_simply_supported` | 0.303042 | 0.3 | 3.042e-03 | 0.01 | PASS |
| `c_springback_rectangle_2kappay` | 7.00677e-06 | 6.99405e-06 | 1.819e-03 | 0.05 | PASS |
| `c_unload_slope_rectangle_2kappay` | 1.4e+13 | 1.4e+13 | 2.790e-16 | 0.01 | PASS |
| `c_springback_rectangle_4kappay` | 2.83266e-05 | 2.83259e-05 | 2.402e-05 | 0.05 | PASS |
| `c_unload_slope_rectangle_4kappay` | 1.4e+13 | 1.4e+13 | 1.395e-16 | 0.01 | PASS |
| `c_springback_rectangle_6kappay` | 5.05158e-05 | 5.05126e-05 | 6.381e-05 | 0.05 | PASS |
| `c_unload_slope_rectangle_6kappay` | 1.4e+13 | 1.4e+13 | 1.395e-16 | 0.01 | PASS |
| `c_springback_ibeam_2kappay` | 6.5683e-06 | 6.5683e-06 | 2.579e-16 | 0.05 | PASS |
| `c_unload_slope_ibeam_2kappay` | 2.26517e+13 | 2.26516e+13 | 3.449e-16 | 0.01 | PASS |
| `c_springback_ibeam_4kappay` | 2.13475e-05 | 2.13475e-05 | 0.000e+00 | 0.05 | PASS |
| `c_unload_slope_ibeam_4kappay` | 2.26516e+13 | 2.26516e+13 | 1.724e-16 | 0.01 | PASS |
| `c_springback_ibeam_6kappay` | 3.62396e-05 | 3.62396e-05 | 0.000e+00 | 0.05 | PASS |
| `c_unload_slope_ibeam_6kappay` | 2.26517e+13 | 2.26516e+13 | 1.724e-16 | 0.01 | PASS |
| `c_springback_circle_2kappay` | 5.66934e-06 | 5.66916e-06 | 3.295e-05 | 0.05 | PASS |
| `c_unload_slope_circle_2kappay` | 1.64935e+13 | 1.64934e+13 | 1.118e-05 | 0.01 | PASS |
| `c_springback_circle_4kappay` | 2.6353e-05 | 2.63528e-05 | 7.809e-06 | 0.05 | PASS |
| `c_unload_slope_circle_4kappay` | 1.64935e+13 | 1.64934e+13 | 1.118e-05 | 0.01 | PASS |
| `c_springback_circle_6kappay` | 4.84063e-05 | 4.84061e-05 | 4.327e-06 | 0.05 | PASS |
| `c_unload_slope_circle_6kappay` | 1.64935e+13 | 1.64934e+13 | 1.118e-05 | 0.01 | PASS |
| `d_energy_section_rectangle` | 1.42027e-05 | 0 | 1.420e-05 | 0.01 | PASS |
| `d_energy_section_ibeam` | 2.19759e-05 | 0 | 2.198e-05 | 0.01 | PASS |
| `d_energy_section_circle` | 1.71686e-05 | 0 | 1.717e-05 | 0.01 | PASS |
| `d_energy_beam` | 8.1002e-05 | 0 | 8.100e-05 | 0.01 | PASS |
| `p_square_plate` | 24 | 24 | 1.480e-16 | 0.001 | PASS |
| `p_circular_plate` | 5.99998 | 6 | 3.125e-06 | 0.001 | PASS |
| `p_strip_plate` | 8 | 8 | 0.000e+00 | 0.001 | PASS |

负控（必须 FAIL）：

| 负控 | 误差 | 门限 | 判定 |
|---|---|---|---|
| `e_nc1_geometry_transcription` | 2.100e-01 | 0.03 | FAIL(预期) |
| `e_nc2_fabricated_hinge` | 8.579e-02 | 0.01 | FAIL(预期) |
| `e_nc3_elastic_springback` | 1.000e+00 | 0.05 | FAIL(预期) |

