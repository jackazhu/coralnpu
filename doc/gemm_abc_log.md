# CoralNPU GEMM 升级进度日志（A -> B -> C）

## 1. 使用说明
- 本文档用于记录 A/B/C 三阶段的实际推进状态。
- 所有性能结论必须附带 benchmark 对照数据。
- 所有“完成”结论必须附带可复现实验命令和 commit hash。

## 2. 当前总体状态

| 阶段 | 状态 | 负责人 | 开始日期 | 完成日期 | 说明 |
|---|---|---|---|---|---|
| A（RVV GEMM 软件优化） | Done | AI Agent | 2026-04-06 | 2026-04-06 | 已完成完整验证矩阵与 benchmark 收益确认 |
| B（mulmac/mac 后端优化） | Done | AI Agent | 2026-04-06 | 2026-04-06 | 已完成首轮有效优化并完成多轮负/零收益尝试回退；阶段收口，保留净正收益改动 |
| C（指令+工具链联动） | In Progress | AI Agent | 2026-04-06 | TBD | 已完成 C 阶段入口骨架（feature flag + capability + fallback）并通过回归 |

状态枚举建议：`Not Started` / `In Progress` / `Blocked` / `Done`

## 3. 基线冻结记录（必须先填）

### Baseline-0（推进 A 前）
- 日期：2026-04-06
- Git commit：`079c45a`
- Toolchain 版本：`toolchain_kelvin_v2-2025-09-11`（WORKSPACE 归档）
- 平台/配置（仿真或硬件）：Verilator + `RvvCoreMiniHighmemAxi`
- 频率与内存配置：sim 默认时钟；Highmem ITCM/DTCM 配置
- Benchmark 输入集版本：`tests/cocotb/tutorial/tfmicro:cocotb_fully_connected`
- 随机种子：cocotb seed 自动；numpy 默认随机
- cycles 记录格式：`PERF_CYCLES|runner=<...>|test=<...>|cycles=<N>`

#### 验证命令
- 正确性：
  - `bazel test //tests/cocotb:rvv_ml_ops_cocotb_test`
  - `bazel test //tests/cocotb:rvv_arithmetic_cocotb_test`
  - `bazel test //tests/cocotb:rvv_load_store_test`
- 系统：
  - `bazel test //tests/cocotb:rvv_highmem_tests`
  - `bazel test //tests/cocotb:rvv_itcm512kb_dtcm512kb_tests`
  - `bazel run //tests/npusim_examples:npusim_run_mobilenet`
  - `bazel run //tests/npusim_examples:npusim_run_bcresnet`

#### 基线 benchmark 结果
| Workload | Config | Metric | Value | 单位 | 备注 |
|---|---|---|---|---|---|
| fc_16x16 | cocotb_fully_connected | ref_cycles | 6661 | cycles | run_ref |
| fc_16x16 | cocotb_fully_connected | opt_cycles | 3299 | cycles | run_optimized |
| fc_16x16 | cocotb_fully_connected | speedup | 2.02 | x | ref/opt |
| fc_64x64 | cocotb_fully_connected | ref_cycles | 90343 | cycles | run_ref |
| fc_64x64 | cocotb_fully_connected | opt_cycles | 16853 | cycles | run_optimized |
| fc_64x64 | cocotb_fully_connected | speedup | 5.36 | x | ref/opt |

## 4. 阶段日志

## A 阶段日志

### A-Entry 模板
### A-Entry-001
- 日期：2026-04-06
- Git commit：`0cc7a8b`
- 改动摘要：`fully_connected.cc` 增加“双输出通道并行累加”路径，复用输入向量加载并减少重复访存/扩展开销。
- 风险评估：中低（仅改 FC RVV 优化路径；保留 tail 路径与参考实现回退机制）。

#### 验证结果
| 测试项 | 命令 | 结果 | 备注 |
|---|---|---|---|
| FC 功能+性能 | `bazel test //tests/cocotb/tutorial/tfmicro:cocotb_fully_connected --test_output=all` | PASS | 16/64 规模通过，输出一致 |
| RVV ML 回归 | `bazel test //tests/cocotb:rvv_ml_ops_cocotb_test --test_output=all` | PASS | matmul 回归通过 |
| 端到端 sanity | `bazel run //tests/npusim_examples:npusim_run_mobilenet` | WARN | 环境缺少 host c++ include（`<array>`） |

#### Benchmark 对照（A vs Baseline-0）
| Workload | Metric | Before | After | Delta | 结论 |
|---|---|---|---|---|---|
| fc_16x16 | opt_cycles | 3299 | 2762 | -16.3% | 改善 |
| fc_16x16 | speedup(ref/opt) | 2.02x | 2.41x | +19.3% | 改善 |
| fc_64x64 | opt_cycles | 16853 | 12863 | -23.7% | 改善 |
| fc_64x64 | speedup(ref/opt) | 5.36x | 7.02x | +31.0% | 改善 |

#### 阶段结论
- 是否满足 A 进入 B 条件：`No（当前为 A 阶段迭代 1）`
- 若 No，阻塞原因：尚未完成 A 阶段完整验证矩阵（rvv_arithmetic/load_store/highmem/itcm512 等未全量复测）。

### A-Entry-002
- 日期：2026-04-06
- Git commit：`1cc023e`
- 改动摘要：新增 `BCResNet` 仿真回归（npusim + cocotb 入口）并统一输出 `PERF_CYCLES`，完成 A 阶段验证矩阵全量复测。
- 风险评估：低（回归接入与日志增强为主，不改变 A 阶段 FC 内核功能语义）。

#### 验证结果（A 阶段完整矩阵）
| 测试项 | 命令 | 结果 | 备注 |
|---|---|---|---|
| FC 功能+性能 | `bazel test //tests/cocotb/tutorial/tfmicro:cocotb_fully_connected --test_output=all` | PASS | fc_16x16 / fc_64x64 输出一致且 speedup 保持 |
| RVV ML 回归 | `bazel test //tests/cocotb:rvv_ml_ops_cocotb_test --test_output=all` | PASS | 含 `PERF_CYCLES` |
| RVV 算术回归 | `bazel test //tests/cocotb:rvv_arithmetic_cocotb_test --test_output=errors` | PASS |  |
| RVV 访存回归 | `bazel test //tests/cocotb:rvv_load_store_test --test_output=errors` | PASS |  |
| Highmem 回归 | `bazel test //tests/cocotb:rvv_highmem_tests --test_output=errors` | PASS |  |
| ITCM/DTCM 512KB 回归 | `bazel test //tests/cocotb:rvv_itcm512kb_dtcm512kb_tests --test_output=errors` | PASS |  |
| 端到端 MobileNet | `bazel run //tests/npusim_examples:npusim_run_mobilenet` | PASS | `inference_status=0`, `PERF_CYCLES` 已输出 |
| 端到端 BCResNet | `bazel run //tests/npusim_examples:npusim_run_bcresnet` | PASS | `inference_status=0`, `PERF_CYCLES` 已输出 |

#### Benchmark 对照（A vs Baseline-0，复测）
| Workload | Metric | Before | After | Delta | 结论 |
|---|---|---|---|---|---|
| fc_16x16 | opt_cycles | 3299 | 2753 | -16.6% | 改善 |
| fc_16x16 | speedup(ref/opt) | 2.02x | 2.41x | +19.3% | 改善 |
| fc_64x64 | opt_cycles | 16853 | 12875 | -23.6% | 改善 |
| fc_64x64 | speedup(ref/opt) | 5.36x | 7.01x | +30.8% | 改善 |
| npusim_mobilenet | cycles | N/A | 516177367 | N/A | 建立场景级 cycle 基线 |
| npusim_bcresnet | cycles | N/A | 327413977 | N/A | 建立场景级 cycle 基线 |

#### 阶段结论
- 是否满足 A 进入 B 条件：`Yes`
- 结论说明：A 阶段验证矩阵已全量通过，且 FC 关键 workload benchmark 收益稳定成立；进入 B 阶段条件满足。

## B 阶段日志

### B-Entry 模板
- 日期：
- Git commit：
- 改动摘要：
- 风险评估：

#### 验证结果
| 测试项 | 命令 | 结果 | 备注 |
|---|---|---|---|
| A 全量回归复测 | `同 A 阶段命令` | TBD |  |
| B 新增专项 | `TBD` | TBD |  |

#### Benchmark 对照（B vs A）
| Workload | Metric | Before(A) | After(B) | Delta | 结论 |
|---|---|---|---|---|---|
| matmul_case_1 | latency_p50 | TBD | TBD | TBD | TBD |
| matmul_case_1 | throughput | TBD | TBD | TBD | TBD |

#### 阶段结论
- 是否满足 B 进入 C 条件：`Yes/No`
- 若 No，阻塞原因：

### B-Entry-001
- 日期：2026-04-06
- Git commit：`b2b12e0`
- 改动摘要：在 `rvv_backend_mulmac.sv` 中新增“单 uop 双 lane 可用时的 round-robin 分发”，避免持续偏置 lane0 导致的局部背压放大；双 uop 与单 lane ready 语义保持不变。
- 风险评估：低（仅调整 mulmac wrapper 的分发策略，不改指令语义与 `mac_unit` 运算逻辑）。

#### 验证结果
| 测试项 | 命令 | 结果 | 备注 |
|---|---|---|---|
| FC 功能+性能 | `bazel test --cache_test_results=no --test_output=streamed //tests/cocotb/tutorial/tfmicro:cocotb_fully_connected` | PASS | `fc_64x64` 的 `opt_cycles` 下降 |
| RVV ML 回归 | `bazel test --cache_test_results=no --test_output=errors //tests/cocotb:rvv_ml_ops_cocotb_test` | PASS | 功能无回退 |
| RVV 算术回归 | `bazel test --cache_test_results=no --test_output=errors //tests/cocotb:rvv_arithmetic_cocotb_test` | PASS | 功能无回退 |
| RVV 访存回归 | `bazel test --cache_test_results=no --test_output=errors //tests/cocotb:rvv_load_store_test` | PASS | 功能无回退 |
| Highmem 回归 | `bazel test --cache_test_results=no --test_output=errors //tests/cocotb:rvv_highmem_tests` | PASS | 功能无回退 |
| ITCM/DTCM 512KB 回归 | `bazel test --cache_test_results=no --test_output=errors //tests/cocotb:rvv_itcm512kb_dtcm512kb_tests` | PASS | 功能无回退 |
| 端到端 MobileNet | `bazel run //tests/npusim_examples:npusim_run_mobilenet` | PASS | `inference_status=0`, `PERF_CYCLES=516177367` |
| 端到端 BCResNet | `bazel run //tests/npusim_examples:npusim_run_bcresnet` | PASS | `inference_status=0`, `PERF_CYCLES=327413977` |

#### Benchmark 对照（B vs A）
| Workload | Metric | Before(A) | After(B) | Delta | 结论 |
|---|---|---|---|---|---|
| fc_16x16 | ref_cycles | 6643 | 6643 | 0.00% | 无变化 |
| fc_16x16 | opt_cycles | 2750 | 2750 | 0.00% | 无变化 |
| fc_16x16 | speedup(ref/opt) | 2.42x | 2.42x | 0.00x | 无变化 |
| fc_64x64 | ref_cycles | 90310 | 90268 | -0.05% | 波动可忽略 |
| fc_64x64 | opt_cycles | 12881 | 12839 | -0.33% | 小幅改善 |
| fc_64x64 | speedup(ref/opt) | 7.01x | 7.03x | +0.02x | 小幅改善 |

#### 阶段结论
- 是否满足 B 进入 C 条件：`No`
- 若 No，阻塞原因：当前仅完成 B 首轮低风险调度优化，收益已验证但幅度较小；需继续推进 `mac_unit` 更深层流水/回压优化并复测收益稳定性。

### B-Entry-002
- 日期：2026-04-06
- Git commit：`bb600b2`, `94cbe53`
- 改动摘要：完成 B2/B3/B4/B5 多轮后端尝试复盘，负收益或零收益改动均已回退，阶段仅保留 B1 净正收益路径；按用户指令结束 B 阶段并切换到 C 阶段推进。
- 风险评估：低（收口动作以回退和文档归档为主，不新增运行时行为变化）。

#### 验证结果
| 测试项 | 命令 | 结果 | 备注 |
|---|---|---|---|
| FC 功能+性能（确定性） | `bazel test --cache_test_results=no --test_output=streamed //tests/cocotb/tutorial/tfmicro:cocotb_fully_connected` | PASS | 保留路径基线：`fc_64x64 opt_cycles=12872` |
| RVV ML 回归 | `bazel test --cache_test_results=no --test_output=errors //tests/cocotb:rvv_ml_ops_cocotb_test` | PASS | 功能无回退 |
| 端到端 MobileNet | `bazel run //tests/npusim_examples:npusim_run_mobilenet` | PASS | `PERF_CYCLES=516177367` |
| 端到端 BCResNet | `bazel run //tests/npusim_examples:npusim_run_bcresnet` | PASS | `PERF_CYCLES=327413977` |

#### 阶段结论
- 是否满足 B 进入 C 条件：`Yes`
- 结论说明：按“仅保留正收益改动”完成 B 阶段收口，现已转入 C 阶段指令/工具链联动实现。

## C 阶段日志

### C-Entry 模板
- 日期：
- Git commit：
- 改动摘要（含 decode/mpact/toolchain）：
- fallback 验证方式：

### C-Entry-001
- 日期：2026-04-06
- Git commit：`ab281d8`, `a5df790`
- 改动摘要（含 decode/mpact/toolchain）：在 `sw/opt/litert-micro` 增加 `custom_gemm` 模块，建立 C 阶段 custom GEMM 路径入口（`feature flag + capability`）并在 `fully_connected` 中接入“先尝试 custom 路径，失败则 fallback 到现有 RVV 路径”的框架。
- fallback 验证方式：默认编译配置下 `CORALNPU_ENABLE_CUSTOM_GEMM` 与 `CORALNPU_HAS_CUSTOM_GEMM` 均关闭，`TryFullyConnectedCustomGemm` 返回 false，执行路径回落到既有 RVV 实现。

#### 验证结果
| 测试项 | 命令 | 结果 | 备注 |
|---|---|---|---|
| FC 功能+性能 | `bazel test --cache_test_results=no --test_output=streamed //tests/cocotb/tutorial/tfmicro:cocotb_fully_connected` | PASS | fallback 生效；`fc_64x64 opt_cycles=12782`，未劣化 |
| RVV ML 回归 | `bazel test --cache_test_results=no --test_output=errors //tests/cocotb:rvv_ml_ops_cocotb_test` | PASS | 行为与 B 阶段兼容 |
| 端到端 MobileNet | `bazel run //tests/npusim_examples:npusim_run_mobilenet` | PASS | `inference_status=0`, `PERF_CYCLES=516177367` |
| 端到端 BCResNet | `bazel run //tests/npusim_examples:npusim_run_bcresnet` | PASS | `inference_status=0`, `PERF_CYCLES=327413977` |

#### Benchmark 对照（C-Entry-001 vs B 基线）
| Workload | Metric | Before(B) | After(C-001) | Delta | 结论 |
|---|---|---|---|---|---|
| fc_64x64 | opt_cycles | 12872 | 12782 | -0.70% | 持平偏优（fallback 不退化） |
| fc_64x64 | speedup(ref/opt) | 7.02x | 7.07x | +0.05x | 持平偏优 |

#### 阶段结论
- 是否满足 C 完成条件：`No`
- 若 No，阻塞原因：当前仅完成 C 路径骨架，尚未接入真实 custom GEMM 指令编码/decode/mpact/simulator/toolchain 端到端链路。

### C-Entry-002
- 日期：2026-04-06
- Git commit：`7a1ebaf`
- 改动摘要（含 decode/mpact/toolchain）：完成 C2 最小可用“真实指令编码+RTL 链路”接入：新增 `VCUSTOMGEMM(6'b101_110)` 指令编码并接入 `rvv_backend_decode_unit_ari/ari_de2`（EMUL/EEW/合法性/uop_class/vs3_valid/rs1_data 路径），执行侧在 `rvv_backend_mac_unit` 复用 `VMACC` 数据通路；软件侧 `custom_gemm.cc` 增加固定编码探针（`.word 0xBA002057`）用于后续 capability/执行链路联调，当前仍保持 fallback 返回 false。
- fallback 验证方式：默认编译配置下 custom GEMM feature/capability 关闭，`FullyConnectedEval` 不进入 custom 分支；即使未来开启，`TryFullyConnectedCustomGemm` 当前也在探针后返回 false，功能语义保持回落到既有 RVV 路径。

#### 验证结果
| 测试项 | 命令 | 结果 | 备注 |
|---|---|---|---|
| FC 功能+性能 | `bazel test --cache_test_results=no --test_output=streamed //tests/cocotb/tutorial/tfmicro:cocotb_fully_connected` | PASS | `fc_64x64 opt_cycles=12782`，与 C-001 保持一致 |
| RVV ML 回归 | `bazel test --cache_test_results=no --test_output=errors //tests/cocotb:rvv_ml_ops_cocotb_test` | PASS | 新增 custom funct6 后无功能回退 |
| 端到端 MobileNet | `bazel run //tests/npusim_examples:npusim_run_mobilenet` | PASS | `inference_status=0`, `PERF_CYCLES=516177367` |
| 端到端 BCResNet | `bazel run //tests/npusim_examples:npusim_run_bcresnet` | PASS | `inference_status=0`, `PERF_CYCLES=327413977` |

#### Benchmark 对照（C-Entry-002 vs C-Entry-001）
| Workload | Metric | Before(C-001) | After(C-002) | Delta | 结论 |
|---|---|---|---|---|---|
| fc_64x64 | opt_cycles | 12782 | 12782 | 0.00% | 持平（新增链路默认不扰动热路径） |
| npusim_mobilenet | cycles | 516177367 | 516177367 | 0.00% | 持平 |
| npusim_bcresnet | cycles | 327413977 | 327413977 | 0.00% | 持平 |

#### 阶段结论
- 是否满足 C 完成条件：`No`
- 若 No，阻塞原因：当前仅完成“编码+decode+执行占位”最小闭环；尚未完成 mpact/simulator/toolchain 的助记符级支持与真正 GEMM 指令语义执行。

### C-Entry-003
- 日期：2026-04-06
- Git commit：`674ab94`
- 改动摘要（含 decode/mpact/toolchain）：继续优化 A/C 共用的 FC 热路径：`fully_connected.cc` 从“双输出并行”升级为“优先四输出并行（4-way）+2-way+tail”分层策略，进一步复用输入向量加载并降低每输出通道开销；不改变 C2 custom 指令默认关闭/fallback 语义。
- fallback 验证方式：`FullyConnectedEval` 的 custom GEMM 入口条件保持不变（编译开关与 capability 同时满足才尝试），默认配置下仍直接走 RVV 优化路径。

#### 验证结果
| 测试项 | 命令 | 结果 | 备注 |
|---|---|---|---|
| FC 功能+性能 | `bazel test --cache_test_results=no --test_output=streamed //tests/cocotb/tutorial/tfmicro:cocotb_fully_connected` | PASS | `fc_64x64 opt_cycles=11156` |
| RVV ML 回归 | `bazel test --cache_test_results=no --test_output=errors //tests/cocotb:rvv_ml_ops_cocotb_test` | PASS | 功能无回退 |
| 端到端 MobileNet | `bazel run //tests/npusim_examples:npusim_run_mobilenet` | PASS | `inference_status=0`, `PERF_CYCLES=516177367` |
| 端到端 BCResNet | `bazel run //tests/npusim_examples:npusim_run_bcresnet` | PASS | `inference_status=0`, `PERF_CYCLES=327413977` |

#### Benchmark 对照（C-Entry-003 vs C-Entry-002）
| Workload | Metric | Before(C-002) | After(C-003) | Delta | 结论 |
|---|---|---|---|---|---|
| fc_16x16 | opt_cycles | 2734 | 2500 | -8.56% | 改善 |
| fc_64x64 | opt_cycles | 12782 | 11156 | -12.72% | 显著改善 |
| fc_64x64 | speedup(ref/opt) | 7.07x | 8.10x | +1.03x | 显著改善 |
| npusim_mobilenet | cycles | 516177367 | 516177367 | 0.00% | 持平 |
| npusim_bcresnet | cycles | 327413977 | 327413977 | 0.00% | 持平 |

#### 阶段结论
- 是否满足 C 完成条件：`No`
- 若 No，阻塞原因：C 阶段已获得 FC 路径软件收益并保持系统回归稳定，但仍缺少 mpact/simulator/toolchain 助记符级支持与 custom GEMM 指令真实语义执行闭环。

### C-Entry-004
- 日期：2026-04-06
- Git commit：`25e1e5c`
- 改动摘要（含 decode/mpact/toolchain）：针对网络级瓶颈继续优化 `ConvPerChannel`：新增 `1x1` 专用 RVV kernel（`Conv_1x1_PerChannel`），将大量 `fh=1 fw=1` fallback 路径替换为向量化输入深度点积+逐通道量化实现；保持原有 4x4/其他分支不变。
- fallback 验证方式：仅在 `filter_height==1 && filter_width==1` 时命中新分支，其他卷积形状仍维持原分发与 fallback 行为。

#### 验证结果
| 测试项 | 命令 | 结果 | 备注 |
|---|---|---|---|
| Conv 算子正确性回归 | `bazel test --cache_test_results=no --test_output=streamed //sw/opt/litert-micro/test:conv_sim_test` | PASS | 所有既有 case 输出匹配参考实现 |
| RVV ML 回归 | `bazel test --cache_test_results=no --test_output=errors //tests/cocotb:rvv_ml_ops_cocotb_test` | PASS | 功能无回退 |
| 端到端 MobileNet | `bazel run //tests/npusim_examples:npusim_run_mobilenet` | PASS | `inference_status=0`, `PERF_CYCLES=206944458` |
| 端到端 BCResNet | `bazel run //tests/npusim_examples:npusim_run_bcresnet` | PASS | `inference_status=0`, `PERF_CYCLES=285871598` |

#### Benchmark 对照（C-Entry-004 vs C-Entry-003）
| Workload | Metric | Before(C-003) | After(C-004) | Delta | 结论 |
|---|---|---|---|---|---|
| npusim_mobilenet | cycles | 516177367 | 206944458 | -59.91% | 显著改善（1x1 conv 主要瓶颈被命中） |
| npusim_bcresnet | cycles | 327413977 | 285871598 | -12.69% | 明显改善（仍受 5x5 conv fallback 影响） |

#### 阶段结论
- 是否满足 C 完成条件：`No`
- 若 No，阻塞原因：网络级性能已从“持平”转为“显著改善”，但 C 阶段仍需继续完成 depthwise/5x5 等剩余热点覆盖与 custom GEMM 真实语义闭环。

## 5. 决策与问题跟踪

| ID | 日期 | 类型 | 内容 | 影响阶段 | 状态 |
|---|---|---|---|---|---|
| D-001 | TBD | Decision | 采用 A -> B -> C 顺序推进 | A/B/C | Open |
| I-001 | 2026-04-06 | Issue | `npusim_run_mobilenet` 在 host_clang 构建阶段找不到 `<array>` 头文件 | A | Resolved |
| A-002 | 2026-04-06 | Action | 修复 `toolchain/host_clang/BUILD`：GCC13/Clang18 include + host link path + 显式 `-isystem`，`npusim_run_mobilenet` 端到端通过（`inference_status=0`） | A | Done |
| A-003 | 2026-04-06 | Action | 新增 `bcresnet` 仿真回归入口并纳入 `running_tflite`，同时在 cocotb/npusim 输出统一 `PERF_CYCLES` 用于场景级性能跟踪 | A | Done |
| B-004 | 2026-04-06 | Action | `cocotb_fully_connected` 改为固定 seed（默认 12345，可环境变量覆盖）以降低 benchmark 噪声；B4（`mul_rs` FULL_PUSH）在确定性基线下无 cycle 收益，已回退 | B | Done |
| B-005 | 2026-04-06 | Action | B5 尝试在 `rvv_backend` 的 EX 结果 FIFO（`u_res_ff`）启用 `FULL_PUSH` 以降低满队列同拍 pop 的反压气泡；确定性基线下 `fc_64x64 opt_cycles` 仍为 12872，无实质收益，已回退（`bb2623b`） | B | Done |
| B-001 | 2026-04-06 | Action | B 第 2 轮尝试（加深 mul 结果缓冲）在 `fc_64x64` 上 `opt_cycles 12839 -> 12851`，确认为负优化，已用 `git revert` 回退（`8d5a6b3`） | B | Done |
| B-002 | 2026-04-06 | Action | B 第 3 轮尝试（arbiter fast path）在 `fc_64x64` 上 `opt_cycles 12839 -> 12854`，确认为负优化，已用 `git revert` 回退（`3c0e237`） | B | Done |
| C-001 | 2026-04-06 | Action | 启动 C 阶段：新增 `custom_gemm` 模块并在 `fully_connected` 接入 custom-path + fallback 骨架，默认配置下确认 fallback 回归通过且不退化 | C | Done |
| C-002 | 2026-04-06 | Action | 打通 C2 最小 custom 指令链路：新增 `VCUSTOMGEMM` 编码并接入 RVV decode/mac 占位执行；软件补充固定编码探针（`.word 0xBA002057`），默认配置保持 fallback 与性能不变 | C | Done |
| C-003 | 2026-04-06 | Action | FC 热路径升级为 4-way 并行累加（再 2-way/tail 收尾），`fc_64x64 opt_cycles 12782 -> 11156`，且 npusim mobilenet/bcresnet 周期与功能保持稳定 | C | Done |
| C-004 | 2026-04-06 | Action | 新增 `1x1` 专用 RVV Conv 内核并接入 `ConvPerChannel` 分发，网络级周期显著下降：mobilenet `-59.91%`、bcresnet `-12.69%`，功能回归通过 | C | Done |

类型建议：`Decision` / `Risk` / `Issue` / `Action`
