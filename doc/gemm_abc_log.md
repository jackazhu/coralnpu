# CoralNPU GEMM 升级进度日志（A -> B -> C）

## 1. 使用说明
- 本文档用于记录 A/B/C 三阶段的实际推进状态。
- 所有性能结论必须附带 benchmark 对照数据。
- 所有“完成”结论必须附带可复现实验命令和 commit hash。

## 2. 当前总体状态

| 阶段 | 状态 | 负责人 | 开始日期 | 完成日期 | 说明 |
|---|---|---|---|---|---|
| A（RVV GEMM 软件优化） | Done | AI Agent | 2026-04-06 | 2026-04-06 | 已完成完整验证矩阵与 benchmark 收益确认 |
| B（mulmac/mac 后端优化） | Not Started | TBD | TBD | TBD | 依赖 A 阶段基线与结果 |
| C（指令+工具链联动） | Not Started | TBD | TBD | TBD | 依赖 B 阶段稳定结果 |

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

## C 阶段日志

### C-Entry 模板
- 日期：
- Git commit：
- 改动摘要（含 decode/mpact/toolchain）：
- fallback 验证方式：

#### 验证结果
| 测试项 | 命令 | 结果 | 备注 |
|---|---|---|---|
| A/B 全量关键回归 | `同前` | TBD |  |
| C 指令链路专项 | `TBD` | TBD |  |
| fallback 对照 | `TBD` | TBD |  |

#### Benchmark 对照（C vs B）
| Workload | Metric | Before(B) | After(C) | Delta | 结论 |
|---|---|---|---|---|---|
| matmul_case_1 | latency_p50 | TBD | TBD | TBD | TBD |
| matmul_case_1 | throughput | TBD | TBD | TBD | TBD |

#### 阶段结论
- 是否满足 C 完成条件：`Yes/No`
- 若 No，阻塞原因：

## 5. 决策与问题跟踪

| ID | 日期 | 类型 | 内容 | 影响阶段 | 状态 |
|---|---|---|---|---|---|
| D-001 | TBD | Decision | 采用 A -> B -> C 顺序推进 | A/B/C | Open |
| I-001 | 2026-04-06 | Issue | `npusim_run_mobilenet` 在 host_clang 构建阶段找不到 `<array>` 头文件 | A | Resolved |
| A-002 | 2026-04-06 | Action | 修复 `toolchain/host_clang/BUILD`：GCC13/Clang18 include + host link path + 显式 `-isystem`，`npusim_run_mobilenet` 端到端通过（`inference_status=0`） | A | Done |
| A-003 | 2026-04-06 | Action | 新增 `bcresnet` 仿真回归入口并纳入 `running_tflite`，同时在 cocotb/npusim 输出统一 `PERF_CYCLES` 用于场景级性能跟踪 | A | Done |

类型建议：`Decision` / `Risk` / `Issue` / `Action`
