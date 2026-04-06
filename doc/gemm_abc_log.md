# CoralNPU GEMM 升级进度日志（A -> B -> C）

## 1. 使用说明
- 本文档用于记录 A/B/C 三阶段的实际推进状态。
- 所有性能结论必须附带 benchmark 对照数据。
- 所有“完成”结论必须附带可复现实验命令和 commit hash。

## 2. 当前总体状态

| 阶段 | 状态 | 负责人 | 开始日期 | 完成日期 | 说明 |
|---|---|---|---|---|---|
| A（RVV GEMM 软件优化） | Not Started | TBD | TBD | TBD | 计划已建立，待基线冻结 |
| B（mulmac/mac 后端优化） | Not Started | TBD | TBD | TBD | 依赖 A 阶段基线与结果 |
| C（指令+工具链联动） | Not Started | TBD | TBD | TBD | 依赖 B 阶段稳定结果 |

状态枚举建议：`Not Started` / `In Progress` / `Blocked` / `Done`

## 3. 基线冻结记录（必须先填）

### Baseline-0（推进 A 前）
- 日期：
- Git commit：
- Toolchain 版本：
- 平台/配置（仿真或硬件）：
- 频率与内存配置：
- Benchmark 输入集版本：
- 随机种子：

#### 验证命令
- 正确性：
  - `bazel test //tests/cocotb:rvv_ml_ops_cocotb_test`
  - `bazel test //tests/cocotb:rvv_arithmetic_cocotb_test`
  - `bazel test //tests/cocotb:rvv_load_store_test`
- 系统：
  - `bazel test //tests/cocotb:rvv_highmem_tests`
  - `bazel test //tests/cocotb:rvv_itcm512kb_dtcm512kb_tests`
  - `bazel run //tests/npusim_examples:npusim_run_mobilenet`

#### 基线 benchmark 结果
| Workload | Config | Metric | Value | 单位 | 备注 |
|---|---|---|---|---|---|
| matmul_case_1 | TBD | latency_p50 | TBD | cycles/ms |  |
| matmul_case_1 | TBD | latency_p95 | TBD | cycles/ms |  |
| matmul_case_1 | TBD | throughput | TBD | GMAC/s |  |
| mobilenet_sim | TBD | latency_p50 | TBD | ms |  |

## 4. 阶段日志

## A 阶段日志

### A-Entry 模板
- 日期：
- Git commit：
- 改动摘要：
- 风险评估：

#### 验证结果
| 测试项 | 命令 | 结果 | 备注 |
|---|---|---|---|
| 单元 | `bazel test //tests/cocotb:rvv_ml_ops_cocotb_test` | TBD |  |
| RVV 算术 | `bazel test //tests/cocotb:rvv_arithmetic_cocotb_test` | TBD |  |
| RVV 访存 | `bazel test //tests/cocotb:rvv_load_store_test` | TBD |  |
| 高内存 | `bazel test //tests/cocotb:rvv_highmem_tests` | TBD |  |

#### Benchmark 对照（A vs Baseline-0）
| Workload | Metric | Before | After | Delta | 结论 |
|---|---|---|---|---|---|
| matmul_case_1 | latency_p50 | TBD | TBD | TBD | TBD |
| matmul_case_1 | throughput | TBD | TBD | TBD | TBD |

#### 阶段结论
- 是否满足 A 进入 B 条件：`Yes/No`
- 若 No，阻塞原因：

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

类型建议：`Decision` / `Risk` / `Issue` / `Action`
