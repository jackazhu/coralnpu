# CoralNPU GEMM 升级计划（A -> B -> C）

## 1. 目标与原则

### 1.1 目标
- 按 **A -> B -> C** 顺序推进 GEMM 升级。
- 每个阶段都必须通过严格验证，并用 benchmark 证明收益。
- 在任何阶段都保持可回退、可复现、可对比。

### 1.2 适用范围
- 硬件：CoralNPU RVV/后端执行路径，以及后续自定义 GEMM 指令路径。
- 软件：`sw/opt/litert-micro` 优化内核与调度路径。
- 工具链：`toolchain`、Bazel 规则、mpact/simulator 链路。

### 1.3 强约束
- 不允许只报告“功能正确”而不报告性能数据。
- 不允许只报告“性能提升”而缺失正确性验证。
- 每个阶段都要保留基线数据与原始日志。

## 2. 阶段定义

## A 阶段：软件优先（RVV GEMM 内核优化）

### A.1 核心改动
- 优化 RVV GEMM micro-kernel（FC/Conv/MatMul 关键路径）。
- 优化数据重排/分块策略，减少 fallback 与无效访存。
- 不引入新 ISA，不改 decode 语义。

### A.2 验收门槛
- 正确性：A 阶段相关回归全部通过。
- 性能：至少 1 个核心 workload 达到明确提升（目标值由基线后设定）。
- 稳定性：连续多次重复测试结果波动在可接受范围内（见第 5 节）。

## B 阶段：后端增强（mulmac/mac 管线优化）

### B.1 核心改动
- 优化 `mulmac`/`mac_unit` 吞吐、回压、流水时序。
- 在不破坏 A 阶段接口的前提下提升执行效率。
- 保留 A 阶段软件路径，确保可对照。

### B.2 验收门槛
- 正确性：A 阶段回归 + B 阶段新增回归全部通过。
- 性能：在 A 阶段基线上继续提升，或在同功耗目标下显著降低延迟。
- 回归风险：非 GEMM 关键测试不得出现新增功能退化。

## C 阶段：指令/工具链联动（自定义 GEMM 指令）

### C.1 核心改动
- 新增 GEMM 指令编码与 decode/dispatch 路径。
- 打通 mpact/simulator/工具链支持。
- 保留 A/B 旧路径作为 fallback（feature flag + capability 判断）。

### C.2 验收门槛
- 正确性：A/B 全量关键回归 + C 新增链路回归通过。
- 性能：在代表性 workload 上相对 B 阶段有可量化收益。
- 兼容性：关闭 C 特性时，行为与 B 阶段一致。

## 3. 验证矩阵（每阶段都必须执行）

### 3.1 正确性验证
- 单元/算子级：
  - `//tests/cocotb:rvv_ml_ops_cocotb_test`
- RVV 关键路径：
  - `//tests/cocotb:rvv_arithmetic_cocotb_test`
  - `//tests/cocotb:rvv_load_store_test`
- 高内存/系统路径：
  - `//tests/cocotb:rvv_highmem_tests`
  - `//tests/cocotb:rvv_itcm512kb_dtcm512kb_tests`

### 3.2 系统级验证
- Simulator 端到端：
  - `//tests/npusim_examples:npusim_run_mobilenet`
  - `//tests/npusim_examples:npusim_run_bcresnet`
- 必要时补充板级执行：
  - `coralnpu_test_utils/run_matmul_test.py`

### 3.3 失败处理
- 任一核心回归失败：禁止进入下一阶段。
- 发现性能倒退：必须给出根因与修复/豁免结论后才能继续。

## 4. Benchmark 规范

### 4.1 基线冻结
- 每阶段开始前，先记录：
  - Git commit hash
  - Toolchain 版本与关键编译参数
  - 目标测试命令、输入规模、随机种子
  - 运行环境（仿真/硬件、频率、内存配置）

### 4.2 指标定义（最少）
- 延迟：`p50` / `p95`（ms 或 cycles）
- 吞吐：GMAC/s 或等效指标
- 能效：mJ/inference（硬件可测时）
- 资源：代码体积、关键内存占用（ITCM/DTCM/arena）
- 回归日志：每个 case 必须输出可解析的 `PERF_CYCLES|...|cycles=<N>` 记录

### 4.3 统计规则
- 每组至少运行 10 次，去除 warmup。
- 固定随机种子。
- 同一阶段比较必须使用同一输入与同一配置。

### 4.4 收益判定
- 报告必须包含“Before vs After”对照表。
- 附带原始日志路径或摘要。
- 明确给出收益结论：提升、持平、退化（及原因）。

## 5. 里程碑与进入条件

### M0（准备完成）
- 计划文档与日志文档创建完成。
- 基线命令可运行，且输出可复现。

### M1（A 完成）
- A 验证矩阵通过。
- A benchmark 报告完成且收益成立。

### M2（B 完成）
- B 验证矩阵通过。
- 相对 A 的 benchmark 收益成立。

### M3（C 完成）
- C 验证矩阵通过。
- 相对 B 的 benchmark 收益成立。
- fallback 验证通过。

## 6. 变更控制与回退策略
- 每个阶段独立 PR，避免把 A/B/C 混在同一大提交。
- 每个阶段保留独立开关（编译开关或运行时 feature flag）。
- 若出现严重回归，先回退到上一里程碑，再定位问题。

## 7. 文档与记录要求
- 进度与结论统一记录在 `doc/gemm_abc_log.md`。
- 每次阶段推进至少补充：
  - 改动摘要
  - 验证结果
  - benchmark 对照
  - 风险与下一步

## 8. 当前进展快照（2026-04-07）
- 里程碑状态：`M0/M1/M2` 已完成，`M3` 进行中。
- C 阶段状态：已完成 `C1~C7`，其中 `C4/C5/C6` 聚焦网络级热点（1x1、grouped-1x1、grouped-5x5）。
- 当前网络周期（最新一轮 C6）：`npusim_mobilenet=204339890`，`npusim_bcresnet=282055557`。
- 风险控制状态：近期多轮“无收益/错误方向”尝试均已回退；仅保留通过正确性回归且网络收益成立的改动。
- 后续重点：继续推进 custom GEMM 真指令语义闭环（mpact/simulator/toolchain 助记符级支持）并保持与网络热点优化并行推进。
