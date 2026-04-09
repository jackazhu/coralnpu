# CoralNPU Phase-2 深度优化计划

## 1. 背景

A→B→C 三阶段（gemm_abc_plan.md）完成后，MobileNet 累积改善 **-93.8%**（516M→31.8M cycles），BCResNet 累积改善 **-22.1%**（327M→254M cycles）。本计划在此基础上继续深化，聚焦新的瓶颈方向。

## 2. 优化项列表（按优先级排序）

| ID | 方向 | 预期难度 | 目标网络 | 核心理由 |
|---|---|---|---|---|
| P1 | 逐层 profiling 框架（rdcycle）| 低 | 全部 | 元收益：精确定位热点，指导后续优先级 |
| P-HOT1 | DW-5x5 depth_multiplier 维向量化 | 中 | BCResNet | P1 发现的最大热点（77.2%），vl=1 退化 |
| P2 | 5x5/grouped filter repack | 中 | BCResNet | 消除 `vlse8` strided load，换 `vle8` 顺序访问 |
| P3 | PostprocessAcc out_w 4x 展开 | 低 | 全部 | out_w 标量循环，4x unroll 减少分支开销 |
| P4 | DepthwiseConv 1D (3x1/1x3) 专用路径 | 低-中 | BCResNet | 分解 DW conv 无特化，走通用 Patch |
| P5 | Conv2D_5x5 空间分块（2-pixel tile）| 低-中 | BCResNet | 25 tap 地址摊销，类比 C8 的 1x1/3x3 |
| P6 | Conv2D_3x3 去 repacked_weights 强依赖 | 中 | 全部 | 内存紧张时 3x3 fallback 到 reference |

## 3. 验收标准（每项）

- 正确性：`conv_sim_test` + `rvv_ml_ops_cocotb_test` + `rvv_highmem` + `rvv_itcm512` 全部通过。
- 性能：mobilenet/bcresnet 端到端 cycles 不退化；每项给出 Before/After 对照。
- 稳定性：连续两次测量周期差 <1%。

## 4. 进展记录

### P1: 逐层 profiling 框架

- 状态：`done`
- 日期：2026-04-08
- 实现：新增 `sw/opt/litert-micro/op_profiler.h`（rdcycle 采样宏）；在 `ConvEval`/`DepthwiseConvEval` 接入 `OP_PROFILE_BEGIN/END`；为 mobilenet/bcresnet 新增 profiling binary targets（`conv_profile`/`depthwise_conv_profile` 库，`defines=["CORALNPU_OP_PROFILE=1"]`）
- 关键发现：
  - BCResNet 最大热点：`dw_conv2d 5x5 depth_multiplier=32`（21.3M cycles，77.2%）
  - BCResNet 第2热点：`conv2d 5x5 grouped id=40 od=40`（1.84M，6.7%）
  - MobileNet 最大热点：`conv2d 3x3 224x224x3`（2.87M，14.5%）+ `conv2d 1x1 14x14x128`（×5，各1.18M）

### P-HOT1: DW-5x5 depth_multiplier=32 向量化

- 状态：`done`
- 日期：2026-04-08
- 实现：新增 `DepthwiseConvPerChannelPatchLargeDM`，当 `depth_multiplier >= 4 && in_d <= 4` 时激活；将向量化维从 `in_ch`（vl=1 退化）改为 `depth_multiplier`（vl=32）；filter 用连续 load
- 结论：**BCResNet 254.9M → 235.3M（-7.7%）**；MobileNet 持平

### P2: 5x5/grouped filter repack

- 状态：`done`
- 日期：2026-04-08
- 实现：`ConvPrepare` 中为 5x5 filter 分配 per-group repack 缓冲区（`[g][ky][kx][fid][fpg]` 布局）；`Conv2D_5x5` 新增 `vle8` 快路径
- 结论：BCResNet 持平（+0.08%）；5x5 filter stride 仅 50，不是主要瓶颈

### P3: PostprocessAcc 4x 展开

- 状态：`done`
- 日期：2026-04-08
- 实现：`accumulator_util.h` 中 `PostprocessAcc` 内层 `out_x` 循环改为 4x 展开（减少分支和循环控制开销）；使用 lambda `quant_store` 共享量化流水逻辑
- 结论：BCResNet -0.16%，MobileNet **-0.73%**，全部通过回归

### P4: DepthwiseConv 1D (3x1/1x3) 专用路径

- 状态：`tried, reverted`
- 日期：2026-04-08
- 结论：实现了 `DepthwiseConvPerChannelPatch1D`，测试显示 BCResNet +1% 轻微退化；lambda 调用开销超过收益；已回退，代码保留在文件但不接入分发

### P5: Conv2D_5x5 空间分块（2-pixel tile）

- 状态：`done`
- 日期：2026-04-08
- 实现：`Conv2D_5x5` 内层 `out_x` 循环增加 2-pixel tile（`dilation==1` 时，2 个相邻像素共享 25 个 filter tap 向量加载）；lambda `compute_acc_5x5` 处理边界像素
- 结论：BCResNet -0.2%，MobileNet 持平

### P6: Conv2D_3x3 去 repacked_weights 强依赖

- 状态：`pending`
- 日期：-
- 结论：优先级降低，当前内存充足时 repack 成功分配，fallback 风险可接受

## 5. 基线冻结（Phase-2 开始前）

- 日期：2026-04-08
- Git commit：`fa343f7`（C8 完成后）
- npusim_mobilenet：40,017,353（C7）→ **31,840,506**（C8，本轮起点）
- npusim_bcresnet：254,702,861（C7）→ **254,941,208**（C8，本轮起点）

## 6. Phase-2 最终快照

- 日期：2026-04-08
- npusim_mobilenet：31,840,506 → **31,636,188**（**-0.64%**）
- npusim_bcresnet：254,941,208 → **234,550,273**（**-8.0%**）
- 累积改善（相对 A 阶段基线 bcresnet=327,413,977）：**-28.4%**
- 累积改善（相对 A 阶段基线 mobilenet=516,177,367）：**-93.9%**
