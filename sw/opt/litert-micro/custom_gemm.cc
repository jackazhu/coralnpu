// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sw/opt/litert-micro/custom_gemm.h"

#include <cstdint>

namespace coralnpu_v2::opt::litert_micro {
namespace {

constexpr uint32_t kRvvOpcode = 0x57;
constexpr uint32_t kOpmvvFunct3 = 0b010;
constexpr uint32_t kCustomGemmFunct6 = 0b101110;
constexpr uint32_t kVmUnmasked = 1;

constexpr uint32_t EncodeRvvInstruction(uint32_t funct6, uint32_t vm,
                                        uint32_t vs2, uint32_t vs1,
                                        uint32_t funct3, uint32_t vd,
                                        uint32_t opcode) {
  return (funct6 << 26) | (vm << 25) | (vs2 << 20) | (vs1 << 15) |
         (funct3 << 12) | (vd << 7) | opcode;
}

constexpr uint32_t kCustomGemmProbeInstruction = EncodeRvvInstruction(
    kCustomGemmFunct6, kVmUnmasked, /*vs2=*/0, /*vs1=*/0, kOpmvvFunct3,
    /*vd=*/0, kRvvOpcode);
static_assert(kCustomGemmProbeInstruction == 0xBA002057,
              "custom gemm probe encoding mismatch");

inline void EmitCustomGemmProbeInstruction() {
#if defined(__riscv)
  // C-phase step2: issue one custom GEMM-encoded RVV op to validate the
  // instruction binary encoding path when the feature is explicitly enabled.
  asm volatile(".word 0xBA002057" ::: "memory");
#endif
}

}  // namespace

bool IsCustomGemmPathEnabled() {
  // Default off in C-phase bootstrap to preserve A/B behavior.
#if defined(CORALNPU_ENABLE_CUSTOM_GEMM) && CORALNPU_ENABLE_CUSTOM_GEMM
  return true;
#else
  return false;
#endif
}

bool HasCustomGemmCapability() {
  // TODO(C-phase): replace with real custom-ISA capability probing.
#if defined(CORALNPU_HAS_CUSTOM_GEMM) && CORALNPU_HAS_CUSTOM_GEMM
  return true;
#else
  return false;
#endif
}

bool TryFullyConnectedCustomGemm(
    const tflite::FullyConnectedParams& params,
    const tflite::RuntimeShape& input_shape, const int8_t* input_data,
    const tflite::RuntimeShape& filter_shape, const int8_t* filter_data,
    const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
    const tflite::RuntimeShape& output_shape, int8_t* output_data) {
  (void)params;
  (void)input_shape;
  (void)input_data;
  (void)filter_shape;
  (void)filter_data;
  (void)bias_shape;
  (void)bias_data;
  (void)output_shape;
  (void)output_data;

  // C-phase step 1: only wire feature/capability gate; execution still falls
  // back to the existing RVV path.
  if (!IsCustomGemmPathEnabled() || !HasCustomGemmCapability()) {
    return false;
  }
  // C-phase step2: custom encoding is plumbed, but compute still falls back
  // until the full GEMM datapath is implemented.
  EmitCustomGemmProbeInstruction();
  return false;
}

}  // namespace coralnpu_v2::opt::litert_micro
