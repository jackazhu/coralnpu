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

namespace coralnpu_v2::opt::litert_micro {

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
  return false;
}

}  // namespace coralnpu_v2::opt::litert_micro
