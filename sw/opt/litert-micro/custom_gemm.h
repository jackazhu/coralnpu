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

#ifndef SW_OPT_LITERT_MICRO_CUSTOM_GEMM_H_
#define SW_OPT_LITERT_MICRO_CUSTOM_GEMM_H_

#include "tensorflow/lite/micro/kernels/fully_connected.h"

namespace coralnpu_v2::opt::litert_micro {

// C phase feature flag: custom GEMM path is compiled in but disabled by default.
bool IsCustomGemmPathEnabled();

// C phase capability gate: placeholder for custom GEMM ISA capability probing.
bool HasCustomGemmCapability();

// Returns true only when custom GEMM instruction path handles the op.
bool TryFullyConnectedCustomGemm(
    const tflite::FullyConnectedParams& params,
    const tflite::RuntimeShape& input_shape, const int8_t* input_data,
    const tflite::RuntimeShape& filter_shape, const int8_t* filter_data,
    const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
    const tflite::RuntimeShape& output_shape, int8_t* output_data);

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_CUSTOM_GEMM_H_
