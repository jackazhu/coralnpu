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

#ifndef SW_OPT_LITERT_MICRO_PAD_H_
#define SW_OPT_LITERT_MICRO_PAD_H_

// Include the upstream pad.h for OpData / PadInit / PadPrepare, and
// kernel_util.h which defines RegisterOp (and its TFLMRegistration return).
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pad.h"

namespace coralnpu_v2::opt::litert_micro {

// RVV-accelerated Pad operator registration.
// Return type is TFLMRegistration (deduced via RegisterOp).
decltype(tflite::micro::RegisterOp(tflite::PadInit, tflite::PadPrepare,
                                   nullptr))
Register_PAD_RVV();

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_PAD_H_
