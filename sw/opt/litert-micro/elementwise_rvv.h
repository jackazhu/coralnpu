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

#ifndef SW_OPT_LITERT_MICRO_ELEMENTWISE_RVV_H_
#define SW_OPT_LITERT_MICRO_ELEMENTWISE_RVV_H_

#include "tensorflow/lite/micro/micro_common.h"

namespace coralnpu_v2::opt::litert_micro {

// RVV-accelerated registrations for elementwise operators.
// All implementations are bit-accurate vs gemmlowp double-rounding (TFLite
// reference) by using vwmul + sign-dependent nudge (SrdmhExact) instead of
// vsmul which has a different tie-breaking rule for negative products.
TFLMRegistration Register_ADD_RVV();
TFLMRegistration Register_MUL_RVV();
TFLMRegistration Register_SUM_RVV();

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_ELEMENTWISE_RVV_H_
