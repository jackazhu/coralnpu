// Copyright 2025 Google LLC
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

#ifndef SW_OPT_LITERT_MICRO_ACCUMULATOR_UTIL_H_
#define SW_OPT_LITERT_MICRO_ACCUMULATOR_UTIL_H_

#include <riscv_vector.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"

#if TFLITE_SINGLE_ROUNDING
#error "TFLITE_SINGLE_ROUNDING is not supported"
#endif  // TFLITE_SINGLE_ROUNDING

namespace coralnpu_v2::opt::litert_micro {
inline void PrepareShiftParams(uint8_t* left, uint8_t* right,
                               const int32_t* shift_in, int out_d) {
  int out_ch = 0;
  size_t out_ch_rem = out_d;
  while (out_ch_rem > 0) {
    const size_t vl = __riscv_vsetvl_e32m8(out_ch_rem);
    const vint32m8_t shift32 = __riscv_vle32_v_i32m8(&shift_in[out_ch], vl);
    const vint16m4_t shift16 = __riscv_vncvt_x_x_w_i16m4(shift32, vl);
    const vint8m2_t shift8 = __riscv_vncvt_x_x_w_i8m2(shift16, vl);
    const vint8m2_t neg = __riscv_vneg_v_i8m2(shift8, vl);
    // Positive values shift to LEFT, negative shift to right.
    const vuint8m2_t shl =
        __riscv_vreinterpret_v_i8m2_u8m2(__riscv_vmax_vx_i8m2(shift8, 0, vl));
    const vuint8m2_t shr =
        __riscv_vreinterpret_v_i8m2_u8m2(__riscv_vmax_vx_i8m2(neg, 0, vl));
    __riscv_vse8_v_u8m2(&left[out_ch], shl, vl);
    __riscv_vse8_v_u8m2(&right[out_ch], shr, vl);
    out_ch += vl;
    out_ch_rem -= vl;
  }
}

// TODO(davidgao): use a param structure for reuse?
// `output_shift` must be the raw per-channel shift from TFLM (same array passed
// to reference_integer_ops::DepthwiseConvPerChannel / ConvPerChannel).
// Do not reconstruct shift from uint8 left/right tables: PrepareShiftParams
// truncates int32 shifts to int8, which breaks channels with |shift| > 127.
inline void PostprocessAcc(const int32_t* accs, const int32_t* bias_data,
                           const int32_t* multiplier, const int32_t* output_shift,
                           int32_t out_offset, int8_t out_min, int8_t out_max,
                           int8_t* out_data, int out_w, int out_d) {
  for (int out_x = 0; out_x < out_w; ++out_x) {
    for (int c = 0; c < out_d; ++c) {
      int32_t acc = accs[out_x * out_d + c];
      if (bias_data) {
        acc += bias_data[c];
      }

      int32_t q = tflite::MultiplyByQuantizedMultiplier(
          acc, multiplier[c], output_shift[c]);
      q += out_offset;
      q = std::max<int32_t>(q, out_min);
      q = std::min<int32_t>(q, out_max);
      out_data[out_x * out_d + c] = static_cast<int8_t>(q);
    }
  }
}

inline void PostprocessAcc16(const int32_t* accs, const int32_t* bias_data,
                             const int32_t* multiplier,
                             const int32_t* output_shift, int32_t out_offset,
                             int16_t out_min, int16_t out_max,
                             int16_t* out_data, int out_w, int out_d) {
  for (int out_x = 0; out_x < out_w; ++out_x) {
    for (int c = 0; c < out_d; ++c) {
      int64_t acc = (int64_t)accs[out_x * out_d + c];
      if (bias_data) {
        acc += (int64_t)bias_data[c];
      }

      int32_t result = tflite::MultiplyByQuantizedMultiplier(
          acc, multiplier[c], output_shift[c]);

      result = std::max<int32_t>(result, out_min);
      result = std::min<int32_t>(result, out_max);
      out_data[out_x * out_d + c] = (int16_t)result;
    }
  }
}
}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_ACCUMULATOR_UTIL_H_
