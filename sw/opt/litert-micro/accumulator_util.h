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

// PostprocessAcc: apply per-channel quantize/clamp to accumulator buffer.
// Layout: accs[out_x * out_d + out_ch] (HWC format).
// Outer loop: out_ch blocks (vl = vsetvl_e32m8); inner: out_x scalar.
//
// Fast path for out_d <= VLMAX (single out_ch pass):
//   Process 4 out_x positions per iteration to reduce loop overhead.
//   Bias/mult/shift vectors are loaded once and reused across out_x.
inline void PostprocessAcc(const int32_t* accs, const int32_t* bias_data,
                           const uint8_t* lshift, const int32_t* multiplier,
                           const uint8_t* rshift, int32_t out_offset,
                           int8_t out_min, int8_t out_max, int8_t* out_data,
                           int out_w, int out_d) {
  constexpr uint32_t vxrm = 0;  // round-to-nearest-up

  int out_ch = 0;
  size_t out_ch_rem = out_d;
  while (out_ch_rem > 0) {
    const size_t vl = __riscv_vsetvl_e32m8(out_ch_rem);
    const vint32m8_t bias_val =
        bias_data ? __riscv_vle32_v_i32m8(&bias_data[out_ch], vl)
                  : __riscv_vmv_v_x_i32m8(0, vl);
    const vint32m8_t mul_val = __riscv_vle32_v_i32m8(&multiplier[out_ch], vl);
    const vuint8m2_t lsh8 = __riscv_vle8_v_u8m2(&lshift[out_ch], vl);
    const vuint8m2_t rsh8 = __riscv_vle8_v_u8m2(&rshift[out_ch], vl);
    const vuint32m8_t lsh32 = __riscv_vzext_vf4_u32m8(lsh8, vl);
    const vuint32m8_t rsh32 = __riscv_vzext_vf4_u32m8(rsh8, vl);

    // Helper: quantize one acc vector and write vl bytes.
    auto quant_store = [&](const vint32m8_t& acc, int8_t* dst) {
      vint32m8_t a = __riscv_vadd_vv_i32m8(acc, bias_val, vl);
      a = __riscv_vsll_vv_i32m8(a, lsh32, vl);
      a = __riscv_vsmul_vv_i32m8(a, mul_val, vxrm, vl);
      a = __riscv_vssra_vv_i32m8(a, rsh32, vxrm, vl);
      a = __riscv_vadd_vx_i32m8(a, out_offset, vl);
      const vint16m4_t a16 = __riscv_vnclip_wx_i16m4(a, 0, vxrm, vl);
      vint8m2_t a8 = __riscv_vnclip_wx_i8m2(a16, 0, vxrm, vl);
      a8 = __riscv_vmax_vx_i8m2(a8, out_min, vl);
      a8 = __riscv_vmin_vx_i8m2(a8, out_max, vl);
      __riscv_vse8_v_i8m2(dst, a8, vl);
    };

    int out_x = 0;
    // 4-pixel unrolled loop: reduces branch/loop overhead by 4x.
    for (; out_x + 3 < out_w; out_x += 4) {
      quant_store(__riscv_vle32_v_i32m8(&accs[out_x * out_d + out_ch], vl),
                  &out_data[out_x * out_d + out_ch]);
      quant_store(__riscv_vle32_v_i32m8(&accs[(out_x + 1) * out_d + out_ch], vl),
                  &out_data[(out_x + 1) * out_d + out_ch]);
      quant_store(__riscv_vle32_v_i32m8(&accs[(out_x + 2) * out_d + out_ch], vl),
                  &out_data[(out_x + 2) * out_d + out_ch]);
      quant_store(__riscv_vle32_v_i32m8(&accs[(out_x + 3) * out_d + out_ch], vl),
                  &out_data[(out_x + 3) * out_d + out_ch]);
    }
    for (; out_x < out_w; ++out_x) {
      quant_store(__riscv_vle32_v_i32m8(&accs[out_x * out_d + out_ch], vl),
                  &out_data[out_x * out_d + out_ch]);
    }

    out_ch += vl;
    out_ch_rem -= vl;
  }
}

inline void PostprocessAcc16(const int32_t* accs, const int32_t* bias_data,
                             const uint8_t* lshift, const int32_t* multiplier,
                             const uint8_t* rshift, int32_t out_offset,
                             int16_t out_min, int16_t out_max,
                             int16_t* out_data, int out_w, int out_d) {
  // Scalar post-processing to ensure absolute bit-exactness with TFLM.
  // The user requested correctness over performance for this stage.
  for (int out_x = 0; out_x < out_w; ++out_x) {
    for (int c = 0; c < out_d; ++c) {
      int64_t acc = (int64_t)accs[out_x * out_d + c];
      if (bias_data) {
        acc += (int64_t)bias_data[c];
      }

      int shift = (int8_t)lshift[c] - (int8_t)rshift[c];
      int32_t result =
          tflite::MultiplyByQuantizedMultiplier(acc, multiplier[c], shift);

      result = std::max<int32_t>(result, out_min);
      result = std::min<int32_t>(result, out_max);
      out_data[out_x * out_d + c] = (int16_t)result;
    }
  }
}
}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_ACCUMULATOR_UTIL_H_
