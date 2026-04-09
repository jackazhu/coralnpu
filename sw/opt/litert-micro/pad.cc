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

// RVV-optimized Pad kernel for CoralNPU.
//
// The TFLite Micro reference Pad implementation uses a 5-level nested loop
// with per-element conditional branches.  For the typical BCResNet usage
// (int8, NHWC layout, zero-pad), the output tensor is a block-structured
// mix of constant "pad_value" regions and copied input rows.  We exploit this
// structure to replace the per-element loop with:
//
//   - RVV vector memset  for pad lines / pad columns (uses vs1r / vmv + vse8)
//   - RVV vector memcpy  for data lines (uses vle8 + vse8 in a tight loop)
//
// Both primitives amortize the loop overhead by VLEN/8 (typically 32 bytes
// per iteration) versus 1 byte in the scalar reference.
//
// Restrictions / fallback:
//   - Only int8 input with <= 4 dimensions is fast-pathed.
//   - The fast path handles the common "pad outer dims only" case where
//     depth padding is zero (left_d_padding == 0 && right_d_padding == 0).
//     This covers all BCResNet Pad ops (3x1/1x3/5x5 convolution padding).
//   - All other cases call the reference implementation.

#include <riscv_vector.h>

#include <cstring>

#include "tensorflow/lite/kernels/internal/reference/pad.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pad.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace coralnpu_v2::opt::litert_micro {

namespace {

// RVV vector memset: fill `n` bytes at `dst` with value `val`.
inline void VectorMemset(int8_t* dst, int8_t val, size_t n) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m8(n);
    vint8m8_t v = __riscv_vmv_v_x_i8m8(val, vl);
    __riscv_vse8_v_i8m8(dst, v, vl);
    dst += vl;
    n -= vl;
  }
}

// RVV vector memcpy: copy `n` bytes from `src` to `dst`.
inline void VectorMemcpy(int8_t* __restrict__ dst,
                         const int8_t* __restrict__ src, size_t n) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m8(n);
    __riscv_vse8_v_i8m8(dst, __riscv_vle8_v_i8m8(src, vl), vl);
    dst += vl;
    src += vl;
    n -= vl;
  }
}

// Fast int8 Pad for <= 4D tensors with no depth (innermost-dim) padding.
//
// The output shape is [B, H_out, W_out, D] where:
//   H_out = left_h + H_in + right_h
//   W_out = left_w + W_in + right_w
//   (and batch / plane paddings fold into outer loops)
//
// For each output row (out_h):
//   - If in the top/bottom padding band → entire row = memset(pad_value)
//   - Otherwise → left pad cols (memset) + data cols (memcpy) + right pad cols
//
// This handles up to 4D (batch, plane, height, width); plane corresponds to
// the P dimension in reference PadImpl's 5-dim expansion.
bool PadInt8Fast(const tflite::PadParams& params,
                 const tflite::RuntimeShape& input_shape,
                 const int8_t* input_data,
                 int8_t pad_value,
                 const tflite::RuntimeShape& output_shape,
                 int8_t* output_data) {
  const int ndim = input_shape.DimensionsCount();
  if (ndim > 4) return false;

  // Extend to 4D by prepending 1s.
  auto dim = [](const tflite::RuntimeShape& s, int i, int rank) -> int {
    int offset = 4 - rank;
    return (i < offset) ? 1 : s.Dims(i - offset);
  };
  auto lpad = [&](int i) -> int {
    int offset = 4 - params.left_padding_count;
    return (i < offset) ? 0 : params.left_padding[i - offset];
  };
  auto rpad = [&](int i) -> int {
    int offset = 4 - params.right_padding_count;
    return (i < offset) ? 0 : params.right_padding[i - offset];
  };

  // Fast path requires no depth-dimension padding.
  if (lpad(3) != 0 || rpad(3) != 0) return false;

  const int B = dim(output_shape, 0, ndim);
  const int P = dim(output_shape, 1, ndim);
  const int H_out = dim(output_shape, 2, ndim);
  const int W_out = dim(output_shape, 3, ndim);
  const int D = dim(input_shape, 3, ndim);

  const int W_in = dim(input_shape, 3, ndim);

  const int left_b = lpad(0), right_b = rpad(0);
  const int left_p = lpad(1), right_p = rpad(1);
  const int left_h = lpad(2), right_h = rpad(2);
  const int left_w = lpad(3), right_w = rpad(3);

  const int row_bytes = W_out * D;           // bytes per output row
  const int left_pad_bytes = left_w * D;     // bytes of left padding per row
  const int data_bytes = W_in * D;           // bytes of data per row
  const int right_pad_bytes = right_w * D;   // bytes of right padding per row

  const int8_t* in_ptr = input_data;

  for (int b = 0; b < B; ++b) {
    for (int p = 0; p < P; ++p) {
      for (int h = 0; h < H_out; ++h) {
        int8_t* out_row = output_data +
                          ((b * P + p) * H_out + h) * row_bytes;

        // Check if this row is in a batch/plane/height padding band.
        bool is_pad_row = (b < left_b || b >= B - right_b ||
                           p < left_p || p >= P - right_p ||
                           h < left_h || h >= H_out - right_h);

        if (is_pad_row) {
          // Entire row is padding.
          VectorMemset(out_row, pad_value, row_bytes);
        } else {
          // Left padding + data + right padding.
          if (left_pad_bytes > 0)
            VectorMemset(out_row, pad_value, left_pad_bytes);
          VectorMemcpy(out_row + left_pad_bytes, in_ptr, data_bytes);
          if (right_pad_bytes > 0)
            VectorMemset(out_row + left_pad_bytes + data_bytes,
                         pad_value, right_pad_bytes);
          in_ptr += data_bytes;
        }
      }
    }
  }
  return true;
}

TfLiteStatus PadEvalOpt(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const tflite::OpData* data =
      static_cast<const tflite::OpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, /*index=*/0);
  const TfLiteEvalTensor* constant_values =
      tflite::NumInputs(node) == 3
          ? tflite::micro::GetEvalInput(context, node, /*index=*/2)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, /*index=*/0);

  if (input->type == kTfLiteInt8) {
    int8_t pad_value;
    if (constant_values == nullptr) {
      pad_value = static_cast<int8_t>(data->output_zero_point);
    } else {
      pad_value = *tflite::micro::GetTensorData<int8_t>(constant_values);
    }

    const tflite::RuntimeShape in_shape =
        tflite::micro::GetTensorShape(input);
    const tflite::RuntimeShape out_shape =
        tflite::micro::GetTensorShape(output);

    // Attempt fast RVV path.
    if (PadInt8Fast(data->params,
                    in_shape,
                    tflite::micro::GetTensorData<int8_t>(input),
                    pad_value,
                    out_shape,
                    tflite::micro::GetTensorData<int8_t>(output))) {
      return kTfLiteOk;
    }

    // Fallback to reference for unsupported shapes.
    if (data->params.resizing_category ==
        tflite::ResizingCategory::kImageStyle) {
      tflite::reference_ops::PadImageStyle(
          data->params, in_shape,
          tflite::micro::GetTensorData<int8_t>(input), &pad_value,
          out_shape, tflite::micro::GetTensorData<int8_t>(output));
    } else {
      tflite::reference_ops::Pad(
          data->params, in_shape,
          tflite::micro::GetTensorData<int8_t>(input), &pad_value,
          out_shape, tflite::micro::GetTensorData<int8_t>(output));
    }
    return kTfLiteOk;
  }

  // Non-int8: delegate to reference via default registration.
  return tflite::Register_PAD().invoke(context, node);
}

}  // namespace

// Return type deduced from RegisterOp to avoid pulling in micro_common.h
// explicitly (TFLMRegistration is defined there but included transitively).
auto Register_PAD_RVV() -> decltype(tflite::micro::RegisterOp(
    tflite::PadInit, tflite::PadPrepare, nullptr)) {
  return tflite::micro::RegisterOp(tflite::PadInit, tflite::PadPrepare,
                                   PadEvalOpt);
}

}  // namespace coralnpu_v2::opt::litert_micro
