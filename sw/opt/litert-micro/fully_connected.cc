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

#include "sw/opt/litert-micro/fully_connected.h"

#include <riscv_vector.h>

#include <algorithm>
#include <cstdint>

#include "sw/opt/litert-micro/accumulator_util.h"
#include "sw/opt/litert-micro/custom_gemm.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace coralnpu_v2::opt::litert_micro {

namespace {

inline int8_t QuantizeAndClampAcc(int32_t acc, const int32_t* bias_data,
                                  int out_c, int32_t output_multiplier,
                                  int output_shift, int32_t output_offset,
                                  int32_t output_activation_min,
                                  int32_t output_activation_max) {
  if (bias_data) {
    acc += bias_data[out_c];
  }

  int32_t acc_scaled = tflite::MultiplyByQuantizedMultiplier(
      acc, output_multiplier, output_shift);
  acc_scaled += output_offset;
  acc_scaled = std::max(acc_scaled, output_activation_min);
  acc_scaled = std::min(acc_scaled, output_activation_max);
  return static_cast<int8_t>(acc_scaled);
}

}  // namespace

void FullyConnected(
    const tflite::FullyConnectedParams& params,
    const tflite::RuntimeShape& input_shape, const int8_t* input_data,
    const tflite::RuntimeShape& filter_shape, const int8_t* filter_data,
    const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
    const tflite::RuntimeShape& output_shape, int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  const int batches =
      tflite::FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);
  const int output_depth =
      output_shape.Dims(output_shape.DimensionsCount() - 1);
  const int accum_depth = filter_shape.Dims(filter_shape.DimensionsCount() - 1);
  const size_t acc_vlmax = __riscv_vsetvlmax_e32m4();
  const vint32m1_t zero_v = __riscv_vmv_v_x_i32m1(0, 1);

  for (int b = 0; b < batches; ++b) {
    const int8_t* batch_input = &input_data[b * accum_depth];
    int8_t* batch_output = &output_data[output_depth * b];
    int out_c = 0;

    // Process four output channels together to maximize input vector reuse.
    for (; out_c + 3 < output_depth; out_c += 4) {
      int d = 0;
      int d_rem = accum_depth;

      vint32m4_t acc0_v = __riscv_vmv_v_x_i32m4(0, acc_vlmax);
      vint32m4_t acc1_v = __riscv_vmv_v_x_i32m4(0, acc_vlmax);
      vint32m4_t acc2_v = __riscv_vmv_v_x_i32m4(0, acc_vlmax);
      vint32m4_t acc3_v = __riscv_vmv_v_x_i32m4(0, acc_vlmax);

      while (d_rem > 0) {
        size_t vl = __riscv_vsetvl_e8m1(d_rem);
        vint8m1_t in_v8 = __riscv_vle8_v_i8m1(&batch_input[d], vl);
        vint16m2_t in_v16 = __riscv_vadd_vx_i16m2(
            __riscv_vsext_vf2_i16m2(in_v8, vl), input_offset, vl);

        vint8m1_t weight0_v8 =
            __riscv_vle8_v_i8m1(&filter_data[out_c * accum_depth + d], vl);
        vint8m1_t weight1_v8 =
            __riscv_vle8_v_i8m1(&filter_data[(out_c + 1) * accum_depth + d], vl);
        vint8m1_t weight2_v8 =
            __riscv_vle8_v_i8m1(&filter_data[(out_c + 2) * accum_depth + d], vl);
        vint8m1_t weight3_v8 =
            __riscv_vle8_v_i8m1(&filter_data[(out_c + 3) * accum_depth + d], vl);

        vint16m2_t weight0_v16 = __riscv_vadd_vx_i16m2(
            __riscv_vsext_vf2_i16m2(weight0_v8, vl), filter_offset, vl);
        vint16m2_t weight1_v16 = __riscv_vadd_vx_i16m2(
            __riscv_vsext_vf2_i16m2(weight1_v8, vl), filter_offset, vl);
        vint16m2_t weight2_v16 = __riscv_vadd_vx_i16m2(
            __riscv_vsext_vf2_i16m2(weight2_v8, vl), filter_offset, vl);
        vint16m2_t weight3_v16 = __riscv_vadd_vx_i16m2(
            __riscv_vsext_vf2_i16m2(weight3_v8, vl), filter_offset, vl);

        acc0_v = __riscv_vwmacc_vv_i32m4(acc0_v, in_v16, weight0_v16, vl);
        acc1_v = __riscv_vwmacc_vv_i32m4(acc1_v, in_v16, weight1_v16, vl);
        acc2_v = __riscv_vwmacc_vv_i32m4(acc2_v, in_v16, weight2_v16, vl);
        acc3_v = __riscv_vwmacc_vv_i32m4(acc3_v, in_v16, weight3_v16, vl);

        d += vl;
        d_rem -= vl;
      }

      int32_t acc0 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vredsum_vs_i32m4_i32m1(acc0_v, zero_v, acc_vlmax));
      int32_t acc1 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vredsum_vs_i32m4_i32m1(acc1_v, zero_v, acc_vlmax));
      int32_t acc2 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vredsum_vs_i32m4_i32m1(acc2_v, zero_v, acc_vlmax));
      int32_t acc3 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vredsum_vs_i32m4_i32m1(acc3_v, zero_v, acc_vlmax));

      batch_output[out_c] = QuantizeAndClampAcc(
          acc0, bias_data, out_c, output_multiplier, output_shift, output_offset,
          output_activation_min, output_activation_max);
      batch_output[out_c + 1] = QuantizeAndClampAcc(
          acc1, bias_data, out_c + 1, output_multiplier, output_shift,
          output_offset, output_activation_min, output_activation_max);
      batch_output[out_c + 2] = QuantizeAndClampAcc(
          acc2, bias_data, out_c + 2, output_multiplier, output_shift,
          output_offset, output_activation_min, output_activation_max);
      batch_output[out_c + 3] = QuantizeAndClampAcc(
          acc3, bias_data, out_c + 3, output_multiplier, output_shift,
          output_offset, output_activation_min, output_activation_max);
    }

    // Process two output channels together to reuse input vector loads.
    for (; out_c + 1 < output_depth; out_c += 2) {
      int d = 0;
      int d_rem = accum_depth;

      vint32m4_t acc0_v = __riscv_vmv_v_x_i32m4(0, acc_vlmax);
      vint32m4_t acc1_v = __riscv_vmv_v_x_i32m4(0, acc_vlmax);

      while (d_rem > 0) {
        size_t vl = __riscv_vsetvl_e8m1(d_rem);
        vint8m1_t in_v8 = __riscv_vle8_v_i8m1(&batch_input[d], vl);
        vint8m1_t weight0_v8 =
            __riscv_vle8_v_i8m1(&filter_data[out_c * accum_depth + d], vl);
        vint8m1_t weight1_v8 =
            __riscv_vle8_v_i8m1(&filter_data[(out_c + 1) * accum_depth + d], vl);

        vint16m2_t in_v16 = __riscv_vadd_vx_i16m2(
            __riscv_vsext_vf2_i16m2(in_v8, vl), input_offset, vl);
        vint16m2_t weight0_v16 = __riscv_vadd_vx_i16m2(
            __riscv_vsext_vf2_i16m2(weight0_v8, vl), filter_offset, vl);
        vint16m2_t weight1_v16 = __riscv_vadd_vx_i16m2(
            __riscv_vsext_vf2_i16m2(weight1_v8, vl), filter_offset, vl);

        acc0_v = __riscv_vwmacc_vv_i32m4(acc0_v, in_v16, weight0_v16, vl);
        acc1_v = __riscv_vwmacc_vv_i32m4(acc1_v, in_v16, weight1_v16, vl);

        d += vl;
        d_rem -= vl;
      }

      int32_t acc0 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vredsum_vs_i32m4_i32m1(acc0_v, zero_v, acc_vlmax));
      int32_t acc1 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vredsum_vs_i32m4_i32m1(acc1_v, zero_v, acc_vlmax));

      batch_output[out_c] = QuantizeAndClampAcc(
          acc0, bias_data, out_c, output_multiplier, output_shift, output_offset,
          output_activation_min, output_activation_max);
      batch_output[out_c + 1] = QuantizeAndClampAcc(
          acc1, bias_data, out_c + 1, output_multiplier, output_shift,
          output_offset, output_activation_min, output_activation_max);
    }

    // Tail channel (odd output depth).
    for (; out_c < output_depth; ++out_c) {
      int d = 0;
      int d_rem = accum_depth;
      vint32m4_t acc_v = __riscv_vmv_v_x_i32m4(0, acc_vlmax);

      while (d_rem > 0) {
        size_t vl = __riscv_vsetvl_e8m1(d_rem);
        vint8m1_t in_v8 = __riscv_vle8_v_i8m1(&batch_input[d], vl);
        vint8m1_t weight_v8 =
            __riscv_vle8_v_i8m1(&filter_data[out_c * accum_depth + d], vl);

        vint16m2_t in_v16 = __riscv_vadd_vx_i16m2(
            __riscv_vsext_vf2_i16m2(in_v8, vl), input_offset, vl);
        vint16m2_t weight_v16 = __riscv_vadd_vx_i16m2(
            __riscv_vsext_vf2_i16m2(weight_v8, vl), filter_offset, vl);
        acc_v = __riscv_vwmacc_vv_i32m4(acc_v, in_v16, weight_v16, vl);

        d += vl;
        d_rem -= vl;
      }

      int32_t acc = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vredsum_vs_i32m4_i32m1(acc_v, zero_v, acc_vlmax));
      batch_output[out_c] = QuantizeAndClampAcc(
          acc, bias_data, out_c, output_multiplier, output_shift, output_offset,
          output_activation_min, output_activation_max);
    }
  }
}

namespace {
TfLiteStatus FullyConnectedEval(TfLiteContext* context, TfLiteNode* node) {
  const auto& data =
      *(static_cast<const tflite::OpDataFullyConnected*>(node->user_data));

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(
      context, node, tflite::kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter = tflite::micro::GetEvalInput(
      context, node, tflite::kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias = tflite::micro::GetEvalInput(
      context, node, tflite::kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(
      context, node, tflite::kFullyConnectedOutputTensor);

  if (input->type == kTfLiteInt8 && filter->type == kTfLiteInt8) {
    const tflite::FullyConnectedParams params =
        tflite::FullyConnectedParamsQuantized(data);
    const tflite::RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
    const tflite::RuntimeShape filter_shape =
        tflite::micro::GetTensorShape(filter);
    const tflite::RuntimeShape bias_shape = tflite::micro::GetTensorShape(bias);
    const tflite::RuntimeShape output_shape =
        tflite::micro::GetTensorShape(output);
    const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
    const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
    const int32_t* bias_data = tflite::micro::GetOptionalTensorData<int32_t>(bias);
    int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

    // C-phase bootstrap: custom GEMM path is compiled behind feature/capability
    // flags so default builds keep the A/B hot path unchanged.
#if defined(CORALNPU_ENABLE_CUSTOM_GEMM) && CORALNPU_ENABLE_CUSTOM_GEMM && \
    defined(CORALNPU_HAS_CUSTOM_GEMM) && CORALNPU_HAS_CUSTOM_GEMM
    if (TryFullyConnectedCustomGemm(params, input_shape, input_data, filter_shape,
                                    filter_data, bias_shape, bias_data,
                                    output_shape, output_data)) {
      return kTfLiteOk;
    }
#endif

    FullyConnected(params, input_shape, input_data, filter_shape, filter_data,
                   bias_shape, bias_data, output_shape, output_data);
    return kTfLiteOk;
  }

  return tflite::Register_FULLY_CONNECTED().invoke(context, node);
}
}  // namespace

TFLMRegistration Register_FULLY_CONNECTED() {
  auto registration = tflite::Register_FULLY_CONNECTED();
  registration.invoke = FullyConnectedEval;
  return registration;
}

}  // namespace coralnpu_v2::opt::litert_micro
