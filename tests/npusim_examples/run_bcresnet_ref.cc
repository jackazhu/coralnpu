// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>

#include <cstring>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tests/npusim_examples/bcresnet_sc35_int8_tflite.h"

namespace {
using BcResnetOpResolver = tflite::MicroMutableOpResolver<10>;

TfLiteStatus RegisterOps(BcResnetOpResolver& op_resolver) {
  // Reference TFLM kernels (no custom optimized registration).
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTranspose());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogSoftmax());
  return kTfLiteOk;
}
}  // namespace

extern "C" {
constexpr size_t kTensorArenaSize = 4 * 1024 * 1024;
int8_t inference_status = -1;
int8_t inference_input[1 * 1 * 40 * 101]
    __attribute__((section(".data"), aligned(16)));
int8_t inference_output[35] __attribute__((section(".data"), aligned(16)));
uint8_t tensor_arena[kTensorArenaSize]
    __attribute__((section(".extdata"), aligned(16)));
}

int main(int argc, char** argv) {
  const tflite::Model* model = tflite::GetModel(g_bcresnet_sc35_int8_model_data);
  BcResnetOpResolver op_resolver;
  RegisterOps(op_resolver);

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return -1;
  }

  TfLiteTensor* input = interpreter.input(0);
  if (input == nullptr) {
    return -1;
  }
  std::memcpy(input->data.data, inference_input, input->bytes);

  if (interpreter.Invoke() != kTfLiteOk) {
    return -1;
  }

  TfLiteTensor* output = interpreter.output(0);
  if (output == nullptr) {
    return -1;
  }
  std::memcpy(inference_output, output->data.data, sizeof(inference_output));
  inference_status = 0;
  return 0;
}
