// Fully reference BCResNet runner (no custom kernels)
#include <stdint.h>
#include <stdio.h>
#include <cstring>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tests/npusim_examples/bcresnet_sc35_int8_tflite.h"
namespace {
using BcResnetOpResolver = tflite::MicroMutableOpResolver<10>;
TfLiteStatus RegisterOps(BcResnetOpResolver& op_resolver) {
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
} // namespace
extern "C" {
constexpr size_t kTensorArenaSize = 4 * 1024 * 1024;
int8_t inference_status = -1;
int8_t inference_input[1*1*40*101] __attribute__((section(".data"), aligned(16)));
int8_t inference_output[35] __attribute__((section(".data"), aligned(16)));
uint8_t tensor_arena[kTensorArenaSize] __attribute__((section(".extdata"), aligned(16)));
}
int main(int argc, char** argv) {
  const tflite::Model* model = tflite::GetModel(g_bcresnet_sc35_int8_model_data);
  BcResnetOpResolver op_resolver;
  RegisterOps(op_resolver);
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  interpreter.AllocateTensors();
  TfLiteTensor* input = interpreter.input(0);
  std::memcpy(input->data.data, inference_input, input->bytes);
  interpreter.Invoke();
  TfLiteTensor* output = interpreter.output(0);
  std::memcpy(inference_output, output->data.data, sizeof(inference_output));
  printf("FULLREF:");
  for (int i = 0; i < 35; i++) printf(" %d", (int)inference_output[i]);
  printf("\n");
  inference_status = 0;
  return 0;
}
