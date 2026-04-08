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
#include <inttypes.h>

#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tests/npusim_examples/bcresnet_sc35_int8_tflite.h"

namespace {
inline uint32_t ReadCycle() {
  uint32_t cycle = 0;
  asm volatile("rdcycle %0" : "=r"(cycle));
  return cycle;
}

class CycleProfiler : public tflite::MicroProfilerInterface {
 public:
  uint32_t BeginEvent(const char* tag) override {
    if (num_events_ >= kMaxEvents) {
      return 0;
    }
    const uint32_t handle = num_events_;
    events_[handle].tag = tag;
    events_[handle].start = ReadCycle();
    events_[handle].end = events_[handle].start;
    ++num_events_;
    return handle;
  }

  void EndEvent(uint32_t event_handle) override {
    if (event_handle >= num_events_) {
      return;
    }
    events_[event_handle].end = ReadCycle();
  }

  void ClearEvents() { num_events_ = 0; }

  void LogTicksPerTagCsv() const {
    printf("\"Unique Tag\",\"Total ticks across all events with that tag.\"\n");

    const char* unique_tags[kMaxEvents];
    uint32_t unique_ticks[kMaxEvents];
    uint32_t unique_count = 0;
    uint32_t total_ticks = 0;
    for (uint32_t i = 0; i < num_events_; ++i) {
      const uint32_t ticks = events_[i].end - events_[i].start;
      total_ticks += ticks;
      uint32_t found_index = unique_count;
      for (uint32_t j = 0; j < unique_count; ++j) {
        if (std::strcmp(unique_tags[j], events_[i].tag) == 0) {
          found_index = j;
          break;
        }
      }
      if (found_index == unique_count) {
        unique_tags[unique_count] = events_[i].tag;
        unique_ticks[unique_count] = 0;
        ++unique_count;
      }
      unique_ticks[found_index] += ticks;
    }

    for (uint32_t i = 0; i < unique_count; ++i) {
      printf("%s, %" PRIu32 "\n", unique_tags[i], unique_ticks[i]);
    }
    printf("\"total number of ticks\", %" PRIu32 "\n", total_ticks);
  }

 private:
  static constexpr uint32_t kMaxEvents = 1024;
  struct Event {
    const char* tag;
    uint32_t start;
    uint32_t end;
  };

  Event events_[kMaxEvents]{};
  uint32_t num_events_ = 0;
};

using BcResnetOpResolver = tflite::MicroMutableOpResolver<10>;
using coralnpu_v2::opt::litert_micro::GetConv2dEvalCount;
using coralnpu_v2::opt::litert_micro::GetConv2dFallbackCount;
using coralnpu_v2::opt::litert_micro::GetDepthwiseConv2dEvalCount;
using coralnpu_v2::opt::litert_micro::GetDepthwiseConv2dFallbackCount;
using coralnpu_v2::opt::litert_micro::Register_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D;
using coralnpu_v2::opt::litert_micro::ResetConv2dEvalCounters;
using coralnpu_v2::opt::litert_micro::ResetDepthwiseConv2dEvalCounters;

TfLiteStatus RegisterOps(BcResnetOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D(Register_CONV_2D()));
  TF_LITE_ENSURE_STATUS(
      op_resolver.AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D()));
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
  (void)argc;
  (void)argv;
  const tflite::Model* model = tflite::GetModel(g_bcresnet_sc35_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("Error: schema mismatch\n");
    return -1;
  }

  BcResnetOpResolver op_resolver;
  if (RegisterOps(op_resolver) != kTfLiteOk) {
    printf("Error: op resolver failed\n");
    return -1;
  }

  CycleProfiler profiler;
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize, nullptr, &profiler);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("Error: AllocateTensors failed\n");
    return -1;
  }

  TfLiteTensor* input = interpreter.input(0);
  if (input == nullptr) {
    printf("Error: input tensor null\n");
    return -1;
  }
  std::memcpy(input->data.data, inference_input, input->bytes);

  profiler.ClearEvents();
  ResetConv2dEvalCounters();
  ResetDepthwiseConv2dEvalCounters();
  if (interpreter.Invoke() != kTfLiteOk) {
    printf("Error: Invoke failed\n");
    return -1;
  }

  TfLiteTensor* output = interpreter.output(0);
  if (output == nullptr) {
    printf("Error: output tensor null\n");
    return -1;
  }
  std::memcpy(inference_output, output->data.data, sizeof(inference_output));
  printf("PROFILE_CSV_BEGIN\n");
  profiler.LogTicksPerTagCsv();
  printf("PROFILE_CSV_END\n");
  printf("FALLBACK_SUMMARY,CONV_2D,%" PRIu32 ",%" PRIu32 "\n",
         GetConv2dFallbackCount(), GetConv2dEvalCount());
  printf("FALLBACK_SUMMARY,DEPTHWISE_CONV_2D,%" PRIu32 ",%" PRIu32 "\n",
         GetDepthwiseConv2dFallbackCount(), GetDepthwiseConv2dEvalCount());
  printf("Invoke successful\n");
  inference_status = 0;
  return 0;
}
