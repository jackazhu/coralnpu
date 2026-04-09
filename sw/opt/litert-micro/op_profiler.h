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

#ifndef SW_OPT_LITERT_MICRO_OP_PROFILER_H_
#define SW_OPT_LITERT_MICRO_OP_PROFILER_H_

// Lightweight per-operator cycle profiler using RISC-V rdcycle CSR.
//
// Usage:
//   #define CORALNPU_OP_PROFILE 1   (build flag or local define)
//   OP_PROFILE_BEGIN(tag);
//   ... kernel work ...
//   OP_PROFILE_END(tag, "conv fh=%d fw=%d id=%d od=%d", fh, fw, id, od);
//
// Output format:
//   OP_CYCLES|tag=<tag>|cycles=<N>|info=<printf-formatted string>
//
// When CORALNPU_OP_PROFILE is not defined, macros expand to nothing.

#include <cstdint>
#include <cstdio>

namespace coralnpu_v2::opt::litert_micro {

inline uint64_t ReadCycle() {
#if defined(__riscv)
  uint32_t lo, hi, hi2;
  // Read mcycle (low 32 bits) and mcycleh (high 32 bits) with retry to
  // handle mid-read carry propagation.
  do {
    asm volatile("csrr %0, mcycleh" : "=r"(hi));
    asm volatile("csrr %0, mcycle" : "=r"(lo));
    asm volatile("csrr %0, mcycleh" : "=r"(hi2));
  } while (hi != hi2);
  return (static_cast<uint64_t>(hi) << 32) | lo;
#else
  return 0;
#endif
}

}  // namespace coralnpu_v2::opt::litert_micro

#if defined(CORALNPU_OP_PROFILE) && CORALNPU_OP_PROFILE

#define OP_PROFILE_BEGIN(tag) \
  const uint64_t _op_profile_t0_##tag = \
      coralnpu_v2::opt::litert_micro::ReadCycle()

#define OP_PROFILE_END(tag, fmt, ...)                              \
  do {                                                             \
    uint32_t _op_profile_dt = (uint32_t)(                         \
        coralnpu_v2::opt::litert_micro::ReadCycle() -             \
        _op_profile_t0_##tag);                                     \
    printf("OP_CYCLES|tag=" #tag "|cycles=%lu|info=" fmt "\n",    \
           (unsigned long)_op_profile_dt, ##__VA_ARGS__);                         \
  } while (0)

#else

#define OP_PROFILE_BEGIN(tag) (void)0
#define OP_PROFILE_END(tag, fmt, ...) (void)0

#endif  // CORALNPU_OP_PROFILE

#endif  // SW_OPT_LITERT_MICRO_OP_PROFILER_H_
