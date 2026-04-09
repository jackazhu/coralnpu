# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-layer profiling runner for BCResNet using CORALNPU_OP_PROFILE=1 binary."""

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np


def run_profile_bcresnet():
    print("Running BCResNet profiling build...")
    npu_sim = CoralNPUV2Simulator(highmem_ld=True, exit_on_ebreak=True)
    r = runfiles.Create()
    elf_file = r.Rlocation(
        'coralnpu_hw/tests/npusim_examples/run_bcresnet_profile_binary.elf')
    entry_point, symbol_map = npu_sim.get_elf_entry_and_symbol(
        elf_file, ['inference_status', 'inference_input', 'inference_output'])
    npu_sim.load_program(elf_file, entry_point)

    if symbol_map.get('inference_input'):
        rng = np.random.default_rng(seed=42)
        input_size = 1 * 40 * 1
        input_data = rng.integers(-128, 127, size=(input_size,), dtype=np.int8)
        npu_sim.write_memory(symbol_map['inference_input'], input_data)

    print("Running simulation (profiling)...", flush=True)
    npu_sim.run()
    npu_sim.wait()
    cycle_count = npu_sim.get_cycle_count()
    print(f"PERF_CYCLES|runner=npusim|test=npusim_profile_bcresnet|cycles={cycle_count}")

    if symbol_map.get('inference_status'):
        inference_status = npu_sim.read_memory(symbol_map['inference_status'], 1)[0]
        print(f"inference_status {inference_status}")
        assert inference_status == 0, f"inference_status={inference_status}"


if __name__ == "__main__":
    run_profile_bcresnet()
