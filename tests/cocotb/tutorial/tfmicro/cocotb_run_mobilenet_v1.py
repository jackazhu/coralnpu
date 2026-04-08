# Copyright 2025 Google LLC
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

import cocotb
import numpy as np

from coralnpu_test_utils.sim_test_fixture import Fixture
from bazel_tools.tools.python.runfiles import runfiles


async def run_tflite_case(
    fixture: Fixture,
    runfiles_root,
    elf_path: str,
    case_name: str,
    status_msg_size: int,
    timeout_cycles: int,
):
    await fixture.load_elf_and_lookup_symbols(
        runfiles_root.Rlocation(elf_path),
        ["inference_status", "inference_status_message"],
    )
    cycle_count = await fixture.run_to_halt(timeout_cycles=timeout_cycles)
    print(
        f"PERF_CYCLES|runner=cocotb|test={case_name}|cycles={cycle_count}",
        flush=True,
    )
    tflite_inference_status = (await fixture.read_word("inference_status")).view(np.int32)
    tflite_inference_message = bytes(
        (await fixture.read("inference_status_message", status_msg_size))
    ).decode(errors="ignore").rstrip("\x00")
    assert tflite_inference_status == 0, tflite_inference_message


@cocotb.test()
async def core_mini_rvv_mobilenet_v1(dut):
    fixture = await Fixture.Create(dut, highmem=True)
    r = runfiles.Create()
    await run_tflite_case(
        fixture,
        r,
        "coralnpu_hw/tests/cocotb/tutorial/tfmicro/run_mobilenet_v1_025_partial_binary.elf",
        "core_mini_rvv_mobilenet_v1",
        status_msg_size=31,
        # NOTE: Running the example in DEBUG mode is too slow.
        timeout_cycles=130_000_000,
    )
    print("\nPartial mobilenet Invoke() successful\n", flush=True)


@cocotb.test()
async def core_mini_rvv_bcresnet(dut):
    fixture = await Fixture.Create(dut, highmem=True)
    r = runfiles.Create()
    await run_tflite_case(
        fixture,
        r,
        "coralnpu_hw/tests/npusim_examples/run_bcresnet_binary.elf",
        "core_mini_rvv_bcresnet",
        status_msg_size=63,
        timeout_cycles=30_000_000,
    )
    print("\nBCResNet Invoke() successful\n", flush=True)