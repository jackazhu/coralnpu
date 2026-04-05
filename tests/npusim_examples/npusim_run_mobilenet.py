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

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np
import json
import time


def append_debug_log(hypothesis_id, location, message, data):
    # region agent log
    with open("/opt/cursor/logs/debug.log", "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": data,
                    "timestamp": int(time.time() * 1000),
                }
            )
            + "\n"
        )
    # endregion


def read_symbol_u32(npu_sim, symbol_map, name):
    addr = symbol_map.get(name)
    if addr is None:
        return None
    raw = npu_sim.read_memory(addr, 4)
    return int.from_bytes(raw, byteorder="little", signed=False)


def is_lsu_accessible(addr):
    if addr is None:
        return None
    return (0x00000000 <= addr < 0x0008AAD4) or (0x20000000 <= addr < 0x20400000)

def run_full_mobilenet():
    print(f"Running full mobilenet...")
    npu_sim = CoralNPUV2Simulator(highmem_ld=True, exit_on_ebreak=True)
    r = runfiles.Create()
    elf_file = r.Rlocation('coralnpu_hw/tests/npusim_examples/run_full_mobilenet_v1_binary.elf')

    debug_symbols = [
        "dbg_conv_hyp_branch",
        "dbg_conv_prepare_filter_h",
        "dbg_conv_prepare_filter_w",
        "dbg_conv_prepare_input_depth",
        "dbg_conv_prepare_output_depth",
        "dbg_conv_prepare_repacked_ptr",
        "dbg_conv_prepare_repacked_size",
        "dbg_conv_perchannel_filter_arg_ptr",
        "dbg_conv_perchannel_bias_arg_ptr",
        "dbg_conv_perchannel_filter_copy_ptr",
        "dbg_conv_perchannel_bias_copy_ptr",
        "dbg_conv1x1_filter_ptr",
        "dbg_conv1x1_repacked_ptr",
        "dbg_conv1x1_input_depth",
        "dbg_conv1x1_output_depth",
        "dbg_conv1x1_first_in_ptr",
        "dbg_conv1x1_first_out_ptr",
        "dbg_conv1x1_first_w_ptr",
        "dbg_conv1x1_last_w_ptr",
        "dbg_conv1x1_vl",
        "dbg_conv1x1_capture_done",
    ]
    entry_point, symbol_map = npu_sim.get_elf_entry_and_symbol(
        elf_file, ['inference_status', 'inference_input', 'inference_output'] + debug_symbols
    )
    # region agent log
    append_debug_log(
        "A",
        "npusim_run_mobilenet.py:run_full_mobilenet",
        "Starting MobileNet simulation with debug symbols",
        {"elf_file": elf_file, "has_debug_symbols": all(s in symbol_map for s in debug_symbols)},
    )
    # endregion
    npu_sim.load_program(elf_file, entry_point)

    if symbol_map.get('inference_input'):
        input_data = np.random.randint(-128, 127, size=(224 * 224 * 3,), dtype=np.int8)
        npu_sim.write_memory(symbol_map['inference_input'], input_data)

    print("Running simulation...", flush=True)
    npu_sim.run()
    npu_sim.wait()
    dbg = {k: read_symbol_u32(npu_sim, symbol_map, k) for k in debug_symbols}
    # region agent log
    append_debug_log(
        "B",
        "npusim_run_mobilenet.py:run_full_mobilenet",
        "Conv prepare and branch state",
        {
            "prepare_filter_hw_id_od": [
                dbg.get("dbg_conv_prepare_filter_h"),
                dbg.get("dbg_conv_prepare_filter_w"),
                dbg.get("dbg_conv_prepare_input_depth"),
                dbg.get("dbg_conv_prepare_output_depth"),
            ],
            "prepare_repacked_ptr": dbg.get("dbg_conv_prepare_repacked_ptr"),
            "prepare_repacked_size": dbg.get("dbg_conv_prepare_repacked_size"),
            "branch": dbg.get("dbg_conv_hyp_branch"),
            "filter_arg_ptr": dbg.get("dbg_conv_perchannel_filter_arg_ptr"),
            "filter_copy_ptr": dbg.get("dbg_conv_perchannel_filter_copy_ptr"),
            "bias_arg_ptr": dbg.get("dbg_conv_perchannel_bias_arg_ptr"),
            "bias_copy_ptr": dbg.get("dbg_conv_perchannel_bias_copy_ptr"),
        },
    )
    # endregion
    # region agent log
    append_debug_log(
        "C",
        "npusim_run_mobilenet.py:run_full_mobilenet",
        "1x1 address window and LSU accessibility",
        {
            "conv1x1_filter_ptr": dbg.get("dbg_conv1x1_filter_ptr"),
            "conv1x1_repacked_ptr": dbg.get("dbg_conv1x1_repacked_ptr"),
            "conv1x1_first_in_ptr": dbg.get("dbg_conv1x1_first_in_ptr"),
            "conv1x1_first_out_ptr": dbg.get("dbg_conv1x1_first_out_ptr"),
            "conv1x1_first_w_ptr": dbg.get("dbg_conv1x1_first_w_ptr"),
            "conv1x1_last_w_ptr": dbg.get("dbg_conv1x1_last_w_ptr"),
            "conv1x1_vl": dbg.get("dbg_conv1x1_vl"),
            "conv1x1_id_od": [
                dbg.get("dbg_conv1x1_input_depth"),
                dbg.get("dbg_conv1x1_output_depth"),
            ],
            "w_ptr_lsu_ok": is_lsu_accessible(dbg.get("dbg_conv1x1_first_w_ptr")),
            "w_last_lsu_ok": is_lsu_accessible(dbg.get("dbg_conv1x1_last_w_ptr")),
            "in_ptr_lsu_ok": is_lsu_accessible(dbg.get("dbg_conv1x1_first_in_ptr")),
            "out_ptr_lsu_ok": is_lsu_accessible(dbg.get("dbg_conv1x1_first_out_ptr")),
        },
    )
    # endregion
    print(f"cycles taken by the simulation {npu_sim.get_cycle_count()}")
    if symbol_map.get('inference_output'):
        output_data = npu_sim.read_memory(symbol_map['inference_output'], 5)
        output_data = np.array(output_data, dtype=np.int8)
        max_idx = np.argmax(output_data)
        print(f"Output info: Top index {max_idx} with value {output_data[max_idx]} from {output_data}")

    if symbol_map.get('inference_status'):
        inference_status = npu_sim.read_memory(symbol_map['inference_status'], 1)[0]
        print(f"inference_status {inference_status}")
        # region agent log
        append_debug_log(
            "D",
            "npusim_run_mobilenet.py:run_full_mobilenet",
            "Simulation finished with inference status",
            {"inference_status": int(inference_status), "cycle_count": int(npu_sim.get_cycle_count())},
        )
        # endregion

if __name__ == "__main__":
  run_full_mobilenet()

