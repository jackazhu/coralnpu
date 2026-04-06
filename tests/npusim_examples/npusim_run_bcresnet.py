from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np


def run_bcresnet():
    print("Running BCResNet KWS...")
    npu_sim = CoralNPUV2Simulator(highmem_ld=True, exit_on_ebreak=True)
    r = runfiles.Create()
    elf_file = r.Rlocation(
        "coralnpu_hw/tests/npusim_examples/run_bcresnet_binary.elf"
    )
    entry_point, symbol_map = npu_sim.get_elf_entry_and_symbol(
        elf_file, ["inference_status", "inference_input", "inference_output"]
    )
    npu_sim.load_program(elf_file, entry_point)

    # BCResNet int8 input shape: [1, 1, 40, 101].
    if symbol_map.get("inference_input"):
        input_data = np.random.randint(
            low=-128, high=127, size=(1 * 1 * 40 * 101), dtype=np.int8
        )
        npu_sim.write_memory(
            symbol_map["inference_input"],
            np.frombuffer(input_data.tobytes(), dtype=np.uint8),
        )

    print("Running simulation...", flush=True)
    npu_sim.run()
    npu_sim.wait()
    print(f"cycles taken by the simulation {npu_sim.get_cycle_count()}")

    if symbol_map.get("inference_output"):
        output_raw = npu_sim.read_memory(symbol_map["inference_output"], 35)
        output = np.frombuffer(output_raw, dtype=np.int8)
        max_idx = int(np.argmax(output))
        print(
            f"Output info: Top index {max_idx} with qvalue {output[max_idx]} "
            f"from first 8 qlogits {output[:8]}"
        )

    if symbol_map.get("inference_status"):
        inference_status = npu_sim.read_memory(symbol_map["inference_status"], 1)[0]
        print(f"inference_status {inference_status}")


if __name__ == "__main__":
    run_bcresnet()
