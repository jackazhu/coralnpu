from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np
import argparse


def _run_single_elf(elf_rlocation: str, input_data: np.ndarray):
    npu_sim = CoralNPUV2Simulator(highmem_ld=True, exit_on_ebreak=True)
    r = runfiles.Create()
    elf_file = r.Rlocation(elf_rlocation)
    entry_point, symbol_map = npu_sim.get_elf_entry_and_symbol(
        elf_file, ["inference_status", "inference_input", "inference_output"]
    )
    npu_sim.load_program(elf_file, entry_point)

    if symbol_map.get("inference_input"):
        npu_sim.write_memory(
            symbol_map["inference_input"],
            np.frombuffer(input_data.tobytes(), dtype=np.uint8),
        )

    npu_sim.run()
    npu_sim.wait()

    output = None
    if symbol_map.get("inference_output"):
        output_raw = npu_sim.read_memory(symbol_map["inference_output"], 35)
        output = np.frombuffer(output_raw, dtype=np.int8)

    status = None
    if symbol_map.get("inference_status"):
        status = int(npu_sim.read_memory(symbol_map["inference_status"], 1)[0])

    return output, status, int(npu_sim.get_cycle_count())


def run_bcresnet(num_samples: int = 5, seed: int = 123, fail_on_diff: bool = True):
    print("Running BCResNet KWS accuracy validation...")
    rng = np.random.default_rng(seed)

    max_abs_diff = 0
    mismatched_samples = 0
    total_l1_diff = 0
    npu_cycles = []
    ref_cycles = []

    for sample_idx in range(num_samples):
        input_data = rng.integers(
            low=-128, high=127, size=(1 * 1 * 40 * 101), dtype=np.int8
        )

        ref_output, ref_status, cyc_ref = _run_single_elf(
            "coralnpu_hw/tests/npusim_examples/run_bcresnet_ref_binary.elf", input_data
        )
        npu_output, npu_status, cyc_npu = _run_single_elf(
            "coralnpu_hw/tests/npusim_examples/run_bcresnet_binary.elf", input_data
        )

        if ref_status != 0 or npu_status != 0:
            raise RuntimeError(
                f"Sample {sample_idx}: invalid status ref={ref_status}, npu={npu_status}"
            )
        if ref_output is None or npu_output is None:
            raise RuntimeError(f"Sample {sample_idx}: missing output buffer")

        diff = npu_output.astype(np.int16) - ref_output.astype(np.int16)
        sample_max = int(np.max(np.abs(diff)))
        sample_l1 = int(np.sum(np.abs(diff)))
        max_abs_diff = max(max_abs_diff, sample_max)
        total_l1_diff += sample_l1
        if sample_max != 0:
            mismatched_samples += 1

        npu_cycles.append(cyc_npu)
        ref_cycles.append(cyc_ref)

        print(
            f"sample={sample_idx} max_abs_diff={sample_max} "
            f"l1_diff={sample_l1} npu_cycles={cyc_npu} ref_cycles={cyc_ref}"
        )

    print("ACCURACY_SUMMARY_BEGIN")
    print(f"num_samples={num_samples}")
    print(f"mismatched_samples={mismatched_samples}")
    print(f"max_abs_diff={max_abs_diff}")
    print(f"total_l1_diff={total_l1_diff}")
    print(
        f"avg_npu_cycles={int(np.mean(np.array(npu_cycles, dtype=np.int64)))} "
        f"avg_ref_cycles={int(np.mean(np.array(ref_cycles, dtype=np.int64)))}"
    )
    print("ACCURACY_SUMMARY_END")

    if fail_on_diff and max_abs_diff != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--allow_diff",
        action="store_true",
        help="Do not fail process when differential mismatch exists",
    )
    args = parser.parse_args()
    run_bcresnet(
        num_samples=args.num_samples,
        seed=args.seed,
        fail_on_diff=not args.allow_diff,
    )
