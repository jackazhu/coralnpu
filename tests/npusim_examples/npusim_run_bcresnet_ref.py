# Copyright 2026 Google LLC
# Reference-only BCResNet runner for bit-accuracy comparison.
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np

def run_bcresnet_ref():
    npu_sim = CoralNPUV2Simulator(highmem_ld=True, exit_on_ebreak=True)
    r = runfiles.Create()
    elf = r.Rlocation('coralnpu_hw/tests/npusim_examples/run_bcresnet_ref_binary.elf')
    entry, syms = npu_sim.get_elf_entry_and_symbol(
        elf, ['inference_status', 'inference_input', 'inference_output'])
    npu_sim.load_program(elf, entry)
    if syms.get('inference_input'):
        rng = np.random.default_rng(seed=42)
        npu_sim.write_memory(syms['inference_input'],
                             rng.integers(-128, 127, 40*101, dtype=np.int8))
    npu_sim.run(); npu_sim.wait()
    if syms.get('inference_output'):
        out = np.frombuffer(npu_sim.read_memory(syms['inference_output'], 35), dtype=np.int8)
        print(f"REF_ALL35: {list(out)}")
    if syms.get('inference_status'):
        s = npu_sim.read_memory(syms['inference_status'], 1)[0]
        print(f"inference_status {s}")

if __name__ == "__main__":
    run_bcresnet_ref()
