from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np

def run():
    npu_sim = CoralNPUV2Simulator(highmem_ld=True, exit_on_ebreak=True)
    r = runfiles.Create()
    elf = r.Rlocation('coralnpu_hw/tests/npusim_examples/run_bcresnet_fullref_binary.elf')
    entry, syms = npu_sim.get_elf_entry_and_symbol(
        elf, ['inference_status', 'inference_input', 'inference_output'])
    npu_sim.load_program(elf, entry)
    if syms.get('inference_input'):
        rng = np.random.default_rng(seed=42)
        inp = rng.integers(-128, 127, 40*101, dtype=np.int8)
        npu_sim.write_memory(syms['inference_input'],
                             np.frombuffer(inp.tobytes(), dtype=np.uint8))
    npu_sim.run(); npu_sim.wait()
    if syms.get('inference_output'):
        out = np.frombuffer(npu_sim.read_memory(syms['inference_output'], 35), dtype=np.int8)
        print(f"FULLREF_ALL35: {list(map(int, out))}")
    if syms.get('inference_status'):
        s = npu_sim.read_memory(syms['inference_status'], 1)[0]
        print(f"inference_status {s}")

if __name__ == "__main__":
    run()
