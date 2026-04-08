# Coral NPU - Development Guide

Coral NPU is a RISC-V based open-source neural processing unit (NPU/AI accelerator) hardware design project by Google Research. The codebase is entirely Bazel-driven. See `README.md` for an overview and quick-start commands.

## Cursor Cloud specific instructions

### System dependencies

The following must be installed on the host (already present in the Cloud Agent VM):

- **Bazel 7.4.1** (pinned in `.bazelversion`; install via `apt install bazel-7.4.1` and symlink to `/usr/local/bin/bazel`)
- **Python 3.9–3.12** (system Python)
- **Java JDK 17** (`openjdk-17-jdk-headless`)
- **SRecord** (`apt install srecord`)
- **GCC/Clang, Make, libelf-dev, libmpfr-dev, autoconf, gawk, xxd, fuse3**

Most other dependencies (Verilator, RISC-V cross-toolchain, Chisel/Scala, cocotb, LLVM firtool, etc.) are fetched hermetically by Bazel on first build.

### Key commands

| Task | Command |
|---|---|
| Run full cocotb test suite | `bazel run //tests/cocotb:core_mini_axi_sim_cocotb` |
| Build an example binary | `bazel build //examples:coralnpu_v2_hello_world_add_floats` |
| Build the Verilator simulator | `bazel build //tests/verilator_sim:core_mini_axi_sim` |
| Run a binary on the simulator | `bazel-out/k8-fastbuild/bin/tests/verilator_sim/core_mini_axi_sim --binary <path-to-elf>` |

### Important caveats

- **First build is slow**: The initial `bazel run` / `bazel build` fetches and compiles Verilator, protobuf, SystemC, and other large C++ dependencies. Expect ~10+ minutes. Subsequent builds are incremental.
- **`bazel-bin` symlink points to the cross-compile config** (`k8-fastbuild-ST-*`), not the host config. Host-built artifacts like the Verilator simulator live under `bazel-out/k8-fastbuild/bin/`.
- **No formal lint tooling**: There is no buildifier/clang-format/verible configuration. Lint targets use VCS (commercial tool, not available in Cloud Agent). The `.bazelrc` excludes VCS and synthesis targets by default (`--build_tag_filters="-vcs,-synthesis"`).
- **`bazel run` vs `bazel build`**: For the cocotb test, use `bazel run` (not `bazel test`). For the simulator, build first then invoke the binary directly to avoid Bazel's `run` config switching overhead.
- **libmpfr compatibility**: The RISC-V toolchain may look for `libmpfr.so.4`. Create a symlink: `sudo ln -sf /lib/x86_64-linux-gnu/libmpfr.so.6 /lib/x86_64-linux-gnu/libmpfr.so.4`.
- **BFD linker required**: The `.bazelrc` specifies `--host_linkopt=-fuse-ld=bfd` and `--linkopt=-fuse-ld=bfd`. Ensure `ld.bfd` is available (comes with `binutils`).
