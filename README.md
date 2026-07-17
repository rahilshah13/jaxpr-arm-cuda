# JAXPR AOT Compiler & Runtime

A pipeline lowering JAX expressions (`jaxpr`) to ARM64 NEON and NVIDIA CUDA, bypassing XLA for bare-metal execution.

---

*   **ARM64 Backend:** Liveness-based allocation (`v8`-`v31`); float loading via `movz`/`movk`; 128-bit vector loops.
*   **CUDA Backend:** Maps array dimensions to SIMT threads; generates C++ operations with managed host-device copies.
*   **Runtime:** Spawns POSIX threads executing CPU and GPU streams concurrently.
*   **AWS Instance:** `g6g.xlarge` (or larger)

### Dependencies
```bash
sudo apt-get update && sudo apt-get install -y gcc g++ build-essential python3-dev nvidia-cuda-toolkit
pip3 install jax jaxlib numpy

# Compile ARM64 shared library
gcc -shared -o ./libarm.so kernel.s

# Compile NVIDIA CUDA shared library
nvcc -shared -o ./libnvidia.so -Xcompiler -fPIC kernel.cu
