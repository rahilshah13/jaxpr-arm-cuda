"""
This script defines an arbitrary mathematical JAX function, compiles its intermediate representation (JAXPR) into host-native ARM64 NEON assembly and discrete NVIDIA CUDA C binaries, and executes both compiled shared libraries concurrently on CPU and GPU hardware threads.
"""

import jax
import jax.numpy as jnp
import numpy as np
from main import compile_closed_jaxpr_to_arm64, compile_closed_jaxpr_to_cuda, HeterogeneousRuntime

def arbitrary_math(x, y, z):
    a = x * y
    b = a + z
    return (b - x) / 2.0

dim = (3, 7)
arr_x = jnp.arange(1, 22, dtype=jnp.float32).reshape(dim)
arr_y = jnp.arange(5, 26, dtype=jnp.float32).reshape(dim)
arr_z = jnp.ones(dim, dtype=jnp.float32) * 10.0

closed_jaxpr = jax.make_jaxpr(arbitrary_math)(arr_x, arr_y, arr_z)

arm_asm = compile_closed_jaxpr_to_arm64(closed_jaxpr)
cuda_code = compile_closed_jaxpr_to_cuda(closed_jaxpr)

np_x = np.array(arr_x)
np_y = np.array(arr_y)
np_z = np.array(arr_z)

runtime = HeterogeneousRuntime(arm_asm, cuda_code)
runtime.compile_and_load()
cpu_res, gpu_res = runtime.execute_concurrently(np_x, np_y, np_z)

expected = arbitrary_math(np_x, np_y, np_z)
print("ARM64 Output Match:", np.allclose(cpu_res, expected))
print("CUDA Output Match:", np.allclose(gpu_res, expected))
