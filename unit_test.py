import jax
import jax.numpy as jnp
import numpy as np
import random
from jaxpr_compiler import (
    compile_closed_jaxpr_to_arm64, 
    compile_closed_jaxpr_to_cuda, 
    HeterogeneousRuntime
)

def get_random_function(depth=50):
    ops = [lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y, lambda x, y: x / 1.1]
    def fn(x, y):
        val = x
        for i in range(depth):
            op = random.choice(ops)
            val = op(val, y)
        return val
    return fn

def main():
    depth = 50 
    random_math = get_random_function(depth)
    data_x = jnp.ones((4, 4), dtype=jnp.float32)
    data_y = jnp.full((4, 4), 0.5, dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(random_math)(data_x, data_y)
    with HeterogeneousRuntime(compile_closed_jaxpr_to_arm64(jaxpr), 
                              compile_closed_jaxpr_to_cuda(jaxpr)) as runtime:
        runtime.compile_and_load()
        cpu_res, gpu_res = runtime.execute_concurrently(np.array(data_x), np.array(data_y))
        expected = random_math(data_x, data_y)
        print(f"Graph Depth: {depth}")
        print("ARM64 Output Match:", np.allclose(cpu_res, expected, atol=1e-3))
        print("CUDA Output Match:", np.allclose(gpu_res, expected, atol=1e-3))

if __name__ == "__main__":
    main()
