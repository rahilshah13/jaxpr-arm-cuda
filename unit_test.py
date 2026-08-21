import jax
import jax.numpy as jnp
import numpy as np
import random
import subprocess
import json
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

def export_jaxpr_to_json(closed_jaxpr, filepath="jaxpr.json"):
    jaxpr = closed_jaxpr.jaxpr
    
    def serialize_var(v):
        return {"Name": v.name}
        
    def serialize_invar(inv):
        if isinstance(inv, jax.core.Literal):
            return {"Type": "Literal", "Val": float(inv.val)}
        else:
            return {"Type": "Var", "Name": inv.name}

    eqns = []
    for eqn in jaxpr.eqns:
        eqns.append({
            "Primitive": {"Name": eqn.primitive.name},
            "Invars": [serialize_invar(inv) for inv in eqn.invars],
            "Outvars": [serialize_var(out) for out in eqn.outvars]
        })

    data = {
        "Invars": [serialize_var(v) for v in jaxpr.invars],
        "Eqns": eqns,
        "Outvars": [serialize_var(v) for v in jaxpr.outvars]
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f)

def run_go_validator(data_x, data_y):
    result = subprocess.run(
        ["go", "run", "jaxpr_compiler.go"],
        input=f"{data_x.size}\n" + " ".join(map(str, data_x.flatten())) + "\n" + " ".join(map(str, data_y.flatten())) + "\n",
        text=True,
        capture_output=True,
        env={"GOEXPERIMENT": "simd", **subprocess.os.environ}
    )
    if result.returncode != 0:
        raise RuntimeError(f"Go validator failed: {result.stderr}")
    return np.array([float(val) for val in result.stdout.strip().split()], dtype=np.float32).reshape(data_x.shape)

def main():
    depth = 50 
    random_math = get_random_function(depth)
    data_x = jnp.ones((4, 4), dtype=jnp.float32)
    data_y = jnp.full((4, 4), 0.5, dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(random_math)(data_x, data_y)
    
    export_jaxpr_to_json(jaxpr)

    with HeterogeneousRuntime(compile_closed_jaxpr_to_arm64(jaxpr), 
                              compile_closed_jaxpr_to_cuda(jaxpr)) as runtime:
        runtime.compile_and_load()
        cpu_res, gpu_res = runtime.execute_concurrently(np.array(data_x), np.array(data_y))
        expected = random_math(data_x, data_y)
        
        go_res = run_go_validator(np.array(data_x), np.array(data_y))

        print(f"Graph Depth: {depth}")
        print("ARM64 Output Match:", np.allclose(cpu_res, expected, atol=1e-3))
        print("CUDA Output Match:", np.allclose(gpu_res, expected, atol=1e-3))
        print("Go SIMD Output Match:", np.allclose(go_res, np.array(expected), atol=1e-3))

if __name__ == "__main__":
    main()
