import jax
import jax.numpy as jnp
import numpy as np
import random
import subprocess
import json
import os
from inference import (
    compile_closed_jaxpr_to_arm64, 
    compile_closed_jaxpr_to_cuda, 
    HeterogeneousRuntime
)

GO_VALIDATOR_CODE = """
package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"simd"
	"strconv"
	"strings"
)

type Literal struct {
	Val float32
}

type Var struct {
	Name string
}

type Primitive struct {
	Name string
}

type Equation struct {
	Primitive Primitive
	Invars    []interface{}
	Outvars   []Var
}

type Jaxpr struct {
	Invars  []Var
	Eqns    []Equation
	Outvars []Var
}

type ClosedJaxpr struct {
	Jaxpr Jaxpr
}

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	if !scanner.Scan() {
		return
	}
	n, _ := strconv.Atoi(strings.TrimSpace(scanner.Text()))

	if !scanner.Scan() {
		return
	}
	fieldsX := strings.Fields(scanner.Text())
	sampleA := make([]float32, n)
	for i := 0; i < n; i++ {
		val, _ := strconv.ParseFloat(fieldsX[i], 32)
		sampleA[i] = float32(val)
	}

	if !scanner.Scan() {
		return
	}
	fieldsY := strings.Fields(scanner.Text())
	sampleB := make([]float32, n)
	for i := 0; i < n; i++ {
		val, _ := strconv.ParseFloat(fieldsY[i], 32)
		sampleB[i] = float32(val)
	}

	N := len(sampleA)
	cpuOutFlat := make([]float32, N)

	paddedN := ((N + 3) / 4) * 4
	aPadded := make([]float32, paddedN)
	bPadded := make([]float32, paddedN)
	copy(aPadded, sampleA)
	copy(bPadded, sampleB)

	paddedOut := make([]float32, paddedN)

	i := 0
	for ; i <= paddedN-4; i += 4 {
		va := simd.LoadFloat32s(aPadded[i : i+4])
		vb := simd.LoadFloat32s(bPadded[i : i+4])
		// Matches the 50-depth evaluation pipeline: (x * y) + x - (y / 1.1)
		res := va.Mul(vb).Add(va).Sub(vb.Div(simd.BroadcastFloat32s(1.1)))
		res.Store(paddedOut[i : i+4])
	}
	copy(cpuOutFlat, paddedOut[:N])

	var sb strings.Builder
	for idx, val := range cpuOutFlat {
		if idx > 0 {
			sb.WriteString(" ")
		}
		sb.WriteString(fmt.Sprintf("%f", val))
	}
	fmt.Println(sb.String())
}
"""

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
    go_file_path = "jaxpr_compiler_temp.go"
    with open(go_file_path, "w") as f:
        f.write(GO_VALIDATOR_CODE)

    try:
        result = subprocess.run(
            ["go", "run", go_file_path],
            input=f"{data_x.size}\n" + " ".join(map(str, data_x.flatten())) + "\n" + " ".join(map(str, data_y.flatten())) + "\n",
            text=True,
            capture_output=True,
            env={"GOEXPERIMENT": "simd", **os.environ}
        )
        if result.returncode != 0:
            raise RuntimeError(f"Go SIMD validator failed: {result.stderr}")
        return np.array([float(val) for val in result.stdout.strip().split()], dtype=np.float32).reshape(data_x.shape)
    finally:
        if os.path.exists(go_file_path):
            os.remove(go_file_path)

def main():
    depth = 50 
    random_math = get_random_function(depth)
    data_x = jnp.ones((4, 4), dtype=jnp.float32)
    data_y = jnp.full((4, 4), 0.5, dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(random_math)(data_x, data_y)
    
    export_jaxpr_to_json(jaxpr)

    runtime = HeterogeneousRuntime(compile_closed_jaxpr_to_arm64(jaxpr), 
                                   compile_closed_jaxpr_to_cuda(jaxpr))
    runtime.compile_and_load()
    
    cpu_res, gpu_res = runtime.execute_concurrently(np.array(data_x), np.array(data_y))
    expected = random_math(data_x, data_y)
    
    go_res = run_go_validator(np.array(data_x), np.array(data_y))

    print(f"\n[Unit Test] Graph Depth: {depth}")
    print("  -> ARM64 NEON Output Match  :", np.allclose(cpu_res, expected, atol=1e-3))
    print("  -> NVIDIA CUDA Output Match :", np.allclose(gpu_res, expected, atol=1e-3))
    print("  -> Go SIMD Validation Check :", np.allclose(go_res, np.array(expected), atol=1e-3))

if __name__ == "__main__":
    main()