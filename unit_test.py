import json
import os
import random
import jax
import jax.numpy as jnp
import numpy as np
from inference import (
    compile_closed_jaxpr_to_arm64,
    compile_closed_jaxpr_to_cuda,
    HeterogeneousRuntime,
)

GO_VALIDATOR_CODE = """
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"simd"
	"strconv"
	"strings"
)

type Var struct {
	Name string `json:"Name"`
}

type Primitive struct {
	Name string `json:"Name"`
}

type InputVar struct {
	Type string  `json:"Type"`
	Name string  `json:"Name"`
	Val  float32 `json:"Val"`
}

type Equation struct {
	Primitive Primitive  `json:"Primitive"`
	Invars    []InputVar `json:"Invars"`
	Outvars   []Var      `json:"Outvars"`
}

type Jaxpr struct {
	Invars  []Var      `json:"Invars"`
	Eqns    []Equation `json:"Eqns"`
	Outvars []Var      `json:"Outvars"`
}

type SimdVector interface {
	Mul(SimdVector) SimdVector
	Add(SimdVector) SimdVector
	Sub(SimdVector) SimdVector
	Div(SimdVector) SimdVector
	Store([]float32)
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

	// Load and parse jaxpr.json exported by Python
	file, err := os.Open("jaxpr.json")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open jaxpr.json: %v\\n", err)
		return
	}
	defer file.Close()

	var jaxpr Jaxpr
	if err := json.NewDecoder(file).Decode(&jaxpr); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to decode jaxpr.json: %v\\n", err)
		return
	}

	N := len(sampleA)
	cpuOutFlat := make([]float32, N)

	paddedN := ((N + 3) / 4) * 4
	aPadded := make([]float32, paddedN)
	bPadded := make([]float32, paddedN)
	copy(aPadded, sampleA)
	copy(bPadded, sampleB)

	paddedOut := make([]float32, paddedN)

	for i := 0; i <= paddedN-4; i += 4 {
		env := make(map[string]SimdVector)
		if len(jaxpr.Invars) >= 2 {
			env[jaxpr.Invars[0].Name] = simd.LoadFloat32s(aPadded[i : i+4])
			env[jaxpr.Invars[1].Name] = simd.LoadFloat32s(bPadded[i : i+4])
		}

		for _, eqn := range jaxpr.Eqns {
			var v1, v2 SimdVector

			if len(eqn.Invars) > 0 {
				if eqn.Invars[0].Type == "Literal" {
					v1 = simd.BroadcastFloat32s(eqn.Invars[0].Val)
				} else {
					v1 = env[eqn.Invars[0].Name]
				}
			}

			if len(eqn.Invars) > 1 {
				if eqn.Invars[1].Type == "Literal" {
					v2 = simd.BroadcastFloat32s(eqn.Invars[1].Val)
				} else {
					v2 = env[eqn.Invars[1].Name]
				}
			}

			var res SimdVector
			prim := strings.ToLower(eqn.Primitive.Name)
			switch prim {
			case "mul", "mul_f32":
				res = v1.Mul(v2)
			case "add", "add_f32":
				res = v1.Add(v2)
			case "sub", "sub_f32":
				res = v1.Sub(v2)
			case "div", "div_f32":
				res = v1.Div(v2)
			default:
				res = v1
			}

			if len(eqn.Outvars) > 0 {
				env[eqn.Outvars[0].Name] = res
			}
		}

		if len(jaxpr.Outvars) > 0 {
			finalVar := jaxpr.Outvars[0].Name
			if finalVec, ok := env[finalVar]; ok {
				finalVec.Store(paddedOut[i : i+4])
			}
		}
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
    random.seed(42)
    ops = [lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y, lambda x, y: x / 1.1]
    def fn(x, y):
        val = x
        for _ in range(depth):
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
	go_res = run_go_validator(np.array(data_x), np.array(data_y))
    expected = random_math(data_x, data_y)
    

    print(f"\n[Unit Test] Graph Depth: {depth}")
    print("  -> ARM64 NEON Output Match  :", np.allclose(cpu_res, expected, atol=1e-3))
    print("  -> NVIDIA CUDA Output Match :", np.allclose(gpu_res, expected, atol=1e-3))
    print("  -> Go SIMD Validation Check :", np.allclose(go_res, np.array(expected), atol=1e-3))

if __name__ == "__main__":
    main()
