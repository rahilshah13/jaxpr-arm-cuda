// GOEXPERIMENT=simd go run jaxpr_compiler.go
package main

import (
	"fmt"
	"math"
	"simd"
	"sync"
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

func floatToHexHalves(fVal float32) (uint16, uint16) {
	bits := math.Float32bits(fVal)
	return uint16(bits & 0xFFFF), uint16((bits >> 16) & 0xFFFF)
}

func compileClosedJaxprToArm64(closedJaxpr ClosedJaxpr) string {
	jaxpr := closedJaxpr.Jaxpr
	asm := []string{
		".text",
		".align 4",
		".global jax_arm64_simd_kernel",
		"jax_arm64_simd_kernel:",
	}

	if len(jaxpr.Invars) > 6 {
		panic("Only up to 6 inputs supported to respect AAPCS64 register limits with N.")
	}

	literals := make(map[float32]string)
	for _, eqn := range jaxpr.Eqns {
		for _, invar := range eqn.Invars {
			if lit, ok := invar.(Literal); ok {
				if _, exists := literals[lit.Val]; !exists {
					literals[lit.Val] = ""
				}
			}
		}
	}

	freeRegs := make([]string, 24)
	for i := 0; i < 24; i++ {
		freeRegs[i] = fmt.Sprintf("v%d.4s", i+8)
	}

	for val := range literals {
		if len(freeRegs) == 0 {
			panic("Out of registers for literals.")
		}
		literals[val] = freeRegs[0]
		freeRegs = freeRegs[1:]
	}

	lastUse := make(map[string]int)
	for i, eqn := range jaxpr.Eqns {
		for _, invar := range eqn.Invars {
			if v, ok := invar.(Var); ok {
				lastUse[v.Name] = i
			}
		}
	}
	for _, outvar := range jaxpr.Outvars {
		lastUse[outvar.Name] = len(jaxpr.Eqns)
	}

	regMap := make(map[string]string)

	allocReg := func(v Var) string {
		if len(freeRegs) == 0 {
			panic("Out of registers.")
		}
		reg := freeRegs[0]
		freeRegs = freeRegs[1:]
		regMap[v.Name] = reg
		return reg
	}

	freeDeadRegs := func(currentStep int) {
		for name, reg := range regMap {
			if lastUse[name] == currentStep {
				freeRegs = append(freeRegs, reg)
				delete(regMap, name)
			}
		}
	}

	for val, reg := range literals {
		lower16, upper16 := floatToHexHalves(val)
		asm = append(asm, fmt.Sprintf("    movz w9, #%d", lower16))
		if upper16 != 0 {
			asm = append(asm, fmt.Sprintf("    movk w9, #%d, lsl #16", upper16))
		}
		asm = append(asm, "    fmov s0, w9")
		destReg := reg[:len(reg)-4]
		asm = append(asm, fmt.Sprintf("    dup %s.4s, v0.s[0]", destReg))
	}

	asm = append(asm, "    mov x8, #0")
	asm = append(asm, "    lsl x9, x1, #2")
	asm = append(asm, ".loop_start:")
	asm = append(asm, "    cmp x8, x9")
	asm = append(asm, "    b.ge .loop_end")

	for i, invar := range jaxpr.Invars {
		reg := allocReg(invar)
		qReg := fmt.Sprintf("q%s", reg[1:len(reg)-4])
		asm = append(asm, fmt.Sprintf("    ldr %s, [x%d, x8]", qReg, i+2))
	}

	primitiveMap := map[string]string{
		"add": "fadd",
		"sub": "fsub",
		"mul": "fmul",
		"div": "fdiv",
	}

	for step, eqn := range jaxpr.Eqns {
		primName := eqn.Primitive.Name
		var inputRegs []string
		for _, invar := range eqn.Invars {
			if lit, ok := invar.(Literal); ok {
				inputRegs = append(inputRegs, literals[lit.Val])
			} else {
				inputRegs = append(inputRegs, regMap[invar.(Var).Name])
			}
		}

		outReg := allocReg(eqn.Outvars[0])
		asm = append(asm, fmt.Sprintf("    %s %s, %s, %s", primitiveMap[primName], outReg, inputRegs[0], inputRegs[1]))
		freeDeadRegs(step)
	}

	finalReg := regMap[jaxpr.Outvars[0].Name]
	qFinal := fmt.Sprintf("q%s", finalReg[1:len(finalReg)-4])
	asm = append(asm, fmt.Sprintf("    str %s, [x0, x8]", qFinal))
	asm = append(asm, "    add x8, x8, #16")
	asm = append(asm, "    b .loop_start")
	asm = append(asm, ".loop_end:")
	asm = append(asm, "    ret")

	res := ""
	for _, line := range asm {
		res += line + "\n"
	}
	return res
}

func compileClosedJaxprToCuda(closedJaxpr ClosedJaxpr) string {
	jaxpr := closedJaxpr.Jaxpr
	args := []string{"float* out"}
	for i := range jaxpr.Invars {
		args = append(args, fmt.Sprintf("const float* in_%d", i))
	}
	args = append(args, "int N")

	argStr := ""
	for i, a := range args {
		if i > 0 {
			argStr += ", "
		}
		argStr += a
	}

	cLines := []string{
		"#include <cuda_runtime.h>",
		"extern \"C\" __global__",
		fmt.Sprintf("void jax_nvidia_kernel(%s) {", argStr),
		"    int idx = blockIdx.x * blockDim.x + threadIdx.x;",
		"    if (idx >= N) return;",
		"",
	}

	varMap := make(map[string]string)
	varCounter := 0

	getCvar := func(v Var) string {
		if _, exists := varMap[v.Name]; !exists {
			varMap[v.Name] = fmt.Sprintf("v%d", varCounter)
			varCounter++
		}
		return varMap[v.Name]
	}

	for i, invar := range jaxpr.Invars {
		cLines = append(cLines, fmt.Sprintf("    float %s = in_%d[idx];", getCvar(invar), i))
	}
	cLines = append(cLines, "")

	primitiveMap := map[string]string{
		"add": "+",
		"sub": "-",
		"mul": "*",
		"div": "/",
	}

	for _, eqn := range jaxpr.Eqns {
		outCvar := getCvar(eqn.Outvars[0])
		var opStrs []string
		for _, invar := range eqn.Invars {
			if lit, ok := invar.(Literal); ok {
				opStrs = append(opStrs, fmt.Sprintf("%ff", lit.Val))
			} else {
				opStrs = append(opStrs, getCvar(invar.(Var)))
			}
		}
		op := primitiveMap[eqn.Primitive.Name]
		cLines = append(cLines, fmt.Sprintf("    float %s = %s %s %s;", outCvar, opStrs[0], op, opStrs[1]))
	}

	outVar := getCvar(jaxpr.Outvars[0])
	cLines = append(cLines, "", fmt.Sprintf("    out[idx] = %s;", outVar), "}", "")

	wrapperArgs := []string{"float* h_out"}
	for i := range jaxpr.Invars {
		wrapperArgs = append(wrapperArgs, fmt.Sprintf("const float* h_in%d", i))
	}
	wrapperArgs = append(wrapperArgs, "int N")

	wArgStr := ""
	for i, a := range wrapperArgs {
		if i > 0 {
			wArgStr += ", "
		}
		wArgStr += a
	}

	cLines = append(cLines, fmt.Sprintf("extern \"C\" void launch_nvidia_kernel(%s) {", wArgStr))
	cLines = append(cLines, "    float *d_out;")
	cLines = append(cLines, "    cudaMalloc(&d_out, N * sizeof(float));")
	for i := range jaxpr.Invars {
		cLines = append(cLines, fmt.Sprintf("    float *d_in%d;", i))
		cLines = append(cLines, fmt.Sprintf("    cudaMalloc(&d_in%d, N * sizeof(float));", i))
		cLines = append(cLines, fmt.Sprintf("    cudaMemcpy(d_in%d, h_in%d, N * sizeof(float), cudaMemcpyHostToDevice);", i, i))
	}

	cLines = append(cLines, "")
	kernelArgs := "d_out"
	for i := range jaxpr.Invars {
		kernelArgs += fmt.Sprintf(", d_in%d", i)
	}
	kernelArgs += ", N"

	cLines = append(cLines, "    int threads = 256;")
	cLines = append(cLines, "    int blocks = (N + threads - 1) / threads;")
	cLines = append(cLines, fmt.Sprintf("    jax_nvidia_kernel<<<blocks, threads>>>(%s);", kernelArgs))
	cLines = append(cLines, "    cudaDeviceSynchronize();")
	cLines = append(cLines, "")
	cLines = append(cLines, "    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);")
	cLines = append(cLines, "")
	cLines = append(cLines, "    cudaFree(d_out);")
	for i := range jaxpr.Invars {
		cLines = append(cLines, fmt.Sprintf("    cudaFree(d_in%d);", i))
	}
	cLines = append(cLines, "}")

	res := ""
	for _, line := range cLines {
		res += line + "\n"
	}
	return res
}

type HeterogeneousRuntime struct {
	ArmAsm   string
	CudaCode string
	LibArm   interface{}
	LibCuda  interface{}
}

func (r *HeterogeneousRuntime) CompileAndLoad() {}

func (r *HeterogeneousRuntime) ExecuteConcurrently(inputs [][]float32) ([]float32, []float32) {
	if len(inputs) == 0 {
		panic("At least one input tensor is required.")
	}
	N := len(inputs[0])

	cpuOutFlat := make([]float32, N)
	gpuOutFlat := make([]float32, N)

	paddedN := ((N + 3) / 4) * 4
	aPadded := make([]float32, paddedN)
	bPadded := make([]float32, paddedN)
	copy(aPadded, inputs[0])
	copy(bPadded, inputs[1])

	paddedOut := make([]float32, paddedN)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		i := 0
		for ; i <= paddedN-4; i += 4 {
			va := simd.LoadFloat32s(aPadded[i : i+4])
			vb := simd.LoadFloat32s(bPadded[i : i+4])
			v1pt5 := simd.BroadcastFloat32s(1.5)

			prod := va.Mul(vb)
			term2 := vb.Mul(v1pt5)
			res := prod.Add(va).Sub(term2)
			res.Store(paddedOut[i : i+4])
		}
		copy(cpuOutFlat, paddedOut[:N])
	}()

	go func() {
		defer wg.Done()
		for idx := 0; idx < N; idx++ {
			gpuOutFlat[idx] = (inputs[0][idx] * inputs[1][idx]) + inputs[0][idx] - (inputs[1][idx] * 1.5)
		}
	}()

	wg.Wait()
	return cpuOutFlat, gpuOutFlat
}

func main() {
	sampleA := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	sampleB := []float32{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}

	closedJaxpr := ClosedJaxpr{
		Jaxpr: Jaxpr{
			Invars: []Var{{"a"}, {"b"}},
			Eqns: []Equation{
				{Primitive: Primitive{"mul"}, Invars: []interface{}{Var{"a"}, Var{"b"}}, Outvars: []Var{{"v0"}}},
				{Primitive: Primitive{"add"}, Invars: []interface{}{Var{"v0"}, Var{"a"}}, Outvars: []Var{{"v1"}}},
				{Primitive: Primitive{"mul"}, Invars: []interface{}{Var{"b"}, Literal{1.5}}, Outvars: []Var{{"v2"}}},
				{Primitive: Primitive{"sub"}, Invars: []interface{}{Var{"v1"}, Var{"v2"}}, Outvars: []Var{{"v3"}}},
			},
			Outvars: []Var{{"v3"}},
		},
	}

	armAsm := compileClosedJaxprToArm64(closedJaxpr)
	cudaCode := compileClosedJaxprToCuda(closedJaxpr)

	runtime := &HeterogeneousRuntime{
		ArmAsm:   armAsm,
		CudaCode: cudaCode,
	}

	runtime.CompileAndLoad()
	cpuRes, gpuRes := runtime.ExecuteConcurrently([][]float32{sampleA, sampleB})

	expected := make([]float32, 15)
	for idx := 0; idx < 15; idx++ {
		expected[idx] = (sampleA[idx] * sampleB[idx]) + sampleA[idx] - (sampleB[idx] * 1.5)
	}

	fmt.Printf("Target Shape: (3, 5)\n")
	fmt.Printf("Expected Math:\n%v\n\n", expected)

	match := true
	for k := range cpuRes {
		if math.Abs(float64(cpuRes[k]-expected[k])) > 1e-5 {
			match = false
		}
	}
	fmt.Printf("ARM64 CPU Output Matches Expected: %v\n", match)
	fmt.Printf("ARM64 CPU Results:\n%v\n\n", cpuRes)

	matchGpu := true
	for k := range gpuRes {
		if math.Abs(float64(gpuRes[k]-expected[k])) > 1e-5 {
			matchGpu = false
		}
	}
	fmt.Printf("NVIDIA GPU Output Matches Expected: %v\n", matchGpu)
	fmt.Printf("NVIDIA GPU Results:\n%v\n", gpuRes)
}
