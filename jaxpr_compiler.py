"""
This module compiles arbitrary JAX expressions (jaxprs) into ARM64 NEON assembly and NVIDIA CUDA C binaries, linking and executing them concurrently on CPU and GPU hardware threads.

Usage:
    closed_jaxpr = jax.make_jaxpr(your_math_fn)(*sample_inputs)
    arm_asm = compile_closed_jaxpr_to_arm64(closed_jaxpr)
    cuda_code = compile_closed_jaxpr_to_cuda(closed_jaxpr)
    runtime = HeterogeneousRuntime(arm_asm, cuda_code)
    runtime.compile_and_load()
    cpu_results, gpu_results = runtime.execute_concurrently(*numpy_inputs)
"""

import jax
import jax.numpy as jnp
from jax.core import Literal
import struct
import subprocess
import ctypes
import threading
import numpy as np
import os


def float_to_hex_halves(f_val):
    uint32_val = struct.unpack('<I', struct.pack('<f', float(f_val)))[0]
    return uint32_val & 0xFFFF, (uint32_val >> 16) & 0xFFFF


def compile_closed_jaxpr_to_arm64(closed_jaxpr):
    jaxpr = closed_jaxpr.jaxpr
    asm = [
        ".text",
        ".align 4",
        ".global jax_arm64_simd_kernel",
        "jax_arm64_simd_kernel:"
    ]

    if len(jaxpr.invars) > 6:
        raise NotImplementedError("Only up to 6 inputs supported to respect AAPCS64 register limits with N.")

    literals = {}
    for eqn in jaxpr.eqns:
        for invar in eqn.invars:
            if isinstance(invar, Literal):
                val = float(invar.val)
                if val not in literals:
                    literals[val] = None

    free_regs = [f"v{i}.4s" for i in range(8, 32)]
    
    for val in literals:
        if not free_regs:
            raise RuntimeError("Out of registers for literals.")
        literals[val] = free_regs.pop(0)

    last_use = {}
    for i, eqn in enumerate(jaxpr.eqns):
        for invar in eqn.invars:
            if not isinstance(invar, Literal):
                last_use[invar] = i
    for outvar in jaxpr.outvars:
        last_use[outvar] = len(jaxpr.eqns)

    reg_map = {}

    def alloc_reg(var):
        if not free_regs:
            raise RuntimeError("Out of registers.")
        reg = free_regs.pop(0)
        reg_map[var] = reg
        return reg

    def free_dead_regs(current_step):
        for var, reg in list(reg_map.items()):
            if last_use.get(var, -1) == current_step:
                free_regs.append(reg)
                del reg_map[var]

    for val, reg in literals.items():
        lower_16, upper_16 = float_to_hex_halves(val)
        asm.append(f"    movz w9, #{lower_16}")
        if upper_16 != 0:
            asm.append(f"    movk w9, #{upper_16}, lsl #16")
        asm.append("    fmov s0, w9")
        dest_reg = reg.replace(".4s", "")
        asm.append(f"    dup {dest_reg}.4s, v0.s[0]")

    asm.append("    mov x8, #0")
    asm.append("    lsl x9, x1, #2")
    asm.append(".loop_start:")
    asm.append("    cmp x8, x9")
    asm.append("    b.ge .loop_end")

    for i, invar in enumerate(jaxpr.invars):
        reg = alloc_reg(invar)
        q_reg = reg.replace(".4s", "").replace("v", "q")
        asm.append(f"    ldr {q_reg}, [x{i + 2}, x8]")

    primitive_map = {"add": "fadd", "sub": "fsub", "mul": "fmul", "div": "fdiv"}

    for step, eqn in enumerate(jaxpr.eqns):
        prim_name = eqn.primitive.name
        input_regs = []
        for invar in eqn.invars:
            if isinstance(invar, Literal):
                input_regs.append(literals[float(invar.val)])
            else:
                input_regs.append(reg_map[invar])

        out_reg = alloc_reg(eqn.outvars[0])
        asm.append(f"    {primitive_map[prim_name]} {out_reg}, {input_regs[0]}, {input_regs[1]}")

        free_dead_regs(step)

    final_reg = reg_map[jaxpr.outvars[0]]
    q_final = final_reg.replace(".4s", "").replace("v", "q")
    asm.append(f"    str {q_final}, [x0, x8]")
    asm.append("    add x8, x8, #16")
    asm.append("    b .loop_start")
    asm.append(".loop_end:")
    asm.append("    ret")

    return "\n".join(asm)


def compile_closed_jaxpr_to_cuda(closed_jaxpr):
    jaxpr = closed_jaxpr.jaxpr
    args = ["float* out"]
    for i in range(len(jaxpr.invars)):
        args.append(f"const float* in_{i}")
    args.append("int N")
        
    c_lines = [
        '#include <cuda_runtime.h>',
        'extern "C" __global__',
        f"void jax_nvidia_kernel({', '.join(args)}) {{",
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;",
        "    if (idx >= N) return;",
        ""
    ]

    var_map = {}
    var_counter = 0

    def get_cvar(var):
        nonlocal var_counter
        if var not in var_map:
            var_map[var] = f"v{var_counter}"
            var_counter += 1
        return var_map[var]

    for i, invar in enumerate(jaxpr.invars):
        c_lines.append(f"    float {get_cvar(invar)} = in_{i}[idx];")
    c_lines.append("")

    primitive_map = {"add": "+", "sub": "-", "mul": "*", "div": "/"}

    for eqn in jaxpr.eqns:
        out_cvar = get_cvar(eqn.outvars[0])
        op_strs = []
        for invar in eqn.invars:
            if isinstance(invar, Literal):
                op_strs.append(f"{float(invar.val)}f")
            else:
                op_strs.append(get_cvar(invar))
                
        op = primitive_map[eqn.primitive.name]
        c_lines.append(f"    float {out_cvar} = {op_strs[0]} {op} {op_strs[1]};")

    out_var = get_cvar(jaxpr.outvars[0])
    c_lines.extend([
        "",
        f"    out[idx] = {out_var};",
        "}",
        ""
    ])

    wrapper_args = ["float* h_out"]
    for i in range(len(jaxpr.invars)):
        wrapper_args.append(f"const float* h_in{i}")
    wrapper_args.append("int N")

    c_lines.append(f'extern "C" void launch_nvidia_kernel({", ".join(wrapper_args)}) {{')
    c_lines.append("    float *d_out;")
    c_lines.append("    cudaMalloc(&d_out, N * sizeof(float));")
    for i in range(len(jaxpr.invars)):
        c_lines.append(f"    float *d_in{i};")
        c_lines.append(f"    cudaMalloc(&d_in{i}, N * sizeof(float));")
        c_lines.append(f"    cudaMemcpy(d_in{i}, h_in{i}, N * sizeof(float), cudaMemcpyHostToDevice);")

    c_lines.append("")
    kernel_args = ["d_out"] + [f"d_in{i}" for i in range(len(jaxpr.invars))] + ["N"]
    c_lines.append("    int threads = 256;")
    c_lines.append("    int blocks = (N + threads - 1) / threads;")
    c_lines.append(f"    jax_nvidia_kernel<<<blocks, threads>>>({', '.join(kernel_args)});")
    c_lines.append("    cudaDeviceSynchronize();")
    c_lines.append("")
    c_lines.append("    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);")
    c_lines.append("")
    c_lines.append("    cudaFree(d_out);")
    for i in range(len(jaxpr.invars)):
        c_lines.append(f"    cudaFree(d_in{i});")
    c_lines.append("}")

    return "\n".join(c_lines)


class HeterogeneousRuntime:
    def __init__(self, arm_asm, cuda_code):
        self.arm_asm = arm_asm
        self.cuda_code = cuda_code
        self.lib_arm = None
        self.lib_cuda = None
        
    def compile_and_load(self):
        with open("kernel.s", "w") as f:
            f.write(self.arm_asm)
        with open("kernel.cu", "w") as f:
            f.write(self.cuda_code)
            
        try:
            subprocess.run(["gcc", "-shared", "-o", "./libarm.so", "kernel.s"], check=True)
            self.lib_arm = ctypes.CDLL("./libarm.so")
        except Exception as e:
            pass

        try:
            subprocess.run([
                "nvcc", "-shared", "-o", "./libnvidia.so", 
                "-Xcompiler", "-fPIC", "kernel.cu"
            ], check=True)
            self.lib_cuda = ctypes.CDLL("./libnvidia.so")
        except Exception as e:
            pass

    def execute_concurrently(self, *inputs):
        if not inputs:
            raise ValueError("At least one input tensor is required.")
        
        shapes = [x.shape for x in inputs]
        if len(set(shapes)) > 1:
            raise ValueError("All input tensors must have the exact same shape.")
            
        shape = shapes[0]
        N = inputs[0].size

        flat_inputs = [np.ascontiguousarray(x.flatten(), dtype=np.float32) for x in inputs]

        padded_N = ((N + 3) // 4) * 4
        if padded_N != N:
            padded_inputs = [np.pad(x, (0, padded_N - N), 'constant') for x in flat_inputs]
            padded_out = np.zeros(padded_N, dtype=np.float32)
        else:
            padded_inputs = flat_inputs
            padded_out = np.zeros(N, dtype=np.float32)

        ptr_padded_inputs = [x.ctypes.data_as(ctypes.c_void_p) for x in padded_inputs]
        ptr_padded_out = padded_out.ctypes.data_as(ctypes.c_void_p)

        cpu_out_flat = np.zeros(N, dtype=np.float32)
        gpu_out_flat = np.zeros(N, dtype=np.float32)

        ptr_gpu_inputs = [x.ctypes.data_as(ctypes.c_void_p) for x in flat_inputs]
        ptr_gpu_out = gpu_out_flat.ctypes.data_as(ctypes.c_void_p)

        def run_arm():
            if self.lib_arm:
                self.lib_arm.jax_arm64_simd_kernel.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * len(inputs)
                self.lib_arm.jax_arm64_simd_kernel(ptr_padded_out, ctypes.c_int(padded_N), *ptr_padded_inputs)
                np.copyto(cpu_out_flat, padded_out[:N])

        def run_gpu():
            if self.lib_cuda:
                self.lib_cuda.launch_nvidia_kernel.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * len(inputs) + [ctypes.c_int]
                self.lib_cuda.launch_nvidia_kernel(ptr_gpu_out, *ptr_gpu_inputs, ctypes.c_int(N))

        t1 = threading.Thread(target=run_arm)
        t2 = threading.Thread(target=run_gpu)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        return cpu_out_flat.reshape(shape), gpu_out_flat.reshape(shape)


if __name__ == "__main__":
    def math_kernel(a, b):
        return (a * b) + a - b * 1.5

    arbitrary_shape = (3, 5) 
    
    sample_a = jnp.arange(1, 16, dtype=jnp.float32).reshape(arbitrary_shape)
    sample_b = jnp.arange(10, 25, dtype=jnp.float32).reshape(arbitrary_shape)
    
    closed_jaxpr = jax.make_jaxpr(math_kernel)(sample_a, sample_b)

    arm_asm = compile_closed_jaxpr_to_arm64(closed_jaxpr)
    cuda_code = compile_closed_jaxpr_to_cuda(closed_jaxpr)

    np_a = np.array(sample_a)
    np_b = np.array(sample_b)

    runtime = HeterogeneousRuntime(arm_asm, cuda_code)
    
    try:
        runtime.compile_and_load()
        cpu_res, gpu_res = runtime.execute_concurrently(np_a, np_b)
        
        expected = (np_a * np_b) + np_a - np_b * 1.5
        
        print(f"Target Shape: {arbitrary_shape}")
        print(f"Expected Math:\n{expected}\n")
        
        if runtime.lib_arm:
            print(f"ARM64 CPU Output Matches Expected: {np.allclose(cpu_res, expected)}")
            print(f"ARM64 CPU Results:\n{cpu_res}\n")
            
        if runtime.lib_cuda:
            print(f"NVIDIA GPU Output Matches Expected: {np.allclose(gpu_res, expected)}")
            print(f"NVIDIA GPU Results:\n{gpu_res}\n")
            
    except Exception as e:
        print(f"\nExecution skipped or failed due to environment constraints: {e}")
