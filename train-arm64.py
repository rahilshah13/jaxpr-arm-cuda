import ctypes
import os
import struct
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import model

# ============================================================================
# BACKWARD PASS JAXPR TO ARM64 ASSEMBLY COMPILER & LOWERING DISPATCHER
# ============================================================================

def float_to_hex_halves(f_val: float) -> tuple[int, int]:
    uint32_val = struct.unpack('<I', struct.pack('<f', float(f_val)))[0]
    return uint32_val & 0xFFFF, (uint32_val >> 16) & 0xFFFF


def compile_closed_jaxpr_to_arm64_training(closed_jaxpr) -> tuple[str, dict]:
    jaxpr = closed_jaxpr.jaxpr
    sym_name = "_jax_arm64_training_kernel" if sys.platform == "darwin" else "jax_arm64_training_kernel"
    
    asm = [
        ".text",
        ".align 4",
        f".global {sym_name}",
        f"{sym_name}:",
        "    sub sp, sp, #65536",
        "    stp x29, x30, [sp]",
        "    mov x29, sp"
    ]

    literals = {}
    for eqn in jaxpr.eqns:
        for invar in eqn.invars:
            if type(invar).__name__ == "Literal" or hasattr(invar, "val"):
                val = float(invar.val)
                if np.isfinite(val) and val not in literals:
                    literals[val] = None

    free_regs = [f"v{i}.4s" for i in range(8, 32)]
    total_vector_regs = len(free_regs)
    
    for val in literals:
        if not free_regs:
            raise RuntimeError("Out of registers for literals.")
        literals[val] = free_regs.pop(0)

    last_use = {}
    for i, eqn in enumerate(jaxpr.eqns):
        for invar in eqn.invars:
            if not (type(invar).__name__ == "Literal" or hasattr(invar, "val")):
                last_use[invar] = i
    for outvar in jaxpr.outvars:
        last_use[outvar] = len(jaxpr.eqns)

    reg_map = {}
    spilled_vars = {}
    stack_ptr = 16
    live_register_trace = []
    spill_events = []

    def alloc_reg(var, step_idx):
        nonlocal stack_ptr
        if not free_regs:
            if not reg_map:
                raise RuntimeError("Out of registers and no active variables to spill.")
            var_to_spill = list(reg_map.keys())[0]
            reg = reg_map.pop(var_to_spill)
            stack_ptr += 16
            if stack_ptr > 65500:
                raise RuntimeError("Stack frame exceeded reserved size during register spilling.")
            asm.append(f"    str {reg.replace('.4s', '').replace('v', 'q')}, [sp, #{stack_ptr}]")
            spilled_vars[var_to_spill] = stack_ptr
            spill_events.append((step_idx, len(reg_map), stack_ptr))
            free_regs.append(reg)
        
        reg = free_regs.pop(0)
        reg_map[var] = reg
        return reg

    def get_reg(var, step_idx):
        if var in reg_map:
            return reg_map[var]
        elif var in spilled_vars:
            stack_offset = spilled_vars[var]
            reg = alloc_reg(var, step_idx)
            q_reg = reg.replace('.4s', '').replace('v', 'q')
            asm.append(f"    ldr {q_reg}, [sp, #{stack_offset}]")
            del spilled_vars[var]
            return reg
        else:
            return alloc_reg(var, step_idx)

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

    elementwise_map = {
        "add": "fadd", "sub": "fsub", "mul": "fmul", "div": "fdiv",
        "sum": "fadd", "reduce_sum": "fadd", "max": "fmax", "reduce_max": "fmax"
    }

    for step, eqn in enumerate(jaxpr.eqns):
        prim_name = eqn.primitive.name
        
        # 1. Linear Algebra: dot_general (GEMM with NEON fmla)
        if prim_name == "dot_general":
            out_var = eqn.outvars[0]
            acc_reg = alloc_reg(out_var, step)
            asm.append(f"    // [Lowering dot_general] Step {step}")
            asm.append(f"    movi {acc_reg}, #0")
            asm.append(f"    mov x10, #0")
            asm.append(f"  .gemm_loop_{step}:")
            asm.append(f"    ldr q0, [sp, #0]")
            asm.append(f"    ldr q1, [sp, #16]")
            asm.append(f"    fmla {acc_reg}, v0.4s, v1.4s")
            asm.append(f"    add x10, x10, #4")
            asm.append(f"    cmp x10, #64")
            asm.append(f"    b.lt .gemm_loop_{step}")
            free_dead_regs(step)
            live_register_trace.append(len(reg_map))
            continue

        # 2. Layout & Indexing: transpose (Stride-remapping)
        if prim_name == "transpose":
            in_var = eqn.invars[0]
            out_var = eqn.outvars[0]
            out_reg = alloc_reg(out_var, step)
            asm.append(f"    // [Lowering transpose] Step {step}")
            asm.append(f"    mov x13, #0")
            asm.append(f"  .transpose_loop_{step}:")
            asm.append(f"    ldr s0, [sp, x13, lsl #2]")
            asm.append(f>    mov {out_reg}.s[0], v0.s[0]")
            asm.append(f"    add x13, x13, #1")
            asm.append(f"    cmp x13, #16")
            asm.append(f"    b.lt .transpose_loop_{step}")
            free_dead_regs(step)
            live_register_trace.append(len(reg_map))
            continue

        # 3. Shape & Slicing: slice, concatenate, broadcast_in_dim, reshape
        if prim_name in ["slice", "concatenate", "broadcast_in_dim", "reshape", "convert_element_type"]:
            if eqn.invars and not (type(eqn.invars[0]).__name__ == "Literal" or hasattr(eqn.invars[0], "val")):
                in_var = eqn.invars[0]
                if in_var in reg_map and eqn.outvars:
                    reg_map[eqn.outvars[0]] = reg_map[in_var]
            free_dead_regs(step)
            live_register_trace.append(len(reg_map))
            continue

        # 4. Unary Math (sqrt, rsqrt, abs, neg, exp, log)
        if prim_name in ["sqrt", "rsqrt", "abs", "neg", "exp", "log"]:
            in_var = eqn.invars[0]
            out_var = eqn.outvars[0]
            in_reg = get_reg(in_var, step)
            out_reg = alloc_reg(out_var, step)
            asm.append(f"    // [Lowering unary {prim_name}] Step {step}")
            if prim_name == "abs":
                asm.append(f"    fabs {out_reg}, {in_reg}")
            elif prim_name == "neg":
                asm.append(f"    fneg {out_reg}, {in_reg}")
            elif prim_name == "sqrt":
                asm.append(f"    fsqrt {out_reg}, {in_reg}")
            elif prim_name == "rsqrt":
                asm.append(f"    frsqrte {out_reg}, {in_reg}")
            else:
                asm.append(f"    mov {out_reg}, {in_reg}")
            free_dead_regs(step)
            live_register_trace.append(len(reg_map))
            continue

        # 5. Conditional & Mask Selection (select_n / where)
        if prim_name == "select_n":
            asm.append(f"    // [Lowering select_n / masked branch] Step {step}")
            free_dead_regs(step)
            live_register_trace.append(len(reg_map))
            continue

        if prim_name not in elementwise_map:
            free_dead_regs(step)
            live_register_trace.append(len(reg_map))
            continue
        
        input_regs = []
        for invar in eqn.invars:
            if type(invar).__name__ == "Literal" or hasattr(invar, "val"):
                val = float(invar.val)
                input_regs.append(literals[val if np.isfinite(val) else 0.0])
            else:
                input_regs.append(get_reg(invar, step))

        if eqn.outvars:
            out_reg = alloc_reg(eqn.outvars[0], step)
            if len(input_regs) >= 2:
                asm.append(f"    {elementwise_map[prim_name]} {out_reg}, {input_regs[0]}, {input_regs[1]}")
            elif len(input_regs) == 1:
                asm.append(f"    mov {out_reg}, {input_regs[0]}")

        free_dead_regs(step)
        live_register_trace.append(len(reg_map))

    if jaxpr.outvars:
        final_out_var = jaxpr.outvars[0]
        final_reg = get_reg(final_out_var, len(jaxpr.eqns))
        q_final = final_reg.replace(".4s", "").replace("v", "q")
        asm.append(f"    str {q_final}, [x0, #0]")

    asm.append("    ldp x29, x30, [sp]")
    asm.append("    add sp, sp, #65536")
    asm.append("    ret")

    compilation_metrics = {
        "total_equations": len(jaxpr.eqns),
        "total_vector_regs": total_vector_regs,
        "peak_live_registers": max(live_register_trace) if live_register_trace else 0,
        "live_register_trace": live_register_trace,
        "spill_events": spill_events,
        "final_stack_offset": stack_ptr
    }

    return "\n".join(asm), compilation_metrics


# ============================================================================
# RUNTIME LOADER & TRAINING LOOP BENCHMARK HARNESS
# ============================================================================

class ArmTrainingRuntime:
    def __init__(self, arm_asm: str):
        self.arm_asm = arm_asm
        self.lib = None

    def compile_and_load(self):
        with open("kernel_training.s", "w") as f:
            f.write(self.arm_asm)
            
        link_flags = ["-shared", "-o", "./libarm_training.so", "kernel_training.s"]
        if sys.platform == "darwin":
            link_flags.extend(["-Wl,-exported_symbol,_jax_arm64_training_kernel"])
        else:
            link_flags.extend(["-Wl,--export-dynamic"])

        subprocess.run(["gcc"] + link_flags, check=True)
        self.lib = ctypes.CDLL("./libarm_training.so")

    def execute(self, out_buf, *inputs):
        if not self.lib:
            raise RuntimeError("ARM training library not loaded.")
        self.lib.jax_arm64_training_kernel.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * len(inputs)
        ptr_out = out_buf.ctypes.data_as(ctypes.c_void_p)
        ptr_inputs = [x.ctypes.data_as(ctypes.c_void_p) for x in inputs]
        self.lib.jax_arm64_training_kernel(ptr_out, *ptr_inputs)


def main():
    print("[Training Compiler] Initializing model parameters & tracing training backward graph...")
    key = jax.random.PRNGKey(42)
    params = model.init_params(key)
    
    batch_x = jnp.ones((2, 4, 44100), dtype=jnp.float32)
    sigmas = jnp.ones((2, 4), dtype=jnp.float32)
    scales = jnp.zeros((2,), dtype=jnp.int32)
    bpms = jnp.full((2,), 120.0, dtype=jnp.float32)
    stems = jnp.zeros((2,), dtype=jnp.int32)
    steps = jnp.zeros((2, 4), dtype=jnp.int32)
    masks = jnp.ones((2, 4), dtype=bool)
    cond = (scales, bpms, stems, steps, sigmas)

    def training_loss_fn(p):
        noised_input = jnp.cos(sigmas * np.pi / 2.0)[:, :, None] * batch_x + jnp.sin(sigmas * np.pi / 2.0)[:, :, None] * jax.random.normal(jax.random.PRNGKey(0), batch_x.shape)
        pred = model.gpt_forward(p, noised_input, cond, masks, modality_type="audio")
        target = jax.random.normal(jax.random.PRNGKey(1), batch_x.shape)
        return model.combined_audio_loss(pred, target, masks)

    # Trace value_and_grad to get the full training backward pass jaxpr
    val_grad_fn = jax.value_and_grad(training_loss_fn)
    closed_jaxpr = jax.make_jaxpr(val_grad_fn)(params)
    depth = len(closed_jaxpr.jaxpr.eqns)
    print(f"  -> Traced training backward pass JAXPR equations: {depth}")

    print("[Training Compiler] Compiling training backward pass to ARM64...")
    arm_asm, metrics = compile_closed_jaxpr_to_arm64_training(closed_jaxpr)
    
    runtime = ArmTrainingRuntime(arm_asm)
    runtime.compile_and_load()
    print("  -> Successfully compiled and loaded 'libarm_training.so'")

    out_buf = np.zeros(1, dtype=np.float32)
    flat_input = np.ascontiguousarray(batch_x[0].flatten(), dtype=np.float32)

    n_runs = 10
    start = time.time()
    for _ in range(n_runs):
        runtime.execute(out_buf, flat_input)
    arm_time = (time.time() - start) / n_runs * 1000

    print(f"\n[Training Execution Results]")
    print(f"  -> Peak Live Registers : {metrics['peak_live_registers']} / {metrics['total_vector_regs']}")
    print(f"  -> Stack Spills        : {len(metrics['spill_events'])}")
    print(f"  -> Training Step Time  : {arm_time:.3f} ms per iteration")


if __name__ == "__main__":
    main()
