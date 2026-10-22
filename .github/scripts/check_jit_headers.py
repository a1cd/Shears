#!/usr/bin/env python3 -I
"""Check that a CUDA tarball carries the headers MLX's run-time (NVRTC) kernel compiler needs, the way a Colab GPU image would see them.

MLX compiles some kernels when they are first used (jit_module.cpp): the sources of mlx/backend/cuda/device/*.cuh are embedded in the
binary and compiled with NVRTC against --include-path=<bundle>/include, <bundle>/include/cccl (else a path baked in at build time) and
the CUDA toolkit's include directory. Those sources include CUTLASS/CuTe and CCCL headers. If the bundle lacks them, the first GEMM
fails on the user's machine with 'cannot open source file "cute/tensor.hpp"' (Colab T4, 2026-10-22).

  1. static: every <cute/..>, <cutlass/..>, <cuda/..>, <nv/..> include of the embedded sources exists in the bundle;
  2. dynamic, without a GPU: compile the kernels with NVRTC (it only needs the library) for sm_75 and sm_80, the way MLX calls it,
     against a toolkit-like include directory that has NO CUTLASS and the stock CCCL;
     (a) with the bundle's headers: must compile; (b) without them: must fail with 'cannot open source file' (the check can fail).

usage: check_jit_headers.py <bundle root> <mlx cuda device dir> <toolkit include dir> <stock cccl dir> <libnvrtc.so.13>
"""
import ctypes, os, re, sys, tempfile, pathlib

bundle, device_dir, toolkit_inc, stock_cccl, nvrtc_path = map(pathlib.Path, sys.argv[1:6])

# The headers MLX embeds (g_include_names in jit_module.cpp).
NAMES = ["atomic_ops.cuh", "binary_ops.cuh", "cast_op.cuh", "config.h", "complex.cuh", "cute_dequant.cuh", "fp16_math.cuh",
         "gemm_sm70.cuh", "hadamard.cuh", "indexing.cuh", "qmm_naive.cuh", "qmm_sm80.cuh", "qmm_sm90.cuh", "scatter_ops.cuh",
         "unary_ops.cuh", "ternary_ops.cuh", "utils.cuh"]
PREFIX = "mlx/backend/cuda/device/"
sources = {n: (device_dir / n).read_text() for n in NAMES}

# ---- 1. static ----------------------------------------------------------------------------------------------------------
wanted = sorted({m for s in sources.values() for m in re.findall(r"#include <([^>]+)>", s)})
bad = []
for inc in wanted:
    top = inc.split("/")[0]
    if top in ("cute", "cutlass"):
        where = bundle / "include" / inc
    elif top in ("cuda", "nv"):
        where = bundle / "include" / "cccl" / inc
    else:
        where = toolkit_inc / inc            # cuda_fp16.h, cuda_bf16.h, cuda_fp8.h, math_constants.h: the toolkit's
    if not where.exists():
        bad.append((inc, str(where)))
print(f"static: {len(wanted)} angle-bracket includes in the embedded sources")
for inc, where in bad:
    print(f"  MISSING {inc}: looked for {where}")
if bad:
    sys.exit("static check failed")
print("static: all present")

# ---- 2. NVRTC -----------------------------------------------------------------------------------------------------------
nvrtc = ctypes.CDLL(str(nvrtc_path))
vmaj, vmin = ctypes.c_int(), ctypes.c_int()
nvrtc.nvrtcVersion(ctypes.byref(vmaj), ctypes.byref(vmin))
print(f"nvrtc {vmaj.value}.{vmin.value}")

def compile_kernel(module, expr, arch, include_args):
    prog = ctypes.c_void_p()
    n = len(NAMES)
    srcs = (ctypes.c_char_p * n)(*[sources[x].encode() for x in NAMES])
    names = (ctypes.c_char_p * n)(*[(PREFIX + x).encode() for x in NAMES])
    main = sources["gemm_sm70.cuh"].encode()
    assert nvrtc.nvrtcCreateProgram(ctypes.byref(prog), main, (module + ".cu").encode(), n, srcs, names) == 0
    nvrtc.nvrtcAddNameExpression(prog, expr.encode())
    opts = ["--device-as-default-execution-space", f"--gpu-architecture={arch}"] + include_args
    arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    rc = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
    size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value + 1)
    nvrtc.nvrtcGetProgramLog(prog, log)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return rc == 0, log.value.decode(errors="replace")

# What MLX passes (include_path_args): <bundle>/include, then <bundle>/include/cccl (else the baked-in cccl dir), then the toolkit's.
def mlx_include_args(root):
    args = []
    inc = root / "include"
    if inc.exists():
        args.append(f"--include-path={inc}")
    cccl = inc / "cccl"
    args.append(f"--include-path={cccl if cccl.exists() else stock_cccl}")
    args.append(f"--include-path={toolkit_inc}")
    return args

# the two kernels: the one Colab failed on (stage C: float32, a and b not transposed, aligned, tile 16x128x64, sm_75), and a bf16 sm_80 one
CASES = [
    ("gemm_sm70_tt_float32_aligned_m16_n128_k64", "sm_75",
     "mlx::core::cu::gemm_sm70_kernel<true, false, true, false, float, cute::Shape<cute::Int<16>, cute::Int<128>, cute::Int<64>>>"),
    ("gemm_sm70_nn_bfloat16_unaligned_m64_n128_k64", "sm_80",
     "mlx::core::cu::gemm_sm70_kernel<false, true, false, true, cutlass::bfloat16_t, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>>"),
]

empty = pathlib.Path(tempfile.mkdtemp())          # a bundle without any headers: what the first tarball was
failed = False
for module, arch, expr in CASES:
    ok, log = compile_kernel(module, expr, arch, mlx_include_args(bundle))
    print(f"[with the bundle's headers] {module} for {arch}: {'compiled' if ok else 'FAILED'}")
    if not ok:
        print(log[-3000:]); failed = True
    ok2, log2 = compile_kernel(module, expr, arch, mlx_include_args(empty))
    print(f"[bundle without headers]    {module} for {arch}: {'compiled (the check cannot fail!)' if ok2 else 'failed as it must'}")
    if ok2 or "cannot open source file" not in log2:
        print(log2[-1500:]); failed = True
    else:
        print("    ->", [l for l in log2.splitlines() if "cannot open source file" in l][0])
if failed:
    sys.exit("NVRTC check failed")
print("NVRTC: the kernels compile with the bundle's headers and not without them")
