# Proof That the Generated .cu Files Are Real CUDA (Not OpenCL)

There are 4 layers of proof, from syntax to runtime.

---

## Proof 1 — CUDA keywords are present, OpenCL keywords are absent

Run these yourself:

```bash
# These MUST exist (CUDA-only syntax)
grep -m1 "__global__ void" ~/CUDAAA/matmul_1.cu
grep -m1 "__shared__"      ~/CUDAAA/matmul_1.cu
grep -m1 "__syncthreads"   ~/CUDAAA/matmul_1.cu
grep -m1 "__device__"      ~/CUDAAA/matmul_1.cu
grep -m1 "blockIdx\|threadIdx\|blockDim" ~/CUDAAA/matmul_1.cu
```

```bash
# These must return 0 (OpenCL-only syntax — none should exist)
grep -c "__kernel\|get_global_id\|CLK_LOCAL_MEM_FENCE\|__local\b" ~/CUDAAA/matmul_1.cu
```

---

## Proof 2 — Side-by-side comparison of the two languages

| Construct | OpenCL (.cl) | CUDA (.cu) | What is in your file |
|---|---|---|---|
| Kernel marker | `__kernel void` | `__global__ void` | `__global__ void` ✓ |
| Shared memory | `__local float x` | `__shared__ float x` | `__shared__` ✓ |
| Thread barrier | `barrier(CLK_LOCAL_MEM_FENCE)` | `__syncthreads()` | `__syncthreads()` ✓ |
| Thread X ID | `get_global_id(0)` | `blockIdx.x * blockDim.x + threadIdx.x` | `blockIdx.x * ...` ✓ |
| Device function | *(no qualifier needed)* | `__device__ inline` | `__device__ inline` ✓ |
| Restrict keyword | `restrict` (no underscores) | `__restrict__` (double underscores) | `__restrict__` ✓ |

---

## Proof 3 — nvcc accepted it, an OpenCL compiler would reject it

`nvcc` is NVIDIA's CUDA-only compiler. It does not understand `__kernel`, `get_global_id()`, or `barrier(CLK_LOCAL_MEM_FENCE)`. The fact that:

```bash
nvcc -c matmul_1.cu -o matmul_1.o -DTYPE_T=float -DTYPE_TS=float \
  -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
  -DG_CB_RES_DEST_LEVEL=2 -DL_CB_RES_DEST_LEVEL=1 -DP_CB_RES_DEST_LEVEL=0 \
  -DG_CB_SIZE_L_1=10 -DG_CB_SIZE_L_2=500 -DG_CB_SIZE_R_1=64 \
  -DL_CB_SIZE_L_1=8  -DL_CB_SIZE_L_2=16  -DL_CB_SIZE_R_1=64 \
  -DP_CB_SIZE_L_1=1  -DP_CB_SIZE_L_2=1   -DP_CB_SIZE_R_1=1  \
  -DNUM_WG_L_1=2 -DNUM_WG_L_2=32 -DNUM_WG_R_1=1 \
  -DNUM_WI_L_1=4 -DNUM_WI_L_2=32 -DNUM_WI_R_1=8 \
  -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_R_1=0
```

...compiled with **zero errors** means the file is valid CUDA.

If you fed actual OpenCL code to nvcc it would throw errors on every kernel line (`__kernel` unknown, `get_global_id` undefined, etc.).

Conversely, an OpenCL compiler (like Intel's `ioc64` or AMD's `clcc`) would reject `__global__ void` and `__syncthreads()` because those are CUDA-only identifiers.

---

## Proof 4 — It ran on a real GPU and gave correct numbers

This is the ultimate proof. The runtime test (`test_matmul.cu`) did the following:

1. Allocated arrays on an NVIDIA GPU via `cudaMalloc`
2. Uploaded input data via `cudaMemcpy`
3. Launched kernel 1 via `matmul_1<<<GRID1, BLOCK1>>>(...)` — CUDA launch syntax
4. Launched kernel 2 via `matmul_2<<<GRID2, BLOCK2>>>(...)` — CUDA launch syntax
5. Downloaded the result via `cudaMemcpy`
6. Compared against a CPU reference matmul

**Result:**
```
Config: L1=10  L2=500  R1=64
K1 Grid: (1,32,2)  Block: (8,32,4)
K2 Grid: (1,32,2)  Block: (1,32,4)

--- kernel 1 ---
int_res: Elements=5000  Max error=5.72e-06  Mismatches=0  -> PASS
--- kernel 2 ---
--- final result (S after kernel 2) ---
Elements checked: 5000  Max error: 5.72e-06  Mismatches: 0
PASS
```

The max error of 5.72e-06 is normal IEEE-754 float32 rounding from parallel reduction — not a correctness error.

OpenCL code cannot be launched via CUDA's runtime API at all. It would crash immediately, not produce correct floating-point results matching a CPU reference.

---

## Summary

The file extension `.cu` is just a convention. The real proof is:

| Check | Result |
|---|---|
| CUDA keywords present (`__global__`, `__shared__`, `__syncthreads__`, `blockIdx`) | Yes |
| OpenCL keywords absent (`__kernel`, `get_global_id`, `CLK_LOCAL_MEM_FENCE`, `__local`) | Yes (count = 0) |
| Compiled by nvcc with zero errors | Yes |
| Ran on NVIDIA GPU via CUDA runtime API | Yes |
| Produced numerically correct output vs CPU reference | Yes (0 mismatches) |
