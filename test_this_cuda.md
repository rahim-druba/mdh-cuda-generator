# How to Generate and Test matmul CUDA Kernels

Copy and paste each command block into your terminal one by one.

---

## Step 1 - Go into the demo folder

```bash
cd ~/CUDAAA
```

---

## Step 2 - Run the CUDA generator

The generator binary is already compiled. Running it from inside `CUDAAA` writes the output files here.

```bash
/home/rahim/mdh-cuda-generator/build/matmul
```

---

## Step 3 - Confirm the files were created

```bash
ls -lh ~/CUDAAA/*.cu
```

You should see two files:
```
matmul_1.cu   (~20 MB)   - kernel 1: main computation
matmul_2.cu   (~14 MB)   - kernel 2: final reduction
```

---

## Step 4 - Peek at the kernel signatures (optional sanity check)

```bash
grep "__global__ void matmul_1" ~/CUDAAA/matmul_1.cu
grep "__global__ void matmul_2" ~/CUDAAA/matmul_2.cu
```

Expected output:
```
__global__ void matmul_1(TYPE_T const * const __restrict__ Z, ...
__global__ void matmul_2(TYPE_T const * const __restrict__ int_res, ...
```

---

## Step 5 - Confirm no OpenCL leftovers (optional)

```bash
grep -c "__kernel\|get_global_id\|CLK_LOCAL_MEM_FENCE\|__local\b" ~/CUDAAA/matmul_1.cu
grep -c "__kernel\|get_global_id\|CLK_LOCAL_MEM_FENCE\|__local\b" ~/CUDAAA/matmul_2.cu
```

Both should print `0`.

---

## Step 6 - Compile both kernels with nvcc (requires CUDA toolkit)

```bash
nvcc -c matmul_1.cu -o matmul_1.o \
  -DTYPE_T=float -DTYPE_TS=float \
  -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
  -DG_CB_RES_DEST_LEVEL=2 -DL_CB_RES_DEST_LEVEL=1 -DP_CB_RES_DEST_LEVEL=0 \
  -DG_CB_SIZE_L_1=10 -DG_CB_SIZE_L_2=500 -DG_CB_SIZE_R_1=64 \
  -DL_CB_SIZE_L_1=8  -DL_CB_SIZE_L_2=16  -DL_CB_SIZE_R_1=64 \
  -DP_CB_SIZE_L_1=1  -DP_CB_SIZE_L_2=1   -DP_CB_SIZE_R_1=1  \
  -DNUM_WG_L_1=2 -DNUM_WG_L_2=32 -DNUM_WG_R_1=1 \
  -DNUM_WI_L_1=4 -DNUM_WI_L_2=32 -DNUM_WI_R_1=8 \
  -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_R_1=0
```

```bash
nvcc -c matmul_2.cu -o matmul_2.o \
  -DTYPE_T=float -DTYPE_TS=float \
  -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
  -DG_CB_RES_DEST_LEVEL=2 -DL_CB_RES_DEST_LEVEL=1 -DP_CB_RES_DEST_LEVEL=0 \
  -DG_CB_SIZE_L_1=10 -DG_CB_SIZE_L_2=500 -DG_CB_SIZE_R_1=64 \
  -DL_CB_SIZE_L_1=8  -DL_CB_SIZE_L_2=16  -DL_CB_SIZE_R_1=64 \
  -DP_CB_SIZE_L_1=1  -DP_CB_SIZE_L_2=1   -DP_CB_SIZE_R_1=1  \
  -DNUM_WG_L_1=2 -DNUM_WG_L_2=32 -DNUM_WG_R_1=1 \
  -DNUM_WI_L_1=4 -DNUM_WI_L_2=32 -DNUM_WI_R_1=8 \
  -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_R_1=0
```

You should see only harmless warnings like:
```
warning #177-D: variable "i_l_cb_l_1" was declared but never referenced
```
No errors means the kernels compile correctly.

---

## Step 7 - Run the full correctness and timing test (requires an NVIDIA GPU)

Copy the test harness into CUDAAA, then build and run:

```bash
cp /home/rahim/mdh-cuda-generator/build/test_matmul.cu ~/CUDAAA/
```

```bash
nvcc test_matmul.cu matmul_1.cu matmul_2.cu -o test_matmul \
  -DTYPE_T=float -DTYPE_TS=float \
  -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
  -DG_CB_RES_DEST_LEVEL=2 -DL_CB_RES_DEST_LEVEL=1 -DP_CB_RES_DEST_LEVEL=0 \
  -DG_CB_SIZE_L_1=10 -DG_CB_SIZE_L_2=500 -DG_CB_SIZE_R_1=64 \
  -DL_CB_SIZE_L_1=8  -DL_CB_SIZE_L_2=16  -DL_CB_SIZE_R_1=64 \
  -DP_CB_SIZE_L_1=1  -DP_CB_SIZE_L_2=1   -DP_CB_SIZE_R_1=1  \
  -DNUM_WG_L_1=2 -DNUM_WG_L_2=32 -DNUM_WG_R_1=1 \
  -DNUM_WI_L_1=4 -DNUM_WI_L_2=32 -DNUM_WI_R_1=8 \
  -DOCL_DIM_L_1=2 -DOCL_DIM_L_2=1 -DOCL_DIM_R_1=0
```

```bash
./test_matmul
```

Expected output:
```
Config: L1=10  L2=500  R1=64
K1 Grid: (1,32,2)  Block: (8,32,4)
K2 Grid: (1,32,2)  Block: (1,32,4)

--- kernel 1 ---
int_res: Elements=5000  Max error=5.72e-06  Mismatches=0  -> PASS
--- kernel 2 ---
--- final result (S after kernel 2) ---
Elements checked: 5000  Max error: 5.72e-06  Mismatches: 0
Matrix Multiplication (MDH) is SUCCESSFUL!
GPU time measurement (MDH): 0.085312 ms
```

---

## To repeat the demo from scratch

Delete everything inside CUDAAA and start again from Step 2:

```bash
rm -rf ~/CUDAAA/*
```

---

## Quick reference - what each file does

| File | What it is |
|---|---|
| `matmul_1.cu` | Generated CUDA kernel 1 - loads Z and W tiles into shared memory, computes partial dot products |
| `matmul_2.cu` | Generated CUDA kernel 2 - reduces the partial results from kernel 1 into the final output matrix S |
| `test_matmul.cu` | Test harness - launches both kernels, checks correctness, and measures GPU execution time |

## Where the generator lives

```
/home/rahim/mdh-cuda-generator/build/matmul               - the generator binary
/home/rahim/mdh-cuda-generator/include/cuda_generator.hpp - the CUDA backend source
/home/rahim/mdh-cuda-generator/src/matmul/matmul.cpp      - the MDH spec for matmul
```

If you ever change the MDH spec or the generator and want to regenerate, rebuild first:

```bash
cd /home/rahim/mdh-cuda-generator/build
make
```

Then start again from Step 2 inside `~/CUDAAA`.
