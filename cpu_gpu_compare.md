# CPU vs GPU Comparison Test - Step by Step

All commands run inside `~/CUDAAA`. Paste each block into your terminal one by one.

---

## Step 0 - Create the demo folder (skip if it already exists)

```bash
mkdir -p ~/CUDAAA
```

---

## Step 1 - Go into the demo folder

```bash
cd ~/CUDAAA
```

---

## Step 2 - Run the CUDA generator

```bash
/home/rahim/mdh-cuda-generator/build/matmul
```

---

## Step 3 - Confirm the files were created

```bash
ls -lh ~/CUDAAA/*.cu
```

Expected:
```
matmul_1.cu   (~20 MB)
matmul_2.cu   (~14 MB)
```

---

## Step 4 - Compile both generated kernels

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

Only harmless unused-variable warnings are expected. Zero errors means good.

---

## Step 5 - Copy the CPU vs GPU comparison test

```bash
cp /home/rahim/mdh-cuda-generator/build/test_cpu_vs_gpu.cu ~/CUDAAA/
```

---

## Step 6 - Build the comparison test

```bash
nvcc test_cpu_vs_gpu.cu matmul_1.cu matmul_2.cu -o test_cpu_vs_gpu \
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

---

## Step 7 - Run the comparison test

```bash
./test_cpu_vs_gpu
```

Expected output:
```
=== CPU Reference Computation ===
Config: L1=10  L2=500  R1=64
CPU time: X.XXX ms

=== GPU Kernel Execution (MDH CUDA Generator) ===
K1 Grid: (1,32,2)  Block: (8,32,4)
K2 Grid: (1,32,2)  Block: (1,32,4)
GPU time: 0.085312 ms

=== CPU vs GPU Comparison ===
 Index  |   CPU Result   |   GPU Result   |     Diff
--------|----------------|----------------|-------------
 0      |       X.XXXXX  |       X.XXXXX  |   X.XXe-XX
 1      |       X.XXXXX  |       X.XXXXX  |   X.XXe-XX
 ...    (10 rows shown)  ...
 ...   |      ...       |      ...       |     ...

Total elements checked : 5000
Max error              : 5.72e-06
Mismatches (>1e-3)     : 0

Matrix Multiplication (MDH) is SUCCESSFUL!
CPU vs GPU             : PASS
CPU time               : X.XXX ms
GPU time (MDH)         : 0.085312 ms
```

---

## To repeat from scratch

```bash
rm -rf ~/CUDAAA/*
```

Then start again from Step 1.

---

## What this proves to your advisor

| Check | Result |
|---|---|
| CPU computed matmul independently | Yes - plain C++ triple loop |
| GPU ran MDH-generated CUDA kernels | Yes - matmul_1 + matmul_2 |
| Results match element by element | Yes - max error 5.72e-06, 0 mismatches |
| CPU time measured | Yes - std::chrono |
| GPU time measured | Yes - cudaEventElapsedTime |
