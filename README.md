# mdh-cuda-generator

A CUDA kernel code generator for the **Multi-Dimensional Homomorphism (MDH)** framework.  
Given a high-level C++ MDH specification, this project generates optimised `.cu` GPU kernel files ready to compile with `nvcc` - no manual CUDA kernel writing required.

---

## What is MDH?

Multi-Dimensional Homomorphism is a mathematical abstraction for expressing a wide class of parallel computations (reductions, scans, stencils, matrix operations) in a hardware-agnostic way. The MDH framework, introduced at PACT 2019, originally generates OpenCL kernels from these specs. This repository adds a **CUDA backend** to that framework.

> See [Acknowledgements](#acknowledgements) for the original MDH framework and paper.

---

## What this repository adds

| | Original MDH (PACT 2019) | This repository |
|---|---|---|
| Target language | OpenCL (`.cl`) | **CUDA (`.cu`)** |
| Thread ID model | `get_global_id(N)` | `blockIdx.x * blockDim.x + threadIdx.x` |
| Shared memory | `__local` | `__shared__` |
| Barriers | `barrier(CLK_LOCAL_MEM_FENCE)` | `__syncthreads()` |
| Device functions | plain `inline` | `__device__ inline` |
| Kernel qualifier | `__kernel void` | `__global__ void` |

The CUDA backend is a drop-in addition - the original OpenCL generator is untouched and still works alongside it.

---

## Prerequisites

| Requirement | Version tested |
|---|---|
| C++ compiler | GCC with C++14 support |
| CMake | ≥ 2.8.11 |
| CUDA Toolkit (for compiling generated kernels) | 11.7 |
| NVIDIA GPU (for running generated kernels) | Any CUDA-capable card |

---

## Repository structure

```
mdh-cuda-generator/
│
├── include/
│   ├── cuda_generator.hpp                  ← CUDA backend (main new file)
│   ├── cuda_input_buffer_wrapper.hpp       ← CUDA input buffer code emitter
│   ├── cuda_result_buffer_wrapper.hpp      ← CUDA result buffer code emitter
│   ├── cuda_input_stencil_buffer_wrapper.hpp
│   │
│   ├── md_hom.hpp                          ← MDH specification interface
│   ├── types.hpp                           ← MDH type system
│   ├── macros.hpp                          ← tiling macro emitter
│   ├── loop_generator.hpp                  ← loop structure generator
│   ├── input_buffer_wrapper.hpp            ← OpenCL input buffer emitter
│   ├── result_buffer_wrapper.hpp           ← OpenCL result buffer emitter
│   ├── ocl_generator.hpp                   ← original OpenCL backend
│   └── ...                                 ← remaining MDH framework headers
│
├── src/
│   ├── matmul/
│   │   └── matmul.cpp                      ← matrix multiply MDH spec (example)
│   ├── helper.cpp
│   ├── input_buffer.cpp
│   ├── input_scalar.cpp
│   ├── md_hom.cpp
│   ├── result_buffer.cpp
│   └── scalar_function.cpp
│
├── build/
│   ├── matmul_1.cu                         ← generated CUDA kernel 1 (example output)
│   ├── matmul_2.cu                         ← generated CUDA kernel 2 (example output)
│   └── test_matmul.cu                      ← runtime correctness test
│
├── md_hom_generator.hpp                    ← top-level include (OCL + CUDA)
├── CMakeLists.txt
│
├── CUDA_steps.md                           ← full implementation log
├── CUDA_proof.md                           ← proof the output is genuine CUDA
├── test_this_cuda.md                       ← step-by-step usage guide
└── implementation_idea.md                  ← original design plan
```

---

## Build

```bash
git clone https://github.com/rahim-druba/mdh-cuda-generator.git
cd mdh-cuda-generator
mkdir build && cd build
cmake ..
make
```

The `matmul` binary is the generator. Running it produces the CUDA kernel files.

---

## Usage — generating CUDA kernels

```bash
cd build
./matmul
```

This reads the MDH spec in `src/matmul/matmul.cpp` and writes two files to the current directory:

```
matmul_1.cu    ← kernel 1: loads input tiles, computes partial results
matmul_2.cu    ← kernel 2: reduces partial results into the final output
```

### Writing your own spec

Create a new file under `src/` modelled on `src/matmul/matmul.cpp`:

```cpp
#include "md_hom_generator.hpp"

int main() {
    // Define your MDH computation here
    auto my_hom = md_hom::md_hom_class<...>(...);

    // Generate CUDA kernels
    auto generator = md_hom::generator::cuda_generator(my_hom);

    std::fstream f1("my_kernel_1.cu", std::fstream::out | std::fstream::trunc);
    f1 << generator.kernel_1_source();

    std::fstream f2("my_kernel_2.cu", std::fstream::out | std::fstream::trunc);
    f2 << generator.kernel_2_source();
}
```

---

## Compiling the generated kernels

Compile with the tile configuration as preprocessor defines (values from your config JSON):

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

Expected: only harmless unused-variable warnings. Zero errors.

---

## Running the correctness test

The test launches both kernels on a real GPU and compares results against a CPU reference matmul:

```bash
cd build

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

./test_matmul
```

**Verified output (NVIDIA GPU, CUDA 11.7):**

| Parameter | Value |
|---|---|
| L1 (batch) | 10 |
| L2 (output features) | 500 |
| R1 (input features) | 64 |
| Kernel 1 grid | (1, 32, 2) |
| Kernel 1 block | (8, 32, 4) |
| Kernel 2 grid | (1, 32, 2) |
| Kernel 2 block | (1, 32, 4) |

| Check | Result |
|---|---|
| Kernel 1 - elements checked | 5000 |
| Kernel 1 - max error | 5.72e-06 |
| Kernel 1 - mismatches | 0 - PASS |
| Kernel 2 - elements checked | 5000 |
| Kernel 2 - max error | 5.72e-06 |
| Kernel 2 - mismatches | 0 - PASS |

The small floating-point error (5.72e-06) is normal IEEE-754 rounding from parallel reduction - not a correctness issue.

---

## How the two-kernel design works

```
Input: Z [L1 × R1],  W [L1 × L2 × R1]
Output: S [L1 × L2]

  ┌─────────────────────────────────┐
  │  kernel 1 (matmul_1.cu)         │
  │  • tiles Z and W into shared    │
  │    memory (__shared__)          │
  │  • accumulates partial dot      │
  │    products per (L1, L2) pair   │
  │  • writes to int_res[]          │
  └────────────────┬────────────────┘
                   │ int_res
  ┌────────────────▼────────────────┐
  │  kernel 2 (matmul_2.cu)         │
  │  • reduces partial sums from    │
  │    int_res across R dimension   │
  │  • writes final result to S[]   │
  └─────────────────────────────────┘
```

---

## Key implementation notes

- **`__device__` qualifier**: Scalar functions `f()` and `g()` must be `__device__` in CUDA. The generator automatically adds this qualifier - it is absent in the OpenCL backend.
- **Kernel 2 block size**: `blockDim.x` for kernel 2 must be `min(NUM_WI_R_1, NUM_WG_R_1)`, not `NUM_WI_R_1`. The shared reduction buffer is sized by `K2_L_NUM_FU_R_1 = min(NUM_WI_R_1, NUM_WG_R_1)` - using more threads overflows it.
- **Result buffer routing**: With `L_CB_RES_DEST_LEVEL=LOCAL`, kernel 1 writes to `int_res`, not `res_g`. The `res_g` parameter exists in the signature but is unused under this configuration.

---

## Documentation

| File | Contents |
|---|---|
| [`CUDA_steps.md`](CUDA_steps.md) | Full step-by-step implementation log with every file, change, error, and fix |
| [`CUDA_proof.md`](CUDA_proof.md) | Evidence that the output is genuine CUDA, not renamed OpenCL |
| [`test_this_cuda.md`](test_this_cuda.md) | Copy-paste terminal guide to generate kernels in any folder |
| [`implementation_idea.md`](implementation_idea.md) | Original design plan for the CUDA backend |

---

## Acknowledgements

This project builds on prior work — the MDH framework, its mathematical foundations, the OpenCL kernel generator, and all supporting infrastructure (`ocl_generator.hpp`, `loop_generator.hpp`, `macros.hpp`, `md_hom.hpp`, `types.hpp`, and related files) were developed by Rasch, Schulze, and Gorlatch and published at PACT 2019:

```bibtex
@inproceedings{rasch2019mdh,
  title={Generating Portable High-Performance Code via Multi-Dimensional Homomorphisms},
  author={Rasch, Ari and Schulze, Richard and Gorlatch, Sergei},
  booktitle={PACT},
  year={2019}
}
```

Artifact repository: [https://gitlab.com/mdh-project/pact_2019_artifact](https://gitlab.com/mdh-project/pact_2019_artifact)

All MDH framework files are included here solely to make this project self-contained and buildable, with full credit to the above authors.
