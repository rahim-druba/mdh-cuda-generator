# MDH CUDA Generator — Full Implementation Log

Every step taken in this project, with file locations, code changes, errors encountered, fixes applied, and verified results.

---

## Background and Goal

The MDH (Multi-Dimensional Homomorphism) framework generates GPU kernels from a high-level C++ specification. It originally only supported OpenCL (`.cl` files). The goal of this project was to add a CUDA backend that generates `.cu` files from the same MDH C++ spec, without touching the original working OpenCL code.

**Original working OpenCL generator location:**
```
/home/rahim/pact_2019_artifact/   ← NEVER modified, kept as-is
```

**Working directory for all CUDA development:**
```
/home/rahim/mdh-cuda-generator/
```

---

## Pre-Work: Safe Isolation

### Problem
Modifying the original `pact_2019_artifact/` would break the working OpenCL generation.

### Solution
Copied the artifact into the new project folder:
```bash
rsync -av --exclude='build/' /home/rahim/pact_2019_artifact/ \
      /home/rahim/mdh-cuda-generator/pact_2019_artifact/
```

Deleted a duplicate in `~/Downloads/pact_2019_artifact-master/` (no longer needed).

The original at `/home/rahim/pact_2019_artifact/` was never touched again.

---

## Step 1 — Study and Map OpenCL → CUDA Syntax

### Files Studied
```
/home/rahim/mdh-cuda-generator/include/ocl_generator.hpp       ← main generator
/home/rahim/mdh-cuda-generator/include/input_buffer_wrapper.hpp
/home/rahim/mdh-cuda-generator/include/result_buffer_wrapper.hpp
/home/rahim/mdh-cuda-generator/include/input_stencil_buffer_wrapper.hpp
/home/rahim/mdh-cuda-generator/include/input_scalar_wrapper.hpp
/home/rahim/mdh-cuda-generator/include/loop_generator.hpp
/home/rahim/mdh-cuda-generator/include/macros.hpp
/home/rahim/mdh-cuda-generator/include/md_hom.hpp
/home/rahim/mdh-cuda-generator/include/types.hpp
```

### Key Translation Table Established

| OpenCL | CUDA |
|---|---|
| `__kernel void name(...)` | `__global__ void name(...)` |
| `__local` | `__shared__` |
| `__private` | *(empty — CUDA registers are implicit)* |
| `__global TYPE*` param | `TYPE*` param (no address space qualifier) |
| `restrict` | `__restrict__` |
| `get_global_id(N)` | `blockIdx.x * blockDim.x + threadIdx.x` (per axis) |
| `get_local_id(N)` | `threadIdx.x` |
| `get_group_id(N)` | `blockIdx.x` |
| `get_global_size(N)` | `gridDim.x * blockDim.x` |
| `get_local_size(N)` | `blockDim.x` |
| `barrier(CLK_LOCAL_MEM_FENCE)` | `__syncthreads()` |
| `barrier(CLK_GLOBAL_MEM_FENCE)` | `__threadfence()` |
| `inline TYPE f(...)` | `__device__ inline TYPE f(...)` |

### Key Architectural Discovery
OpenCL uses `get_global_id(N)` where N is an integer. CUDA uses **named axes** (`x`, `y`, `z`). The framework maps MDH dimensions to hardware axes via `OCL_DIM_L_1`, `OCL_DIM_L_2`, `OCL_DIM_R_1` (integers 0/1/2). This requires `#if OCL_DIM_L_1==0 ... #elif==1 ... #elif==2 ... #endif` chains in the generated kernel.

---

## Step 2 — Create Project Structure

### CMakeLists.txt
**File:** `/home/rahim/mdh-cuda-generator/CMakeLists.txt` (created)

```cmake
cmake_minimum_required(VERSION 2.8.11)
project(mdh_cuda_generator)
set(CMAKE_CXX_STANDARD 14)
include_directories(.)
include_directories(include)
file(GLOB SOURCE_FILES src/*.cpp)
add_library(mdh_cuda_generator STATIC ${SOURCE_FILES})
add_executable(matmul src/matmul/matmul.cpp)
target_link_libraries(matmul mdh_cuda_generator)
```

**Error encountered:** `types.hpp: No such file or directory`
**Root cause:** `src/*.cpp` files use `#include <types.hpp>` which needs the `include/` directory on the path.
**Fix:** Added `include_directories(include)` as a second include line.

### md_hom_generator.hpp
**File:** `/home/rahim/mdh-cuda-generator/md_hom_generator.hpp` (modified)

Added CUDA generator include after the OCL one:
```cpp
#include "include/ocl_generator.hpp"
#include "include/cuda_generator.hpp"   // ← added
```

### matmul.cpp
**File:** `/home/rahim/mdh-cuda-generator/src/matmul/matmul.cpp` (copied unchanged from artifact)

Describes a 2L + 1R matmul:
- `L1` = batch dimension, `L2` = output features, `R1` = input features (reduced)
- `Z[L1, R1]`, `W[L1, L2, R1]` → `S[L1, L2]`
- `f = Z_val * W_val`, `g = identity`

---

## Step 3 — CUDA Thread-ID Macro Emission

### Problem
`ocl_generator.hpp` lines 77–175 build a `std::stringstream ocl_function_macros` that emits:
```
#define GET_GLOBAL_ID_L_1   get_global_id(OCL_DIM_L_1)
```
This is OpenCL-only. CUDA needs named-axis macros with `#if` chains.

### Solution
Created function `emit_cuda_function_macros<L_DIMS, R_DIMS>()` in `cuda_generator.hpp`.

For ≤3 MDH dimensions (the normal case), emits per dimension:
```
#if   OCL_DIM_L_1 == 0
#define GET_GLOBAL_ID_L_1   (blockIdx.x * blockDim.x + threadIdx.x)
#define GET_LOCAL_ID_L_1    (threadIdx.x)
#define GET_GROUP_ID_L_1    (blockIdx.x)
#define GET_GLOBAL_SIZE_L_1 (gridDim.x * blockDim.x)
#define GET_LOCAL_SIZE_L_1  (blockDim.x)
#elif OCL_DIM_L_1 == 1
... (y axis)
#elif OCL_DIM_L_1 == 2
... (z axis)
#endif
```

For >3 MDH dimensions, mirrors the OpenCL stride-arithmetic z-axis packing from `ocl_generator.hpp` lines 99–174, but using `blockIdx.z / threadIdx.z` instead of `get_group_id(2) / get_local_id(2)`.

**File:** `/home/rahim/mdh-cuda-generator/include/cuda_generator.hpp` (function written at top of file)

---

## Step 4 — Inspect loop_generator.hpp and macros.hpp

### Finding
Both files contain **zero OpenCL syntax**. They are pure C++ template code that generates generic tiling loop structures. They can be reused unchanged by `cuda_generator_class`.

**Files confirmed as pure C++, no changes needed:**
```
/home/rahim/mdh-cuda-generator/include/loop_generator.hpp
/home/rahim/mdh-cuda-generator/include/macros.hpp
```

The two OpenCL barriers injected by `loop_generator` hooks (lines 388 and 412 of `ocl_generator.hpp`) are injected by the **generator class itself**, not by `loop_generator.hpp`. So the fix belongs in the CUDA generator.

---

## Step 5 — Create CUDA Wrapper Files

Three wrapper files contain OpenCL-specific strings and needed CUDA versions.

### 5a. cuda_input_buffer_wrapper.hpp

**File:** `/home/rahim/mdh-cuda-generator/include/cuda_input_buffer_wrapper.hpp` (created)

**Source:** Copy of `input_buffer_wrapper.hpp` with:
- Header guard: `MD_BLAS_INPUT_BUFFER_WRAPPER_HPP` → `MDH_CUDA_INPUT_BUFFER_WRAPPER_HPP`
- Line 171: `"__local"` → `"__shared__"`, `"__private"` → `""`
- Global word-boundary rename: `input_buffer_wrapper` → `cuda_input_buffer_wrapper`
  (used Python `re.sub(r'\binput_buffer_wrapper\b', 'cuda_input_buffer_wrapper', content)`)

**Error encountered:** `redefinition of 'class md_hom::generator::input_buffer_wrapper'`
**Root cause:** First rename attempt missed the constructor (partial rename).
**Fix:** Global word-boundary replacement covered all occurrences including the constructor.

### 5b. cuda_result_buffer_wrapper.hpp

**File:** `/home/rahim/mdh-cuda-generator/include/cuda_result_buffer_wrapper.hpp` (created)

**Source:** Copy of `result_buffer_wrapper.hpp` with:
- Header guard: → `MDH_CUDA_RESULT_BUFFER_WRAPPER_HPP`
- Lines 527, 541: `"__local"` → `"__shared__"`, `"__private"` → `""`
- 12 barrier replacement sites:
  ```
  stringf("barrier(CLK_%s_MEM_FENCE);", long_level(level))
  →
  (level == LEVEL::LOCAL ? "__syncthreads();" : "__threadfence();")
  ```
- Global word-boundary rename: `result_buffer_wrapper` → `cuda_result_buffer_wrapper`

### 5c. cuda_input_stencil_buffer_wrapper.hpp

**File:** `/home/rahim/mdh-cuda-generator/include/cuda_input_stencil_buffer_wrapper.hpp` (created)

**Source:** Copy of `input_stencil_buffer_wrapper.hpp` with:
- Header guard: → `MDH_CUDA_INPUT_STENCIL_BUFFER_WRAPPER_HPP`
- Line 231: `"__local"` → `"__shared__"`, `"__private"` → `""`
- Global word-boundary rename: `input_stencil_buffer_wrapper` → `cuda_input_stencil_buffer_wrapper`

**Verification after each file:** `grep "__local\|CLK_"` returned zero results.

---

## Step 5 (continued) — Write cuda_generator.hpp (Main Work)

**File:** `/home/rahim/mdh-cuda-generator/include/cuda_generator.hpp`

Modeled **directly** on `ocl_generator.hpp`. The class `cuda_generator_class` is a line-for-line copy of `ocl_generator_class` with exactly 7 targeted changes:

### Change 1 — Include CUDA wrappers instead of OCL wrappers
```cpp
// REMOVED:
#include "input_buffer_wrapper.hpp"
#include "result_buffer_wrapper.hpp"

// ADDED:
#include "cuda_input_buffer_wrapper.hpp"
#include "cuda_input_stencil_buffer_wrapper.hpp"
#include "cuda_result_buffer_wrapper.hpp"
```

### Change 2 — Thread ID macros
```cpp
// REMOVED (ocl_generator.hpp lines 77-175):
std::stringstream ocl_function_macros;
ocl_function_macros << concat(multi_stringf(...get_global_id(OCL_DIM_...)...));
// ...
ocl_function_macros.str()   // passed to stringf

// ADDED:
emit_cuda_function_macros<L_DIMS, R_DIMS>()   // passed to stringf
```

### Change 3 — Kernel signature
```cpp
// REMOVED (ocl_generator.hpp line 277):
"__kernel void %s_%d(..., __global TYPE_TS * const restrict res_g, "
"__global TYPE_TS * const restrict %s%s)"

// ADDED:
"__global__ void %s_%d(..., TYPE_TS * const __restrict__ res_g, "
"TYPE_TS * const __restrict__ %s%s)"
```

### Change 4 — int_res parameter
```cpp
// REMOVED (ocl_generator.hpp line 284):
"__global TYPE_T const * const restrict int_res"

// ADDED:
"TYPE_T const * const __restrict__ int_res"
```

### Change 5 — Comment
```cpp
// REMOVED: "// map md_hom dimensions to OpenCL dimensions"
// ADDED:   "// map md_hom dimensions to CUDA dimensions"
```

### Change 6 — Barriers (2 sites in loop generator hooks)
```cpp
// REMOVED (ocl_generator.hpp lines 388, 412):
caching_code.append("barrier(CLK_LOCAL_MEM_FENCE);\n");
code.append("barrier(CLK_LOCAL_MEM_FENCE);\n");

// ADDED:
caching_code.append("__syncthreads();\n");
code.append("__syncthreads();\n");
```

### Change 7 — Wrapper class instances
```cpp
// REMOVED in wrap_inputs_impl():
new input_buffer_wrapper<L_DIMS, R_DIMS>(buffer, _macros)
new input_stencil_buffer_wrapper<NT, L_DIMS, R_DIMS>(buffer, _macros)

// ADDED:
new cuda_input_buffer_wrapper<L_DIMS, R_DIMS>(buffer, _macros)
new cuda_input_stencil_buffer_wrapper<NT, L_DIMS, R_DIMS>(buffer, _macros)

// REMOVED in get_input_names_and_definitions_impl():
"__global TYPE_T const * const restrict " + buffer.name()

// ADDED:
"TYPE_T const * const __restrict__ " + buffer.name()

// REMOVED in result wrapper member:
const result_buffer_wrapper<L_DIMS, R_DIMS> _result_wrapper;

// ADDED:
const cuda_result_buffer_wrapper<L_DIMS, R_DIMS> _result_wrapper;
```

### Convenience function
```cpp
template <unsigned int L_DIMS, unsigned int R_DIMS, typename T, typename... Ts>
auto cuda_generator(const md_hom_class<L_DIMS, R_DIMS, T, Ts...> &md_hom) {
    return cuda_generator_class<L_DIMS, R_DIMS, T, Ts...>(md_hom);
}
```

---

## Step 6 — Update matmul.cpp

**File:** `/home/rahim/mdh-cuda-generator/src/matmul/matmul.cpp`

Three lines changed:
```cpp
// BEFORE:
auto generator = md_hom::generator::ocl_generator(md_hom_matmul);
kernel_file.open("matmul_1.cl", std::fstream::out | std::fstream::trunc);
kernel_file.open("matmul_2.cl", std::fstream::out | std::fstream::trunc);

// AFTER:
auto generator = md_hom::generator::cuda_generator(md_hom_matmul);
kernel_file.open("matmul_1.cu", std::fstream::out | std::fstream::trunc);
kernel_file.open("matmul_2.cu", std::fstream::out | std::fstream::trunc);
```

Everything else in `matmul.cpp` is identical.

---

## Step 7 — Test and Verify

### 7a. Build Verification
```bash
cd /home/rahim/mdh-cuda-generator/build
cmake ..
make
```
**Result:** Clean build. Only 2 pre-existing `-Wformat-security` warnings from `helper.hpp` (not from our code).

### 7b. Generator Run
```bash
./matmul
```
**Output files created:**
```
/home/rahim/mdh-cuda-generator/build/matmul_1.cu   (20 MB)
/home/rahim/mdh-cuda-generator/build/matmul_2.cu   (14 MB)
```
Note: The files are large because the framework unrolls all tiling combinations as `#if/#elif` chains at compile time.

Old OpenCL files remain alongside them:
```
/home/rahim/mdh-cuda-generator/build/matmul_1.cl   (20 MB)
/home/rahim/mdh-cuda-generator/build/matmul_2.cl   (14 MB)
```

### 7c. Correctness Grep Checks
```bash
grep "__global__ void matmul_1" matmul_1.cu        # kernel signature
grep "__shared__" matmul_1.cu                       # shared memory
grep "__global__\|__restrict__" matmul_1.cu         # CUDA syntax
grep "__syncthreads" matmul_1.cu                    # barriers
grep "__kernel\|get_global_id\|CLK_LOCAL_MEM_FENCE" matmul_1.cu  # no OCL leftovers
```

**Results:**
- `__global__ void matmul_1(TYPE_T const * const __restrict__ Z, ...)` ✓
- `__shared__ TYPE_T cb_l_Z[...]`, `__shared__ TYPE_T cb_l_W[...]` ✓
- `__syncthreads()` present throughout ✓
- Zero OpenCL strings remaining ✓
- Both `matmul_1.cu` and `matmul_2.cu` clean ✓

### 7d. CUDA Thread ID Macros in Output
```
#if   OCL_DIM_L_1 == 0
#define GET_GLOBAL_ID_L_1   (blockIdx.x * blockDim.x + threadIdx.x)
...
#elif OCL_DIM_L_1 == 1
#define GET_GLOBAL_ID_L_1   (blockIdx.y * blockDim.y + threadIdx.y)
...
#elif OCL_DIM_L_1 == 2
#define GET_GLOBAL_ID_L_1   (blockIdx.z * blockDim.z + threadIdx.z)
...
#endif
```
The `#if` chains are correct for all 3 axes. ✓

---

## Step 7e — nvcc Compile Test

### First attempt (no config defines)
```bash
nvcc -c matmul_1.cu -o matmul_1.o
```
**Error:** `division by zero in #if` for macros like `K1_L_NUM_CB_L_1 (K1_G_CB_SIZE_L_1 / K1_L_CB_SIZE_L_1 / ...)`.

**Root cause:** Tile-size macros (`G_CB_SIZE_L_1`, `L_CB_SIZE_L_1`, etc.) are all 0 because they weren't defined. Same requirement as the OpenCL kernels — config must be supplied at compile time.

### Config Source
Found real GPU default config at:
```
/home/rahim/mdh-cuda-generator/pact_2019_artifact/defaults/gpu/md_hom_initial/gemm_10x500x64_config.json
```
```json
{
  "G_CB_SIZE_L_1": 10, "G_CB_SIZE_L_2": 500, "G_CB_SIZE_R_1": 64,
  "L_CB_SIZE_L_1": 8,  "L_CB_SIZE_L_2": 16,  "L_CB_SIZE_R_1": 64,
  "P_CB_SIZE_L_1": 1,  "P_CB_SIZE_L_2": 1,   "P_CB_SIZE_R_1": 1,
  "NUM_WG_L_1": 2, "NUM_WG_L_2": 32, "NUM_WG_R_1": 1,
  "NUM_WI_L_1": 4, "NUM_WI_L_2": 32, "NUM_WI_R_1": 8,
  "OCL_DIM_L_1": 2, "OCL_DIM_L_2": 1, "OCL_DIM_R_1": 0,
  "G_CB_RES_DEST_LEVEL": 2, "L_CB_RES_DEST_LEVEL": 1, "P_CB_RES_DEST_LEVEL": 0
}
```

### Second attempt (with config, new error)
```bash
nvcc -c matmul_1.cu -o matmul_1.o -DTYPE_T=float -DTYPE_TS=float \
  -DG_CB_SIZE_L_1=10 ... (all config values)
```
**Error:**
```
calling a __host__ function("f(float, float)") from a __global__ function("matmul_1")
identifier "f" is undefined in device code
```

**Root cause:** `scalar_function_wrapper::definition()` (defined in `ocl_generator.hpp`) emits:
```cpp
inline TYPE_TS f(...)
```
In CUDA, plain `inline` means `__host__` — invisible to device code.

**Fix location:** `cuda_generator.hpp` in the `cuda_generator_class` constructor where `f()` and `g()` definitions are appended to the kernel source.

**Fix applied:**
```cpp
// Added lambda wrapper before appending scalar function definitions:
auto cuda_device_fn = [](const std::string &def) {
    return search_and_replace(def, "inline TYPE_TS ", "__device__ inline TYPE_TS ");
};

// Then use it:
kernel_source.append(cuda_device_fn(_scalar_function_wrapper->definition()));
kernel_source.append(search_and_replace(
    cuda_device_fn(_result_function_wrapper->definition()), "f(", "g("));
```

**File modified:** `/home/rahim/mdh-cuda-generator/include/cuda_generator.hpp` (around line 257)

### Third attempt (after __device__ fix)
```bash
nvcc -c matmul_1.cu -o matmul_1.o -DTYPE_T=float ... (all config values)
```
**Result:** 0 errors, 6 pre-existing `warning #177: variable declared but never referenced` (same warnings the OCL generator produces — loop variables for dimensions with size 1 are optimised away). ✓

---

## Step 7f — Runtime Test (test_matmul.cu)

**File created:** `/home/rahim/mdh-cuda-generator/build/test_matmul.cu`

### Buffer Layout (discovered from generated kernel macros)

| Buffer | Source macro | Layout |
|---|---|---|
| `Z` | `K1_G_BUFFER_Z(i,k) = Z[(i)*NR1+(k)]` | Row-major `[L1][R1]` |
| `W` | `K1_G_BUFFER_W(i,j,k) = W[(i*NL2+j)*NR1+k]` | Row-major `[L1][L2][R1]` |
| `int_res` (K1 output) | `int_res[(i)*NL2+(j)]` when `K1_G_NUM_FU_R_1==1` | Row-major `[L1][L2]` |
| `S` (K2 output) | `S[(i)*NL2+(j)]` when `K2_G_NUM_FU_R_1==1` | Row-major `[L1][L2]` |

### Key Architectural Finding: res_g vs int_res

Inspecting the generated kernel around the `K1_RES_G_BUFFER_NAME()` definition:
```c
#if (L_CB_RES_DEST_LEVEL == GLOBAL && K1_L_NUM_FU_R_1 > 1) || \
    (P_CB_RES_DEST_LEVEL == GLOBAL && K1_P_NUM_FU_R_1 > 1)
  #define K1_RES_G_BUFFER_NAME() res_g      ← used for global-level partial sums
#else
  #define K1_RES_G_BUFFER_NAME() int_res    ← used when reduction is at LOCAL/PRIVATE
  #if K1_G_NUM_FU_R_1 == 1
    #define K1_G_BUFFER(i,j,...) int_res[(i)*NL2+(j)]   ← row-major final result
  #endif
#endif
```

With config `L_CB_RES_DEST_LEVEL=1 (LOCAL)` and `P_CB_RES_DEST_LEVEL=0 (PRIVATE)`:
- Condition is **FALSE** → kernel_1 writes to **`int_res`**, not `res_g`
- `res_g` parameter exists in the signature but is unused in this config
- First test attempt read `d_res` → all zeros (wrong buffer)
- **Fix:** Read from `d_int` after kernel_1

### Grid/Block Dimensions

**Derived from OCL_DIM_* assignments:**
```
OCL_DIM_R_1=0 → x axis
OCL_DIM_L_2=1 → y axis
OCL_DIM_L_1=2 → z axis
```

| | x (R1) | y (L2) | z (L1) |
|---|---|---|---|
| **kernel_1 blockDim** | NUM_WI_R_1=8 | NUM_WI_L_2=32 | NUM_WI_L_1=4 |
| **kernel_1 gridDim** | NUM_WG_R_1=1 | NUM_WG_L_2=32 | NUM_WG_L_1=2 |
| **kernel_2 blockDim** | **1** | NUM_WI_L_2=32 | NUM_WI_L_1=4 |
| **kernel_2 gridDim** | 1 | NUM_WG_L_2=32 | NUM_WG_L_1=2 |

`kernel_1 blockDim.x * blockDim.y * blockDim.z = 8 * 32 * 4 = 1024` ✓ (CUDA max)

### kernel_2 Block Dimension Bug Found and Fixed

**First run:** kernel_2 crashed with `illegal memory access`.

**Root cause:** kernel_2's shared memory reduction buffer for R1 is sized as:
```
__shared__ float K2_L_REDUCTION_MEM[K2_L_CB_SIZE_L_1][K2_L_CB_SIZE_L_2][K2_L_NUM_FU_R_1]
```
With our config:
- `K2_L_NUM_FU_R_1 = min(NUM_WI_R_1, CEIL(K2_L_CB_SIZE_R_1, K2_P_CB_SIZE_R_1))`
- `K2_L_CB_SIZE_R_1 = K2_G_CB_SIZE_R_1 = NUM_WG_R_1 = 1`
- `K2_L_NUM_FU_R_1 = min(8, CEIL(1,1)) = min(8, 1) = 1`
- Array is `[8][16][1]` — only 1 slot in R1 direction

Launching with `blockDim.x = 8` means threads 1–7 have `threadIdx.x ∈ {1..7}` → they write to `shared[...][1..7]` which is **out of bounds** → illegal memory access.

**Fix:**
```cpp
// kernel_2 blockDim.x = min(NUM_WI_R_1, NUM_WG_R_1) = min(8, 1) = 1
static const int  K2_BLOCK_X = (NUM_WG_R_1 < NUM_WI_R_1) ? NUM_WG_R_1 : NUM_WI_R_1;
static const dim3 BLOCK2 { (unsigned)K2_BLOCK_X, NUM_WI_L_2, NUM_WI_L_1 };
static const dim3 GRID2  { 1, NUM_WG_L_2, NUM_WG_L_1 };
```

### CPU Reference Formula
```cpp
// Z row-major [L1][R1]:   Z[l1 * NR1 + r1]
// W row-major [L1][L2][R1]: W[(l1*NL2 + l2) * NR1 + r1]
// result row-major [L1][L2]: ref[l1 * NL2 + l2]
for (int l1 = 0; l1 < NL1; l1++)
    for (int l2 = 0; l2 < NL2; l2++) {
        float s = 0.0f;
        for (int r1 = 0; r1 < NR1; r1++)
            s += Z[l1*NR1+r1] * W[(l1*NL2+l2)*NR1+r1];
        ref[l1*NL2+l2] = s;
    }
```

### Build Command (full pipeline test)
```bash
cd /home/rahim/mdh-cuda-generator/build
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

### Final Test Output
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

Max error 5.72e-06 is normal IEEE-754 float32 rounding from non-associative parallel reduction — not a correctness error.

---

## Complete File Inventory

### New files created
| File | Description |
|---|---|
| `/home/rahim/mdh-cuda-generator/CMakeLists.txt` | Build system |
| `/home/rahim/mdh-cuda-generator/include/cuda_generator.hpp` | Main CUDA generator class |
| `/home/rahim/mdh-cuda-generator/include/cuda_input_buffer_wrapper.hpp` | CUDA version of input buffer wrapper |
| `/home/rahim/mdh-cuda-generator/include/cuda_result_buffer_wrapper.hpp` | CUDA version of result buffer wrapper |
| `/home/rahim/mdh-cuda-generator/include/cuda_input_stencil_buffer_wrapper.hpp` | CUDA version of stencil wrapper |
| `/home/rahim/mdh-cuda-generator/build/test_matmul.cu` | Runtime test harness |

### Files modified
| File | Change |
|---|---|
| `/home/rahim/mdh-cuda-generator/md_hom_generator.hpp` | Added `#include "include/cuda_generator.hpp"` |
| `/home/rahim/mdh-cuda-generator/src/matmul/matmul.cpp` | 3 lines: `ocl_generator` → `cuda_generator`, `.cl` → `.cu` |

### Generated output files (in build/)
| File | Size | Description |
|---|---|---|
| `matmul_1.cu` | 20 MB | CUDA kernel 1 (main computation) |
| `matmul_2.cu` | 14 MB | CUDA kernel 2 (final reduction) |
| `matmul_1.cl` | 20 MB | Original OpenCL kernel 1 (unchanged) |
| `matmul_2.cl` | 14 MB | Original OpenCL kernel 2 (unchanged) |

### Files never touched (OpenCL generator still works)
```
/home/rahim/pact_2019_artifact/                     ← original, never modified
/home/rahim/mdh-cuda-generator/include/ocl_generator.hpp
/home/rahim/mdh-cuda-generator/include/input_buffer_wrapper.hpp
/home/rahim/mdh-cuda-generator/include/result_buffer_wrapper.hpp
/home/rahim/mdh-cuda-generator/include/input_stencil_buffer_wrapper.hpp
/home/rahim/mdh-cuda-generator/include/loop_generator.hpp
/home/rahim/mdh-cuda-generator/include/macros.hpp
```

---

## Summary of All Bugs Found and Fixed

| # | Bug | Location | Fix |
|---|---|---|---|
| 1 | `types.hpp: No such file` | `CMakeLists.txt` | Added `include_directories(include)` |
| 2 | `redefinition of input_buffer_wrapper` | `cuda_input_buffer_wrapper.hpp` | Word-boundary Python regex rename of class + constructor |
| 3 | `__device__` missing on `f()` / `g()` | `cuda_generator.hpp` line ~257 | `cuda_device_fn` lambda wraps `definition()` output |
| 4 | Reading wrong buffer (`res_g` all zeros) | `test_matmul.cu` | Read `d_int` (kernel_1 writes to `int_res`, not `res_g`, with this config) |
| 5 | kernel_2 illegal memory access | `test_matmul.cu` | `blockDim.x = min(NUM_WI_R_1, NUM_WG_R_1)` — shared array is only `NUM_WG_R_1` wide |

---

## nvcc Version Used
```
NVIDIA CUDA Compiler Driver
Release 11.7, V11.7.99
Built on Wed_Jun__8_16:49:14_PDT_2022
Location: /usr/local/cuda-11.7/bin/nvcc
```
