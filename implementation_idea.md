# Implementation Idea: MDH to CUDA Generator

**Goal:** Build a new backend for the MDH framework that generates CUDA kernels
directly from the same high-level MDH C++ specification used for OpenCL generation.

---

## The Big Picture

```
matmul.cpp (MDH spec)
        |
        v
  [YOUR CUDA GENERATOR]        <-- what you build
        |
        v
  matmul_1.cu + matmul_2.cu    <-- CUDA kernels ready to compile with nvcc
```

The MDH spec does not change at all.
Only the generator backend changes - from OpenCL syntax to CUDA syntax.

---

## How the Existing Generator Works (What to Learn First)

Before writing anything, study these files in order:

```
pact_2019_artifact/preliminary/MLP&SVM/md_hom_generator/
|
|- md_hom_generator.hpp          -- 1. start here, top-level include
|- include/
|   |- md_hom.hpp                -- 2. understand the core MDH abstraction
|   |- ocl_generator.hpp         -- 3. most important - this is what you replace
|   |- loop_generator.hpp        -- 4. understand how tiling loops are built
|   |- macros.hpp                -- 5. understand the macro emission system
|   |- configuration_generator.hpp  -- 6. understand tile config enumeration
|   |- input_buffer.hpp          -- 7. how input buffers are described
|   |- result_buffer.hpp         -- 8. how output buffers are described
|   |- scalar_function.hpp       -- 9. how f() and g() are stored and emitted
```

The single most important file is `ocl_generator.hpp`.
Everything it does for OpenCL, your `cuda_generator.hpp` will do for CUDA.

---

## Architecture of Your Generator

Your generator has three layers:

```
LAYER 1 - Input (reuse from MDH, do not touch)
    md_hom spec --> md_hom_class object
    (Z, W buffers, f and g functions, dimension info)

LAYER 2 - Your CUDA Generator (what you build)
    cuda_generator.hpp
    cuda_loop_generator.hpp
    cuda_macros.hpp

LAYER 3 - Output
    kernel_1() --> matmul_1.cu
    kernel_2() --> matmul_2.cu
```

---

## Step-by-Step Implementation Plan

---

### Step 1 - Study and Map OpenCL to CUDA Syntax

Before writing any code, make a complete translation table.
Every OpenCL construct the existing generator emits must have a CUDA equivalent.

| OpenCL (emitted by ocl_generator.hpp) | CUDA (your generator emits) |
|---|---|
| `__kernel void name(...)` | `__global__ void name(...)` |
| `__global TYPE_T *buf` | `TYPE_T *buf` |
| `__local TYPE_T buf[N]` | `__shared__ TYPE_T buf[N]` |
| `__private TYPE_T x` | `TYPE_T x` |
| `get_global_id(0)` | `blockIdx.x * blockDim.x + threadIdx.x` |
| `get_global_id(1)` | `blockIdx.y * blockDim.y + threadIdx.y` |
| `get_global_id(2)` | `blockIdx.z * blockDim.z + threadIdx.z` |
| `get_local_id(0)` | `threadIdx.x` |
| `get_local_id(1)` | `threadIdx.y` |
| `get_local_id(2)` | `threadIdx.z` |
| `get_group_id(0)` | `blockIdx.x` |
| `get_group_id(1)` | `blockIdx.y` |
| `get_group_id(2)` | `blockIdx.z` |
| `get_local_size(0)` | `blockDim.x` |
| `get_global_size(0)` | `gridDim.x * blockDim.x` |
| `barrier(CLK_LOCAL_MEM_FENCE)` | `__syncthreads()` |
| `barrier(CLK_GLOBAL_MEM_FENCE)` | `__threadfence()` |
| `#pragma OPENCL EXTENSION ...` | not needed in CUDA |
| `(void)` kernel return | same - `void` |

---

### Step 2 - Create the Project Structure

```
mdh-cuda-generator/
|
|- md_hom_generator.hpp          -- copy from original, add cuda_generator include
|- include/
|   |- md_hom.hpp                -- COPY as-is from original (no changes needed)
|   |- input_buffer.hpp          -- COPY as-is
|   |- input_buffer_wrapper.hpp  -- COPY as-is
|   |- result_buffer.hpp         -- COPY as-is
|   |- scalar_function.hpp       -- COPY as-is
|   |- types.hpp                 -- COPY as-is
|   |- helper.hpp                -- COPY as-is
|   |- loop_generator.hpp        -- ADAPT (change OpenCL thread calls to CUDA)
|   |- macros.hpp                -- ADAPT (change OpenCL macros to CUDA macros)
|   |- cuda_generator.hpp        -- WRITE FROM SCRATCH (your main contribution)
|- src/
|   |- matmul/
|       |- matmul.cpp            -- your spec file, unchanged
|- CMakeLists.txt
```

Files you copy unchanged: md_hom.hpp, input_buffer, result_buffer, scalar_function, types, helper.
Files you adapt: loop_generator.hpp, macros.hpp.
File you write from scratch: cuda_generator.hpp.

---

### Step 3 - Adapt macros.hpp

The existing `macros.hpp` emits OpenCL `#define` blocks for thread ID lookups.

Change every emitted macro to use CUDA equivalents.

Example - existing OpenCL macro emission:
```cpp
// ocl_generator emits this string:
"#define GET_GLOBAL_ID_L_1 get_global_id(OCL_DIM_L_1)"
```

Your CUDA version emits:
```cpp
// your generator emits this string:
"#define GET_GLOBAL_ID_L_1 (blockIdx.x * blockDim.x + threadIdx.x)"
```

The macro names stay the same - only the right-hand side changes.
This means the loop body code that uses these macros does not need to change.

---

### Step 4 - Adapt loop_generator.hpp

The loop generator builds the tiling loop structure.
Most of it is pure C logic (for loops, index arithmetic) which is identical in CUDA.

The only parts to change are:
- Barrier calls: `barrier(CLK_LOCAL_MEM_FENCE)` becomes `__syncthreads()`
- Memory space qualifiers when declaring local buffers

---

### Step 5 - Write cuda_generator.hpp (Main Work)

This is your primary contribution. Model it directly on `ocl_generator.hpp`.

The class structure:

```cpp
namespace md_hom {
namespace generator {

template<unsigned int L_DIMS, unsigned int R_DIMS, typename T, typename... Ts>
class cuda_generator_class {
public:
    cuda_generator_class(const md_hom_class<L_DIMS, R_DIMS, T, Ts...>& hom);

    std::string kernel_1();   // returns .cu kernel 1 as string
    std::string kernel_2();   // returns .cu kernel 2 as string

private:
    std::string emit_kernel_header(int kernel_num);
    std::string emit_shared_memory_declarations();
    std::string emit_thread_index_mappings();
    std::string emit_scalar_functions();
    std::string emit_tiling_loops();
    std::string emit_reduction();
};

// convenience function - mirrors ocl_generator()
template<unsigned int L_DIMS, unsigned int R_DIMS, typename T, typename... Ts>
auto cuda_generator(const md_hom_class<L_DIMS, R_DIMS, T, Ts...>& hom) {
    return cuda_generator_class<L_DIMS, R_DIMS, T, Ts...>(hom);
}

} // namespace generator
} // namespace md_hom
```

Key method - `emit_kernel_header()` - this is where the biggest difference lives:

```cpp
// OpenCL version emits:
"__kernel void matmul_1(__global TYPE_T const * const restrict Z, ...)"

// Your CUDA version emits:
"__global__ void matmul_1(TYPE_T const * const __restrict__ Z, ...)"
```

Note the differences:
- `__kernel` becomes `__global__`
- `__global` buffer qualifier is removed (CUDA pointers are global by default)
- `restrict` becomes `__restrict__`

---

### Step 6 - Update the Spec File

Change only the generator call and output file extensions:

```cpp
// matmul.cpp - only these two lines change:

// OLD:
auto generator = md_hom::generator::ocl_generator(md_hom_matmul);
kernel_file.open("matmul_1.cl", ...);

// NEW:
auto generator = md_hom::generator::cuda_generator(md_hom_matmul);
kernel_file.open("matmul_1.cu", ...);
```

Everything else in matmul.cpp stays identical.

---

### Step 7 - Test and Verify

#### Correctness test - compare structure:
```bash
# check kernel name exists
grep "__global__ void matmul_1" matmul_1.cu

# check shared memory used (not __local)
grep "__shared__" matmul_1.cu

# check no OpenCL leftovers
grep "__kernel\|get_global_id\|CLK_LOCAL_MEM_FENCE" matmul_1.cu
# should return nothing
```

#### Compile test:
```bash
nvcc -c matmul_1.cu -o matmul_1.o
```
If it compiles without errors, the syntax is correct.

#### Runtime test (if NVIDIA GPU available):
Write a small test harness that:
1. Allocates Z and W on GPU with `cudaMalloc`
2. Launches `matmul_1` kernel
3. Copies result back
4. Compares against a CPU reference matmul
5. Checks values match within floating point tolerance

---

## What You Reuse vs What You Build

| Component | Action | Effort |
|---|---|---|
| `md_hom.hpp` - core abstraction | Copy unchanged | None |
| `input_buffer.hpp` - buffer descriptors | Copy unchanged | None |
| `scalar_function.hpp` - f() and g() | Copy unchanged | None |
| `types.hpp`, `helper.hpp` | Copy unchanged | None |
| `macros.hpp` | Adapt - change RHS of macros | Low |
| `loop_generator.hpp` | Adapt - change barriers and qualifiers | Low |
| `ocl_generator.hpp` | Use as reference, write cuda_generator.hpp | High |
| `cuda_generator.hpp` | Write from scratch | High |
| `matmul.cpp` spec | Change 3 lines | Trivial |

---

## Phases Summary

```
Phase 1 - Study (1-2 weeks)
    Read ocl_generator.hpp thoroughly
    Understand how every emitted string is constructed
    Build your full OpenCL to CUDA translation table

Phase 2 - Setup (2-3 days)
    Create project structure
    Copy unchanged files
    Set up CMakeLists.txt

Phase 3 - Core Build (2-4 weeks)
    Adapt macros.hpp
    Adapt loop_generator.hpp
    Write cuda_generator.hpp

Phase 4 - Test (1 week)
    Structural tests (grep checks)
    Compile tests (nvcc)
    Runtime correctness tests

Phase 5 - Document and Compare (1 week)
    Compare generated CUDA vs generated OpenCL side by side
    Document what changed and why
    Measure performance if possible
```

---

## The Simplest Possible First Version

Do not try to handle everything at once.
Start with this minimal goal:

> Generate a CUDA kernel for matmul that compiles with nvcc, even if not fully optimized.

That means:
1. Get the kernel signature right (`__global__` instead of `__kernel`)
2. Get shared memory right (`__shared__` instead of `__local`)
3. Get thread indexing right (CUDA threadIdx/blockIdx instead of get_local_id/get_global_id)
4. Get barriers right (`__syncthreads()`)
5. Get the scalar functions f() and g() emitted correctly

Once it compiles and produces correct results, then optimize.

---

## Key Files to Read Right Now

Start with these two files from the cloned repo:

```
/home/rahim/pact_2019_artifact/preliminary/MLP&SVM/md_hom_generator/include/ocl_generator.hpp
/home/rahim/pact_2019_artifact/preliminary/MLP&SVM/md_hom_generator/include/macros.hpp
```

These two files together contain 80% of what you need to understand
before you can write a single line of your CUDA generator.
