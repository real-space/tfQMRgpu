#pragma once
// This file is part of tfQMRgpu under MIT-License

//
// cudaStubs: replacement for CUDA infrastructure functions when compiling without CUDA
//

#include <cstdlib> // std::malloc, std::size_t, std::free
#include <cstring> // std::memcpy, std::memset

typedef int cudaStream_t;
typedef int cudaError_t;
cudaError_t constexpr cudaSuccess = 0;

enum  cudaMemcpyKind : char
{
      cudaMemcpyHostToHost = '_',
      cudaMemcpyDeviceToDevice = '-',
      cudaMemcpyDeviceToHost = 'v',
      cudaMemcpyHostToDevice = '^'
}; // cudaMemcpyKind

inline char const* cudaGetErrorString(cudaError_t const err) { return err ? "cudaStubs::Error!" : "cudaStubs::Success"; }
inline cudaError_t cudaGetLastError(void) { return cudaSuccess; }

inline cudaError_t cudaMalloc(void* *d, std::size_t const size_in_Byte) {
    *d = std::malloc(size_in_Byte);
    return (nullptr == *d);
} // cudaMalloc

inline cudaError_t cudaMallocManaged(void* *d, std::size_t const size_in_Byte) {
    return cudaMalloc(d, size_in_Byte);
} // cudaMallocManaged

inline cudaError_t cudaFree(void* d) {
    if (d) std::free(d); 
    return (nullptr == d); 
} // cudaFree


inline cudaError_t cudaStreamCreate(cudaStream_t *s) { *s = 0; return cudaSuccess; }

inline cudaError_t cudaDeviceSynchronize(void) { return cudaSuccess; }

inline cudaError_t cudaMemcpy(void *dest, void const *src, std::size_t count, cudaMemcpyKind kind) {
    if (src != dest) std::memcpy(dest, src, count);
    return cudaSuccess;
} // cudaMemcpy

inline cudaError_t cudaMemcpyAsync(void *dest, void const *src, std::size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    return cudaMemcpy(dest, src, count, kind);
} // cudaMemcpyAsync

inline cudaError_t cudaMemset(void* dest, int ch, std::size_t count) {
    std::memset(dest, ch, count);
    return cudaSuccess;
} // cudaMemset

inline cudaError_t cudaMemsetAsync(void* dest, int ch, std::size_t count, cudaStream_t stream) { 
    return cudaMemset(dest, ch, count);
} // cudaMemsetAsync

inline void __syncthreads(void) {}

// NVTX markers
inline void PUSH_RANGE(char const *name) {}
inline void POP_RANGE(void) {}

int constexpr cudaFuncAttributeMaxDynamicSharedMemorySize = 0;

template <class WhatEver>
inline void cudaFuncSetAttribute(WhatEver, int, size_t) {}

#define __global__
#define __device__
#define __host__
