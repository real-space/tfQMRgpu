#pragma once
//
// cudaStubs: replacement for CUDA infrastructure functions when compiling without CUDA
//

#include <cstdlib> // std::malloc, std::size_t, std::free
#include <cstring> // std::memcpy, std::memset

typedef int cudaStream_t;
typedef int cudaError;
cudaError constexpr cudaSuccess = 0;

enum  cudaMemcpyKind : char
{
      cudaMemcpyDeviceToDevice = '-',
      cudaMemcpyDeviceToHost = 'v',
      cudaMemcpyHostToDevice = '^'
}; // cudaMemcpyKind

inline char const * cudaGetErrorString(cudaError const err) { return "cudaStubs::Error!"; }

inline cudaError cudaMalloc(void* *d, std::size_t const size_in_Byte) {
    *d = std::malloc(size_in_Byte);
    return (nullptr == *d);
} // cudaMalloc

inline cudaError cudaMallocManaged(void* *d, std::size_t const size_in_Byte) {
    return cudaMalloc(d, size_in_Byte);
} // cudaMallocManaged

inline cudaError cudaFree(void* d) {
    if (d) std::free(d); 
    return (nullptr == d); 
} // cudaFree

inline cudaError cudaStreamCreate(cudaStream_t *s) { *s = 0; return cudaSuccess; }

inline cudaError cudaDeviceSynchronize(void) { return cudaSuccess; }

inline cudaError cudaMemcpy(void *dest, void const *src, std::size_t count, cudaMemcpyKind kind) {
    if (src != dest) std::memcpy(dest, src, count);
    return cudaSuccess;
} // cudaMemcpy

inline cudaError cudaMemcpyAsync(void *dest, void const *src, std::size_t count, char kind, cudaStream_t stream) {
    return cudaMemcpy(dest, src, count, kind);
} // cudaMemcpyAsync

inline cudaError cudaMemset(void* dest, int ch, std::size_t count) {
    std::memset(dest, ch, count);
    return cudaSuccess;
} // cudaMemset

inline cudaError cudaMemsetAsync(void* dest, int ch, std::size_t count, cudaStream_t stream) { 
    return cudaMemset(dest, ch, count);
} // cudaMemsetAsync

inline void __syncthread(void) {}

// NVTX markers
inline void PUSH_RANGE(char const *name) {}
inline void POP_RANGE(void) {}

#define __global__
#define __device__
#define __host__
