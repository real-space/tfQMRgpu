#pragma once
// This file is part of tfQMRgpu under MIT-License

#ifndef   HAS_NO_CUDA

    #include <cuda.h>
    #ifdef    USE_NVTX
        #include "nvToolsExt.h"

        #define POP_RANGE() nvtxRangePop();
        #define PUSH_RANGE(name) \
        { \
            nvtxEventAttributes_t eventAttrib = {0}; \
            eventAttrib.version = NVTX_VERSION; \
            eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
            eventAttrib.colorType = NVTX_COLOR_ARGB; \
            eventAttrib.color = (uint32_t) rand(); \
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
            eventAttrib.message.ascii = name; \
            nvtxRangePushEx(&eventAttrib); \
        }

    #else  // USE_NVTX

        #define PUSH_RANGE(name)
        #define POP_RANGE()

    #endif // USE_NVTX

#else  // HAS_NO_CUDA

    #include "tfqmrgpu_cudaStubs.hxx" // replaces cuda.h

#endif // HAS_NO_CUDA

extern "C" {
    #include "tfqmrgpu.h" // C-interface for tfqmrgpu
} // extern "C"

// // mark device pointers == pointers to GPU memory
#define devPtr const __restrict__


#ifdef    _OPENMP
    #include <omp.h> // OpenMP threading library
    #define getTime omp_get_wtime // use the OpenMP internal timer
#else  // _OPENMP
    #include <ctime> // clock
    inline double getTime() { return double(clock())/double(CLOCKS_PER_SEC); }
    inline int omp_get_num_threads() { return 1; }
#endif // _OPENMP


#ifdef    __CUDA_ARCH__
    #define UNROLL _Pragma("unroll")
#else  // __CUDA_ARCH__
    #define UNROLL
#endif // __CUDA_ARCH__

typedef uint16_t colIndex_t;
