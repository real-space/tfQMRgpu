#pragma once

#ifndef HAS_NO_CUDA
    #include <cuda.h>
    #ifdef USE_NVTX
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

    #else

        #define PUSH_RANGE(name)
        #define POP_RANGE()

    #endif
#else

    #include "tfqmrgpu_cudaStubs.hxx" // replaces cuda.h

#endif

extern "C" {
    #include "tfqmrgpu.h" // C-interface for tfqmrgpu
} // extern "C"

#define devPtr const __restrict__

// #include <cmath>
// #include <cstdio>
// #include <cstdlib>
// #include <cstring>
// #include <ctime>

#ifdef _MPI
	#define getTime MPI_Wtime // use the MPI internal timer
#else
// #ifdef _OMP
	#include <omp.h> // OpenMP threading
	#define getTime omp_get_wtime // use the OpenMP internal timer
// #else
// 	#define getTime cpu_time
// #endif
#endif

#ifdef __CUDA_ARCH__
    #define UNROLL _Pragma("unroll")
#else
    #define UNROLL
#endif

