#pragma once

#include "tfqmrgpu_memWindow.h" // memWindow_t

    // FlopChar allows to distinguish single and double precision floating point operations as Gflop and GFlop
    template <typename T> inline char FlopChar         () { return '?'; }
    template <>           inline char FlopChar<int>    () { return 'i'; }
    template <>           inline char FlopChar<float>  () { return 'f'; }
    template <>           inline char FlopChar<double> () { return 'F'; }

    char constexpr IgnoreCase = 'a' - 'A'; // compare e.g. 'a' == (c | IgnoreCase)

	// Helper /////////////////////////////////////////////////////////////////////
	

    #define CCheck(err) __cudaSafeCall((err), __FILE__, __LINE__)
	inline void __cudaSafeCall(cudaError const err, char const *const file, int const line) {
#ifndef NDEBUG
		if (cudaSuccess != err) {
			printf("[ERROR] Cuda call in %s:%d failed, cudaErrorString= %s\n", file, line, cudaGetErrorString(err));
			exit(0);
		}
#endif // DEBUG
	} // __cudaSafeCall

#ifndef HAS_NO_CUDA
    inline void __device__ check_launch_params(dim3 const grid, dim3 const blk) {
#ifdef  DEBUG
        assert(grid.x == gridDim.x); assert(grid.y == gridDim.y); assert(grid.z == gridDim.z);
        assert(blk.x == blockDim.x); assert(blk.y == blockDim.y); assert(blk.z == blockDim.z);
#endif // DEBUG
    } // check_launch_params
#endif // HAS_CUDA

	// Memory management /////////////////////////////////////////////////////////
	template <typename T>
	void copy_data_to_gpu(T (*devPtr d), T const *const h, size_t const size=1, cudaStream_t const stream=0, char const *const name="") {
#ifdef DEBUGGPU
		printf("# transfer %lu x %.3f kByte from %p @host to %p @device %s\n", size, 1e-3*sizeof(T), h, d, name);
#endif // DEBUGGPU
        CCheck(cudaMemcpyAsync(d, h, size*sizeof(T), cudaMemcpyHostToDevice, stream));
	} // copy_data_to_gpu

	template <typename T>
	void get_data_from_gpu(T *const h, T const (*devPtr d), size_t const size=1, cudaStream_t const stream=0, char const *const name="") {
#ifdef DEBUGGPU
		printf("# transfer %lu x %.3f kByte from %p @device to %p @host %s\n", size, 1e-3*sizeof(T), d, h, name);
#endif // DEBUGGPU
		CCheck(cudaMemcpyAsync(h, d, size*sizeof(T), cudaMemcpyDeviceToHost, stream));
	} // get_data_from_gpu

    inline void tfqmrgpu_memAlign(char* &address) {
        size_t const a = size_t(address); // cast pointer into a size_t
        size_t const mask = (1ul << TFQMRGPU_MEMORY_ALIGNMENT) - 1;
        if (a & mask) { // probe the last e.g. 8 bits
            auto const aa = (((a >> TFQMRGPU_MEMORY_ALIGNMENT) + 1) << TFQMRGPU_MEMORY_ALIGNMENT);
            address = (char*)aa; // modify
        } // if
    } // align

    template <typename T> inline
    T* take_gpu_memory(
          char* &buffer // take GPU memory from this buffer instead of calling cudaMalloc or cudaMallocManaged
        , size_t const size=1 // how many elements of type T
        , memWindow_t* win=nullptr // optional: store memory window information
        , char const *const win_name="" // for debugging
    ) {
        size_t const total_size_inByte = size*sizeof(T);
        tfqmrgpu_memAlign(buffer);
        auto const d = (T*)buffer; // place the object here
        if (nullptr != win) {
            win->offset = size_t((char*)d);
            win->length = total_size_inByte;
#ifdef DEBUGGPU
            printf("%s:  new window [%p, %p) %s\n", __func__, (char*)win->offset, (char*)(win->offset + win->length), win_name);
            fflush(stdout);
#endif // DEBUGGPU
        } // win
        buffer += total_size_inByte;
        tfqmrgpu_memAlign(buffer);
        return d;
    } // take_gpu_memory

    // clear on gpu
    template <typename T> inline
    void clear_on_gpu(
          T (*devPtr d) // pointer in GPU memory
        , size_t const size=1 // how many elements of type T
        , cudaStream_t const stream=0
    ) {
        CCheck(cudaMemset(d, 0x0, size*sizeof(T)));
//      CCheck(cudaMemsetAsync(d, 0x0, size*sizeof(T), stream)); // also allowed?
    } // clear_on_gpu

    /////////////////////////////////////// debug helpers //////////////////////////////////
#ifdef  DEBUGGPU    
    template <typename T, int Dim>
    void __global__ print_array( // GPU kernel, must be launched with <<< 1, 1 >>>
          T const (*devPtr array)[Dim] // any array[][Dim]
        , int const num
        , char const name // only a single character
    ) {
#ifndef HAS_NO_CUDA
        if (0 == threadIdx.x)
#endif // HAS_CUDA          
        {
            for(auto i = 0; i < num; ++i) {
                printf("# %c[%d] ", name, i); 
                for(auto d = 0; d < Dim; ++d) {
                    printf(" %f", array[i][d]); 
                } // d
                printf(" \n"); 
            } // i
        } // master
    } // print_array
#endif // DEBUGGPU
    
    // absolute square of a complex number computed in double
    inline __host__ __device__ double abs2(double const zRe, double const zIm) { return zRe*zRe + zIm*zIm; }
