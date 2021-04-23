#pragma once

#include "tfqmrgpu_util.hxx" // check_launch_params

#define devPtr const __restrict__

    /* GPU kernel that accepts real and imaginary part split --> coalesced loads possible, use shared memory */
    template <typename real_t, int LM, int NA> // NA=number of accumulators
    void __global__ gemmNxNf( // GPU kernel, must be launched with <<< {ncmat, 1, 1}, { LM, LM/NA, 1 } >>>
          real_t       (*devPtr Y)[2][LM][LM] // result, real and imaginary parts are split
        , real_t const (*devPtr A)[2][LM][LM] // scalar argument, conjugated (A^t)
        , real_t const (*devPtr X)[2][LM][LM] // vectorized argument (X)
        , uint32_t const (*devPtr pairs) // pairs[nPairs*2]
        , uint32_t const (*devPtr start) // start[ncmat + 1] indices s into pairs[s*2]
    ) {
        check_launch_params({gridDim.x, 1, 1}, {LM, LM/NA, 1}); // warning: gridDim.x == ncmat is not checked!
        int const jLM = threadIdx.x;
        int const iLM = threadIdx.y;
        int const ri = iLM & 0x1; // 0: real, 1: imaginary part (relevant only during the pre-loading)

        __shared__ real_t A_sk[2][LM], X_sk[2][LM]; // for real_t=double, LM=32 this are 1024 Byte static shared memory 

        real_t Yre[NA], Yim[NA]; // accumulators
        UNROLL
        for(int ii = 0; ii < NA; ++ii) {
            Yre[ii] = 0;
            Yim[ii] = 0;
        } // ii

        int const icmat = blockIdx.x;
        for(int ipair = start[icmat]; ipair < start[icmat + 1]; ++ipair) { // BSR-type loop
            int const iAmat = pairs[ipair*2 + 0];
            int const iXmat = pairs[ipair*2 + 1];

            for(int k = 0; k < LM; ++k) {

                __syncthreads(); // all threads synchronize

                // four vectors of length LM do the pre-loading
                A_sk[ri][jLM] = A[iAmat][ri][k][jLM]; // coalesced load from global memory into shared memory
                X_sk[ri][jLM] = X[iXmat][ri][k][jLM]; // coalesced load from global memory into shared memory

                __syncthreads(); // all other LM*(LM-4) threads wait here

                // load vectorized matrix elements from shared memory into registers
                real_t const Xre_j = X_sk[0][jLM];
                real_t const Xim_j = X_sk[1][jLM];

                UNROLL
                for(int ii = 0; ii < NA; ++ii) {
                    // load scalar matrix elements from shared memory into registers
                    real_t const Are_i = A_sk[0][NA*iLM + ii];
                    real_t const Aim_i = A_sk[1][NA*iLM + ii];

                    // complex multiplication, 8 Flop
                    Yre[ii] += Are_i * Xre_j - Aim_i * Xim_j; // Real part
                    Yim[ii] += Are_i * Xim_j + Aim_i * Xre_j; // Imag part
                } // ii

            } // k

        } // ipair

        // store result to global memory
        UNROLL
        for(int ii = 0; ii < NA; ++ii) {
            Y[icmat][0][NA*iLM + ii][jLM] = Yre[ii];
            Y[icmat][1][NA*iLM + ii][jLM] = Yim[ii];
        } // ii

    } // gemmNxNf
