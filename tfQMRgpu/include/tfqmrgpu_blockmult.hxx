#pragma once

#include "tfqmrgpu_util.hxx" // check_launch_params

#define devPtr const __restrict__

    /* GPU kernel that accepts real and imaginary part split --> coalesced loads possible, uses shared memory */
    template <typename real_t, int LM, int LN, int NA, typename double_t=real_t> // NA=number of accumulators
    void __global__ gemmNxNf( // GPU kernel, must be launched with <<< {nnzbY, 1, 1}, { LN, LM/NA, 1 } >>>
          real_t         (*devPtr Y)[2][LM][LN] // result, real and imaginary parts are split
        , real_t   const (*devPtr A)[2][LM][LM] // scalar argument, conjugated (A^t)
        , real_t   const (*devPtr X)[2][LM][LN] // vectorized argument (X)
        , uint32_t const (*devPtr pairs) // pairs[nPairs*2]
        , uint32_t const (*devPtr start) // start[nnzbY + 1] indices s into pairs[s*2]
    ) {
        check_launch_params({gridDim.x, 1, 1}, {LN, LM/NA, 1}); // warning: gridDim.x == nnzbY is not checked!
        int const jLN = threadIdx.x; // vectorization of X and Y over LN
        int const iLM = threadIdx.y;
//      int const ri = iLM & 0x1; // 0: real, 1: imaginary part (relevant only during the pre-loading)

//      full_debug_printf("# %s start with block=%i threads=%i %i\n", __func__, blockIdx.x, iLM, jLN);

        __shared__ real_t X_sk[2][LN], A_sk[2][LM]; // for real_t=double, LM=LN=32 this is 1 kiByte static shared memory

//      typedef double double_t; // we can perform the computation in double even if real_t==float
//      typedef real_t double_t; // default behaviour

        double_t Yre[NA], Yim[NA]; // accumulators
        UNROLL
        for(int ii = 0; ii < NA; ++ii) {
            Yre[ii] = 0;
            Yim[ii] = 0;
        } // ii

        auto const iYmat = blockIdx.x;
        for(auto ipair = start[iYmat]; ipair < start[iYmat + 1]; ++ipair) { // BSR-type loop
            auto const iAmat = pairs[ipair*2 + 0];
            auto const iXmat = pairs[ipair*2 + 1];

            for(int kLM = 0; kLM < LM; ++kLM) {

                __syncthreads(); // all threads synchronize

                // coalesced load from global memory into shared memory
                if (0 == iLM) {
                    UNROLL
                    for (int ri = 0; ri < 2; ++ri) {
                        if (jLN < LM)
                        A_sk[ri][jLN] = A[iAmat][ri][kLM][jLN];
                        X_sk[ri][jLN] = X[iXmat][ri][kLM][jLN];
                    } // ri
                } // 0 == threadIdx.y

                __syncthreads(); // wait until shared memory is filled

                // load vectorized matrix element X_{kj} from shared memory into registers
                double_t const Xre_j = X_sk[0][jLN];
                double_t const Xim_j = X_sk[1][jLN];

                UNROLL
                for(int ii = 0; ii < NA; ++ii) {
                    // load scalar matrix element A_{ki} from shared memory into registers
                    double_t const Are_i = A_sk[0][iLM*NA + ii];
                    double_t const Aim_i = A_sk[1][iLM*NA + ii];
 
//                  full_debug_printf("# %s block=%i threads=%i %i adds %g * %g for k=%i\n", __func__, blockIdx.x, iLM, jLN, Are_i, Xre_j, kLM);

                    // complex multiplication, 8 Flop
                    Yre[ii] += Are_i * Xre_j - Aim_i * Xim_j; // Real part
                    Yim[ii] += Are_i * Xim_j + Aim_i * Xre_j; // Imag part
                } // ii

            } // kLM

        } // ipair

        // store result to global memory
        UNROLL
        for(int ii = 0; ii < NA; ++ii) {
            Y[iYmat][0][iLM*NA + ii][jLN] = Yre[ii];
            Y[iYmat][1][iLM*NA + ii][jLN] = Yim[ii];
        } // ii

    } // gemmNxNf
