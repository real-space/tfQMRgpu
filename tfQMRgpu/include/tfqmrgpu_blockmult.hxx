#pragma once
// This file is part of tfQMRgpu under MIT-License

#include "tfqmrgpu_util.hxx" // check_launch_params

#define devPtr const __restrict__

    /* GPU kernel that accepts real and imaginary part split --> coalesced loads possible, uses shared memory */
    template <typename real_t, unsigned LM, unsigned LN, unsigned NA, typename double_t=real_t> // NA=number of accumulators
    void __global__ gemmNxNf( // GPU kernel, must be launched with <<< {nnzbY, 1, 1}, { LN, LM/NA, 1 } >>>
          real_t         (*devPtr Y)[2][LM][LN] // result, real and imaginary parts are split
        , real_t   const (*devPtr A)[2][LM][LM] // scalar argument, conjugated (A^t)
        , real_t   const (*devPtr X)[2][LM][LN] // vectorized argument (X)
        , uint32_t const (*devPtr pairs) // pairs[nPairs*2]
        , uint32_t const (*devPtr start) // start[nnzbY + 1] indices into pairs[s*2]
    ) {
        check_launch_params({gridDim.x, 1, 1}, {LN, LM/NA, 1}); // warning: gridDim.x == nnzbY is not checked
        int const jLN = threadIdx.x; // vectorization of X and Y over LN
        int const ilm = threadIdx.y;

//      full_debug_printf("# %s start with block=%i threads=%i %i\n", __func__, blockIdx.x, ilm, jLN);

        __shared__ real_t X_sk[2][LN], A_sk[2][LM]; // for real_t=double, LM=LN=32 this is 1 kiByte static shared memory

//      typedef double double_t; // we can perform the computation in double even if real_t==float
//      typedef real_t double_t; // default behaviour

        double_t Yij_re[NA], Yij_im[NA]; // complex accumulators
        UNROLL
        for(int ia = 0; ia < NA; ++ia) {
            Yij_re[ia] = 0; // initialize zero
            Yij_im[ia] = 0; // initialize zero
        } // ia

        auto const iYmat = blockIdx.x;
        for(auto ipair = start[iYmat]; ipair < start[iYmat + 1]; ++ipair) { // BSR-type loop
            auto const iAmat = pairs[ipair*2 + 0];
            auto const iXmat = pairs[ipair*2 + 1];

            // Block-times-Block-multiplication:
            //
            // equation: Y_{ij} += sum_k A_{ik} * X_{kj}
            //

            for(int kLM = 0; kLM < LM; ++kLM) {

                __syncthreads(); // all threads synchronize

                // coalesced loads from global memory into shared memory
                if (0 == ilm) {
                    UNROLL
                    for (int ri = 0; ri < 2; ++ri) {
                        if (jLN < LM)
                        A_sk[ri][jLN] = A[iAmat][ri][kLM][jLN]; // blocks of A are stored column-major
                        X_sk[ri][jLN] = X[iXmat][ri][kLM][jLN];
                    } // ri
                } // 0 == threadIdx.y

                __syncthreads(); // wait until shared memory is filled

                // load vectorized matrix element X_{kj} from shared memory into registers
                double_t const Xkj_re = X_sk[0][jLN];
                double_t const Xkj_im = X_sk[1][jLN];

                UNROLL
                for(int ia = 0; ia < NA; ++ia) {
                    // load scalar matrix element A_{ki} from shared memory into registers
                    auto const iLM = ilm*NA + ia;
                    double_t const Aik_re = A_sk[0][iLM];
                    double_t const Aik_im = A_sk[1][iLM];
 
//                  full_debug_printf("# %s block=%i threads=%i %i adds %g * %g for k=%i\n", __func__, blockIdx.x, iLM, jLN, Aik_re, Xkj_re, kLM);
// std::printf("# %s Y[%i][%i][%i] += %g * %g for k=%i\n", __func__, iYmat, iLM, jLN, Aik_re, Xkj_re, kLM); // real part only

                    // complex multiplication, 8 Flop
                    Yij_re[ia] += Aik_re * Xkj_re - Aik_im * Xkj_im; // Real part
                    Yij_im[ia] += Aik_re * Xkj_im + Aik_im * Xkj_re; // Imag part
                } // ia

            } // kLM

        } // ipair

        // store result to global memory
        UNROLL
        for(int ia = 0; ia < NA; ++ia) {
            auto const iLM = ilm*NA + ia;
            Y[iYmat][0][iLM][jLN] = Yij_re[ia];
            Y[iYmat][1][iLM][jLN] = Yij_im[ia];
// std::printf("# %s Y[%i][%i][%i]= %g\n", __func__, iYmat, iLM, jLN, Y[iYmat][0][iLM][jLN]); // real part only
        } // ia

    } // gemmNxNf
