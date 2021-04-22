#pragma once

#ifndef HAS_NO_CUDA
    #include <curand.h> // random number generator for CUDA
#endif // HAS_CUDA

#include <cmath> // std::abs

#include "tfqmrgpu.hxx"           // includes cuda.h and tfqmrgpu.h
#include "tfqmrgpu_util.hxx"      // common utilities: FlopChar, copy_data_to_gpu, get_data_from_gpu, abs2
#include "bsr.hxx"                // bsr_t, find_in_array
#include "tfqmrgpu_memWindow.h"   // memWindow_t
#include "tfqmrgpu_plan.hxx"      // bsrsv_plan_t
#include "tfqmrgpu_handle.hxx"    // tfq_handle_t

#ifdef DEBUG
    #define debug_printf(...) std::printf(__VA_ARGS__)
#else  // DEBUG
    #define debug_printf(...)
#endif // DEBUG

namespace tfqmrgpu {
   
    // tfQMR decision sections ////////////////////////////////////////////////////////////////////////
    
#define EPSILON 2.5e-308


    template <typename real_t, int LM>
    void __global__ tfQMRdec35( // GPU kernel, must be launched with <<< nCols, LM >>>
          int          (*devPtr status)[LM] // tfQMR status
        , real_t       (*devPtr rho)[2][LM] // rho  (inout)
        , real_t       (*devPtr bet)[2][LM] // beta (out)
        , double const (*devPtr z35)[2][LM] // inner product v3.v5
        , uint32_t const nCols
    ) {
#ifndef HAS_NO_CUDA      
        check_launch_params( { nCols, 1, 1 }, { LM, 1, 1 } );
        { int const i = blockIdx.x;
            { int const j = threadIdx.x;
#else  // HAS_CUDA
        for (uint32_t i = 0; i < nCols; ++i) {
            for (int j = 0; j < LM; ++j) {
#endif // HAS_CUDA
                double const rho_Re = double(rho[i][0][j]),
                             rho_Im = double(rho[i][1][j]);
                double const abs2rho = abs2(rho_Re, rho_Im);
                double const z35_Re = z35[i][0][j], 
                             z35_Im = z35[i][1][j]; // z35 == v3.v5
                double const abs2z35 = abs2(z35_Re, z35_Im);
                
                if ((abs2z35 < EPSILON) || (abs2rho < EPSILON)) { 
                    status[i][j] = -1; // severe breakdown in dec35
                    bet[i][0][j] = 0; bet[i][1][j] = 0; // beta := 0
                    rho[i][0][j] = 0; rho[i][1][j] = 0; // rho  := 0
                } else {
                    double const rho_denom = 1./abs2rho;
                    // beta := z35 / rho, complex divison
                    bet[i][0][j] = real_t((z35_Re*rho_Re + z35_Im*rho_Im) * rho_denom);
                    bet[i][1][j] = real_t((z35_Im*rho_Re - z35_Re*rho_Im) * rho_denom);
                    // rho := z35
                    rho[i][0][j] = z35_Re; rho[i][1][j] = z35_Im;
                }
            } // threads j
        } // blocks i 
    } // dec35

    template <typename real_t, int LM>
    void __global__ tfQMRdec34( // GPU kernel, must be launched with <<< nCols, LM >>>
          int          (*devPtr status)[LM] // tfQMR status
        , real_t       (*devPtr c67)[2][LM] // c67  (out)
        , real_t       (*devPtr alf)[2][LM] // alfa (out)
        , real_t const (*devPtr rho)[2][LM] // rho
        , real_t const (*devPtr eta)[2][LM] // eta
        , double const (*devPtr z34)[2][LM] // inner product v3.v4
        , double const (*devPtr var)[LM] // var
        , uint32_t const nCols
    ) {
#ifndef HAS_NO_CUDA      
        check_launch_params( { nCols, 1, 1 }, { LM, 1, 1 } );
        { int const i = blockIdx.x;
            { int const j = threadIdx.x;
#else  // HAS_CUDA
        for (uint32_t i = 0; i < nCols; ++i) {
            for (int j = 0; j < LM; ++j) {
#endif // HAS_CUDA
                double const rho_Re = double(rho[i][0][j]), rho_Im = double(rho[i][1][j]); // load rho
                double const abs2rho = abs2(rho_Re, rho_Im);
                double const z34_Re = z34[i][0][j], z34_Im = z34[i][1][j]; // load z34
                double const abs2z34 = abs2(z34_Re, z34_Im);

                if ((abs2z34 < EPSILON) || (abs2rho < EPSILON)) { 
                    status[i][j] = -2; // severe breakdown in dec34
                    alf[i][0][j] = 0; alf[i][1][j] = 0; // alfa := 0
                    c67[i][0][j] = 0; c67[i][1][j] = 0; // c67 := 0
                } else {
                    double const eta_Re = double(eta[i][0][j]), eta_Im = double(eta[i][1][j]); // load eta

                    double const z34_denom = -1./abs2z34;
                    // alfa := - rho / z34, complex divison
                    alf[i][0][j] = real_t((rho_Re*z34_Re + rho_Im*z34_Im) * z34_denom);
                    alf[i][1][j] = real_t((rho_Im*z34_Re - rho_Re*z34_Im) * z34_denom);

                    double const vrho_denom = var[i][j]/abs2rho;
                    // compute tmp := var * eta / rho, complex divison
                    double const tmp_Re = (eta_Re*rho_Re + eta_Im*rho_Im) * vrho_denom;
                    double const tmp_Im = (eta_Im*rho_Re - eta_Re*rho_Im) * vrho_denom;

                    // c67 := z34 * (var * eta / rho) = z34 * tmp, complex multiplication
                    c67[i][0][j] = real_t(z34_Re*tmp_Re - z34_Im*tmp_Im);
                    c67[i][1][j] = real_t(z34_Im*tmp_Re + z34_Re*tmp_Im);
                }
            } // threads j
        } // blocks i 
    } // dec34
    
    
    template <typename real_t, int LM>
    void __global__ tfQMRdecT( // GPU kernel, must be launched with <<< nCols, LM >>>
          int          (*devPtr status)[LM] // tfQMR status
        , real_t       (*devPtr c67)[2][LM] // c67 (optional out)
        , real_t       (*devPtr eta)[2][LM] // eta (out)
        , double       (*devPtr var)   [LM] // var (out)
        , double       (*devPtr tau)   [LM] // tau (inout)
        , real_t const (*devPtr alf)[2][LM] // alfa
        , double const (*devPtr d55)[1][LM] // |v5|
        , uint32_t const nCols
    ) {
#ifndef HAS_NO_CUDA
        check_launch_params( { nCols, 1, 1 }, { LM, 1, 1 } );
        { int const i = blockIdx.x;
            { int const j = threadIdx.x;
#else  // HAS_CUDA
        for (unsigned i = 0; i < nCols; ++i) {
            for (unsigned j = 0; j < LM; ++j) {
#endif // HAS_CUDA
                double cosi;
                real_t r67;
                double const Tau = tau[i][j]; // load
                if (std::abs(Tau) > EPSILON) {
                    double const D55 = d55[i][0][j]; // load
                    double const Var = D55 / Tau;
                    cosi = 1./(1. + Var);
                    var[i][j] = Var; // store
                    tau[i][j] = D55 * cosi; // store
                    r67 = real_t(Var * cosi);
                } else {
                    status[i][j] = -3; // early convergence or breakdown(stagnation)
                    cosi = 0;
                    var[i][j] = 0; // store
                    tau[i][j] = 0; // store
                    r67 = 1;
                }

                if (status[i][j] < 0) { 
                    eta[i][0][j] = 0; eta[i][1][j] = 0;
                } else {
                    eta[i][0][j] = real_t(-cosi * double(alf[i][0][j])); 
                    eta[i][1][j] = real_t(-cosi * double(alf[i][1][j])); 
                }

                if (nullptr != c67) { c67[i][0][j] = r67; c67[i][1][j] = 0; } // no imaginary part given
            } // threads j
        } // blocks i 
    } // decT

    
    
    // basis linear algebra kernels ////////////////////////////////////////////////////////////////////////

    
    template <typename real_in_t, typename real_out_t>
    void __global__ convert_precision( // GPU kernel, must be launched with <<< { any, 1, 1 }, { any, 1, 1 } >>>
          real_out_t      (*devPtr out) // result, out
        , real_in_t const (*devPtr  in) // input,  in
        , size_t const n // number of elements to be converted
        , double const scaling = 1. // global scaling factor
    ) {
#ifndef HAS_NO_CUDA
        check_launch_params( {gridDim.x, 1, 1}, {blockDim.x, 1, 1} );
        for(size_t i = blockIdx.x*blockDim.x + threadIdx.x; 
            i < n; i += gridDim.x*blockDim.x) // grid stride loop over blocks
#else  // HAS_CUDA
        for(size_t i = 0; i < n; ++i)
#endif // HAS_CUDA
        {
            auto const tmp = scaling * in[i];
            out[i] = real_out_t(tmp);
        } // i

    } // convert_precision

    template <typename T>
    inline void __device__ gpu_swap(T &a, T &b) { T const c = a; a = b; b = c; }

    template <typename real_t>
    void __global__ transpose_blocks_kernel( // GPU kernel, must be launched with <<< {any,1,1}, {nCols,nRows,1}, 2*nRows*nCols*sizeof(real_t) >>>
          real_t (*devPtr array) // input and result
        , size_t const nnzb // number of nonzero blocks
        , double const scal_real // global scaling factor for the real part
        , double const scal_imag // global scaling factor for the imaginary part
        , tfqmrgpuDataLayout_t const layout_in  // data layout input
        , tfqmrgpuDataLayout_t const layout_out // data layout output
        , char const trans // transpositions, allowed are {'n','N','t','T'}
        , uint16_t const nCols_per_block=0
        , uint16_t const nRows_per_block=0
    ) {
#ifndef HAS_NO_CUDA
        // number of columns and rows is transmitted via the launch parameters
        auto const nCols = blockDim.x;
        auto const nRows = blockDim.y;
        check_launch_params( {gridDim.x, 1, 1}, {nCols, nRows, 1} );
#else  // HAS_CUDA
        uint32_t const nCols = nCols_per_block; assert(nCols > 0);
        uint32_t const nRows = nRows_per_block; assert(nRows > 0);
#endif // HAS_CUDA
        // TFQMRGPU_LAYOUT_RRRRIIII internal format: blocks of real parts, blocks of imaginary parts
        // TFQMRGPU_LAYOUT_RRIIRRII intermediate format: vectors separated between real and imag
        // TFQMRGPU_LAYOUT_RIRIRIRI Fortran complex(4 or 8), C++ std::complex<real_t>

        uint32_t oNi, oNj, oNc; // output
        switch (layout_out) {
            case TFQMRGPU_LAYOUT_RRRRIIII: oNc = nRows*nCols; oNi = nCols; oNj = 1; break;
            case TFQMRGPU_LAYOUT_RRIIRRII: oNi =     2*nCols; oNc = nCols; oNj = 1; break;
            case TFQMRGPU_LAYOUT_RIRIRIRI: oNi = nCols*    2; oNj =     2; oNc = 1; break;
        } // switch layout_out

        uint32_t iNi, iNj, iNc; // input
        switch (layout_in) {
            case TFQMRGPU_LAYOUT_RRRRIIII: iNc = nRows*nCols; iNi = nCols; iNj = 1; break;
            case TFQMRGPU_LAYOUT_RRIIRRII: iNi =     2*nCols; iNc = nCols; iNj = 1; break;
            case TFQMRGPU_LAYOUT_RIRIRIRI: iNi = nCols*    2; iNj =     2; iNc = 1; break;
        } // switch layout_in

        double scaler[2] = {scal_real, scal_imag}; // should scaler be of real_t?

        // 'n' is non-transposed, 't' is transposed
        if ('t' == (trans | IgnoreCase)) { // transposition
            gpu_swap(iNi, iNj);
        } else {
           assert('n' == (trans | IgnoreCase)); // other characters are not permitted
           // complex conjugation can be done by a negative sign to scal_imag
        }

        size_t const blockSize = 2*nRows*nCols;

#ifndef HAS_NO_CUDA
        auto const j = threadIdx.x; // out col index
        auto const i = threadIdx.y; // out row index

        extern __shared__ char shared_buffer[];
        auto const temp = (real_t*) shared_buffer; // cast pointer
 
        for(auto inzb = blockIdx.x; inzb < nnzb; inzb += gridDim.x) { // grid stride loop over non-zero blocks
            size_t const boff = inzb*blockSize; // block offset
            for(int c = 0; c < 2; ++c) { // loop over real and imaginary part
                auto const iin  = iNi*i + iNj*j + iNc*c;
                auto const iout = oNi*i + oNj*j + oNc*c;
                // store in shared memory
                temp[iout] = scaler[c] * array[boff + iin];
            } // c
            __syncthreads(); // wait unit the entire block is in the shared memory
            for(int c = 0; c < 2; ++c) { // loop over real and imaginary part
                auto const iout = (c*nRows + i)*nCols + j;
                // coalesced reads from shared memory and coalesced writes to global memory
                array[boff + iout] = temp[iout];
            } // c
        } // inzb
#else  // HAS_CUDA
        for(size_t inzb = 0; inzb < nnzb; ++inzb) { // parallel
            size_t const boff = inzb*blockSize; // block offset
            std::vector<real_t> temp(blockSize);
            for(uint32_t i = 0; i < nRows; ++i) {
                for(uint32_t j = 0; j < nCols; ++j) {
                    for(int c = 0; c < 2; ++c) { // loop over real and imaginary part
                        auto const iin  = iNi*i + iNj*j + iNc*c;
                        auto const iout = oNi*i + oNj*j + oNc*c;
                        temp[iout] = scaler[c] * array[boff + iin];
                    } // c
                } // j
            } // i
            for(uint32_t i = 0; i < nRows; ++i) {
                for(uint32_t j = 0; j < nCols; ++j) {
                    for(int c = 0; c < 2; ++c) { // loop over real and imaginary part
                        auto const iout = (c*nRows + i)*nCols + j;
                        array[boff + iout] = temp[iout];
                    } // c
                } // j
            } // i
        } // inzb
#endif // HAS_CUDA
    } // transpose_blocks_kernel

    void transpose_blocks( // driver
          char (*devPtr ptr)
        , size_t const nnzb // number of non-zero blocks
        , tfqmrgpuDataLayout_t const l_in  // data layout input
        , tfqmrgpuDataLayout_t const l_out // data layout output
        , uint32_t const nRows // Rows per block
        , uint32_t const nCols // Columns per block
        , char const doublePrecision='z' // should be 'z' or 'Z' for double and 'c' or 'C' for float
        , double const scal_imag=1 // use -1 for complex conjugation
        , char const Trans='n' // should be 'n' or 'N' or 't' or 'T'
        , cudaStream_t const streamId=0
    ) {
        if ('z' == (doublePrecision | IgnoreCase)) {
//          assert(nnzb * 2 * nRows * nCols * sizeof(double) == size);
            transpose_blocks_kernel<double>
#ifndef HAS_NO_CUDA
                <<<nnzb, {nCols, nRows, 1}, 2*nRows*nCols*sizeof(double), streamId>>>
#endif // HAS_CUDA
                ((double*) ptr, nnzb, 1, scal_imag, l_in, l_out, Trans, nCols, nRows);
        } else {
  //        assert(nnzb * 2 * nRows * nCols * sizeof(float)  == size);
            transpose_blocks_kernel<float>
#ifndef HAS_NO_CUDA
                <<<nnzb, {nCols, nRows, 1}, 2*nRows*nCols*sizeof(float) , streamId>>>
#endif // HAS_CUDA
                ((float *) ptr, nnzb, 1, scal_imag, l_in, l_out, Trans, nCols, nRows); 
        }
    } // transpose_blocks

    
#ifndef HAS_NO_CUDA
    template <typename real_t, int LM>
    void __global__ add_RHS_kernel( // GPU kernel, must be launched with <<< {any}, { LM, LM, 1 } >>>
          real_t       (*devPtr v)[2][LM][LM] // result, v[nnzv][2][LM][LM]
        , real_t const (*devPtr b)[2][LM][LM] // input,  b[nnzb][Re:Im][LM][LM]
        , real_t const scal // global scaling factor, no imaginary part
        , uint32_t const (*devPtr subset) // subset index list[nnzb]
        , uint32_t const nnzb // number of nonzero blocks in B
    ) {
        check_launch_params( gridDim, { LM, LM, 1 } );
        int const j = threadIdx.x, i = threadIdx.y;

        __shared__ int inzv; // ToDo: check if we need it to be so complicated
        for(int inzb = blockIdx.x; inzb < nnzb; inzb += gridDim.x) { // grid stride loop over blocks
            __syncthreads();
            if ((0 == j) && (0 == i)) inzv = subset[inzb]; // block master loads index
            __syncthreads();
            v[inzv][0][i][j] += scal*b[inzb][0][i][j];
            v[inzv][1][i][j] += scal*b[inzb][1][i][j];
        } // inzb
    } // add_RHS_kernel
#endif // HAS_CUDA

    template <typename real_t, int LM>
    void __host__ add_RHS(
          real_t       (*devPtr v)[2][LM][LM] // result, v[nnzv][2][LM][LM]
        , real_t const (*devPtr b)[2][LM][LM] // input,  b[nnzb][Re:Im][LM][LM]
        , real_t const scal // global scaling factor, no imaginary part
        , uint32_t const (*devPtr subset) // subset index list[nnzb]
        , uint32_t const nnzb // number of nonzero blocks in B
        , cudaStream_t const streamId=0
    ) {
#ifndef HAS_NO_CUDA
        add_RHS_kernel<real_t,LM> <<< nnzb, { LM, LM, 1 }, 0, streamId >>> (v, b, scal, subset, nnzb);
#else  // HAS_CUDA
        for(uint32_t inzb = 0; inzb < nnzb; ++inzb) {
            auto const inzv = subset[inzb]; // load index
            for(int cij = 0; cij < 2*LM*LM; ++cij) {
                v[inzv][0][0][cij] += scal*b[inzb][0][0][cij];
            } // cij
        } // inzb
#endif // HAS_CUDA
    } // add_RHS


    // linear algebra functions ////////////////////////////////////////////////////////////////////////////////////////

#ifndef HAS_NO_CUDA

    template <typename real_t, int LM, int D2>
    void __global__ col_inner( // GPU kernel, must be launched with <<< {anypowerof2}, { LM, 1, 1 } >>>
          double       (*devPtr dots)[D2][LM] // result, dots[2^p*nCols][D2][LM], D2 is 2==Re:Im for v*w and 1 for norm |v|^2
        , real_t const (*devPtr v)[2][LM][LM] // input, v[nnz][Re:Im][LM][LM]
        , float  const (*devPtr w)[2][LM][LM] // input, w[nnz][Re:Im][LM][LM], only read if D2==2, always float
        , uint16_t const (*devPtr ColInd) // column index
        , uint32_t const nnz // number of nonzero blocks
        , uint32_t const nCols // number of block columns
    ) {
        check_launch_params( {gridDim.x, 1, 1}, { LM, 1, 1 } );
        int const j = threadIdx.x; // vectorization
        int const iput = blockIdx.x;

        // dots must be set to zero before calling this kernel

        for(int inz = blockIdx.x; inz < nnz; inz += gridDim.x) { // grid stride loop over blocks
            auto const icol = ColInd[inz];
            double dr{0}, di{0};
            for(int k = 0; k < LM; ++k) {
                double const vRe = v[inz][0][k][j], 
                             vIm = v[inz][1][k][j]; // coalesced loads
                if (2 == D2) {
                    // inner product between two different vectors v and w, unconjugated
                    double const wRe = w[inz][0][k][j], 
                                 wIm = w[inz][1][k][j]; // coalesced loads
                    dr += vRe*wRe - vIm*wIm; // complex multiplication
                    di += vRe*wIm + vIm*wRe;
                } else {
                    // square norm of only one vector v, ignore w
                    dr += abs2(vRe, vIm); // square norm of v
                    // no need to compute an imaginary part
                } // D2
            } // k

            // now store
            dots[iput*nCols + icol][0][j] = dr; // no race condition here
            if (2 == D2) {
                dots[iput*nCols + icol][1][j] = di; // no race condition here
            } // D2
           
        } // inz

    } // col_inner

    template <typename real_t, int LM, int D2>
    void __global__ col_reduction( // GPU kernel, must be launched with <<< {nCols, 2^(p-1), 1}, { LM, 1, D2 } >>>
          double (*devPtr a)[D2][LM] // in/out, a[2^p*nCols][D2][LM], D2 is 2==Re:Im for v*w and 1 for norm |v|^2
        , uint32_t const nCols // number of block columns
    ) {
        check_launch_params( {nCols, gridDim.y, 1}, { LM, 1, D2 } );

        int const j = threadIdx.x; // vectorization
        int const ri = threadIdx.z; // real/imag part
        int const icol = blockIdx.x;

        int const iput = blockIdx.y;       // in [0, 2^(p-1)-1]
        int const iget = iput + gridDim.y; // in [2^(p-1), 2^p-1]

        a[iput*nCols + icol][ri][j] += a[iget*nCols + icol][ri][j];

    } // col_reduction

#endif // HAS_CUDA

    template <typename real_t, int LM> inline
    double __host__ dotp(
          double       (*devPtr a)[2][LM]     // result, a[2^p*nCols][Re:Im][LM]
        , real_t const (*devPtr x)[2][LM][LM] // input,  x[nnz][Re:Im][LM][LM]
        , float  const (*devPtr y)[2][LM][LM] // input,  y[nnz][Re:Im][LM][LM], always float (v3)
        , uint16_t const (*devPtr ColInd) // column index
        , uint32_t const nnz // number of nonzero blocks
        , uint32_t const nCols // number of columns
        , uint32_t const p2 // number of reduction levels
        , cudaStream_t const streamId=0
    ) {
        int constexpr D2 = 2;
#ifndef HAS_NO_CUDA
        uint32_t const np2 = (1 << p2); // 2^p
        clear_on_gpu<double[D2][LM]>(a, np2*nCols, streamId);
        col_inner <real_t, LM, D2> <<< np2, LM, 0, streamId >>> (a, x, y, ColInd, nnz, nCols);
        for(uint32_t np = np2 >> 1; np > 0; np >>= 1) { // reduce from 2*np to np
            col_reduction <double,LM,D2> <<< { nCols, np, 1 }, { LM, 1, D2 }, 0, streamId >>> (a, nCols);
        } // level
#else  // HAS_CUDA
        for (uint32_t icj = 0; icj < nCols*D2*LM; ++icj) a[0][0][icj] = 0; // init
        for (uint32_t inz = 0; inz < nnz; ++inz) {
            auto const icol = ColInd[inz];
            for(int j = 0; j < LM; ++j) { // vectorized
                double dr{0}, di{0};
                for(int k = 0; k < LM; ++k) { // contraction index
                    dr += x[inz][0][k][j]*y[inz][0][k][j] - x[inz][1][k][j]*y[inz][1][k][j]; // complex multiplication
                    di += x[inz][0][k][j]*y[inz][1][k][j] + x[inz][1][k][j]*y[inz][0][k][j];
                } // k
                a[icol][0][j] += dr;
                a[icol][1][j] += di;
            } // j
        } // inz
#endif // HAS_CUDA
        return nnz*4.*D2*LM*LM; // returns the number of Flops
    } // dotp = <x|y>

    template <typename real_t, int LM> inline
    double __host__ nrm2(
          double       (*devPtr a)[1][LM]     // result, a[2^p*nCols][1][LM]
        , real_t const (*devPtr x)[2][LM][LM] // input,  x[nnz][Re:Im][LM][LM]
        , uint16_t const (*devPtr ColInd) // column index
        , uint32_t const nnz // number of nonzero blocks
        , uint32_t const nCols // number of columns
        , uint32_t const p2 // number of reduction levels
        , cudaStream_t const streamId=0
    ) {
        int constexpr D2 = 1;
#ifndef HAS_NO_CUDA
        unsigned const np2 = (1 << p2); // 2^p
        clear_on_gpu<double[D2][LM]>(a, np2*nCols, streamId);
        col_inner <real_t, LM, D2> <<< np2, LM, 0, streamId >>> (a, x, 0x0, ColInd, nnz, nCols);
        for(unsigned np = np2 >> 1; np > 0; np >>= 1) { // reduce from 2*np to np
            col_reduction <double,LM,D2> <<< { nCols, np, 1 }, { LM, 1, D2 }, 0, streamId >>> (a, nCols);
        } // level
#else  // HAS_CUDA
        for (uint32_t icj = 0; icj < nCols*D2*LM; ++icj) a[0][0][icj] = 0; // init
        for (uint32_t inz = 0; inz < nnz; ++inz) {
            auto const icol = ColInd[inz];
            for(int j = 0; j < LM; ++j) { // vectorized
                double dr{0};
                for(int k = 0; k < LM; ++k) { // contraction index
                    dr += abs2(x[inz][0][k][j], x[inz][1][k][j]); // square norm of x
                } // k
                a[icol][0][j] += dr;
            } // j
        } // inz
#endif // HAS_CUDA
        return nnz*4.*D2*LM*LM; // returns the number of Flops
    } // nrm2 = <x|x>


    template <typename real_t, int LM, bool ScaleX>
    void __global__ col_axpay( // GPU kernel, must be launched with <<< {any, 1, 1}, { LM, 1, 1 } >>>
          real_t       (*devPtr y)[2][LM][LM] // in/out, y[nnz][Re:Im][LM][LM]
        , real_t const (*devPtr x)[2][LM][LM] // input,  x[nnz][Re:Im][LM][LM]
        , real_t const (*devPtr a)[2][LM]     // input,  a[nCols][Re:Im]  [LM]
        , uint16_t const (*devPtr ColInd) // column index
        , uint32_t const nnz // number of nonzero blocks
    ) { //
#ifndef HAS_NO_CUDA
        check_launch_params( {gridDim.x, 1, 1}, { LM, 1, 1 } );

        int const j = threadIdx.x; // vectorization
        for(auto inz = blockIdx.x; inz < nnz; inz += gridDim.x) { // grid stride loop over blocks
#else  // HAS_CUDA
        for(uint32_t inz = 0; inz < nnz; ++inz)
          for(int j = 0; j < LM; ++j) {
#endif // HAS_CUDA
            auto const icol = ColInd[inz];
            real_t const aRe = a[icol][0][j], 
                        aIm = a[icol][1][j];
            for(int i = 0; i < LM; ++i) {
                real_t const xRe = x[inz][0][i][j],
                            xIm = x[inz][1][i][j];
                real_t const yRe = y[inz][0][i][j], 
                            yIm = y[inz][1][i][j];
                if (ScaleX) {
                    // axpy: scale the x-vector with a and add to y
                    y[inz][0][i][j] = aRe*xRe - aIm*xIm + yRe;
                    y[inz][1][i][j] = aIm*xRe + aRe*xIm + yIm;
                } else {
                    // xpay: re-scale the y-vector with a and add x
                    y[inz][0][i][j] = xRe + aRe*yRe - aIm*yIm;
                    y[inz][1][i][j] = xIm + aIm*yRe + aRe*yIm;
                } // axpy or xpay
            } // i

        } // inz
    } // col_axpay

    template <typename real_t, int LM> inline
    double __host__ axpy(
          real_t       (*devPtr y)[2][LM][LM] // in/out, y[nnz][Re:Im][LM][LM]
        , real_t const (*devPtr x)[2][LM][LM] // input,  x[nnz][Re:Im][LM][LM]
        , real_t const (*devPtr a)[2][LM]     // input,  a[nCols][Re:Im]  [LM]
        , uint16_t const (*devPtr ColInd) // column index
        , uint32_t const nnz // number of nonzero blocks
        , cudaStream_t const streamId=0
    ) {
        col_axpay <real_t, LM, true >
#ifndef HAS_NO_CUDA
            <<<nnz, LM, 0, streamId>>>
#endif // HAS_CUDA
            (y, x, a, ColInd, nnz);

        return nnz*8.*LM*LM; // returns the number of Flops
    } // axpy   y := a*x + y

    template <typename real_t, int LM> inline
    double __host__ xpay(
          real_t       (*devPtr y)[2][LM][LM] // in/out, y[nnz][Re:Im][LM][LM]
        , real_t const (*devPtr a)[2][LM]     // input,  a[nCols][Re:Im]  [LM]
        , real_t const (*devPtr x)[2][LM][LM] // input,  x[nnz][Re:Im][LM][LM]
        , uint16_t const (*devPtr ColInd) // column index
        , uint32_t const nnz // number of nonzero blocks
        , cudaStream_t const streamId=0
    ) {
        col_axpay <real_t, LM, false>
#ifndef HAS_NO_CUDA
            <<<nnz, LM, 0, streamId>>>
#endif // HAS_CUDA
            (y, x, a, ColInd, nnz);

        return nnz*8.*LM*LM; // returns the number of Flops
    } // xpay   y := x + a*y


    // basis linear algebra level 3 kernels ////////////////////////////////////////////////////////////////////////
    
#ifndef HAS_NO_CUDA    
    template <typename real_t, int LM>
    void __global__ set_complex_value_kernel(
          real_t (*devPtr array)[2][LM] // 1D launch with correct size
        , real_t const re
        , real_t const im=0
    ) {
        array[blockIdx.x][0][threadIdx.x] = re;
        array[blockIdx.x][1][threadIdx.x] = im;
    } // set_complex_value_kernel
#endif // HAS_CUDA

    template <typename real_t, int LM>
    void __host__ set_complex_value(
          real_t (*devPtr array)[2][LM] // array[nblocks][2][LM]
        , uint32_t const nblocks
        , real_t const re
        , real_t const im=0
        , cudaStream_t const streamId=0
    ) {
#ifndef HAS_NO_CUDA
        set_complex_value_kernel<real_t,LM> <<< nblocks, LM, 0, streamId >>> (array, re, im); // needs to run every time!
#else  // HAS_CUDA
        for(uint32_t iblock = 0; iblock < nblocks; ++iblock) {
            for(int j = 0; j < LM; ++j) {
                array[iblock][0][j] = re;
                array[iblock][1][j] = im;
            } // j
        } // iblock
#endif // HAS_CUDA
    } // set_complex_value

    inline tfqmrgpuStatus_t create_random_numbers(
          float (*devPtr v3)
        , size_t length // number of floats in v3
        , cudaStream_t const streamId=0
        , unsigned long long seed=1234ull // random seed
    ) {
#ifndef HAS_NO_CUDA
        tfqmrgpuStatus_t stat{0};
        curandGenerator_t gen;
        #define CURAND_CALL(x) { stat = (x); \
            if (0 != stat) return TFQMRGPU_UNDOCUMENTED_ERROR + __LINE__ * TFQMRGPU_CODE_LINE; }
        /* Create pseudo-random number generator */
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        /* Set seed */
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
        /* Generate floats on device */
        CURAND_CALL(curandGenerateUniform(gen, v3, length));
        /* Cleanup */
        CURAND_CALL(curandDestroyGenerator(gen));
        #undef  CURAND_CALL
#else  // HAS_CUDA
        auto const denom = 1./float(RAND_MAX);
        for(size_t i = 0; i < length; ++i) {
            v3[i] = rand()*denom;
        } // i
#endif // HAS_CUDA
        return TFQMRGPU_STATUS_SUCCESS;    
    } // create_random_numbers

    
    template <typename plan_t>
    inline void transfer_index_lists(
          cudaStream_t const streamId
        , plan_t *p
    ) {
        // here we transfer the integer vectors that are filled during the analysis step.
        #define transfer(INTVEC) \
                debug_printf(#INTVEC" has pointer %p @host\n", p->INTVEC.data()); \
                debug_printf(#INTVEC" has offset %.3f kBytes from pBuffer %p @device\n", (p->INTVEC##win.offset)*.001, p->pBuffer); \
                debug_printf(#INTVEC" transfer to %p @device\n", p->pBuffer + p->INTVEC##win.offset); \
                assert(p->INTVEC.size()*sizeof(p->INTVEC.value_type) == p->INTVEC##win.length); \
                copy_data_to_gpu<char>(p->pBuffer + p->INTVEC##win.offset, \
                    (char*)p->INTVEC.data(), p->INTVEC##win.length, streamId)
//         transfer(pairs);  // moved into action_t
//         transfer(starts); // moved into action_t
//         transfer(subset);
//         transfer(colindx);
        #undef  transfer
        copy_data_to_gpu<char>(p->pBuffer + p->colindxwin.offset, (char*)p->colindx.data(), p->colindxwin.length, streamId, "colindx");
        copy_data_to_gpu<char>(p->pBuffer + p->subsetwin.offset, (char*)p->subset.data(), p->subsetwin.length, streamId, "subset");
    } // transfer_index_lists


    inline int __host__ highestbit(unsigned const n) {
        unsigned nn{n};
        int l2{-1}; // -1 is the return value for n==0
        while (nn > 0) {
            ++l2;
            nn >>= 1; // divide by two
        } // while
        return l2;
    } // highestbit

} // namespace tfqmrgpu
