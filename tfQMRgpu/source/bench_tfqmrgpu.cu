#include <cstdio> // std::printf
#include <cstdlib> // std::atoi
#include <iostream> // std::cout, std::endl
#include <fstream> // std::ifstream
#include <algorithm> // std::max
#include <cmath> // std::sqrt
#include <vector> // std::vector<T>
#include <cassert> // assert

// #define DEBUG

#include "tfqmrgpu.hxx" // includes cuda.h (or tfqmrgpu_cudaStubs.hxx) and tfqmrgpu.h
#include "bsr.hxx" // bsr_t block-sparse row matrices
#include "tfqmrgpu_example_reader.hxx" // ::read_in()
#include "tfqmrgpu_example_xml_reader.hxx" // ::read_in()

#include "tfqmrgpu_util.hxx" // FlopChar, CCheck, copy_data_to_gpu, get_data_from_gpu
#ifndef HAS_NO_CUDA
    #include "tfqmrgpu_blockmult.hxx" // gemmNxNf
#endif // HAS_CUDA

#ifdef DEBUG
    #define debug_printf(...) std::printf(__VA_ARGS__)
#else  // DEBUG
    #define debug_printf(...)
#endif // DEBUG


namespace GPUbench {

    // Example routine using the tfQMRgpu library's C-interface

    int benchmark_tfQMRgpu_library(
          bsr_t const ABX[3]
        , double const tolerance=1.0e-6
        , int const maxIterations=999
        , int const nRepetitions=1
        , char const doublePrecision='z' // beware: 'c' is not fully implemented!
    ) {

        PUSH_RANGE(__func__); // NVTX range markers for nvvp
        std::printf("\n# %s on GPU !!!!\n", __func__);

        auto const A = &(ABX[0]), B = &(ABX[1]), X = &(ABX[2]); // abbreviations

#define callAndCheck(FUN) \
        { \
            debug_printf("\n# Start "#FUN"\n"); \
            auto const stat = FUN; \
            debug_printf("# Done  "#FUN"\n"); \
            tfqmrgpuPrintError(stat); \
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat; \
        }

        // step 1: create a handle for the function call
        tfqmrgpuHandle_t handle{0};
        callAndCheck(  tfqmrgpuCreateHandle(&handle)  )

        // step 2: create a CUDA stream to work on
        cudaStream_t streamId{0};
        if (1) {
            auto const cudaErr = cudaStreamCreate(&streamId);
            if (cudaSuccess != cudaErr) {
                std::printf("[ERROR] CUDA call failed to create a stream in %s:%d\n", __FILE__, __LINE__);
                return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__;
            }
        } else  std::printf("# Warning! GPU works on default stream!\n");

        // step 3: register the CUDA stream in the handle
        callAndCheck(  tfqmrgpuSetStream(handle, streamId)  )

        if (1) { // sanity check
            auto streamId_copy{streamId};
            callAndCheck(  tfqmrgpuGetStream(handle, &streamId_copy)  )
            assert(streamId == streamId_copy);
        } // sanity check

        // step 4: analyse the blocks-sparse-row matrix patterns and create a bsrsv-plan
        tfqmrgpuBsrsvPlan_t plan{0};
        std::printf("\n# nnzb for A=%d, X=%d, B=%d\n", A->nnzb, X->nnzb, B->nnzb);
        callAndCheck(  tfqmrgpu_bsrsv_createPlan(handle, &plan, 
           A->nRows, // the number of block rows in A, X and B, also the number of block columns in A
           A->RowPtr.data(), A->nnzb, A->ColInd.data(), // block-sparse-row structure for A
           X->RowPtr.data(), X->nnzb, X->ColInd.data(), // block-sparse-row structure for X
           B->RowPtr.data(), B->nnzb, B->ColInd.data(), // block-sparse-row structure for B
           0)  ) // indexOffset=0(C-style) or indexOffset=1(Fortran)

        // step 5: compute the GPU memory requirement based on block sizes
        int const blockDim = A->fastBlockDim;
        size_t pBufferSize{0}; // in Bytes
        std::printf("# compute the GPU memory requirements for blockDim=%d\n", blockDim);
        callAndCheck(  tfqmrgpu_bsrsv_bufferSize(handle, plan, 
            blockDim, // Leading dimension for blocks in matrix A.
            blockDim, // Block dimension of matrix A, blocks in A are square blocks. blockDim <= ldA
            blockDim, // Leading dimension for blocks in matrix B or X.
            blockDim, // Fast block dimension of matrix B or X, RhsBlockDim <= ldB.
            doublePrecision, // Solver precision 'c':complex<float>, 'z':complex<double>
            &pBufferSize)  )

        // step 6: allocate the GPU memory
        void* pBuffer{nullptr};
        {
            auto const cudaErr = cudaMalloc(&pBuffer, pBufferSize);
            if (cudaSuccess != cudaErr) {
                std::printf("[ERROR] CUDA call failed to allocate %.6f GByte in %s:%d\n", pBufferSize*1e-9, __FILE__, __LINE__);
                return TFQMRGPU_STATUS_ALLOCATION_FAILED;
            } else {
                debug_printf("# allocated %.6f GByte GPU memory at %p in %s:%d\n", pBufferSize*1e-9, pBuffer, __FILE__, __LINE__);
                std::printf("# use %.6f GByte GPU memory\n", pBufferSize*1e-9);
            }
        }

        // step 7: register the GPU memory buffer in the bsrsv-plan
        callAndCheck(  tfqmrgpu_bsrsv_setBuffer(handle, plan, pBuffer)  )

        if (1) { // sanity check
            auto pBuffer_copy{pBuffer};
            callAndCheck(  tfqmrgpu_bsrsv_getBuffer(handle, plan, &pBuffer_copy)  )
            assert(pBuffer == pBuffer_copy);
        } // sanity check

        // step 8a: upload the values for the matrix A
        // values come from Fortran, so they are in RIRIRIRI layout
        callAndCheck(  tfqmrgpu_bsrsv_setMatrix(handle, plan, 'A',
            (char*)A->mat.data(), 'z', blockDim, 't', TFQMRGPU_LAYOUT_RIRIRIRI)  )

        // step 8b: upload the values for the right-hand side vectors B
        // values come from Fortran, so we need to transpose the blocks of B
        callAndCheck(  tfqmrgpu_bsrsv_setMatrix(handle, plan, 'B',
            (char*)B->mat.data(), 'z', blockDim, 't', TFQMRGPU_LAYOUT_RIRIRIRI)  )

        // [optional ]step 8x: upload the values for the initial vectors X

        // step 9: envoke the transpose-free Quasi Minimal Residual solver
        double solver_time = - getTime(); // start timer
        callAndCheck(  tfqmrgpu_bsrsv_solve(handle, plan, tolerance, maxIterations)  )
        solver_time += getTime(); // stop timer

        // step a: spare line
        // step b: spare line
        // step c: spare line

        // compare matX and matR (the reference matrix)
        auto const sizeX = X->mat.size(); 
        std::vector<double> Xref(X->mat); // copy constructor

        // step d: retrieve the result vectors X 
        // convert the blocks into ColMajor and RIRIRIRI to match the Fortran data layout
        callAndCheck(  tfqmrgpu_bsrsv_getMatrix(handle, plan, 'X',
            (char*)X->mat.data(), 'z', blockDim, 't', TFQMRGPU_LAYOUT_RIRIRIRI)  )

        { // scope:
            PUSH_RANGE("compare@CPU");
            double alldev{0}, allval{0}, maxdev{0}, maxrel{0};
            for(auto cij = 0ull; cij < sizeX; ++cij) {
                double const dev = std::abs(X->mat[cij] - Xref[cij]);
                maxdev = std::max(maxdev, dev);
                if (0.0 != Xref[cij]) maxrel = std::max(maxrel, dev/Xref[cij]);
                alldev += dev;
                allval += 1.0;
            } // cij
            std::printf("# GPU maxdev %g avgdev %g maxrel %g\n", maxdev, alldev/allval, maxrel);
            POP_RANGE(); // end of NVTX range

            if (maxdev < 1e-5) {
                // seems correct, report performance
                int iterations_needed{0};
                double flops_performed{0}, residuum_reached{1};
                callAndCheck(  tfqmrgpu_bsrsv_getInfo(handle, plan, &residuum_reached, 
                                            &iterations_needed, &flops_performed, 0x0)
                            )
                std::printf("# GPU converged in %d iterations\n", iterations_needed);
                char const fF = ('z' == (doublePrecision | 32))? 'F' : 'f'; // F:double, f:float
                double const TFlop = 1e-12*flops_performed;
                double const performance = TFlop/std::max(solver_time, 1e-6);
                std::printf("# GPU performed %.3f T%clop in %.3f seconds = %.3f T%clop/s\n", 
                                       TFlop, fF, solver_time, performance, fF);
            } // maxdev
        } // scope


        // step e: destroy the plan
        callAndCheck(  tfqmrgpu_bsrsv_destroyPlan(handle, plan)  )
        plan = 0;

        // step f: destroy the handle
        callAndCheck(  tfqmrgpuDestroyHandle(handle)  )
        handle = 0;

        // last step, free GPU memory
        CCheck(cudaFree(pBuffer));

#undef  callAndCheck
        POP_RANGE(); // end of NVTX range
        return TFQMRGPU_STATUS_SUCCESS;
    } // benchmark_tfQMRgpu_library
















    // Multiplication benchmark: measure the performance of Y = A*X

    template <typename T>
    T* get_gpu_memory(size_t const size=1) {
#ifdef DEBUGGPU
        std::printf("#  cudaMalloc: %lu x %.3f kByte = \t%.3E MByte", size, 1e-3*sizeof(T), size*1e-6*sizeof(T));
#endif // DEBUGGPU
        void* d = nullptr;
        CCheck(cudaMalloc(&d, size*sizeof(T)));
#ifdef DEBUGGPU
        std::printf(" @ %p through %p \n", d, (char*)d + size*sizeof(T) - 1);
#endif // DEBUGGPU
        return (T*)d;
    } // get_gpu_memory

    template <typename T>
    void free_gpu_memory(T*& d) {
        CCheck(cudaFree(d));
        d = nullptr;
    } // free_gpu_memory

    template <typename T>
    T* create_on_gpu(T const *const h, size_t const size=1, cudaStream_t const stream=0) {
        T* d = get_gpu_memory<T>(size);
        copy_data_to_gpu<T>(d, h, size, stream); // start copying to the GPU, async!
        return d;
    } // create_on_gpu

    template <typename T>
    T* create_on_cpu(T const (*devPtr d), size_t const size=1, cudaStream_t const stream=0) {
        T* h = (T*) malloc(size*sizeof(T)); // c-style allocation
        get_data_from_gpu<T>(h, d, size, stream); // start copying from the GPU, async!
        return h;
    } // create_on_cpu

#ifndef HAS_NO_CUDA
    template <typename real_t, int LM, int LN>
    void __global__ // GPU kernel, must be launched with <<< {nmat, 1, 1}, {LN, any, 1} >>>
    fill_cos_sin(real_t (*devPtr c)[2][LM][LN]) {
        // fill GPU arrays with non-trivial but deterministic values
        int const m = blockIdx.x;
        int const j = threadIdx.x;
        for(int i = threadIdx.y; i < LM; i += blockDim.y) { // grid stride loop
            auto const arg = double((m*LM + i)*LN + j);
            c[m][0][i][j] = std::cos(arg);
            c[m][1][i][j] = std::sin(arg);
        } // i
    } // fill_cos_sin
#endif // HAS_CUDA

    template <typename real_t, int LM, int LN=LM>
    double bench_multi( // returns the average time needed per kernel call
          unsigned const nnzbY
        , uint32_t const (*const starts_h)
        , size_t const nPairs
        , uint32_t const (*const pairs_h)
        , unsigned const nnzbA
        , unsigned const nnzbX
        , int const nRepetitions=1 // Number of iterations of the same procedure
        , int const nSamples=1 // Number of samples taken for timings
    ) {
        std::printf("\n# %s<%d> on GPU !!!!\n", __func__, LM);

        std::printf("# Execute %d repetitions, sample %d times.\n", nRepetitions, nSamples);

        size_t const mem = (nnzbY + nnzbX)*sizeof(real_t[2][LM][LN]) + nnzbA*sizeof(real_t[2][LM][LM]);
        std::printf("# Try to allocate %.3f GByte for %d + %d complex matrices of dim=%d x %d and "
                    "%d complex matrices of dim=%d x %d\n", mem*1e-9, nnzbY, nnzbX, LM, LN, nnzbA, LM, LM);
        auto matY = get_gpu_memory<real_t[2][LM][LN]>(nnzbY);
        auto matA = get_gpu_memory<real_t[2][LM][LM]>(nnzbA);
        auto matX = get_gpu_memory<real_t[2][LM][LN]>(nnzbX);

        auto starts_d = create_on_gpu<uint32_t>(starts_h, nnzbY + 1);
        auto pairs_d  = create_on_gpu<uint32_t>(pairs_h, nPairs*2);

#ifndef HAS_NO_CUDA
        fill_cos_sin<real_t,LM,LM> <<< nnzbA, {LM, 1024/LM, 1} >>> (matA);
        fill_cos_sin<real_t,LM,LN> <<< nnzbX, {LN, 1024/LM, 1} >>> (matX);

        int constexpr TUNE = 2;
        // TUNE == 2 performance up to 3.8 TFlop/s for LM=32 on V100
        // TUNE == 4 performance up to 4.3 TFlop/s for LM=32 on V100, does not work for LM=6
        dim3 const threads = { LN, TUNE, 1 };
        std::printf("# CUDA Launch <<< %d, { %d, %d, %d } >>>, TUNE = %d\n",
                              nnzbY, threads.x, threads.y, threads.z, TUNE);
#endif // HAS_CUDA
        assert(nnzbX == nnzbY); // operator A must be logically square

        double nFlop{0};
        double time_sum{0}, time_rms{0}; // timing stats
        PUSH_RANGE("GPU benchmarks gemmNxNf");
        for(int sample = 0; sample < nSamples; ++sample) {
            double time{-getTime()}; // start
            for(int repetition = 0; repetition < nRepetitions; ++repetition) {
                // test the small matrix-matrix multiplications
#ifndef HAS_NO_CUDA
                gemmNxNf<real_t,LM,LN,LM/TUNE> <<< nnzbY, threads >>> (matY, matA, matX, pairs_d, starts_d);
                nFlop += nPairs*(8.*LM)*(LM*LN);
#endif // HAS_CUDA
            } // repetition
            CCheck(cudaDeviceSynchronize());
            time += getTime(); // stop
            time_sum += time; time_rms += time*time; // add to timing stats
        } // sample
        POP_RANGE(); // end of NVTX range

        double const time_avg = time_sum / double(nSamples); // average
        time_rms = std::sqrt(std::max(0., time_rms/double(nSamples) - time_avg*time_avg)); // rms
        std::printf("# GPU needed %.3f seconds, %.6f +/- %.6f sec per sample, %.1f%% dev\n",
                  time_sum, time_avg, time_rms, time_rms*100./time_avg);

        bool correct{true};
#ifdef  SKIP_CORRECTNESS_CHECK
        std::printf("# Warning! Correctness checks are deactivated with -D SKIP_CORRECTNESS_CHECK!\n");
#else // SKIP_CORRECTNESS_CHECK
        PUSH_RANGE("CPU checks correctness");
        double time_chk =- getTime();
        // check if matY has the correct values
        double maxdev={-1}, alldev{0}, allval{0};
        int nthreads{1};
#pragma omp parallel
        { nthreads = omp_get_num_threads(); }
        std::printf("# CPU %d threads check for correct results\n", nthreads);
        { // correctness check scope
            auto const matY_h = create_on_cpu<real_t[2][LM][LN]>(matY, nnzbY);
            auto const matA_h = create_on_cpu<real_t[2][LM][LM]>(matA, nnzbA);
            auto const matX_h = create_on_cpu<real_t[2][LM][LN]>(matX, nnzbX);
#pragma omp parallel for reduction(+:alldev,allval) reduction(max:maxdev)
            for(auto iY = 0u; iY < nnzbY; ++iY) {
                auto matY_r = new real_t[2][LM][LN]; // thread-private reference result
                for(auto i = 0; i < LM; ++i) {
                    for(auto j = 0; j < LN; ++j) {
                        matY_r[0][i][j] = 0; matY_r[1][i][j] = 0; // clear real and imaginary part
                    } // j
                } // i
                for(auto ipair = starts_h[iY]; ipair < starts_h[iY + 1]; ++ipair) {
                    auto const iA = pairs_h[ipair*2 + 0], iX = pairs_h[ipair*2 + 1];
                    for(auto i = 0; i < LM; ++i) {
                        for(auto j = 0; j < LN; ++j) {
                            real_t cr{0}, ci{0};
                            for(auto k = 0; k < LM; ++k) {
                                real_t const srei = matA_h[iA][0][k][i], simi = matA_h[iA][1][k][i];
                                real_t const vrej = matX_h[iX][0][k][j], vimj = matX_h[iX][1][k][j];
                                cr += srei * vrej - simi * vimj; // Real part
                                ci += srei * vimj + simi * vrej; // Imag part
                            } // k
                            matY_r[0][i][j] += cr; matY_r[1][i][j] += ci;
                        } // j
                    } // i
                } // ipair
                for(auto c = 0; c < 2; ++c) {
                    for(auto i = 0; i < LM; ++i) {
                        for(auto j = 0; j < LN; ++j) {
                            double const dev = std::abs(matY_r[c][i][j] - matY_h[iY][c][i][j]);
                            maxdev = std::max(maxdev, dev);
                            alldev += dev;
                            allval += 1.0;
                        } // j
                    } // i
                } // c
                delete[] matY_r;
            } // iY
            delete[] matY_h;
            delete[] matA_h;
            delete[] matX_h;
        } // scope

        time_chk += getTime();
        POP_RANGE(); // end of NVTX range

        std::printf("# GPU maxdev %g avgdev %g\n", maxdev, alldev/allval);
        if (maxdev > 1e-4) {
            std::printf("# Warning! GPU result has large deviations (%g) for blockDim=%d x %d\n", maxdev, LM, LN);
            correct = false; // do not show the performance of wrong results
        } else {
            std::printf("# CPU result checking with %d threads took %.3f sec\n", nthreads, time_chk);
        }
#endif // SKIP_CORRECTNESS_CHECK

        if (correct) { // print performance scope
            char const fF = FlopChar<real_t>();
            std::printf("# GPU performed %.3f T%clop in %.3f seconds, i.e. %.1f G%clop/sec\n", 
                                  nFlop*1e-12, fF, time_sum, nFlop*1e-9/time_sum, fF);
        } // scope

        std::printf("# %s: free GPU memory\n", __func__);
        free_gpu_memory(matX);
        free_gpu_memory(matA);
        free_gpu_memory(matY);
        free_gpu_memory(pairs_d);
        free_gpu_memory(starts_d);

        std::printf("# %s: deviceSynchronize\n", __func__);
        CCheck(cudaDeviceSynchronize());
        std::printf("# %s: done\n", __func__);
        return time_avg;
    } // bench_multi

    int benchmark_blockMatrixMatrixMultiplication(int const argc, char const *const argv[]) {
        // ToDo: use control::get environment
                                 assert( 'm' == *argv[1] ); // 'multiplication' task
        char const *fnm  = (argc > 2)?           argv[2]  : "plan"; // inputfile
        char const fF    = (argc > 3)?          *argv[3]  : 'f'; // {f,F,c,C, d,D,z,Z} = float or double
        int const nrep   = (argc > 4)? std::atoi(argv[4]) : 1; // number or repetitions
        int const nsamp  = (argc > 5)? std::atoi(argv[5]) : 1; // number of sampling
        int const lm     = (argc > 6)? std::atoi(argv[6]) : 16; // block rows
        int const ln     = (argc > 7)? std::atoi(argv[7]) : lm; //  block cols

        bool const doublePrecision = (('d' == (fF | 32)) || ('z' == (fF | 32)));

        // read multiplication plan from input file
        std::ifstream input(fnm, std::ifstream::in);
        if (input.fail()) {
            std::cout << argv[0] << ": error: did not find file" << std::endl; 
            exit(-3);
        } // input file not found

        std::string str;
        unsigned nnzY, nnzA, nnzX;
        input >> str >> nnzY >> nnzA >> nnzX;
        bool const info = false;
        if (info) {
            std::cout << "# nnz Y " << nnzY << std::endl;
            std::cout << "# nnz A " << nnzA << std::endl;
            std::cout << "# nnz X " << nnzX << std::endl;
        } // info

        std::vector<uint32_t> pairs;
        std::vector<uint32_t> starts;
        int64_t iY, iA, iX;
        int beta;
        int nzpr{-1};
        std::vector<int> hist(96, 0); // histogram
        int irow{-1}; //
        int64_t iYprev{-1}; // init with an impossible index value
        while (input >> iY >> iA >> iX >> beta) {
    //      std::cout << iY << " " << iA << " " << iX << " " << beta << " " << std::endl; // echo the input file structure
            if (iY != iYprev) {
                assert(0 == beta);
                starts.push_back(pairs.size()/2);
                if (-1 == iYprev) { ++hist[nzpr]; nzpr = 0; } // update histogram
                iYprev = iY;
                ++irow;
            } else {
                assert(1 == beta);
            }
    //      assert(iY == irow); // this will fail if the iY indices do not come in order
            ++nzpr; // number of non-zero entries per row
            pairs.push_back(iA); // add new small matrix-matrix-multiplications between block iA and block iX
            pairs.push_back(iX); // add new small matrix-matrix-multiplications between block iA and block iX
        } // while
        if (info) std::cout << "# found " << starts.size() << " result elements" << std::endl;
        starts.push_back(pairs.size()/2); // final
        assert(starts.size() == nnzY + 1); // we need one more for the sparse format

        if (info) {
            // show histogram about number of non-zero elements per row
            for(nzpr = 0; nzpr < 96; ++nzpr) {
                if (0 < hist[nzpr]) {
                    std::cout << "# found " << hist[nzpr] << " elements with " << nzpr << " operations" << std::endl; 
                } // nonzero
            } // nzpr
        } // info

        auto const nPairs = pairs.size()/2; // number of small matrix-matrix-multiplications
        if (info) std::cout << "# found " << nPairs << " operations" << std::endl;
#ifdef  FULLDEBUG
            std::cout << "# rows start at ";
            for(auto rs : starts) {
                std::cout << " " << rs;
            } // rs
            std::cout << std::endl;
#endif // FULLDEBUG

        switch (lm*1000 + ln) { // blocksize
#define call_it(REAL_t,LM,LN) bench_multi<REAL_t,LM,LN>(nnzY, starts.data(), nPairs, pairs.data(), nnzA, nnzX, nrep, nsamp)
#define decide_precision(LM,LN) if (doublePrecision) { call_it(double,LM,LN); } else { call_it(float,LM,LN); }
            case   4004:  decide_precision(  4,  4); break; // Lmax=1
            case   8008:  decide_precision(  8,  8); break; // Lmax=1, noco
            case  16016:  decide_precision( 16, 16); break; // Lmax=3
            case  32032:  decide_precision( 32, 32); break; // Lmax=3, noco
            case  64064:  decide_precision( 64, 64); break; // Lmax=7
            case 128128:  decide_precision(128,128); break; // Lmax=7, noco

            // with a single prime factor 3
            case   6006:  decide_precision(  6,  6); break;
            case  12012:  decide_precision( 12, 12); break;
            case  24024:  decide_precision( 24, 24); break;
            case  48048:  decide_precision( 48, 48); break;
            case  96096:  decide_precision( 96, 96); break;

            // rectangular cases
            case   4032:  decide_precision(  4, 32); break;
            case   8032:  decide_precision(  8, 32); break;
            case  16032:  decide_precision( 16, 32); break;

#undef  decide_precision
#undef  call_it
            default : std::cout << "ERROR: Case not implemented lm = " << lm << " ln = " << std::max(lm,ln) << std::endl; return 1; 
        } // switch lm

        std::cout << "# done " << __func__ << std::endl;
        return 0; // 0:success
    } // benchmark_blockMatrixMatrixMultiplication

} // namespace GPUbench


int main(int const argc, char const *const argv[]) {

    if (argc < 2) { 
        std::printf("Usage:  %s  [tfQMR/multiply]  [file]  [float/double]  "
                    "[#repetitions]  [#iterations]  [#blocksize]\n", argv[0]);
        exit(1);
    } // not enough command line args passed

    char const bench   = (argc > 1)?          *argv[1]  : 't'; // t=tfQMR, m=multiplication
    if ('m' == bench) return GPUbench::benchmark_blockMatrixMatrixMultiplication(argc, argv);

    char const *fnm    = (argc > 2)?           argv[2]  : "problem"; // inputfile
    char const flouble = (argc > 3)?        ((*argv[3]) | 32) : 'z'; // z:double, c:float
    int  const nrep    = (argc > 4)? std::atoi(argv[4]) : 1; // number of repetitions
    int  const MaxIter = (argc > 5)? std::atoi(argv[5]) : 2000; // max. number of iteration

    std::printf("\n# read file '%s' as input.\n", fnm);
    bsr_t ABX[3]; // three block-sparse operators
    double tolerance{0};
    if (std::string(fnm).find("xml") != std::string::npos) {
        tolerance = tfqmrgpu_example_xml_reader::read_in(ABX, fnm);
    } else {
        tolerance = tfqmrgpu_example_reader::read_in(ABX, fnm);
    }
    std::printf("# found tolerance %g\n", tolerance);
    std::printf("# Execute %d repetitions with max. %d iterations.\n", nrep, MaxIter);
    std::printf("# requested precision = %c for LM = %d\n", flouble, ABX[0].fastBlockDim);

    return GPUbench::benchmark_tfQMRgpu_library(ABX, tolerance, MaxIter, nrep, flouble);
} // main
