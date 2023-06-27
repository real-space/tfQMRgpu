#pragma once

// #define FULLDEBUG

#include "tfqmrgpu_plan.hxx" // bsrsv_plan_t, colIndex_t
#include "tfqmrgpu_util.hxx" // take_gpu_memory<T>, copy_data_to_gpu<T>, print_array

#ifndef HAS_NO_CUDA
    #include "tfqmrgpu_blockmult.hxx" // gemmNxNf
#endif // HAS_CUDA

template <typename floating_point_t, int block_rows, int block_cols, typename double_t=floating_point_t>
class blocksparse_action_t {
  public:
      typedef floating_point_t real_t;
      static int constexpr LM = block_rows,
                           LN = block_cols; // in principle only multiply depends on LN, so one could move that template there
  //
  // This action is an explicit block-sparse matrix multiplication.
  // Blocks of A are sized [LM][LM].
  // Blocks of X are sized [LM][LN].
  // Arithmetic according to complex<real_t>
  // with real_t either float or double
  //

  private: // private members

    bsrsv_plan_t* p; // ToDo: would it be better to have a reference here?

    real_t (* matA_d)[2][LM][LM]; // matrix data in GPU memory
    // ToDo: try how fast it is with pairs and starts in managed memory (UVM)
    uint32_t (* pairs_d); // pairs in GPU memory
    uint32_t (* starts_d); // starts in GPU memory

  public:

    blocksparse_action_t(bsrsv_plan_t* const plan) : p(plan) {
        assert(nullptr != p);
        assert(LM == p->LM);
        assert(LM > 0);
        assert(LN >= LM); // this constraint is only for the block-sparse operator and stems from the specific implementation of the kernel gemmNxNf
        p->precision = (sizeof(real_t) == 8) ? 'z' : 'c';
    } // constructor

    void take_memory(char* &buffer) {
        // bring the matrix data and index lists into GPU memory
        starts_d = take_gpu_memory<uint32_t>(buffer, p->starts.size(), &p->startswin, "starts");
        pairs_d  = take_gpu_memory<uint32_t>(buffer, p->pairs.size(),  &p->pairswin,   "pairs");
        matA_d   = take_gpu_memory<real_t[2][LM][LM]>(buffer, p->nnzbA, &p->matAwin, "A");
    } // take_memory

    void transfer(char* const buffer, cudaStream_t const streamId=0) {
        // transfer index lists to GPU memory, could be cached, i.e. stored in the plan if this transfer has already been done
    #ifdef  DEBUG
        printf("# p->pairs.data()  = %p\n", (void*)(p->pairs.data()));
        printf("# p->starts.data() = %p\n", (void*)(p->starts.data()));
        printf("# buffer = %p\n", buffer);
        printf("# buffer + startswin.offset = %p\n", buffer + p->startswin.offset);
        printf("# buffer +  pairswin.offset = %p\n", buffer + p-> pairswin.offset);
        fflush(stdout);
    #endif // DEBUG
        // matA_d is filled during setMatrix operation
        copy_data_to_gpu<char>(buffer + p->startswin.offset, (char*)(p->starts.data()), p->startswin.length, streamId, "starts");
        copy_data_to_gpu<char>(buffer + p-> pairswin.offset, (char*)(p-> pairs.data()), p-> pairswin.length, streamId,  "pairs");
    } // transfer

    bool has_preconditioner() const { return false; }

    // driver
    double multiply(
          real_t         (*devPtr y)[2][LM][LN] // result, y[nnzbY][Re:Im][LM][LN]
        , real_t   const (*devPtr x)[2][LM][LN] // input,  x[nnzbX][Re:Im][LM][LN]
        , colIndex_t const (*devPtr colIndex) // column indices, needed if the operator internally has reductions
        , uint32_t const nnzbY // == nnzbX
        , uint32_t const nCols=1
        , unsigned const l2nX=0
        , cudaStream_t const streamId=0
        , bool const precondition=false
    ) {
        // how to multiply the action A onto x
#ifndef HAS_NO_CUDA
        // CUDA version

        // int  constexpr TUNE = 2; // TUNE = 2 does not launch for 16x16 and 64x64
        // int  constexpr TUNE = 1; // TUNE = 2 does not launch for 16x16 and 64x64
        // int  constexpr TUNE = 4; // TUNE = 4 does not work for LM==6
        int  constexpr TUNE = ((16 == LM) || (64 == LM)) ? 4 : 2; // fix
        dim3 const     nblocks(nnzbY, 1, 1); // number of blocks
        dim3 constexpr threads(LN, TUNE, 1); // threads per block
    #ifdef  FULLDEBUG
    
        { // scope: check if a kernel before this one failed
            auto const err = cudaGetLastError();
            if (cudaSuccess != err) {
                auto const errString = cudaGetErrorString(err);
                printf("[ERROR] in %s:%d cudaError \"%s\" in last kernel before gemmNxNf\n", __FILE__, __LINE__, errString);
            } // error
        } // scope
    
        printf("# [info] launch gemmNxNf <real_t=%s,LM=%d,LN=%d,LM/TUNE=%d,double_t=%s> "
               "<<< nblocks=(%d,%d,%d), threads=(%d,%d,%d) >>>\n",
               (8 == sizeof(real_t))?"double":"float", LM, LN, LM/TUNE, (8 == sizeof(double_t))?"double":"float",
               nblocks.x, nblocks.y, nblocks.z, threads.x, threads.y, threads.z);
//      printf("# [info] launch gemmNxNf(y=%p, matA_d=%p, x=%p, pairs_d=%p, starts_d=%p);\n",
//             y, matA_d, x, pairs_d, starts_d);
        cudaDeviceSynchronize();
    #endif // FULLDEBUG

        gemmNxNf <real_t,LM,LN,LM/TUNE,double_t> <<< nblocks, threads, 0, streamId >>> (y, matA_d, x, pairs_d, starts_d);

    #ifdef  FULLDEBUG
        // cudaDeviceSynchronize(); // necessary?
        // auto const err = cudaGetLastError();
        auto const err = cudaDeviceSynchronize();
        if (cudaSuccess != err) {
            auto const errString = cudaGetErrorString(err);
            printf("[ERROR] in %s:%d cudaError \"%s\" after kernel call!\n", __FILE__, __LINE__, errString);
        } else {
            cudaDeviceSynchronize(); // necessary?
          #ifdef  EXTREMEDEBUG
            printf("\n\n# multiply:\n");
            for(int i{0}; i < nnzbY; ++i) {
                printf("# from [%d to %d)\n", p->starts[i], p->starts[i + 1]);
                for(int j = p->starts[i]; j < p->starts[i + 1]; ++j) {
                    printf("#   pair %i %i\n", p->pairs[2*j], p->pairs[2*j + 1]);
                } // j
            } // i
            print_array<uint32_t,1> <<< 1, 1, 0, streamId >>> ((uint32_t(*)[1])starts_d, nnzbY + 1, 's', 'i');
            print_array<uint32_t,2> <<< 1, 1, 0, streamId >>> ((uint32_t(*)[2])pairs_d, p->starts[nnzbY], 'p', 'i');
            print_array<real_t, LM> <<< 1, 1, 0, streamId >>> (matA_d[0][0], p->nnzbA*2*LM, 'A', 'g');
            print_array<real_t, LN> <<< 1, 1, 0, streamId >>> (x[0][0], nnzbY*2*LM, 'x', 'g');
            cudaDeviceSynchronize(); // necessary?
            print_array<real_t, LN> <<< 1, 1, 0, streamId >>> (y[0][0], nnzbY*2*LM, 'y', 'g');
            cudaDeviceSynchronize(); // necessary?
            printf("\n");
          #endif // EXTREMEDEBUG
        } // true or false
    #endif // FULLDEBUG


#else  // HAS_CUDA

        // CPU version
        auto const A = matA_d; // works since host and device memory are the same if HAS_CUDA is not defined
        for(int iYmat = 0; iYmat < nnzbY; ++iYmat) {

            real_t Yb[2][LM][LN]; // temporary for a block of result operator Y
            for(int cij = 0; cij < 2*LM*LN; ++cij) {
                Yb[0][0][cij] = 0;
            } // cij

//          // WARNING: BLAS-version using dgemm_ so far cannot treat LM != LN
//          char const tA = 'n', tx = 'n';
//          int32_t const n = LM;
//          real_t const alpha = 1, minus = -1;
//          real_t beta{0};

            for(auto ipair = p->starts[iYmat]; ipair < p->starts[iYmat + 1]; ++ipair) { // contract over block elements
                auto const iAmat = p->pairs[ipair*2 + 0];
                auto const iXmat = p->pairs[ipair*2 + 1];
    #if 1
                for(int i = 0; i < LM; ++i) {
                    for(int j = 0; j < LN; ++j) {
                        real_t Yre{0}, Yim{0};
                        for(int k = 0; k < LM; ++k) { // contract over k
                            auto const Are = A[iAmat][0][i][k], // ToDo: check block transposition of A
                                       Aim = A[iAmat][1][i][k];
                            auto const Xre = x[iXmat][0][k][j],
                                       Xim = x[iXmat][1][k][j];
                            // complex multiplication, 8 Flop
                            Yre += Are * Xre - Aim * Xim; // Real part
                            Yim += Are * Xim + Aim * Xre; // Imag part
                        } // k
                        Yb[0][i][j] += Yre;
                        Yb[1][i][j] += Yim;
                    } // j
                } // i
    #else  // 1
                dgemm_(&tA, &tx, &n, &n, &n, &alpha, A[iAmat][0], &n, x[iXmat][0], &n, &beta, yb[0], &n);
                dgemm_(&tA, &tx, &n, &n, &n, &minus, A[iAmat][1], &n, x[iXmat][1], &n, &beta, yb[0], &n);
                beta = 1;
                dgemm_(&tA, &tx, &n, &n, &n, &alpha, A[iAmat][1], &n, x[iXmat][0], &n, &beta, yb[1], &n);
                dgemm_(&tA, &tx, &n, &n, &n, &alpha, A[iAmat][0], &n, x[iXmat][1], &n, &beta, yb[1], &n);
    #endif // 1
            } // ipair

            // copy block into result array
            for(int cij = 0; cij < 2*LM*LN; ++cij) {
                y[iYmat][0][0][cij] = Yb[0][0][cij];
            } // cij

        } // iYmat
        // end of CPU version

#endif // HAS_CUDA

        return p->pairs.size()*.5*LM*8.*LM*LN; // returns the number of Flops or flops: 8 per complex
    } // multiply

    bsrsv_plan_t* get_plan() const { return p; }

}; // class blocksparse_action_t

