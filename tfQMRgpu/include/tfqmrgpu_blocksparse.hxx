#pragma once

#include "tfqmrgpu_plan.hxx" // bsrsv_plan_t
#include "tfqmrgpu_memWindow.h" // memWindow_t
#include "tfqmrgpu_util.hxx" // take_gpu_memory<T>, copy_data_to_gpu<T>

#ifndef HAS_NO_CUDA
    #include "tfqmrgpu_blockmult.hxx" // gemmNxNf
#endif // HAS_CUDA

template <typename floating_point_t, int block_size>
class action_t {
  public:
      typedef floating_point_t real_t;
      static int constexpr LM = block_size;
  // 
  // This action is an explicit block-sparse matrix multiplication.
  // Blocks are sized [LM][LM].
  // Arithmetic according to complex<real_t> 
  // with real_t either float or double
  //

  private: // members

    bsrsv_plan_t* p; // ToDo: would it be better to have a reference here?
//  std::vector<int_t> p->pairs; // [nPairs*2], each pair is one block-times-block mutliplication
//  std::vector<int_t> p->starts; // [nnzbX + 1] number of target elements plus one

    real_t (* matA_d)[2][LM][LM]; // matrix data in GPU memory
    // ToDo: try how fast it is with pairs and starts in managed memory (UVM)
    uint32_t (* pairs_d); // pairs in GPU memory
    uint32_t (* starts_d); // starts in GPU memory

  public:

    action_t(bsrsv_plan_t* const plan) : p(plan) {
        assert(nullptr != p);
        assert(LM == p->LM);
        p->doublePrecision = (sizeof(real_t) > 4)? 'Z' : 'C';
    } // constructor

    void take_memory(char* &buffer) {
        // bring the matrix data and index lists into GPU memory
        starts_d = take_gpu_memory<uint32_t>(buffer, p->starts.size(), &p->startswin, "starts");
        pairs_d  = take_gpu_memory<uint32_t>(buffer, p->pairs.size(),  &p->pairswin,   "pairs");
        matA_d   = take_gpu_memory<real_t[2][LM][LM]>(buffer, p->nnzbA, &p->matAwin, "A");
    } // take_memory

    void transfer(char* const buffer, cudaStream_t const streamId=0) {
        // transfer index lists to GPU memory, could be cached, i.e. stored in the plan if this transfer has already been done
#ifdef DEBUG
        printf("# p->pairs.data()  = %p\n", (char*)(p->pairs.data()));
        printf("# p->starts.data() = %p\n", (char*)(p->starts.data()));
        printf("# buffer = %p\n", buffer);
        printf("# buffer + startswin.offset = %p\n", buffer + p->startswin.offset);
        printf("# buffer +  pairswin.offset = %p\n", buffer + p-> pairswin.offset);
        fflush(stdout);
#endif // DEBUG
        copy_data_to_gpu<char>(buffer + p->startswin.offset, (char*)(p->starts.data()), p->startswin.length, streamId, "starts");
        copy_data_to_gpu<char>(buffer + p-> pairswin.offset, (char*)(p-> pairs.data()), p-> pairswin.length, streamId,  "pairs");
    } // transfer

    bool has_preconditioner() const { return false; }

    // driver
    double multiply(   
          real_t       (*devPtr y)[2][LM][LM] // result, y[nnzbY][2][LM][LM]
        , real_t const (*devPtr x)[2][LM][LM] // input,  x[nnzbX][2][LM][LM]
        , uint16_t const (*devPtr colIndex) // column indices, uint16_t allows up to 65,536 block columns
        , uint32_t const nnzbY // == nnzbX
        , uint32_t const nCols=1
        , unsigned const l2nX=0
        , cudaStream_t const streamId=0
        , bool const precondition=false
    ) {
        // how to multiply the action onto x
#ifndef HAS_NO_CUDA
        int constexpr TUNE = 2; // 2 has been found to yield the best performance for LM==32
        gemmNxNf <real_t,LM,LM/TUNE> <<< nnzbY, { LM, TUNE, 1 }, 0, streamId >>> (y, matA_d, x, pairs_d, starts_d);
#else  // HAS_CUDA
        // CPU version
        auto const A = matA_d; // works since host and device memory are the same if HAS_CUDA is not defined
        for(int iYmat = 0; iYmat < nnzbY; ++iYmat) {

            real_t Yb[2][LM][LM]; // temporary for a block of result operator Y
            for(int cij = 0 ; cij < 2*LM*LM; ++cij) Yb[0][0][cij] = 0;

//          char const tA = 'n', tx = 'n';
//          int32_t const n = LM;
//          real_t const alpha = 1, minus = -1;
//          real_t beta{0};
            for(auto ipair = p->starts[iYmat]; ipair < p->starts[iYmat + 1]; ++ipair) { // contract over block elements
                auto const iAmat = p->pairs[ipair*2 + 0];
                auto const iXmat = p->pairs[ipair*2 + 1];
#if 1
                for(int i = 0; i < LM; ++i) {
                    for(int j = 0; j < LM; ++j) {
                        real_t Yre{0}, Yim{0};
                        for(int k = 0; k < LM; ++k) { // contract over k
                            real_t const Are = A[iAmat][0][i][k],
                                         Aim = A[iAmat][1][i][k];
                            real_t const Xre = x[iXmat][0][k][j],
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
            for(int cij = 0 ; cij < 2*LM*LM; ++cij) {
                y[iYmat][0][0][cij] = Yb[0][0][cij];
            } // cij

        } // iYmat
#endif // HAS_CUDA
        auto const nPairs = p->pairs.size()/2;
        return nPairs*LM*8.*LM*LM; // returns the number of Flops
    } // multiply

    bsrsv_plan_t* get_plan() const { return p; }

}; // class action_t

