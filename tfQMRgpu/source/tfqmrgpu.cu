#include <cstdio> // std::printf, std::fflush, stdout
#include <vector> // std::vector<T>
#include <cassert> // assert

// #define DEBUG

#include "tfqmrgpu.hxx" // includes cuda.h and tfqmrgpu.h
#include "tfqmrgpu_linalg.hxx" // tfqmrgpu::transpose_blocks_kernel
#include "tfqmrgpu_core.hxx"   // tfqmrgpu::solve<action_t>
#include "tfqmrgpu_blocksparse.hxx" // blocksparse_action_t
#include "tfqmrgpu_util.hxx" // IgnoreCase


    template <typename real_t, int LM, int LN=LM, typename double_t=real_t>
    tfqmrgpuStatus_t mysolve_real_LM_LN (
          cudaStream_t streamId // stream
        , bsrsv_plan_t* p // plan
        , double const tolerance
        , int const MaxIterations
        , bool const memcount
    ) {
        blocksparse_action_t<real_t,LM,LN,double_t> action(p);
        return tfqmrgpu::solve(action, memcount ? nullptr : p->pBuffer, tolerance, MaxIterations, streamId);
    } // mysolve_real_LM_LN


    template <int LM, int LN=LM>
    tfqmrgpuStatus_t mysolve_LM_LN (
          cudaStream_t streamId // stream
        , bsrsv_plan_t* p // plan
        , double const tolerance
        , int const MaxIterations
        , bool const memcount
    ) {
        switch (p->doublePrecision | IgnoreCase) {
          case 'z': return mysolve_real_LM_LN<double,LM,LN>(streamId, p, tolerance, MaxIterations, memcount);
          case 'm': return mysolve_real_LM_LN<float,LM,LN,double>(streamId, p, tolerance, MaxIterations, memcount); // mixed precision: load float, multipy-accumulate double, store float
          default : return mysolve_real_LM_LN<float,LM,LN>(streamId, p, tolerance, MaxIterations, memcount);
        }
    } // mysolve_LM_LN


    tfqmrgpuStatus_t mysolve (
          cudaStream_t streamId
        , bsrsv_plan_t* p // plan
        , double const tolerance
        , int const MaxIterations
        , bool const memcount=false
    ) {
        switch (p->LM*1000 + p->LN) {
#define     allow_block_size(LM,LN) \
            case   LM*1000 + LN: return mysolve_LM_LN<LM,LN>(streamId, p, tolerance, MaxIterations, memcount)

            // list all the allowed block sizes here as allow_block_size(ldA, ldB);
#include    "allowed_block_sizes.h"
//          allow_block_size( 4, 4);
//          allow_block_size( 4,32); // blocks in X and B are rectangular
//          allow_block_size( 8, 8);
//          allow_block_size( 8,32); // blocks in X and B are rectangular
//          allow_block_size(16,16);
//          allow_block_size(16,32); // blocks in X and B are rectangular
//          allow_block_size(32,32);
//          allow_block_size(64,64);

#undef      allow_block_size
            default: return TFQMRGPU_BLOCKSIZE_MISSING + TFQMRGPU_CODE_CHAR*p->LM + TFQMRGPU_CODE_LINE*p->LN; // also say which blocksize was requested
        } // switch LM
    } // mysolve





    // library peripherals ////////////////////////////////////////
    template <typename T>
    T tfqmrgpu_mem_align(T a) { return (((a - 1) >> TFQMRGPU_MEMORY_ALIGNMENT) + 1) << TFQMRGPU_MEMORY_ALIGNMENT; }

    tfqmrgpuStatus_t tfqmrgpuPrintError(tfqmrgpuStatus_t const status) {
        tfqmrgpuStatus_t stat{status};
        char const key = stat / TFQMRGPU_CODE_CHAR;
                  stat -= key * TFQMRGPU_CODE_CHAR;
        uint32_t const line = stat / TFQMRGPU_CODE_LINE;
                  stat -= line * TFQMRGPU_CODE_LINE;
        switch (stat) {
            case TFQMRGPU_STATUS_SUCCESS:          debug_printf("tfQMRgpu: Success!\n");           break;
            case TFQMRGPU_STATUS_ALLOCATION_FAILED: std::printf("tfQMRgpu: Allocation failed!\n"); break;
            case TFQMRGPU_POINTER_INVALID:          std::printf("tfQMRgpu: Pointer invalid!\n");   break;
            case TFQMRGPU_STATUS_MAX_ITERATIONS:    std::printf("tfQMRgpu: Max number of iterations exceeded!\n");       break;
            case TFQMRGPU_STATUS_BREAKDOWN:         std::printf("tfQMRgpu: All components have broken down!\n");         break;
            case TFQMRGPU_NO_IMPLEMENTATION:        std::printf("tfQMRgpu: Missing implementation at line %d!\n", line); break;
            case TFQMRGPU_UNDOCUMENTED_ERROR:       std::printf("tfQMRgpu: Undocumented error at line %d!\n",     line); break;
            case TFQMRGPU_BLOCKSIZE_MISSING:        std::printf("tfQMRgpu: Missing blocksize %d x %d!\n",              key, line); break;
            case TFQMRGPU_TANSPOSITION_UNKNOWN:     std::printf("tfQMRgpu: Unknown transposition '%c' at line %d!\n",  key, line); break;
            case TFQMRGPU_VARIABLENAME_UNKNOWN:     std::printf("tfQMRgpu: Unknown variable name '%c' at line %d!\n",  key, line); break;
            case TFQMRGPU_DATALAYOUT_UNKNOWN:       std::printf("tfQMRgpu: Unknown data layout '%c' at line %d!\n", 20+key, line); break;
            case TFQMRGPU_PRECISION_MISSMATCH:      std::printf("tfQMRgpu: Missmatch in precision '%c' at line %d!\n", key, line); break;
            default:                                std::printf("tfQMRgpu: Unknown status= %d at line %d!\n", status, line); break;
        } // switch status
        std::fflush(stdout);
        return TFQMRGPU_STATUS_SUCCESS;
    } // printError

    tfqmrgpuStatus_t tfqmrgpuCreateHandle(tfqmrgpuHandle_t *handle) { // out: opaque handle for the tfqmrgpu library.
        if (nullptr != *handle) return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__;
        *handle = (tfqmrgpuHandle_t) new tfq_handle_t(); // create new and cast pointer
        return (nullptr != *handle)? TFQMRGPU_STATUS_SUCCESS : TFQMRGPU_STATUS_ALLOCATION_FAILED;
    } // createHandle

    tfqmrgpuStatus_t tfqmrgpuDestroyHandle(tfqmrgpuHandle_t handle) { // inout: opaque handle for the tfqmrgpu library.
        if (nullptr == handle) return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__;
        delete (tfq_handle_t*) handle; // cast pointer and delete
        return TFQMRGPU_STATUS_SUCCESS;
    } // destroyHandle


    tfqmrgpuStatus_t tfqmrgpuSetStream(tfqmrgpuHandle_t handle, // inout: opaque handle for the tfqmrgpu library.
        cudaStream_t const streamId) { // in: GPU stream to be used by tfqmrgpu
        ((tfq_handle_t*) handle)->streamId = streamId;
        return TFQMRGPU_STATUS_SUCCESS;
    } // setStream

    tfqmrgpuStatus_t tfqmrgpuGetStream(tfqmrgpuHandle_t handle, // in: opaque handle for the tfqmrgpu library.
        cudaStream_t      *streamId) { // out: GPU stream used by tfqmrgpu
        *streamId = ((tfq_handle_t*) handle)->streamId;
        return TFQMRGPU_STATUS_SUCCESS;
    } // getStream

    tfqmrgpuStatus_t tfqmrgpu_bsrsv_createPlan(
          tfqmrgpuHandle_t handle // none: opaque handle for the tfqmrgpu library.
        , tfqmrgpuBsrsvPlan_t *plan // out: newly created plan
        , int     const mb          // in: number of block rows in A, X and B == number of block columns in A
        , int32_t const *bsrRowPtrA // in: integer array of mb+1 elements that contains the start of every block row of A and the end of the last block row of A plus one.
        , int     const nnzbA       // in: number of nonzero blocks of matrix A
        , int32_t const *bsrColIndA // in: integer array of nnzbA ( = bsrRowPtrA[mb] - bsrRowPtrA[0] ) column indices of the nonzero blocks of matrix A.
        , int32_t const *bsrRowPtrX // in: integer array of mb+1 elements that contains the start of every block row of X and the end of the last block row of X plus one.
        , int     const nnzbX       // in: number of nonzero blocks of matrix X
        , int32_t const *bsrColIndX // in: integer array of nnzbX ( = bsrRowPtrX[mb] - bsrRowPtrX[0] ) column indices of the nonzero blocks of matrix X.
        , int32_t const *bsrRowPtrB // in: integer array of mb+1 elements that contains the start of every block row of B and the end of the last block row of B plus one.
        , int     const nnzbB       // in: number of nonzero blocks of matrix B, nnzbB must be less or equal to nnzbX.
        , int32_t const *bsrColIndB // in: integer array of nnzbB ( = bsrRowPtrB[mb] - bsrRowPtrB[0] ) column indices of the nonzero blocks of matrix B.
        , int     const indexOffset // in: indexOffset=0(C-style) or indexOffset=1(Fortran) for RowPtr and ColInd arrays    
    ) {
        debug_printf("tfqmrgpu_bsrsv_createPlan(handle=%p, *plan=%p, mb=%d, \n"
               "         bsrRowPtrA=%p, nnzbA=%d, bsrColIndA=%p, \n"
               "         bsrRowPtrX=%p, nnzbX=%d, bsrColIndX=%p, \n"
               "         bsrRowPtrB=%p, nnzbB=%d, bsrColIndB=%p, indexOffset=%d)\n",
               handle, *plan, mb,             bsrRowPtrA, nnzbA, bsrColIndA, 
               bsrRowPtrX, nnzbX, bsrColIndX, bsrRowPtrB, nnzbB, bsrColIndB, indexOffset);
        std::fflush(stdout);

        if (nullptr != *plan)   return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__; // requirement that *plan == nullptr on entry.

        // compute Y = A*X, minimize |Y - B| to solve A*X == B
        // nnzbY == nnzbX

        // static plausibility checks
        int const nnzbY = nnzbX; // copy number of non-zero elements
        if (mb < 1)             return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__; // at least one row/column needs to be there
        if (nnzbB > nnzbY)      return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__; // the non-zero pattern of B must be a true subset of that of X or Y.
        if (nnzbA > mb*mb)      return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__; // the operator A is assumed logically square, mb*mb is the upper bound.
        if (nnzbA != bsrRowPtrA[mb] - bsrRowPtrA[0])  return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__; // the operator A is not sane
        if (nnzbX != bsrRowPtrX[mb] - bsrRowPtrX[0])  return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__; // the operator X is not sane
        if (nnzbB != bsrRowPtrB[mb] - bsrRowPtrB[0])  return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__; // the operator B is not sane

        bsrsv_plan_t* const p = new bsrsv_plan_t(); // allocate the plan in host memory
        p->cpu_mem = sizeof(bsrsv_plan_t); // init host memory usage with the memory capacity required by the struct itself
        p->gpu_mem = 0; // initialize for GPU memory count

        p->nRows = mb;
        p->nnzbA = nnzbA;

        int const C0F1 = indexOffset; // abbreviate start index C/C++:0, Fortran:1

        { // in this scope we compute the multiplication index pair list for Y = A * X

            // the bsrY sparsity pattern is equal to the bsrX sparsity pattern
            auto const bsrRowPtrY = bsrRowPtrX; // copy pointer
            auto bsrColIndY = bsrColIndX; // copy pointer

            p->pairs.clear();
            size_t const estimate_n_pairs = (nnzbY * nnzbA) / mb; // approximate number of block operations
            debug_printf("tfqmrgpu_bsrsv_createPlan tries to reserve %ld pairs\n", estimate_n_pairs);
            p->pairs.reserve(2 * estimate_n_pairs); // factor 2 as we always save pairs of indices

            p->starts.clear();
            p->starts.reserve(nnzbY + 1); // exact size

            for (auto irow = 0; irow < mb; ++irow) {
                for (auto inzy = bsrRowPtrY[irow] - C0F1; inzy < bsrRowPtrY[irow + 1] - C0F1; ++inzy) {
                    auto const jcol = bsrColIndY[inzy]; // warning, jcol starts from 1 in Fortran
                    // now compute Y[irow][jcol] = sum_k A[irow][kcol] * X[krow][jcol] with k==kcol==krow
                    {
                        int const start_index = p->pairs.size()/2;
                        p->starts.push_back(start_index);
                    }
                    for (auto inza = bsrRowPtrA[irow] - C0F1; inza < bsrRowPtrA[irow + 1] - C0F1; ++inza) {
                        auto const kcol = bsrColIndA[inza] - C0F1;
                        auto const krow = kcol;
                        auto const inzx = find_in_array(bsrRowPtrX[krow] - C0F1, // begin
                                                        bsrRowPtrX[krow + 1] - C0F1, // end
                                                        jcol, // try to find this value
                                                        bsrColIndX); // in this array
                        if (inzx >= 0) {
                            p->pairs.push_back(inza);
                            p->pairs.push_back(inzx);
                        } // exists
                    } // inza
                } // inzy
            } // irow
            {   // this last entry is very important for the sparse matrix format
                int const start_index = p->pairs.size()/2;
                p->starts.push_back(start_index);
            }
            assert(nnzbY + 1 == p->starts.size()); // sanity check

            debug_printf("# found %ld pairs in A*X multiplication\n", p->pairs.size()/2); // log output

            p->pairs.shrink_to_fit(); // free unused host memory
#ifdef DEBUG
            std::printf("# p->pairs.data()  = %p\n", (char*)(p->pairs.data()));
            std::printf("# p->starts.data() = %p\n", (char*)(p->starts.data()));
#endif // DEBUG
            p->cpu_mem += p->starts.size() * sizeof(int); // register host memory usage in Byte
            p->cpu_mem += p->pairs.size()  * sizeof(int); // register host memory usage in Byte
        } // scope


        { // in this scope we check if B is a true subset of X
          // and compute the sparse subset list for operations of type X -= B or X += B
            p->subset.clear();
            p->subset.reserve(nnzbB); // exact size
            for (auto irow = 0; irow < mb; ++irow) {
                for (auto inzb = bsrRowPtrB[irow] - C0F1; inzb < bsrRowPtrB[irow + 1] - C0F1; ++inzb) {
                    auto const inzx = find_in_array(bsrRowPtrX[irow] - C0F1, // begin
                                                    bsrRowPtrX[irow + 1] - C0F1, // end
                                                    bsrColIndB[inzb], // try to find this value
                                                    bsrColIndX); // in this array
                    if (inzx < 0) return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__; // B is not a true subset of X
                    p->subset.push_back(inzx); // store the block index into the value array of X at which B is also non-zero.
                } // inzb
            } // irow

            p->cpu_mem += p->subset.size() * sizeof(int); // register host memory usage in Byte
        } // scope


        int nb{0}; // number of block columns

        { // in this scope we try to find the number of block columns in X and B
          // and we create a compressed copy of the bsrColIndX list called colindx
            int nc{0}; // preliminary number of columns computed via the range of indices

            int min_colInd = 2e9, max_colInd = -min_colInd; // init as close to the largest int32_t
            for (auto inzx = 0; inzx < nnzbX; ++inzx) {
                auto const jcol = bsrColIndX[inzx]; // we do not need to subtract the Fortran 1 here.
                min_colInd = std::min(min_colInd, jcol); // find the minimum index
                max_colInd = std::max(max_colInd, jcol); // find the maxmimum index
            } // inzx
            nc = 1 + max_colInd - min_colInd; // preliminary number of columns computed via the range of indices
            if (nc < 1) return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__; // at least one column must be in X and B

            // check if all indices in the range [min_colInd, max_colInd] are touched
            std::vector<int> nRowsPerColX(nc, 0);
            for (auto inzx = 0; inzx < nnzbX; ++inzx) {
                auto const jc = bsrColIndX[inzx] - min_colInd;
                ++nRowsPerColX[jc];
            } // inzx

            std::vector<int> translate_jc2jb(nc);
            unsigned nempty{0};
            nb = 0;
            for (auto jc = 0; jc < nc; ++jc) {
                if (nRowsPerColX[jc] < 1) {
                    translate_jc2jb[jc] = -1; // empty column
                    ++nempty;
                } else {
                    translate_jc2jb[jc] = nb; // valid column
                    ++nb;
                }
            } // jc
            // now nb is the number of block columns after filtering the empty columns

            // warn if there are empty columns as these should be erased before. Is erasing really necessary?
            if (nempty > 0) {
                debug_printf("# found %d columns without non-zero entries!\n", nempty); // warning output
            } // nempty

            p->colindx.clear();
            p->colindx.resize(nnzbX); // exact size

            p->original_bsrColIndX.clear();
            p->original_bsrColIndX.resize(nb); // exact size

            for (auto inzx = 0; inzx < nnzbX; ++inzx) {
                auto const jc = bsrColIndX[inzx] - min_colInd; // jc in [0, nc)
                auto const jb = translate_jc2jb[jc]; // jb in [0, nb)
                p->colindx[inzx] = jb; // or p->colindx.push_back(jb); // but then we need reserve instead of resize above
                assert(-1 != jb);
                p->original_bsrColIndX[jb] = bsrColIndX[inzx]; // retrieval information for debugging
            } // inzx

            p->cpu_mem += p->colindx.size() * sizeof(int); // register host memory usage in Byte
            p->cpu_mem += p->original_bsrColIndX.size() * sizeof(int); // register host memory usage in Byte
        } // scope
        p->nCols = nb; // store number of block columns

        p->pBuffer = nullptr; // init pointer copy to device memory (which will be allocated by the user)

        p->flops_performed_all = 0; // init
        p->flops_performed    = -1; // init impossible
        p->iterations_needed  = -1; // init impossible

        debug_printf("# found %ld non-zero entries in X\n", p->colindx.size());
        assert(p->colindx.size() == nnzbX);

        *plan = (tfqmrgpuBsrsvPlan_t) p; // cast into opaque pointer type

        debug_printf("done tfqmrgpu_bsrsv_createPlan(handle=%p, *plan=%p, [internal p=%p] ...)\n", handle, *plan, p);
        return TFQMRGPU_STATUS_SUCCESS;
    } // analysis

    tfqmrgpuStatus_t tfqmrgpu_bsrsv_destroyPlan(
          tfqmrgpuHandle_t handle  // none: opaque handle for the tfqmrgpu library.
        , tfqmrgpuBsrsvPlan_t plan // plan is destroyed
    ) {
        auto const p = (bsrsv_plan_t*) plan;
        if (nullptr == p) return TFQMRGPU_POINTER_INVALID;
        delete p;
        return TFQMRGPU_STATUS_SUCCESS;
    } // destroyPlan


    tfqmrgpuStatus_t tfqmrgpu_bsrsv_bufferSize(
          tfqmrgpuHandle_t handle // in: opaque handle for the tfqmrgpu library.
        , tfqmrgpuBsrsvPlan_t plan // inout: plan becomes enriched by LM and doublePrecision
        , int const ldA         // in: Leading dimension for blocks in matrix A.
        , int const blockDim    // in: Block dimension of matrix A, blocks in A are square blocks. blockDim <= ldA
        , int const ldB         // in: Leading dimension for blocks in matrix B or X.
        , int const RhsBlockDim // in: Fast block dimension of matrix B or X
        , char const doublePrecision // in: Solver precision 'C':complex<float>, 'Z':complex<double>, 'M':load float and compute double.
        , size_t *pBufferSizeInBytes // out: number of bytes of the buffer used in the setMatrix, getMatrix and solve.
    ) {
        // query the necessary GPU memory buffer size
        int const LM = ldA;
        int const LN = ldB;
        if (LM != blockDim)     return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__; // so far, this library is not that flexible
        if (LM > LN)            return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__; // so far, this library is not that flexible
        if (LN != RhsBlockDim)  return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__; // so far, this library is not that flexible

        auto const p = (bsrsv_plan_t*)plan;

        switch (doublePrecision | IgnoreCase) {
            case 'f':
            case 'c': p->doublePrecision = 'c'; break;  // single precision complex
            case 'm': p->doublePrecision = 'm'; break;  // mixed  precision complex, ToDo: test
            case 'd':
            case 'z': p->doublePrecision = 'z'; break;  // double precision complex
            default : p->doublePrecision = 'z'; // default double precision complex
        } // doublePrecision

        p->LM = LM; // store the block size and precision information in the plan
        p->LN = LN; // store the number of columns in each block of X or B

        cudaStream_t streamId;
        {   auto const stat = tfqmrgpuGetStream(handle, &streamId);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }

        if (nullptr == pBufferSizeInBytes) return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__;

        bool const memcount = true;
        auto const status = mysolve(streamId, p, 0.0, 0, memcount); // call the solver in memcount-mode

        *pBufferSizeInBytes = p->gpu_mem; // requested minimum number of Bytes in device memory
        debug_printf("# plan for doublePrecision= %c and LM= %d, LN= %d needs %.3f MByte device memory\n",
                                     p->doublePrecision, p->LM, p->LN, p->gpu_mem*1e-6);
        return status;
    } // bufferSize


    tfqmrgpuStatus_t tfqmrgpu_bsrsv_setBuffer(
          tfqmrgpuHandle_t handle // in: opaque handle for the tfqmrgpu library.
        , tfqmrgpuBsrsvPlan_t plan // inout: set the plan-internal buffer variable
        , void* const pBuffer // in: pointer to GPU memory
    ) {
        if (nullptr == pBuffer) return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__;

        cudaStream_t streamId{0};
        {   auto const stat = tfqmrgpuGetStream(handle, &streamId);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }

        auto const p = (bsrsv_plan_t*) plan;
        p->pBuffer = (char*)pBuffer; // buffer setting

        { // random number generation scope
            auto const n_floats_in_v3 = p->vec3win.length/sizeof(float);
            auto const v3 = (float*)(p->pBuffer + p->vec3win.offset);
            debug_printf("# v3 has address %p\n", (void*)v3);
            auto const stat = tfqmrgpu::create_random_numbers(v3, n_floats_in_v3, streamId);
            {
                float first{0}, flast{0};
                get_data_from_gpu<float>(&first, v3, 1, 0, "first of v3");
                get_data_from_gpu<float>(&flast, &v3[n_floats_in_v3 - 1], 1, 0, "last of v3");
                debug_printf("# v3 has values %g ... %g\n", first, flast);
            }
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        } // scope

        { // in this scope we transfer the integer vectors 
          // that are filled during the analysis step.
            tfqmrgpu::transfer_index_lists(streamId, p);
        } // scope

        return TFQMRGPU_STATUS_SUCCESS;
    } // setBuffer

    // registers the GPU memory buffer pointer in the handle and calls the random number generator.
    tfqmrgpuStatus_t tfqmrgpu_bsrsv_getBuffer(
          tfqmrgpuHandle_t handle // none: opaque handle for the tfqmrgpu library.
        , tfqmrgpuBsrsvPlan_t plan // in: plan for bsrsv, read the plan-internal buffer variable
        , void* *pBuffer // out: pointer to GPU memory
    ) {
        *pBuffer = (void*)((bsrsv_plan_t*)plan)->pBuffer;
        if (nullptr == *pBuffer) return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_LINE*__LINE__;
        return TFQMRGPU_STATUS_SUCCESS;
    } // getBuffer


    tfqmrgpuStatus_t varname_filter(
          char const var
        , bsrsv_plan_t const *const p
        , size_t &size
        , char* &ptr
        , uint32_t &nnzb
        , uint32_t &nRows
        , uint32_t &nCols
        , unsigned const line
    ) {
        nRows = p->LM;
        nCols = p->LN;
        switch (var | IgnoreCase) {
            case 'a': ptr += p->matAwin.offset; size = p->matAwin.length; nnzb = p->nnzbA; nCols = p->LM; break;
            case 'b': ptr += p->matBwin.offset; size = p->matBwin.length; nnzb = p->subset.size();  break;
            case 'x': ptr += p->matXwin.offset; size = p->matXwin.length; nnzb = p->colindx.size(); break;
            // the passed variable name does not carry a meaning
            default: return TFQMRGPU_VARIABLENAME_UNKNOWN + TFQMRGPU_CODE_CHAR*var + TFQMRGPU_CODE_LINE*line; 
        } // switch var
        return TFQMRGPU_STATUS_SUCCESS;
    } // varname_filter

    tfqmrgpuStatus_t datalayout_filter(
          tfqmrgpuDataLayout_t const layout
        , unsigned const line
    ) {
        switch (layout) {
            case TFQMRGPU_LAYOUT_RRRRIIII: break; // native for this GPU solver
//          case TFQMRGPU_LAYOUT_RRIIRRII: break; // not implemented
            case TFQMRGPU_LAYOUT_RIRIRIRI: break; // native for e.g. Fortran complex arrays
            default: return TFQMRGPU_DATALAYOUT_UNKNOWN + TFQMRGPU_CODE_CHAR*layout + TFQMRGPU_CODE_LINE*line;
        } // switch var
        return TFQMRGPU_STATUS_SUCCESS;
    } // datalayout_filter

    tfqmrgpuStatus_t transposition_filter(
          char &trans // in: 'n','N', 't','T', '*', 'h','H','c','C', out: 'n' or 't'
        , double &sign_imag
        , unsigned const line
    ) {
        trans |= IgnoreCase; // convert trans to lowercase
        sign_imag = 1;
        switch (trans) {
            case 'h':
            case 'c': sign_imag = -1; trans = 't'; break; // transpose + conjugate // LAPACK uses 'c' for the Hermitian adjoint, but allow also 'H' or 'h'
            case '*': sign_imag = -1; trans = 'n'; break; //        only conjugate
            case 't': break; // transpose
            case 'n': break; // non-transpose
            default: return TFQMRGPU_TANSPOSITION_UNKNOWN + TFQMRGPU_CODE_CHAR*trans + TFQMRGPU_CODE_LINE*line;
        } // switch trans
        return TFQMRGPU_STATUS_SUCCESS;
    } // transposition_filter

    // asynchronous setting of matrix operands
    tfqmrgpuStatus_t tfqmrgpu_bsrsv_set_or_getMatrix(
          tfqmrgpuHandle_t handle // in: opaque handle for the tfqmrgpu library.
        , tfqmrgpuBsrsvPlan_t plan // inout: plan for bsrsv
        , char const var // in: selector which variable, only {'A', 'X', 'B'} allowed.
        , void*const values // pointer to read-only values, pointer is casted to float* or double*
        , char const doublePrecision // in: 'C','c':complex<float>, 'Z','z':complex<double>, 's' and 'd' are not supported.
        , int const ld // in: leading dimension of blocks in array val, ignored, use the block dimensions stored in plan
        , int const d2 // in:  second dimension of blocks in array val, ignored, use the block dimensions stored in plan
        , char const trans // in: transposition of the input matrix blocks.
        , tfqmrgpuDataLayout_t const layout
        , bool const is_get=false
    ) {
        debug_printf("# %s for operator \'%c\'  values=%p\n", __func__, var, values);
        {   auto const stat = datalayout_filter(layout, __LINE__);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }
        uint32_t nnzb{0}, nRows{0}, nCols{0};
        auto const p = (bsrsv_plan_t const*) plan;
        char* ptr = is_get ? nullptr : p->pBuffer;
        size_t size{0}; // size in Byte
        {   auto const stat = varname_filter(var, p, size, ptr, nnzb, nRows, nCols, __LINE__);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }
        if (nnzb < 1) return TFQMRGPU_STATUS_SUCCESS; // nothing to do
        assert(nullptr != ptr);

        double scal_imag{1};
        char Trans = trans | IgnoreCase; // non-const copy
        {   auto const stat = transposition_filter(Trans, scal_imag, __LINE__);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }

        if ('a' == (var | IgnoreCase)) {
            // internally, operator A is stored column major for coalesced memory access on the GPU
            if      ('n' == Trans) { Trans = 't'; }
            else if ('t' == Trans) { Trans = 'n'; }
            else { return TFQMRGPU_TANSPOSITION_UNKNOWN + TFQMRGPU_CODE_CHAR*trans + TFQMRGPU_CODE_LINE*__LINE__; }
            debug_printf("# tfqmrgpu_bsrsv_setMatrix: flip transposition "
              "'%c' to internal '%c' for operator '%c'\n", trans, Trans, var);
        } // only operator A

        auto const dp = ('z' == p->doublePrecision);
        if (('z' == (doublePrecision | IgnoreCase)) != dp) {
            std::printf("# mismatch: \'%c\' and plan says \'%c\'\n", doublePrecision, p->doublePrecision);
            return TFQMRGPU_PRECISION_MISSMATCH + TFQMRGPU_CODE_CHAR*doublePrecision + TFQMRGPU_CODE_LINE*__LINE__;
        }

        auto const byte_per_block = 2*nRows*nCols*(dp ? sizeof(double) : sizeof(float));
        assert(nnzb*byte_per_block == size);

        cudaStream_t streamId{0};
        {   auto const stat = tfqmrgpuGetStream(handle, &streamId);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }

        bool trans_in{false}, trans_out{false};
        tfqmrgpuDataLayout_t l_in, l_out;
        if (is_get) {
            trans_in = ('t' == Trans);
            l_in = TFQMRGPU_LAYOUT_RRRRIIII;
            l_out = layout;
        } else {
            l_in = layout;
            l_out = TFQMRGPU_LAYOUT_RRRRIIII;
            trans_out = ('t' == Trans);
            debug_printf("# start asynchronous memory transfer from the host to the GPU for operator '%c'\n", var);
            copy_data_to_gpu<char>(ptr, (char*)values, size, streamId);
            debug_printf("#  done asynchronous memory transfer from the host to the GPU for operator '%c'\n", var);
        } // set

        // for each block change data layout and (if necessary) transpose in-place on the GPU
        if (dp) {
            tfqmrgpu::transpose_blocks_kernel<double>
#ifndef HAS_NO_CUDA
                <<< nnzb, {nCols,nRows,1}, byte_per_block, streamId >>>
#endif // HAS_CUDA
                ((double*) ptr, nnzb, 1, scal_imag, l_in, l_out, trans_in, trans_out, var, nRows, nCols);
        } else {
            tfqmrgpu::transpose_blocks_kernel<float>
#ifndef HAS_NO_CUDA
                <<< nnzb, {nCols,nRows,1}, byte_per_block, streamId >>>
#endif // HAS_CUDA
                ((float *) ptr, nnzb, 1, scal_imag, l_in, l_out, trans_in, trans_out, var, nRows, nCols);
        } // dp

        if (is_get) {
            // start asynchronous memory transfer from the GPU to the host
            get_data_from_gpu<char>((char*)values, ptr, size, streamId);
        } // get

        return TFQMRGPU_STATUS_SUCCESS;
    } // set_or_getMatrix



    // asynchronous setting of matrix operands
    tfqmrgpuStatus_t tfqmrgpu_bsrsv_setMatrix(
          tfqmrgpuHandle_t handle // in: opaque handle for the tfqmrgpu library.
        , tfqmrgpuBsrsvPlan_t plan // inout: plan for bsrsv
        , char const var // in: selector which variable, only {'A', 'X', 'B'} allowed.
        , void const *const values // in: pointer to read-only values, pointer is casted to float* or double*
        , char const doublePrecision // in: 'c':complex<float>, 'z':complex<double>, 's' and 'd' are not supported.
        , int const ld // in: leading dimension of blocks in array val.
        , int const d2 // in:  second dimension of blocks in array val.
        , char const trans // in: transposition of the input matrix blocks.
        , tfqmrgpuDataLayout_t const layout
    ) {
#if 0
        uint32_t nnzb{0}, nRows{0}, nCols{0};

        auto const p = (bsrsv_plan_t*) plan;
        auto ptr = (char*) p->pBuffer; // device pointer
        size_t size{0}; // size in Byte
        {   auto const stat = varname_filter(var, p, size, ptr, nnzb, nRows, nCols, __LINE__);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }

        double scal_imag{1};
        char Trans = trans | IgnoreCase; // non-const copy
        {   auto const stat = transposition_filter(Trans, scal_imag, __LINE__);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }
        if ('a' == (var | IgnoreCase)) {
            // internally, operator A is stored column major for coalesced memory access on the GPU
            if      ('n' == Trans) { Trans = 't'; }
            else if ('t' == Trans) { Trans = 'n'; }
            else { return TFQMRGPU_TANSPOSITION_UNKNOWN + TFQMRGPU_CODE_CHAR*Trans + TFQMRGPU_CODE_LINE*__LINE__; }
            debug_printf("tfqmrgpu_bsrsv_setMatrix: flip transposition "
              "'%c' to internal '%c' for operator '%c'\n", trans, Trans, var);
        } // only operator A

        if (('z' == (doublePrecision | IgnoreCase)) != ('z' == p->doublePrecision)) {
            std::printf("# mismatch: \'%c\' and plan says \'%c\'\n", doublePrecision, p->doublePrecision);
            return TFQMRGPU_PRECISION_MISSMATCH + TFQMRGPU_CODE_CHAR*doublePrecision + TFQMRGPU_CODE_LINE*__LINE__;
        }

        cudaStream_t streamId{0};
        {   auto const stat = tfqmrgpuGetStream(handle, &streamId);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }

        debug_printf("start asynchronous memory transfer from the host to the GPU for operator '%c'\n", var);
        copy_data_to_gpu<char>(ptr, (char*)values, size, streamId);
        debug_printf(" done asynchronous memory transfer from the host to the GPU for operator '%c'\n", var);

        // change data layout and (if necessary) transpose in-place on the GPU
        auto const l_in = TFQMRGPU_LAYOUT_RIRIRIRI,
                  l_out = TFQMRGPU_LAYOUT_RRRRIIII;
        if ('z' == p->doublePrecision) {
            assert(nnzb*2*nRows*nCols*sizeof(double) == size);
            tfqmrgpu::transpose_blocks_kernel<double>
#ifndef HAS_NO_CUDA
                <<< nnzb, {nCols,nRows,1}, 2*nRows*nCols*sizeof(double), streamId >>>
#endif // HAS_CUDA
                ((double*) ptr, nnzb, 1, scal_imag, l_in, l_out, Trans, nCols, nRows);
        } else {
            assert(nnzb*2*nRows*nCols*sizeof(float)  == size);
            tfqmrgpu::transpose_blocks_kernel<float>
#ifndef HAS_NO_CUDA
                <<< nnzb, {nCols,nRows,1}, 2*nRows*nCols*sizeof(float) , streamId >>>
#endif // HAS_CUDA
                ((float *) ptr, nnzb, 1, scal_imag, l_in, l_out, Trans, nCols, nRows); 
        }

        return TFQMRGPU_STATUS_SUCCESS;
#else
        return tfqmrgpu_bsrsv_set_or_getMatrix(handle, plan, var, (void*)values, doublePrecision, ld, d2, trans, layout, false);
#endif
    } // setMatrix

    tfqmrgpuStatus_t tfqmrgpu_bsrsv_getMatrix(
          tfqmrgpuHandle_t handle // in: opaque handle for the tfqmrgpu library.
        , tfqmrgpuBsrsvPlan_t plan // in: plan for bsrsv
        , char const var // in: selector which variable, only 'X' or 'x' supported.
        , void       *const values // out: pointer to writeable values, pointer is casted to float* or double*
        , char const doublePrecision // in: 'c':complex<float>, 'z':complex<double>, 's' and 'd' are not supported.
        , int const ld // in: leading dimension of blocks in array val.
        , int const d2 // in:  second dimension of blocks in array val.
        , char const trans // in: transposition of the output matrix blocks.
        , tfqmrgpuDataLayout_t const layout
    ) {
#if 0
        {   auto const stat = datalayout_filter(layout, __LINE__);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }

        double scal_imag{1};
        char Trans = trans; // non-const copy
        {   auto const stat = transposition_filter(Trans, scal_imag, __LINE__);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }

        cudaStream_t streamId{0};
        {   auto const stat = tfqmrgpuGetStream(handle, &streamId);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }

        auto const p = (bsrsv_plan_t*) plan;
        if (('z' == (doublePrecision | IgnoreCase)) != ('z' == p->doublePrecision)) {
            std::printf("# mismatch: \'%c\' and plan says \'%c\'\n", doublePrecision, p->doublePrecision);
            return TFQMRGPU_PRECISION_MISSMATCH + TFQMRGPU_CODE_CHAR*doublePrecision + TFQMRGPU_CODE_LINE*__LINE__;
        }

        uint32_t nnzb{0}, nRows{0}, nCols{0};
        char* ptr = nullptr; // after solving, the memory window offsets are at the correct absolute positions
        // ToDo: we have to make sure that getMatrix is not called before the solver has been called with memcount=false

        size_t size{0}; // size in Byte
        {   auto const stat = varname_filter(var, p, size, ptr, nnzb, nRows, nCols, __LINE__);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }
        if ('x' != (var | IgnoreCase)) {
            // Only the download of operator 'X' is allowed.
            // Internally, operator A is stored column major, so downloading in e.g. with trans 'n'
            // would first modify the value of the operator A in-place on the GPU, so
            // solving again, e.g. with a modified right hand side B might lead to unexpected results
            // therefore, we do not allow downloading of operator A
            // similarly, B, therefore, we only allow to download operator X
            return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_CHAR*var + TFQMRGPU_CODE_LINE*__LINE__;
        } // only operator A

        // change data layout and (if necessary) transpose in-place on the GPU
        auto const l_in = TFQMRGPU_LAYOUT_RRRRIIII,
                  l_out = TFQMRGPU_LAYOUT_RIRIRIRI;
        if ('z' == p->doublePrecision) {
            assert(nnzb*2*nRows*nCols*sizeof(double) == size);
            tfqmrgpu::transpose_blocks_kernel<double>
#ifndef HAS_NO_CUDA
                <<< nnzb, {nCols,nRows,1}, 2*nRows*nCols*sizeof(double), streamId >>>
#endif // HAS_CUDA
                ((double*) ptr, nnzb, 1, scal_imag, l_in, l_out, Trans, nCols, nRows);
        } else {
            assert(nnzb*2*nRows*nCols*sizeof(float)  == size);
            tfqmrgpu::transpose_blocks_kernel<float>
#ifndef HAS_NO_CUDA
                <<< nnzb, {nCols,nRows,1}, 2*nRows*nCols*sizeof(float) , streamId >>>
#endif // HAS_CUDA
                ((float *) ptr, nnzb, 1, scal_imag, l_in, l_out, Trans, nCols, nRows); 
        }

        // start asynchronous memory transfer from the GPU to the host
        get_data_from_gpu<char>((char*)values, ptr, size, streamId);

        return TFQMRGPU_STATUS_SUCCESS;
#else
        if ('x' != (var | IgnoreCase)) {
            // Only the download of operator 'X' is allowed.
            // Internally, operator A is stored column major, so downloading in e.g. with trans 'n'
            // would first modify the value of the operator A in-place on the GPU, so
            // solving again, e.g. with a modified right hand side B might lead to unexpected results
            // therefore, we do not allow downloading of operator A
            // similarly, B, therefore, we only allow to download operator X
            return TFQMRGPU_UNDOCUMENTED_ERROR + TFQMRGPU_CODE_CHAR*var + TFQMRGPU_CODE_LINE*__LINE__;
        } // only operator A
        return tfqmrgpu_bsrsv_set_or_getMatrix(handle, plan, var, values, doublePrecision, ld, d2, trans, layout, true);
#endif
    } // getMatrix


    tfqmrgpuStatus_t tfqmrgpu_bsrsv_solve(
          tfqmrgpuHandle_t handle // in: opaque handle for the tfqmrgpu library.
        , tfqmrgpuBsrsvPlan_t plan // inout: plan for bsrsv
        , double const threshold // in: convergence threshold
        , int const maxIterations // in: maximum number of solver iterations
    ) {
        cudaStream_t streamId;
        {   auto const stat = tfqmrgpuGetStream(handle, &streamId);
            if (TFQMRGPU_STATUS_SUCCESS != stat) return stat;
        }

        auto const p = (bsrsv_plan_t*) plan;

        return mysolve(streamId, p, threshold, maxIterations);
    } // solve (wrapper)

    tfqmrgpuStatus_t tfqmrgpu_bsrsv_getInfo(
          tfqmrgpuHandle_t handle // in: no function
        , tfqmrgpuBsrsvPlan_t plan // in: contains state
        , double *residuum_reached // out: residuum after iterations
        , int32_t *iterations_needed // out: number of iterations needed to converge
        , double *flops_performed // out: number of floating pointer operations performed for the last run
        , double *flops_performed_all // out: number of floating pointer operations performed since createPlan
    ) {
        auto const p = (bsrsv_plan_t*) plan; // convert opaque plan object
        int any{0};
        if (nullptr != residuum_reached   ) { ++any; *residuum_reached    = p->residuum_reached; }
        if (nullptr != iterations_needed  ) { ++any; *iterations_needed   = p->iterations_needed; }
        if (nullptr != flops_performed    ) { ++any; *flops_performed     = p->flops_performed;    }
        if (nullptr != flops_performed_all) { ++any; *flops_performed_all = p->flops_performed_all; }

        return any ? TFQMRGPU_STATUS_SUCCESS : TFQMRGPU_STATUS_NO_INFO_PASSED;
    } // getInfo

    // utilities for the Fortran interface
    tfqmrgpuStatus_t tfqmrgpuCreateWorkspace(
          void* *pBuffer
        , size_t const pBufferSizeInBytes
        , char const MemoryType
    ) {
        cudaError err;
        if ('m' == (MemoryType | IgnoreCase)) { // 'm' or 'M' stand for "managed"
            err = cudaMallocManaged(pBuffer, pBufferSizeInBytes);
        } else {
            err = cudaMalloc(pBuffer, pBufferSizeInBytes);
        }
        return (cudaSuccess == err) ? TFQMRGPU_STATUS_SUCCESS : TFQMRGPU_STATUS_ALLOCATION_FAILED;
    } // createWorkspace

    tfqmrgpuStatus_t tfqmrgpuDestroyWorkspace(void* pBuffer) {
        return cudaFree(pBuffer);
    } // destroyWorkspace


    tfqmrgpuStatus_t tfqmrgpu_bsrsv_z(
          int const mb // number of block rows and block columns in A
        , int const ldA // number of rows in a block
        , int const ldB // number of columns in a block of X or B
        , int32_t const *const rowPtrA
        , int const nnzbA
        , int32_t const *const colIndA
        , double  const *const Amat // assumed data layout double A[nnzbA][ldA][ldA][2]
        , char const transA
        , int32_t const *const rowPtrX
        , int const nnzbX
        , int32_t const *const colIndX
        , double        *const Xmat // assumed data layout double X[nnzbX][ldA][ldB][2]
        , char const transX
        , int32_t const *const rowPtrB
        , int const nnzbB
        , int32_t const *const colIndB
        , double  const *const Bmat // assumed data layout double B[nnzbB][ldA][ldB][2]
        , char const transB
        , int32_t *const iterations // on entry: max. number of iterations, on exit: needed number of iterations
        , float *const residual // on entry: required residuum for convergence, on exit: residdum reached
        , int const echo // verbosity to stdout
    ) {
        if (echo > 0) std::printf("# %s: mb= %d, ldA= %d, ldB= %d, iterations= %d, residual= %.1e\n", __func__, mb, ldA, ldB, *iterations, *residual);
        tfqmrgpuHandle_t handle{0};
        auto stat = tfqmrgpuCreateHandle(&handle);
        if (stat) { if (echo > 0) std::printf("# %s: tfqmrgpuCreateHandle returned %d\n", __func__, stat); return stat; }

        stat = tfqmrgpuSetStream(handle, 0); // set default stream
        if (stat) { if (echo > 0) std::printf("# %s: tfqmrgpuSetStream returned %d\n", __func__, stat); return stat; }

        tfqmrgpuBsrsvPlan_t plan{0};
        stat = tfqmrgpu_bsrsv_createPlan(handle, &plan, mb
                                  , rowPtrA, nnzbA, colIndA
                                  , rowPtrX, nnzbX, colIndX
                                  , rowPtrB, nnzbB, colIndB, 0);
        if (stat) { if (echo > 0) std::printf("# %s: tfqmrgpu_bsrsv_createPlan returned %d\n", __func__, stat); return stat; }

        size_t gpu_memory_size{0};
        stat = tfqmrgpu_bsrsv_bufferSize(handle, plan, ldA, ldA, ldB, ldB, 'z', &gpu_memory_size);
        if (stat) { if (echo > 0) std::printf("# %s: tfqmrgpu_bsrsv_bufferSize returned %d\n", __func__, stat); return stat; }

        void* gpu_memory_buffer{nullptr};
        stat = tfqmrgpuCreateWorkspace(&gpu_memory_buffer, gpu_memory_size, 'd'); // device memory
        if (stat) {
            if (echo > 0) std::printf("# %s: tfqmrgpuCreateWorkspace returned %d\n", __func__, stat);
            if (echo > 3) std::printf("# %s: probably running on hardware without GPUs\n", __func__);
            return stat;
        } // stat

        stat = tfqmrgpu_bsrsv_setBuffer(handle, plan, gpu_memory_buffer);
        if (stat) { if (echo > 0) std::printf("# %s: tfqmrgpu_bsrsv_setBuffer returned %d\n", __func__, stat); return stat; }

        stat = tfqmrgpu_bsrsv_setMatrix(handle, plan, 'A', Amat, 'z', ldA, ldA, 'n', TFQMRGPU_LAYOUT_RIRIRIRI);
        if (stat) { if (echo > 0) std::printf("# %s: tfqmrgpu_bsrsv_setMatrix(\'A\') returned %d\n", __func__, stat); return stat; }

        stat = tfqmrgpu_bsrsv_setMatrix(handle, plan, 'B', Bmat, 'z', ldB, ldA, 'n', TFQMRGPU_LAYOUT_RIRIRIRI);
        if (stat) { if (echo > 0) std::printf("# %s: tfqmrgpu_bsrsv_setMatrix(\'B\') returned %d\n", __func__, stat); return stat; }

        double const threshold = (nullptr != residual) ? *residual : 1e-9;
        int const maxiter = (nullptr != iterations) ? *iterations : 200;
        stat = tfqmrgpu_bsrsv_solve(handle, plan, threshold, maxiter);
        if (stat) { if (echo > 0) std::printf("# %s: tfqmrgpu_bsrsv_solve returned %d\n", __func__, stat); return stat; }

        double residuum{0}, flops{0}, flops_all{0};
        int32_t needed{0};
        stat = tfqmrgpu_bsrsv_getInfo(handle, plan, &residuum, &needed, &flops, &flops_all);
        if (stat) { if (echo > 0) std::printf("# %s: tfqmrgpu_bsrsv_getInfo returned %d\n", __func__, stat); return stat; }
        if (echo > 1) std::printf("# tfQMRgpu needed %d iterations to converge to %.1e using %g GFlop\n", needed, residuum, flops*1e-9);
        if (nullptr != residual) *residual = residuum;
        if (nullptr != iterations) *iterations = needed;

        stat = tfqmrgpu_bsrsv_getMatrix(handle, plan, 'X', Xmat, 'z', ldB, ldA, 'n', TFQMRGPU_LAYOUT_RIRIRIRI);
        if (stat) { if (echo > 0) std::printf("# %s: tfqmrgpu_bsrsv_getMatrix returned %d\n", __func__, stat); return stat; }

        stat = tfqmrgpuDestroyWorkspace(gpu_memory_buffer);
        if (stat) { if (echo > 0) std::printf("# %s: tfqmrgpuDestroyWorkspace returned %d\n", __func__, stat); return stat; }

        stat = tfqmrgpu_bsrsv_destroyPlan(handle, plan);
        if (stat) { if (echo > 0) std::printf("# %s: tfqmrgpu_bsrsv_destroyPlan returned %d\n", __func__, stat); return stat; }

        stat = tfqmrgpuDestroyHandle(handle);
        if (stat) { if (echo > 0) std::printf("# %s: tfqmrgpuDestroyHandle returned %d\n", __func__, stat); return stat; }

        return TFQMRGPU_STATUS_SUCCESS;
    } // tfqmrgpu_bsrsv_z
