#ifndef TFQMRGPU_H
#define TFQMRGPU_H

    typedef int   tfqmrgpuStatus_t; // return codes
    // opaque structs
    typedef void* tfqmrgpuHandle_t; // globale library handle pointer, need only one per application execution
    typedef int*  tfqmrgpuBsrsvPlan_t; // plan for bsrsv
    typedef int   tfqmrgpuDataLayout_t; // data layout of a block of complex numbers

    //
    //
    // tfqmrgpu API documentation
    //
    //
    tfqmrgpuStatus_t tfqmrgpuPrintError(tfqmrgpuStatus_t const status);

    // tfqmrgpuHandle_t handle = NULL; // user must perform this
    tfqmrgpuStatus_t tfqmrgpuCreateHandle(tfqmrgpuHandle_t *handle); // out: opaque handle for the tfqmrgpu library. 
    //               handle must be NULL on entry, handle needs to be passed in as pointer
    tfqmrgpuStatus_t tfqmrgpuDestroyHandle(tfqmrgpuHandle_t handle); // inout


    tfqmrgpuStatus_t tfqmrgpuSetStream(tfqmrgpuHandle_t handle, // inout
        cudaStream_t const streamId); // in: GPU stream to be used by tfqmrgpu
    tfqmrgpuStatus_t tfqmrgpuGetStream(tfqmrgpuHandle_t handle, // in
        cudaStream_t      *streamId); // out: GPU stream used by tfqmrgpu

    tfqmrgpuStatus_t tfqmrgpuCreateWorkspace(void* *pBuffer, size_t const pBufferSizeInBytes, char const memType);
    tfqmrgpuStatus_t tfqmrgpuDestroyWorkspace(void* pBuffer);

    ////////////////////// bsrsv specific routines //////////////////////////////////////////////////
    // bsrsv is a linear solve of A * X == B
    // with A, X and B are BSR (block compressed sparse row) formatted operators.
    // 
    // the tfqmrgpu_bsrsv_* routines are listed in the order of how they should be called in a default use case.

    // tfqmrgpuBsrsvPlan_t plan = NULL; // user must perform this before calling createPlan
    tfqmrgpuStatus_t tfqmrgpu_bsrsv_createPlan(tfqmrgpuHandle_t handle, // no interaction
        tfqmrgpuBsrsvPlan_t *plan, // out: pointer to the newly created plan
        int const mb, // in: number of block rows in A, X and B == number of block columns in A
        int const *bsrRowPtrA, // in: integer array of mb+1 elements that contains the start of every block row of A and the end of the last block row of A plus one.
        int const nnzbA,       // in: number of nonzero blocks of matrix A
        int const *bsrColIndA, // in: integer array of nnzbA ( = bsrRowPtrA[mb] - bsrRowPtrA[0] ) column indices of the nonzero blocks of matrix A.
        int const *bsrRowPtrX, // in: integer array of mb+1 elements that contains the start of every block row of X and the end of the last block row of X plus one.
        int const nnzbX,       // in: number of nonzero blocks of matrix X
        int const *bsrColIndX, // in: integer array of nnzbX ( = bsrRowPtrX[mb] - bsrRowPtrX[0] ) column indices of the nonzero blocks of matrix X.
        int const *bsrRowPtrB, // in: integer array of mb+1 elements that contains the start of every block row of B and the end of the last block row of B plus one.
        int const nnzbB,       // in: number of nonzero blocks of matrix B, nnzbB must be less or equal to nnzbX.
        int const *bsrColIndB, // in: integer array of nnzbB ( = bsrRowPtrB[mb] - bsrRowPtrB[0] ) column indices of the nonzero blocks of matrix B.
        int const indexOffset); // in: indexOffset=0(C-style) or indexOffset=1(Fortran) for RowPtr and ColInd arrays

    tfqmrgpuStatus_t tfqmrgpu_bsrsv_destroyPlan(tfqmrgpuHandle_t handle, // no interaction
                                                tfqmrgpuBsrsvPlan_t plan); // in: pointer to the plan to be destroyed
    // plan = NULL; // user must perform this after calling destroyPlan

    tfqmrgpuStatus_t tfqmrgpu_bsrsv_bufferSize(tfqmrgpuHandle_t handle, // in
        tfqmrgpuBsrsvPlan_t plan, // inout: plan becomes enriched by block size information
        int const ldA,         // in: Leading dimension for blocks in matrix A.
        int const blockDim,    // in: Block dimension of matrix A, blocks in A are square blocks. blockDim <= ldA
        int const ldB,         // in: Leading dimension for blocks in matrix B or X.
        int const RhsBlockDim, // in: Fast block dimension of matrix B or X, RhsBlockDim <= ldB.
        char const doublePrecision, // in: Solver precision 'C':complex<float>, 'Z':complex<double>, 'M':start with float and converge double.
        size_t *pBufferSizeInBytes); // out: number of bytes of the buffer used in the setMatrix, getMatrix and solve.
    // returns the computed size to be allocated by cudaMalloc

    // void* pBuffer = cudaMalloc(pBufferSizeInBytes); // user must perform this

    // registers the GPU memory buffer pointer in the handle and calls the random number generator.
    tfqmrgpuStatus_t tfqmrgpu_bsrsv_setBuffer(tfqmrgpuHandle_t handle, // in
        tfqmrgpuBsrsvPlan_t plan, // inout: the buffer is registered inside the plan
        void* const pBuffer); // in: pointer to GPU memory buffer

    tfqmrgpuStatus_t tfqmrgpu_bsrsv_getBuffer(tfqmrgpuHandle_t handle, // in
        tfqmrgpuBsrsvPlan_t plan, // in
        void* *pBuffer); // out: pointer to GPU memory saved in handle

    tfqmrgpuStatus_t tfqmrgpu_bsrsv_setMatrix(tfqmrgpuHandle_t handle, // inout
        tfqmrgpuBsrsvPlan_t plan, // inout:
        char const var, // in: selector which variable, only {'A', 'X', 'B'} allowed.
        void const *val, // in: pointer to read-only values, pointer is casted to double* if 'Z'==doublePrecision or to float* if 'C'==doublePrecision
        char const doublePrecision, // in: 'C':complex<float>, 'Z':complex<double>, 'S' and 'D' are not supported.
        int const ld, // in: leading dimension of blocks in array val.
        char const trans, // in: transposition of the input matrix blocks.
        tfqmrgpuDataLayout_t const layout); // in: input data layout {RIRIRIRI(Fortran), RRRRIIII(native)}

    tfqmrgpuStatus_t tfqmrgpu_bsrsv_getMatrix(tfqmrgpuHandle_t handle, // inout
        tfqmrgpuBsrsvPlan_t plan, // in:
        char const var, // in: selector which variable, only 'X' supported.
        void       *val, // in: pointer to values to be written, pointer is casted to double* if 'Z'==doublePrecision or to float* if 'C'==doublePrecision
        char const doublePrecision, // in: 'C':complex<float>, 'Z':complex<double>, 'S' and 'D' are not supported.
        int const ld, // in: leading dimension of blocks in array val. -> See my comment above.
        char const trans, // in: transposition of the output matrix blocks.
        tfqmrgpuDataLayout_t const layout); // in: output data layout {RIRIRIRI(Fortran), RRRRIIII(native)}

    tfqmrgpuStatus_t  tfqmrgpu_bsrsv_solve(tfqmrgpuHandle_t handle, // inout
        tfqmrgpuBsrsvPlan_t plan, // inout:
        double const threshold, // in: convergence threshold
        int const maxIterations); // in: maximum number of solver iterations

    tfqmrgpuStatus_t tfqmrgpu_bsrsv_getInfo(tfqmrgpuHandle_t handle, // inout
        tfqmrgpuBsrsvPlan_t plan, // in:
        double *residuum_reached, // out: residuum after iterations
        int *iterations_needed, // out: number of iterations needed to converge
        double *flops_performed, // out: number of floating pointer operations performed for the last run
        double *flops_performed_all); // out: number of floating pointer operations performed since createPlan



    // tfqmrgpu CONSTANTS

    tfqmrgpuStatus_t const TFQMRGPU_STATUS_SUCCESS = 0;

    // error codes
    tfqmrgpuStatus_t const TFQMRGPU_STATUS_MAX_ITERATIONS    = 9; // ToDo: adjust value of this flag
    tfqmrgpuStatus_t const TFQMRGPU_STATUS_BREAKDOWN         = 6;
    tfqmrgpuStatus_t const TFQMRGPU_STATUS_ALLOCATION_FAILED = 4;
    tfqmrgpuStatus_t const TFQMRGPU_STATUS_RANDOM_GEN_FAILED = 5;
    tfqmrgpuStatus_t const TFQMRGPU_POINTER_INVALID          = 7;
    // for the following error codes, the throwing source line can be extracted from bit #8 on
    tfqmrgpuStatus_t const TFQMRGPU_CODE_LINE             = 1 << 8; // roughly 4 decimal digits
    tfqmrgpuStatus_t const TFQMRGPU_NO_IMPLEMENTATION     = 19; //
    tfqmrgpuStatus_t const TFQMRGPU_BLOCKSIZE_MISSING     = 12; //
    tfqmrgpuStatus_t const TFQMRGPU_UNDOCUMENTED_ERROR    = 14; //
    // for the following error codes, a char can be extracted from bin #24 on
    tfqmrgpuStatus_t const TFQMRGPU_CODE_CHAR             = 1 << 24; // only valid ASCII chars are expected
    tfqmrgpuStatus_t const TFQMRGPU_TANSPOSITION_UNKNOWN  = 17; //
    tfqmrgpuStatus_t const TFQMRGPU_VARIABLENAME_UNKNOWN  = 18; //
    tfqmrgpuStatus_t const TFQMRGPU_DATALAYOUT_UNKNOWN    = 15; //
    tfqmrgpuStatus_t const TFQMRGPU_PRECISION_MISSMATCH   = 16; //

    // block shape is assumed 2x2 for simplicity here. 0:real part, 1:imaginary part
    tfqmrgpuDataLayout_t const TFQMRGPU_LAYOUT_RRRRIIII = 15; // 0b00001111; // native layout for the GPU version, real and imag part of each block are separated.
    tfqmrgpuDataLayout_t const TFQMRGPU_LAYOUT_RRIIRRII = 51; // 0b00110011; // intermediate layout, ever used?
    tfqmrgpuDataLayout_t const TFQMRGPU_LAYOUT_RIRIRIRI = 85; // 0b01010101; // default host layout, real and imag parts are interleaved.


    // tfqmrgpu configuration:

    size_t const TFQMRGPU_MEMORY_ALIGNMENT = 8; // 8:256 Byte
    int    const TFQMRGPU_NUMBER_OF_INSTANCES_OF_X = 7; // need 7+1 when preprocessing is active

#endif // TFQMRGPU_H
