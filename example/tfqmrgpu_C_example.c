#include <stdio.h> // printf
#include <stdint.h> // int32_t, int8_t
#include <stdlib.h> // atoi, atof, rand, RAND_MAX
#include <math.h> // sqrt
#include <assert.h> // assert

#define   DEBUG
#ifdef    DEBUG
    #define debug_printf(...) printf(__VA_ARGS__)
#else  // DEBUG
    #define debug_printf(...)
#endif // DEBUG

#ifdef    HAS_TFQMRGPU
    typedef size_t cudaStream_t;
    #include "../tfQMRgpu/include/tfqmrgpu.h" // C-interface for the tfQMRgpu library
#endif // HAS_TFQMRGPU

int main(int const argc, char const *const argv[]) {

    int const nRows    = (argc > 1)? atoi(argv[1]) : 0; // number of block rows
    int const ldA      = (argc > 2)? atoi(argv[2]) : 8; // number of rows per block
    int const ldB      = (argc > 3)? atoi(argv[3]) : ldA; // number of columns per block in X and B
    float const pA     = (argc > 4)? atof(argv[4]) : .125; // filling factor for A
    float const pX     = (argc > 5)? atof(argv[5]) : .500; // filling factor for X
    float const pB     = (argc > 6)? atof(argv[6]) : .125; // filling factor for B, relative to X
    int const max_it   = (argc > 7)? atoi(argv[7]) : 100;  // max. number of iterations
    float const thres  = (argc > 8)? atof(argv[8]) : 1e-6; // threshold for convergence
    int const echo     = (argc > 9)? atoi(argv[9]) : 0; // verbosity, 0: no output, 9:debug
    int const nCols    = (argc >10)? atoi(argv[10]) : nRows/2+1; // number of block columns

    if (nRows < 1) {
        printf("# Usage: %s nRows>0 ldA=%d ldB=%d fillA=%.3f fillX=%.3f fillB=%.3f "
               "max_iterations=%d threshold=%.1e echo=%d nCols=nRows/2+1\n",
               argv[0], ldA, ldB, pA, pX, pB, max_it, thres, echo);
        return -1; // error
    }

    debug_printf("# %s: nRows=%d, ldA=%d, ldB=%d, "
        "A %.1f%% filled, X %.1f%% filled, B %.1f%% of X, "
        "max_iterations=%d, threshold=%.1e, echo=%d, nCols=%d\n",
        __FILE__, nRows, ldA, ldB, pA*100, pX*100, pB*100, max_it, thres, echo, nCols);

    int32_t* const rowPtrA = (int32_t*)malloc((nRows + 1)*sizeof(int32_t));
    int32_t* const rowPtrX = (int32_t*)malloc((nRows + 1)*sizeof(int32_t));
    int32_t* const rowPtrB = (int32_t*)malloc((nRows + 1)*sizeof(int32_t));

    // create a random operator shapes for A, X and B
    int8_t* const nzA = (int8_t*)malloc(nRows*nRows*sizeof(int8_t));
    int8_t* const nzX = (int8_t*)malloc(nRows*nCols*sizeof(int8_t));
    int8_t* const nzB = (int8_t*)malloc(nRows*nCols*sizeof(int8_t));

    // convert operators into block-sparse row (BSR) format
    int nnzbA = 0;
    int nnzbX = 0;
    int nnzbB = 0;

    rowPtrA[0] = 0; // initialize the prefetch sum
    rowPtrX[0] = 0; // initialize the prefetch sum
    rowPtrB[0] = 0; // initialize the prefetch sum
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nRows; ++j) {
            nzA[i*nRows + j] = (rand() < .125*RAND_MAX) + 2*(i == j);
            nnzbA += (nzA[i*nRows + j] != 0); // count the number of nonzero blocks in A
        }
        for (int j = 0; j < nCols; ++j) {
            int8_t const x = (rand() < .5*RAND_MAX) + (i == j);
            nzX[i*nCols + j] = x;
            nzB[i*nCols + j] = x*(rand() < .125*RAND_MAX) + (i == j);
            nnzbX += (nzX[i*nCols + j] != 0); // count the number of nonzero blocks in X
            nnzbB += (nzB[i*nCols + j] != 0); // count the number of nonzero blocks in B
        }
        rowPtrA[i + 1] = nnzbA;
        rowPtrX[i + 1] = nnzbX;
        rowPtrB[i + 1] = nnzbB;
    }

    debug_printf("# %s: nonzero blocks %d of %d in A, %d of %d in X, %d of %d in B\n",
                   __FILE__, nnzbA, nRows*nRows, nnzbX, nRows*nCols, nnzbB, nnzbX);

    int32_t* const colIndA = (int32_t*)malloc(nnzbA*sizeof(int32_t));
    int32_t* const colIndX = (int32_t*)malloc(nnzbX*sizeof(int32_t));
    int32_t* const colIndB = (int32_t*)malloc(nnzbB*sizeof(int32_t));

    double* const Amat = (double*)malloc(nnzbA*ldA*ldA*2*sizeof(double));
    double* const Xmat = (double*)malloc(nnzbX*ldA*ldB*2*sizeof(double));
    double* const Bmat = (double*)malloc(nnzbB*ldA*ldB*2*sizeof(double));

    double const rand_denom = 1./RAND_MAX;
    for (int i = 0; i < nRows; ++i) {
        int inzbA = rowPtrA[i];
        int inzbX = rowPtrX[i];
        int inzbB = rowPtrB[i];
        for (int j = 0; j < nRows; ++j) {
            if (nzA[i*nRows + j] != 0) {
                colIndA[inzbA] = j;
                // fill block Amat[inzbA]
                for (int jic = 0; jic < ldA*ldA*2; ++jic) {
                    Amat[inzbA*ldA*ldA*2 + jic] = rand()*2*rand_denom - 1.; // in [-1., 1.]
                }
                ++inzbA;
            }
        }
        for (int j = 0; j < nCols; ++j) {
            if (nzX[i*nCols + j] != 0) {
                colIndX[inzbX] = j;
                // we could initialize Xmat[inzbX] here, but no need
                ++inzbX;
            }
            if (nzB[i*nCols + j] != 0) {
                colIndB[inzbB] = j;
                // fill block Bmat[inzbB]
                for (int ijc = 0; ijc < ldA*ldB*2; ++ijc) {
                    Bmat[inzbB*ldA*ldB*2 + ijc] = rand()*rand_denom - .5; // in [-.5, .5]
                }
                ++inzbB;
            }
        }
        assert(rowPtrA[i + 1] == inzbA);
        assert(rowPtrX[i + 1] == inzbX);
        assert(rowPtrB[i + 1] == inzbB);
    }

    free(nzA);
    free(nzX);
    free(nzB);

    int32_t iterations = max_it;
    float residual = thres;

#ifdef    HAS_TFQMRGPU
    int const status = tfqmrgpu_bsrsv_z(
            nRows, ldA, ldB,
            rowPtrA, nnzbA, colIndA, Amat, 't',
            rowPtrX, nnzbX, colIndX, Xmat, 'n',
            rowPtrB, nnzbB, colIndB, Bmat, 'n', 
            &iterations, &residual, 0, echo);
#else  // HAS_TFQMRGPU
    int const status = -1;
#endif // HAS_TFQMRGPU

    if (0 == status) {
        printf("# tfQMRgpu converged to %.1e in %d iterations\n", residual, iterations);
    } else {
#ifdef    HAS_TFQMRGPU
        char const *const msg = tfqmrgpuGetErrorString(status);
#else  // HAS_TFQMRGPU
        char const *const msg = "tfQMRgpu interface not included!";
#endif // HAS_TFQMRGPU
        printf("# tfQMRgpu returned an error status= %i, message=\"%s\"\n", status, msg);
    }

    free(rowPtrA);
    free(rowPtrX);
    free(rowPtrB);

    free(colIndA);
    free(colIndX);
    free(colIndB);

    free(Amat);
    free(Xmat);
    free(Bmat);

    return status;
} // main
