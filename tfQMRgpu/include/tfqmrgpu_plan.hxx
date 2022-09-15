#pragma once

#include <cstdint> // uint16_t, uint32_t, int32_t, int64_t
#include <vector> // std::vector<T>
#include "tfqmrgpu_memWindow.h" // memWindow_t 

struct bsrsv_plan_t {

    char* pBuffer; // device memory buffer

    uint32_t nRows; // number of block rows
    uint32_t nCols; // number of block columns

    uint16_t LM; // rows in each block 
    uint16_t LN; // columns in each block
    char doublePrecision; // solve in 'C', 'Z' or 'M' (single, double or mixed precision)

    // for the matrix-matrix addition:
    std::vector<uint32_t> subset; // [nnzbB], list of inzbX-indices where B is also non-zero

    // for the inner products and axpy/xpay
    std::vector<uint16_t> colindx; // [nnzbX] compressed copy of input bsrColIndX, 1 is subtracted for Fortran

    // retrieval information for debugging
    std::vector<int32_t> original_bsrColIndX; // [nCols]

    // for memory management:
    size_t cpu_mem; // host memory requirement in Byte, including the struct itself
    size_t gpu_mem; // device memory requirement in Byte

    // memory positions
    memWindow_t matBwin;
    memWindow_t matXwin;
    memWindow_t vec3win;
    memWindow_t subsetwin;
    memWindow_t colindxwin;

    // stats:
    double residuum_reached;
    double flops_performed;
    double flops_performed_all;
    int iterations_needed;

    // the following members are required in the case of the
    // block-sparse matrix-matrix multiplication, but do not need to be used:
    std::vector<uint32_t> starts; // [nnzbX + 1] number of target elements plus one
    memWindow_t startswin;
    std::vector<uint32_t> pairs; // [nPairs*2], each pair is one block-times-block mutliplication
    memWindow_t pairswin;
    uint32_t nnzbA; // number of non-zero blocks in A
    memWindow_t matAwin;

}; // struct bsrsv_plan_t
