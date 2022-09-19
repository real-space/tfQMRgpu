#pragma once

#include <vector> // std::vector<T>
#include <string> // std::string
#ifdef  FULL_DEBUG
  #include <cstdio> // std::printf
#endif // FULL_DEBUG

struct bsr_t {
    // sparse matrix structure
    unsigned nRows; // number of Rows
    unsigned nCols; // number of block columns
    unsigned nnzb;  // number of non-zero blocks
    std::vector<int> RowPtr; // [nRows + 1]
    std::vector<int> ColInd; // [nnzb]

    // block sparse matrix values
    unsigned fastBlockDim; // number of columns per block
    unsigned slowBlockDim; // number of rows per block
    std::vector<double> mat;

    std::string name;
}; // struct bsr_t


    template <typename int_t>
    inline int find_in_array(int const begin, int const end, int const value,
                             int_t const *array, int const not_found=-1) {
#ifdef  FULL_DEBUG
        std::printf("find_in_array(begin=%d, end=%d, value=%d, array=%p)\n", begin, end, value, array);
#endif // FULL_DEBUG

        // ToDo: the ColInd list is sorted ascendingly, so bisection search will be faster
        for(auto ind = begin; ind < end; ++ind) {
            if (value == array[ind]) return ind;
        } // ind
        return not_found;
    } // find_in_array

