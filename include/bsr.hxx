#pragma once

#include <vector> // std::vector<T>
#include <string> // std::string

struct bsr_t {
    // sparse matrix structure
    unsigned nRows; // number of Rows
    unsigned nCols; // number of block columns
    unsigned nnzb;  // number of non-zero blocks
    std::vector<int> RowPtr; // [nRows + 1]
    std::vector<int> ColInd; // [nnzb]

    // block sparse matrix values
    unsigned fastBlockDim;
    unsigned slowBlockDim;
    std::vector<double> mat;

    std::string name;
}; // struct bsr_t


    inline int find_in_array(int const begin, int const end, int const value,
                             int const *array, int const not_found=-1) {
#ifdef  FULL_DEBUG    
        printf("find_in_array(begin=%d, end=%d, value=%d, array=%p)\n", begin, end, value, array);
#endif
        // ToDo: the ColInd list is sorted ascendingly, so bisection search will be faster
        for(auto ind = begin; ind < end; ++ind) {
            if (value == array[ind]) return ind;
        } // ind
        return not_found;
    } // find_in_array

