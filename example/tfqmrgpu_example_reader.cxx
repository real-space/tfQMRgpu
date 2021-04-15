// This code tests the legacy input reader for tfQMRgpu
// g++ -std=c++11 -I../include tfqmrgpu_example_reader.cxx && ./a.out tfqmrgpu_problem.0

#include <cmath> // std::abs, std::log10

#include "tfqmrgpu_example_reader.hxx" // ::read_in, bsr_t

int main(int const argc, char const *const argv[]) {
    bsr_t ABX[3];
    auto const tolerance = tfqmrgpu_example_reader::read_in(ABX, argv[1]);
    // echo
    for (auto op = ABX; op < 3+ABX; ++op) {
        for (auto iRow = 0; iRow < op->nRows; ++iRow) {
            std::cout << "# row " << iRow << std::endl;
            for (auto inzb = op->RowPtr[iRow]; inzb < op->RowPtr[iRow + 1]; ++inzb) {
                auto const iCol = op->ColInd[inzb];
                std::cout << "# row " << iRow << " col " << iCol << std::endl;
                for (auto islow = 0; islow < op->slowBlockDim; ++islow) {
                    for (auto ifast = 0; ifast < op->fastBlockDim; ++ifast) {
                        auto const idx = (inzb * op->slowBlockDim + islow) * op->fastBlockDim + ifast;
                        double const re = op->mat[idx*2], im = op->mat[idx*2 + 1];
                        printf("(%.1f,%.1f) ", -std::log10(std::abs(re)), -std::log10(std::abs(im))); // real and imaginary part
                    } // ifast
                    std::cout << std::endl;
                } // islow
            } // inzb
        } // iRow
    } // op
    return 0;
} // main
