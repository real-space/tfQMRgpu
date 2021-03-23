#include <cstdio> // printf
#include <cstdlib> // std::atoi

#include "bsr.hxx" // bsr_t (Block-compressed Sparse Row descriptor)
#include "tfqmrgpu_example_reader.hxx" // tfqmrgpu_example_reader::read_in()

#ifdef  hasGPUbenchmarks
namespace GPUbench {

    // main benchmark for the performance of the tfQMR solver using the library interface
    int benchmark_tfQMRgpu_library(bsr_t const ABX[3], double const, int const, int const, char const);

    // side-benchmark for the performance of the block-matrix-matrix-multiplication only
    int benchmark_blockMatrixMatrixMultiplication(int const argc, char const *const argv[]);

} // GPUbench
#endif

int main(int const argc, char const *const argv[]) {

    if (argc < 2) { 
        printf("Usage:  %s  [file]  [tfQMR/multiply]  [float/double]  "
                        "[#repetitions]  [#iterations]  [#blocksize]\n", argv[0]);
        exit(1);
    } // not enough command line args passed

    char const *fnm   = (argc > 1)?  argv[1] : "problem"; // inputfile
    char const bench  = (argc > 2)? *argv[2] : 't'; // t=tfQMR, m=multiplication
#ifdef  hasGPUbenchmarks
    if ('m' == bench) return GPUbench::benchmark_blockMatrixMatrixMultiplication(argc, argv);
#else
    printf(" please activate -D hasGPUbenchmarks to run Benchmark='%c'\n", bench); return 41;
#endif

    char const flouble = (argc > 3)?          *argv[3]  : 'z'; // z:double, c:float
    int const nrep     = (argc > 4)? std::atoi(argv[4]) : 1; // number of repetitions
    int const MaxIter  = (argc > 5)? std::atoi(argv[5]) : 2000; // max. number of iteration

    printf("\n# read file %s as input.\n", fnm);
    bsr_t ABX[3];
    auto const tolerance = tfqmrgpu_example_reader::read_in(ABX, fnm);
    printf("# found tolerance %g\n", tolerance);
    printf("# Execute %d repetitions with max. %d iterations.\n", nrep, MaxIter);
    int const lsmd = ABX[0].fastBlockDim;

    printf("# requested precision = %c for LM = %d\n", flouble, lsmd);
#ifdef  hasGPUbenchmarks
    return GPUbench::benchmark_tfQMRgpu_library(ABX, tolerance, MaxIter, nrep, flouble);
#else
    printf(" please activate -D hasGPUbenchmarks\n"); return 42;
#endif
} // main
