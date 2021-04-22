/*
 *  This program reads example problems
 *  stored in the extendable markup languange (XML)
 *  for the tfQMRgpu library which solves A*X==B.
 *          tfQMRgpu stands for the 
 *          transpose-free quasi minimal residual (QMR)
 *          method implemented on CUDA-enabled GPUs.
 *
 */

// To test the functionality of tfqmrgpu_example_xml_reader.hxx:
//      g++ -std=c++11 -I../include tfqmrgpu_example_xml_reader.cxx && ./a.out

#include "tfqmrgpu_example_xml_reader.hxx" // ::read_in, bsr_t

int main(int argc, char *argv[]) {
    // parse command line arguments or set default values
    char const *executable = (argc > 0) ? argv[0] : __FILE__;
    char const *filename   = (argc > 1) ? argv[1] : "FD_problem.xml";
    int  const echo        = (argc > 2) ? std::atoi(argv[2]) : 5;

    bsr_t ABX[3];
    if (echo > 0) std::printf("# %s: read_in(\"%s\")\n\n", __FILE__, filename);
    auto const tolerance = tfqmrgpu_example_xml_reader::read_in(ABX, filename, echo);
    if (echo > 0) std::printf("\n# %s: tolerance= %.3e\n", __FILE__, tolerance);
    return 0;
} // main
