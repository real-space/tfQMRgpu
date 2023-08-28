#pragma once
// This file is part of tfQMRgpu under MIT-License

#include <iostream> // std::cout
#include <cstdint> // uint64_t
#include <fstream> // std::ifstream
#include <string> // std::string
#include <sstream> // std::stringstream
#include <vector> // std::vector<T>
#include <cmath> // std::sqrt
#include <cassert> // assert

#include "bsr.hxx" // bsr_t

namespace tfqmrgpu_example_reader {

  // this is a modified version of https://stackoverflow.com/questions/16388510/evaluate-a-string-with-a-switch-in-c
  inline uint64_t constexpr str2ull(char const *const str, int const h=0) {
      // simple string hashing function
      return str[h] ? (str2ull(str, h + 1) * 33) ^ str[h] : 5381;
  } // str2ull

  inline int  mapA0B1X2(char const c) { return (c > 'B')?  2  : (c - 'A'); }
  inline char map0A1B2X(int  const i) { return (i >  1 )? 'X' : (i + 'A'); }


  inline double average_and_deviation(
        size_t const num_samples
      , double const sum_samples
      , double const sum_squares
      , double *dev=nullptr
  ) {
      if (num_samples < 1) { if (dev) *dev = -1.; return 0.; }
      double const denom = 1./num_samples;
      double const avg = sum_samples*denom;
      if(dev) *dev = std::sqrt(std::max(0., sum_squares*denom - avg*avg));
      return avg;
  } // average_and_deviation


  inline double read_in( // returns tolerance
        bsr_t ABX[3]
      , char const *const filename
  ) {
      double tolerance{0}; // init return value

      std::ifstream input(filename);
      if (!input) {
          std::cout << "# cannot read from file '" << filename << "'!" << std::endl;
          exit(__LINE__);
      }

      unsigned block_size{0}, nCols{0};
      bsr_t* op;

      std::string line;
      while (std::getline(input, line)) {
          std::string keyword;
          std::stringstream(line) >> keyword;

          switch (str2ull(keyword.c_str())) {
              case str2ull("nRHSs"):      std::stringstream(line) >> keyword >> block_size;   break; // == block size, sorry, bad naming
              case str2ull("nCols"):      std::stringstream(line) >> keyword >> nCols;        break;
              case str2ull("tolerance"):  std::stringstream(line) >> keyword >> tolerance;    break;
                        // 012 4
              case str2ull("bsr_A%nCols"):
              case str2ull("bsr_B%nCols"):
              case str2ull("bsr_X%nCols"): {
                  op = &ABX[mapA0B1X2(keyword[4])];
                  int n1{0};
                  std::stringstream(line) >> keyword >> n1;
                  op->nCols = n1;
                  op->name = keyword[4];
              }   break;
                        // 0123456 8
              case str2ull("sizebsr_A%RowStart"):
              case str2ull("sizebsr_B%RowStart"):
              case str2ull("sizebsr_X%RowStart"): {
                  op = &ABX[mapA0B1X2(keyword[8])];
                  int n1{0};
                  std::stringstream(line) >> keyword >> n1;
                  op->nRows = n1 - 1;
                  op->RowPtr.reserve(n1);
                  for (auto k = 0; k < n1; ++k) {
                      int ival{0};
                      input >> ival;
                      op->RowPtr.push_back(ival - 1); // subtract 1 for the conversion from Fortran to C/C++
                  } // k
              }   break;
                        // 0123456 8
              case str2ull("sizebsr_A%ColIndex"):
              case str2ull("sizebsr_B%ColIndex"):
              case str2ull("sizebsr_X%ColIndex"): {
                  op = &ABX[mapA0B1X2(keyword[8])];
                  int n1{0};
                  std::stringstream(line) >> keyword >> n1;
                  op->nnzb = n1;
                  op->ColInd.reserve(n1);
                  for (auto k = 0; k < n1; ++k) {
                      int ival{0};
                      input >> ival;
                      op->ColInd.push_back(ival - 1); // subtract 1 for the conversion from Fortran to C/C++
                  } // k
              }   break;
                        // 01234567 9
              case str2ull("shapemat_A"):
              case str2ull("shapemat_B"):
              case str2ull("shapemat_X"): {
                  op = &ABX[mapA0B1X2(keyword[9])];
                  int n1{0}, n2{0}, n3{0};
                  std::stringstream(line) >> keyword >> n1 >> n2 >> n3;
                  assert(block_size == unsigned(n1));
                  assert(block_size == unsigned(n2));
                  op->fastBlockDim = n1;
                  op->slowBlockDim = n2;
                  { // scope
                      auto const nall = n3*n2*n1*2;
                      size_t nonz{0};
                      std::cout << "# allocate " << nall*8e-6 << " MByte for operator " << keyword[9]
                                << ", fast=" << n1 << " slow=" << n2 << " blocks=" << n3 << std::endl;
                      op->mat.reserve(nall);
                      for (auto k = 0; k < nall; ++k) {
                          double dval{0};
                          input >> dval;
                          nonz += (dval != 0);
                          op->mat.push_back(dval); // values in op->mat are ordered ColMajor within each block and in RIRIRIRI(Fortran) layout
                      } // k
                      std::cout << "# found " << nonz*1e-6 << " M of " 
                                              << nall*1e-6 << " M elements nonzero" << std::endl; 
                  } // scope
              }   break;
              case str2ull("") : 
  // #ifdef DEBUG
  //              std::cout << "# empty " << std::endl;
  // #endif // DEBUG
                  break; // do nothing on empty lines
              default: std::cout << "# keyword " << keyword << " unknown!" << std::endl;
          } // switch
      }
      input.close();

      // sanity checks
      for (bsr_t const* op = ABX; op < 3 + ABX; ++op) {
          assert(op->RowPtr.size() == op->nRows + 1);
          assert(op->ColInd.size() == op->nnzb);
          assert(op->mat.size() == op->nnzb * op->slowBlockDim * op->fastBlockDim * 2);
          // show some stats of the operators
          std::cout << "# stats for the " << op->nnzb << " non-zero entries of " << op->name << std::endl;
          // traverse the operator lists
          std::vector<int> nzpc(op->nCols, 0); // number of non-zeros per column
          int nzpr0{0}, nzpr1{0}, nzpr2{0}; // number of non-zeros per row squared
          for (int iRow = 0; iRow < op->nRows; ++iRow) {
              int nzpr{0};
              for (int inzb = op->RowPtr[iRow]; inzb < op->RowPtr[iRow + 1]; ++inzb) {
                  int const iCol = op->ColInd[inzb];
                  ++nzpc[iCol];
                  ++nzpr;
              } // inzb
              nzpr2 += nzpr*nzpr;
              nzpr1 += nzpr;
              nzpr0 += (nzpr > 0);
          } // iRow
          int nzpc0{0}, nzpc1{0}, nzpc2{0}; // number of non-zeros per column squared
          for (int iCol = 0; iCol < op->nCols; ++iCol) {
              auto const nz = nzpc[iCol];
              nzpc2 += nz*nz; 
              nzpc1 += nz;
              nzpc0 += (nz > 0); 
          } // iCol
          double dev_nzpr; double const avg_nzpr = average_and_deviation(nzpr0, nzpr1, nzpr2, &dev_nzpr);
          std::cout << "# non-zeros " << avg_nzpr << " +/- " << dev_nzpr << " in " << nzpr0 << " of " << op->nRows << " rows" << std::endl;
          double dev_nzpc; double const avg_nzpc = average_and_deviation(nzpc0, nzpc1, nzpc2, &dev_nzpc);
          std::cout << "# non-zeros " << avg_nzpc << " +/- " << dev_nzpc << " in " << nzpc0 << " of " << op->nCols << " columns" << std::endl;

          std::cout << std::endl;
      } // op

      auto const A = &(ABX[0]), B = &(ABX[1]), X = &(ABX[2]);

      assert(B->nCols == nCols); // number of right hand sides
      assert(X->nCols == nCols); // number of right hand sides, redundant info, sorry
      assert(X->nRows == A->nCols); // multiplication of A*X must be well-defined

      assert(A->nRows == A->nCols); // A is assmed to be a square operator here
      assert(A->fastBlockDim == A->slowBlockDim); // A is assmed to be a square operator here

      assert(A->slowBlockDim == X->fastBlockDim); // multiplication of A*X must be well-defined
      assert(B->fastBlockDim == X->fastBlockDim); // X and B must match in blocks
      assert(B->slowBlockDim == X->slowBlockDim); // X and B must match in blocks

      assert(X->slowBlockDim == block_size);

      if (B->nRows < X->nRows) {
          auto const rpB = B->RowPtr[B->nRows];
          std::cout << "# add " << X->nRows - B->nRows << " empty rows to B" << std::endl;
          for (auto k = B->nRows; k < X->nRows; ++k) {
              B->RowPtr.push_back(rpB);
          } // k
          B->nRows = X->nRows;
          assert(B->nRows == A->nRows); // multiplication of A*X must be well-defined

          if (0) {
              std::cout << std::endl << "# number of nonzeros per row for A, B, X" << std::endl;
              for (auto k = 0; k < X->nRows; ++k) {
                  for (bsr_t const* op = ABX; op < 3 + ABX; ++op) {
                      std::cout << op->RowPtr[k + 1] - op->RowPtr[k] << " \t";
                  } // op
                  std::cout << std::endl;
              }
              std::cout << std::endl;
          } // 0

      } // elongate the B operator

      return tolerance;
  } // read_in

} // namespace tfqmrgpu_example_reader
