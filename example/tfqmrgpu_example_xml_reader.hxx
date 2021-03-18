#pragma once
/*
 *  This program reads example problems
 *  stored in the extendable markup languange (XML)
 *  for the tfQMRgpu library which solves A*X==B.
 *          tfQMRgpu stands for the 
 *          transpose-free quasi minimal residual (QMR)
 *          method implemented on CUDA-enabled GPUs.
 *
 */

#include <cstdio> // std::printf
#include <cstdio> // std::fBSMen, std::fprintf, std::fclose
#include <cassert> // assert
#include <complex> // std::complex<T>
#include <vector> // std::vector<T>
#include <cerrno> // errno, ERANGE
#include <numeric> // std::iota
#include <algorithm> // std::max_element, std::min_element

#ifndef HAS_NO_RAPIDXML
  #include <cstdlib> // std::atof, std::strtod
  #include <cstring> // std::strcmp

  // from https://sourceforge.net/projects/rapidxml/files/latest/download
  #include "../include/tools/rapidxml/rapidxml.hpp" // ::xml_document<>
  #include "../include/tools/rapidxml/rapidxml_utils.hpp" // ::file<>
#else
  #error "Example reader needs the rapidxml library!"
#endif

#ifdef  HAS_BSR
    // use the definition of the Block-compressed Sparse Row format from the include path
    #include "bsr.hxx" // bsr_t
#else // HAS_BSR

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
  }; // bsr_t

#endif // HAS_BSR

namespace tfqmrgpu_example_xml_reader {

    char const empty_string[] = "";
  
    inline char const * find_attribute(
          rapidxml::xml_node<> const *node
        , char const *const name
        , char const *const default_value=""
        , int const echo=0
    ) { 
        if (nullptr == node) return empty_string;
        for (auto attr = node->first_attribute(); attr; attr = attr->next_attribute()) {
            if (0 == std::strcmp(name, attr->name())) {
                return attr->value();
            } // found
        } // attr
        return default_value;
    } // find_attribute 
  
    inline rapidxml::xml_node<> const * find_child(
          rapidxml::xml_node<> const *node
        , char const *const name
        , int const echo=0
    ) { 
        if (nullptr == node) return nullptr;
        for (auto child = node->first_node(); child; child = child->next_sibling()) {
            if (0 == std::strcmp(name, child->name())) {
                return child;
            } // found
        } // attr
        return nullptr;
    } // find_child 

    template <typename real_t>
    std::vector<real_t> read_sequence(
          char const *sequence
        , int const echo=0
        , size_t const reserve=0
    ) {
        // read data from a (potentially long) char sequence into a vector
        char *end;
        char const *seq{sequence};
        std::vector<real_t> v;
        v.reserve(reserve);
        for (double f = std::strtod(seq, &end);
             seq != end;
             f = std::strtod(seq, &end)) {
            seq = end;
            if (errno == ERANGE){
                std::fprintf(stderr, "range error, got %g", f);
                errno = 0;
            } else {
                v.push_back(real_t(f));
            } // errno
        } // f
        return v;
    } // read_sequence

//   template <typename real_t> inline real_t constexpr pow2(real_t const x) { return x*x; }
  
  inline double read_in( // returns tolerance
        bsr_t ABX[3]
      , char const *const filename
      , int const echo=0
  ) {
      double tolerance{0}; // init return value
      if (nullptr == filename) {
          std::fprintf(stderr, "%s: filename is null", __func__);
          return 0;
      } // !filename

      if (echo > 0) std::printf("# read file \"%s\" using rapidxml\n", filename);
      rapidxml::file<> infile(filename);

      // create the root node
      rapidxml::xml_document<> doc;
      
      if (echo > 0) std::printf("# parse file content using rapidxml\n");
      doc.parse<0>(infile.data());

      auto const LinearProblem = doc.first_node("LinearProblem");
      if (!LinearProblem) return 0;
      
      auto const tolerance_string = find_attribute(LinearProblem, "tolerance", "0", echo);
      tolerance = std::atof(tolerance_string);
      if (echo > 3) {
          std::printf("# tolerance= %.3e (\"%s\")\n", tolerance, tolerance_string);
          auto const problem_kind = find_attribute(LinearProblem, "problem_kind", "?", echo);
          std::printf("# problem_kind= %s\n", problem_kind);
          auto const generator_version = find_attribute(LinearProblem, "generator_version", "?", echo);
          std::printf("# generator_version= %s\n", generator_version);
      } // echo

// <LinearProblem type="A*X==B" generator_version="0.0" tolerance="1.000e-09">
//   <BlockSparseMatrix id="A">
//     <SparseMatrix type="CSR">
//       <CompressedSparseRow>
//         <NonzerosPerRow rows="7">
//         </NonzerosPerRow>
//         <ColumnIndex nonzeros="19">
//         </ColumnIndex>
//       </CompressedSparseRow>
//     </SparseMatrix>
      
      double scale_values[] = {1, 1, 1};
      std::vector<unsigned> indirect[3];
      
      for(auto BSM = LinearProblem->first_node(); BSM; BSM = BSM->next_sibling()) {
          auto const id = find_attribute(BSM, "id", "?", echo);
          if (echo > 5) std::printf("# BlockSparseMatrix id= %s\n", id);
          int const abx = ('A' == *id) ? 0 : (('B' == *id) ? 1 : 2);
          bsr_t & bsr = ABX[abx];
          bsr.name = id;

          auto const SparseMatrix = find_child(BSM, "SparseMatrix", echo);
          if (SparseMatrix) {
#if 0
              auto const type = find_attribute(SparseMatrix, "type", "?", echo);
              if (std::string("CSR") != type) {
                  std::printf("# SparseMatrix not in CSR format!\n");
                  return 0;
              } // not CompressedSparseRow formatted
#endif
              auto const csr = find_child(SparseMatrix, "CompressedSparseRow", echo);
              if (!csr) {
                  std::printf("# Cannot find CompressedSparseRow in SparseMatrix\n");
                  return 0;
              } // no csr found
              
              auto const nzpr = find_child(csr, "NonzerosPerRow", echo);
              if (nzpr) {
                  int const nrows = std::atoi(find_attribute(nzpr, "rows", "0", echo));
                  auto const nonzero_element_per_row = read_sequence<int>(nzpr->value(), echo, nrows);
                  bsr.nRows = nonzero_element_per_row.size();
                  bsr.RowPtr.resize(bsr.nRows + 1);
                  bsr.RowPtr[0] = 0;
                  for(int irow = 0; irow < bsr.nRows; ++irow) {
                      bsr.RowPtr[irow + 1] = bsr.RowPtr[irow] + nonzero_element_per_row[irow];
                  } // irow
              } else {
                  auto const RowStart = find_child(csr, "RowStart", echo);
                  if (RowStart) {
                      int const nrows = std::atoi(find_attribute(RowStart, "rows", "0", echo));
                      bsr.RowPtr = read_sequence<int>(RowStart->value(), echo, nrows + 1);
                      bsr.nRows = bsr.RowPtr.size() - 1;
                  } else {
                      std::printf("# Cannot find NonzerosPerRow nor RowStart in CompressedSparseRow\n");
                      return 0;
                  } // RowStart
              } // nzpr
              if (echo > 4) std::printf("# number of rows in %s is %d\n", id, bsr.nRows);

              auto const ColumnIndex = find_child(csr, "ColumnIndex", echo);
              if (!ColumnIndex) {
                  std::printf("# Cannot find ColumnIndex in CompressedSparseRow\n");
                  return 0;
              } // no ColumnIndex
              size_t const nnz = std::atoi(find_attribute(ColumnIndex, "nonzeros", "0", echo));
              bsr.ColInd = read_sequence<int>(ColumnIndex->value(), echo, nnz);
              bsr.nnzb = bsr.ColInd.size();
              
              auto const highest_column_index = *std::max_element(bsr.ColInd.begin(), bsr.ColInd.end());
              auto const lowest_column_index  = *std::min_element(bsr.ColInd.begin(), bsr.ColInd.end());
              bsr.nCols = highest_column_index + 1 - lowest_column_index;
              if (echo > 4) std::printf("# number of columns in %s is %d\n", id, bsr.nCols);
              if (echo > 4) std::printf("# number of nonzeros in %s is %d\n", id, bsr.nnzb);

              unsigned highest_index{bsr.nnzb - 1};
              auto const Indirection = find_child(SparseMatrix, "Indirection", echo);
              if (Indirection) {
                  indirect[abx] = read_sequence<unsigned>(Indirection->value(), echo, bsr.nnzb);
                  assert(indirect[abx].size() == bsr.nnzb);
                  highest_index = *std::max_element(indirect[abx].begin(), indirect[abx].end());
              } else {
                  indirect[abx] = std::vector<unsigned>(bsr.nnzb);
                  // create a trivial indirection vector, i.e. 0,1,2,3,...
                  std::iota(indirect[abx].begin(), indirect[abx].end(), 0);
              } // Indirection
              if (1) { // analyze the indirection table
                  std::vector<uint16_t> stats(bsr.nnzb, 0);
                  for(auto i : indirect[abx]) {
                      assert(0 <= i); assert(i < bsr.nnzb);
                      ++stats[i];
                  } // i
                  std::vector<unsigned> occurence(96, 0);
                  for(auto s : stats) {
                      assert(s < 96);
                      ++occurence[s];
                  } // s
                  for(int h = 0; h < 96; ++h) {
                      if (occurence[h] > 0) {
                          std::printf("# %s occurence[%i] = %d\n", id, h, occurence[h]);
                      } // occurred at least once
                  } // h
                  if (!Indirection) {
                      // the result of std::iota or other permutations must produce each number exactly once
                      assert(occurence[1] == bsr.nnzb);
                  } // no indirection
              } // analysis

          } else { // SparseMatrix
              std::printf("# Cannot find a SparseMatrix for operator \'%s\'\n", id);
          } // SparseMatrix
          
//     <DataTensor type="real" rank="3" dimensions="19 1 1" scale="1.984126984126984e-04">
          auto const DataTensor = find_child(BSM, "DataTensor", echo);
          if (DataTensor) {
              scale_values[abx] = std::atof(find_attribute(DataTensor, "scale", "1", echo));
              auto const type = find_attribute(DataTensor, "type", "complex", echo);
              int const rank = std::atoi(find_attribute(DataTensor, "rank", "3", echo));
              auto const dim_string = find_attribute(DataTensor, "dimensions", "0 0 0", echo);
              auto const dims = read_sequence<int>(dim_string, echo, rank);
              assert(dims.size() == rank);
              std::printf("# Found DataTensor[%d][%d][%d] for operator \'%s\'\n", dims[0], dims[1], dims[2], id);
              if (bsr.nnzb != dims[0]) {
                  std::printf("# DataTensor[%d] dimension differs from SparseMatrix.nnz = %d of operator \'%s\'\n", dims[0], bsr.nnzb, id);
              } // different
              bsr.slowBlockDim = dims[1];
              bsr.fastBlockDim = dims[2];
          } else {
              std::printf("# Cannot find a DataTensor for operator \'%s\'\n", id);
          } // DataTensor

      } // BSM

      return tolerance;
  } // read_in

} // namespace tfqmrgpu_example_xml_reader
