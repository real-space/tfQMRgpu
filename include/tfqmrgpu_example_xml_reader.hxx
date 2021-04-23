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

  // git clone https://github.com/dwd/rapidxml
  #include "../external/rapidxml/rapidxml.hpp" // ::xml_document<>
  #include "../external/rapidxml/rapidxml_utils.hpp" // ::file<>
#else  // HAS_RAPIDXML
  #error "Example reader needs the rapidxml library!"
#endif // HAS_RAPIDXML

// use the definition of the Block-compressed Sparse Row format from the include path
#include "bsr.hxx" // bsr_t

namespace tfqmrgpu_example_xml_reader {

  inline char const * find_attribute(
        rapidxml::xml_node<> const *node
      , char const *const attribute_name
      , char const *const default_value=""
      , int const echo=0
  ) {
      if (nullptr != node) {
          for (auto attr = node->first_attribute(); attr; attr = attr->next_attribute()) {
              if (0 == std::strcmp(attribute_name, attr->name())) {
                  if (echo > 9) std::printf("# %s: node '%s' has attribute '%s' with value '%s'\n",
                                            __func__, node->name(), attribute_name, attr->value());
                  return attr->value();
              } // found
          } // attr
          if (echo > 7) std::printf("# %s: node '%s' has no attribute '%s', use default '%s'\n",
                                        __func__, node->name(), attribute_name, default_value);
      } else { // node
          if (echo > 7) std::printf("# %s: node = NULL, use default '%s'\n", __func__, default_value);
      } // node
      return default_value;
  } // find_attribute

  inline rapidxml::xml_node<> const * find_child(
        rapidxml::xml_node<> const *node
      , char const *const child_name
      , int const echo=0
  ) { 
      if (nullptr != node) {
          for (auto child = node->first_node(); child; child = child->next_sibling()) {
              if (0 == std::strcmp(child_name, child->name())) {
                  if (echo > 9) std::printf("# %s: node '%s' has child '%s'\n",
                                            __func__, node->name(), child_name);
                  return child;
              } // found
          } // attr
          if (echo > 7) std::printf("# %s: node '%s' has no child '%s'\n", __func__, node->name(), child_name);
      } else { // node
          if (echo > 7) std::printf("# %s: node = NULL\n", __func__);
      } // node
      return nullptr;
  } // find_child

  template <typename real_t>
  std::vector<real_t> read_sequence(
        char const *const sequence
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
              if (echo > 0) std::fprintf(stderr, "range error, got %g", f);
              errno = 0;
          } else {
              v.push_back(real_t(f));
          } // errno
      } // f
      return v;
  } // read_sequence

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

      if (echo > 0) std::printf("# read file '%s' using rapidxml\n", filename);
      rapidxml::file<> infile(filename);

      // create the root node
      rapidxml::xml_document<> doc;
      
      if (echo > 0) std::printf("# parse file content using rapidxml\n");
      doc.parse<0>(infile.data());

      auto const LinearProblem = doc.first_node("LinearProblem");
      if (!LinearProblem) {
          std::printf("\n# Warning! Cannot find LinearProblem in file '%s'\n\n", filename);
          return 0;
      } // no linear problem found

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

      for (auto BSM = LinearProblem->first_node(); BSM; BSM = BSM->next_sibling()) {
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
#endif // 0
              auto const csr = find_child(SparseMatrix, "CompressedSparseRow", echo);
              if (!csr) {
                  std::printf("\n# Warning! Cannot find CompressedSparseRow in SparseMatrix\n\n");
                  return 0;
              } // no csr found
              
              auto const nzpr = find_child(csr, "NonzerosPerRow", echo);
              if (nzpr) {
                  int const nrows = std::atoi(find_attribute(nzpr, "rows", "0", echo));
                  auto const nonzero_element_per_row = read_sequence<int>(nzpr->value(), echo, nrows);
                  bsr.nRows = nonzero_element_per_row.size();
                  bsr.RowPtr.resize(bsr.nRows + 1);
                  bsr.RowPtr[0] = 0;
                  for (int irow = 0; irow < bsr.nRows; ++irow) {
                      bsr.RowPtr[irow + 1] = bsr.RowPtr[irow] + nonzero_element_per_row[irow];
                  } // irow
              } else {
                  auto const RowStart = find_child(csr, "RowStart", echo);
                  if (RowStart) {
                      int const nrows = std::atoi(find_attribute(RowStart, "rows", "0", echo));
                      bsr.RowPtr = read_sequence<int>(RowStart->value(), echo, nrows + 1);
                      bsr.nRows = bsr.RowPtr.size() - 1;
                  } else {
                      std::printf("\n# Warning! Cannot find NonzerosPerRow nor RowStart in CompressedSparseRow\n\n");
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
                  for (auto i : indirect[abx]) {
                      assert(0 <= i); assert(i < bsr.nnzb);
                      ++stats[i];
                  } // i
                  std::vector<unsigned> occurence(96, 0);
                  for (auto s : stats) {
                      if (s >= occurence.size()) occurence.resize(s + 1);
                      ++occurence[s];
                  } // s
                  for (int h = 0; h < occurence.size(); ++h) {
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
              std::printf("\n# Warning! Cannot find a SparseMatrix for operator %s\n\n", id);
          } // SparseMatrix
          
          auto const DataTensor = find_child(BSM, "DataTensor", echo);
          if (DataTensor) {
              scale_values[abx] = std::atof(find_attribute(DataTensor, "scale", "1", echo));
              auto const type = find_attribute(DataTensor, "type", "complex", echo);
              int const r1c2 = ('c' == (*type | 32)) ? 2 : 1; // 1:real or 2:complex
              int const rank = std::atoi(find_attribute(DataTensor, "rank", "3", echo));
              auto const dim_string = find_attribute(DataTensor, "dimensions", "0 0 0", echo);
              auto const dims = read_sequence<int>(dim_string, echo, rank);
              assert(dims.size() == rank);
              std::printf("# Found DataTensor[%d][%d][%d] (type=%s) for operator %s\n", dims[0], dims[1], dims[2], type, id);
              bsr.slowBlockDim = dims[1];
              bsr.fastBlockDim = dims[2];
              auto const block2 = dims[1] * dims[2];
              auto const source_size = size_t(dims[0])  * block2;
              auto const target_size = size_t(bsr.nnzb) * block2;
              auto const data = read_sequence<double>(DataTensor->value(), echo, source_size*r1c2);
              assert(data.size() == source_size*r1c2);
              bsr.mat = std::vector<double>(target_size*2, 0.0); // always complex (in RIRIRIRI data layout)
              if (bsr.nnzb != dims[0]) {
                  std::printf("# DataTensor[%d] dimension differs from SparseMatrix.nnz = %d of operator %s\n", dims[0], bsr.nnzb, id);
              } else {
                  double const scale_factor = scale_values[abx];
                  auto const indirection = indirect[abx];
                  for (size_t inzb = 0; inzb < bsr.nnzb; ++inzb) {
                      auto const iblock = indirection[inzb];
                      for (int ij = 0; ij < block2; ++ij) {
                          for (int ri = 0; ri < r1c2; ++ri) { // real [and imaginary] part
                              bsr.mat[(inzb*block2 + ij)*2 + ri] = data[(iblock*block2 + ij)*r1c2 + ri] * scale_factor;
                          } // ri
                      } // ij
                  } // inzb
              } // different

          } else {
              std::printf("\n# Warning! Cannot find a DataTensor for operator %s\n\n", id);
          } // DataTensor

      } // BSM

      return tolerance;
  } // read_in

} // namespace tfqmrgpu_example_xml_reader
