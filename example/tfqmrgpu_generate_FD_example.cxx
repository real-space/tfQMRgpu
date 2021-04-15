/*
 *  This program generates an example problem 
 *  stored in the extendable markup languange (XML)
 *  for the tfQMRgpu library which solves A*X==B.
 *          tfQMRgpu stands for the 
 *          transpose-free quasi minimal residual (QMR)
 *          method implemented on CUDA-enabled GPUs.
 *
 *  The type of operator is a scaled Laplacian
 *  approximated by a 9-point finite-difference (FD) stencil 
 *  and can generate 2-dimensional or 3-dimensional example problems.
 *
 *  For reasons of performance, grid points are grouped into blocks
 *  (squares in 2D and cubes in 3D) with the parameter block_edge.
 *
 *  We can control two radii:
 *
 *      - radius_source_blocks: all cubes inside that radius
 *                          are treated as right hand side B.
 *
 *      - radius_target_blocks: all cubes within that radius
 *          of a source block are treated as nonzero target blocks of X.
 *
 *  This generator does not offer a reference solution in X,
 *  i.e. the operator X is stored without data.
 *
 *  Also, the data blocks of B are not store explicitly since they are 
 *  unit matrix blocks.
 *
 *  The data blocks of A currently lead to very large output files
 *  with many repeating copies of the same block
 *  which could be avoided using an indirection method.
 *
 */

// compile and run:
// g++ -std=c++11 -D__MAIN__ tfqmrgpu_generate_FD_example.cxx && ./a.out

#include <cstdio> // std::printf, ::fopen, ::fprintf, ::fclose
#include <iostream> // std::cout
#include <cstdlib> // std::exit
#include <cassert> // assert
#include <complex> // std::complex<T>
#include <vector> // std::vector<T>

#ifdef  HAS_BSR
    // use the definition of the Block-compressed Sparse Row format from the include path
    #include "bsr.hxx" // bsr_t
#else // HAS_BSR

  struct bsr_t { // block sparse row matrix structure
    unsigned nRows; // number of Rows
    unsigned nCols; // number of block columns
    unsigned nnzb;  // number of non-zero blocks
    std::vector<int> RowPtr; // [nRows + 1]
    std::vector<int> ColInd; // [nnzb]
  }; // bsr_t

#endif // HAS_BSR

  template <typename real_t> inline real_t constexpr pow2(real_t const x) { return x*x; }

  // The purpose of this is to create a test for the tfQMR solver
  // therefore, we use a Hamilton operator originating from 8th order finite-difference

  template <int BlockSize, typename data_t>
  struct DenseBlock {
      static int constexpr BS = BlockSize;
      data_t data[BS][BS];
      DenseBlock(data_t const init_value=data_t(0)) {
          for(int ij = 0; ij < BS*BS; ++ij) data[0][ij] = init_value;
      } // default constructor and inital value constructor
  }; // DenseBlock


  template <int BlockSize, typename BaseData_t>
  struct BlockSparseOperator {
      static int constexpr BS = BlockSize;
      typedef BaseData_t complex_t; // abbreviate
      BlockSparseOperator(char id_char) { id[0] = id_char; } // constructor
  // members
      std::vector<DenseBlock<BS, complex_t>*> blocks;
      std::vector<uint32_t> indirection;
      bsr_t bsr;
      double scale_data = 1;
      char id[2] = {'?', '\0'};
  }; // BlockSparseOperator

  
  template <typename complex_t> inline char const * complex_name()     { return "real"; }
  template <> inline char const * complex_name<std::complex<double>>() { return "complex64"; }
  template <> inline char const * complex_name<std::complex<float >>() { return "complex32"; }

  
  void xml_export_bsr(
        FILE* f
      , bsr_t const & bsr
      , char const *spaces=""
      , uint32_t const indirection[]=nullptr
  ) {
      std::fprintf(f, "%s<SparseMatrix type=\"CSR\">\n", spaces);
      std::fprintf(f, "%s  <CompressedSparseRow>\n", spaces);

      // we could export the content of RowPtr, however, this produces many long numbers
      // while the increments between the numbers are usually small, so we export only the increments
      std::fprintf(f, "%s    <NonzerosPerRow rows=\"%ld\">", spaces, bsr.RowPtr.size() - 1);
      for(size_t i = 0; i < bsr.RowPtr.size() - 1; ++i) {
          char const blank = (i & 0xf) ? ' ' : '\n'; // newline every 16 entries
          std::fprintf(f, "%c%d", blank, bsr.RowPtr[i + 1] - bsr.RowPtr[i]);
      } // i
      std::fprintf(f, "\n%s    </NonzerosPerRow>\n", spaces);

      std::fprintf(f, "%s    <ColumnIndex nonzeros=\"%ld\">", spaces, bsr.ColInd.size());
      for(size_t i = 0; i < bsr.ColInd.size(); ++i) {
          char const blank = (i & 0xf) ? ' ' : '\n'; // newline every 16 entries
           std::fprintf(f, "%c%d", blank, bsr.ColInd[i]);
      } // irow
      std::fprintf(f, "\n%s    </ColumnIndex>\n", spaces);

      std::fprintf(f, "%s  </CompressedSparseRow>\n", spaces);
      
      if (indirection) {
          std::fprintf(f, "%s  <Indirection nonzeros=\"%ld\">", spaces, bsr.ColInd.size());
          for(size_t i = 0; i < bsr.ColInd.size(); ++i) {
              char const blank = (i & 0xf) ? ' ' : '\n'; // newline every 16 entries
              std::fprintf(f, "%c%d", blank, indirection[i]);
          } // irow
          std::fprintf(f, "\n%s  </Indirection>\n", spaces);
      } // indirection

      std::fprintf(f, "%s</SparseMatrix>\n", spaces);
  } // xml_export_bsr
  
  
  template <class operator_t>
  void xml_export_operator(
        FILE* f
      , operator_t const & op
  ) {
      auto const type = complex_name<typename operator_t::complex_t>();
      auto const is_complex = ('c' == *type);
      auto const nblocks = op.blocks.size();
      int constexpr BS = operator_t::BS;
      std::fprintf(f, "  <BlockSparseMatrix id=\"%s\">\n", op.id);
      auto const ind_ptr = (op.indirection.size() == op.bsr.nnzb) ?
                            op.indirection.data() : nullptr;
      xml_export_bsr(f, op.bsr, "    ", ind_ptr);
      if (nblocks < 1) { std::fprintf(f, "  </BlockSparseMatrix>\n"); return; }
      std::fprintf(f, "    <DataTensor type=\"%s\"", type);
      std::fprintf(f, " rank=\"3\" dimensions=\"%ld %d %d\"", nblocks, BS, BS);
      if (op.scale_data != 1) {
          std::fprintf(f, " scale=\"%.15e\"", op.scale_data);
      } // scaling
      std::fprintf(f, ">\n");
      for(size_t iblock = 0; iblock < nblocks; ++iblock) {
          auto const block = op.blocks[iblock];
          assert(nullptr != block);
          for(int i = 0; i < BS; ++i) {
              for(int j = 0; j < BS; ++j) {
                  std::fprintf(f, "%g ",   double(std::real(block->data[i][j])));
                  if (is_complex)
                  std::fprintf(f, " %g  ", double(std::imag(block->data[i][j])));
              } // j
              std::fprintf(f, "\n");
          } // i
          std::fprintf(f, "\n");
      } // iblock
      std::fprintf(f, "    </DataTensor>\n");
      std::fprintf(f, "  </BlockSparseMatrix>\n");
  } // xml_export_operator

  
  union index4_t {
      // Beware, this is a union, not a struct,
      // so the following 4 lines refer to the same 4 Byte in memory
      uint32_t  ui32;
      int32_t    i32;
      int8_t   i8[4];
      uint8_t ui8[4];
      // constructors
      index4_t(uint32_t i=0) : ui32(i) {} // also the default constructor
      index4_t( int32_t i)   :  i32(i) {}
      index4_t(uint8_t x, uint8_t y, uint8_t z, uint8_t t=0) { ui8[0] = x; ui8[1] = y; ui8[2] = z; ui8[3] = t; }
      index4_t( int8_t x,  int8_t y,  int8_t z,  int8_t t=0) {  i8[0] = x;  i8[1] = y;  i8[2] = z;  i8[3] = t; }
  }; // index4_t


  template <typename center_t, int Dimension=3>
  size_t create_cluster(
      std::vector<uint32_t> & index // output
    , center_t const center[3]
    , float const radius
    , std::vector<bool> *nonzero=nullptr // output (size should be 2^24 needs to be initialized as false)
    , int const echo=0
  ) {
      uint32_t const max_nonzero = nonzero ? nonzero->size() : 0;
      index.clear();
      index.reserve(4.1888*pow2(radius)*radius); // 4.1888 == 4*pi/3
      size_t nblocks{0};
      int  const irad = std::ceil(radius);
      auto const rad2 = pow2(radius);
      int8_t minima[3] = { 127,  127,  127};
      int8_t maxima[3] = {-128, -128, -128};
      int limits[3][2] = {{0,0}, {0,0}, {0,0}};
      for(int dir = 0; dir < Dimension; ++dir) {
          limits[dir][0] = center[dir] - irad;
          limits[dir][1] = center[dir] + irad;
      } // dir

      for(int z = limits[2][0]; z <= limits[2][1]; ++z) {
          auto const z2 = pow2(center[2] - z)*(3 == Dimension);
      for(int y = limits[1][0]; y <= limits[1][1]; ++y) {
          auto const y2 = pow2(center[1] - y);
      for(int x = limits[0][0]; x <= limits[0][1]; ++x) {
          auto const x2 = pow2(center[0] - x);

          auto const dist2 = x2 + y2 + z2;
          if (dist2 <= rad2) { // we need <= so we can use rsb=0 for exactly 1 RHS
              index4_t const i4(int8_t(x), int8_t(y), int8_t(z*(3 == Dimension)));
              if (echo > 5) std::printf("# block %li at %i %i %i (0x%6.6x)\n", index.size(), i4.i8[0],i4.i8[1],i4.i8[2], i4.ui32);
              for(int d = 0; d < 3; ++d) {
                  minima[d] = std::min(minima[d], i4.i8[d]);
                  maxima[d] = std::max(maxima[d], i4.i8[d]);
              } // d
              index.push_back(i4.ui32);
              assert(i4.ui32 < (1ul << 24) && "maybe BigEndian needs to be adjusted");
              if (i4.ui32 < max_nonzero) (*nonzero)[i4.ui32] = true;
              ++nblocks;
          } // inside
      }}} // x y z
      if (echo > 3) {
          std::printf("# block centered at %g %g %g has %ld entries inside [%i %i %i, %i %i %i]\n"
                  , 1.*center[0], 1.*center[1], 1.*center[2] // converted to double
                  , nblocks
                  , minima[0], minima[0], minima[0]
                  , maxima[0], maxima[1], maxima[2]);
      } // echo
      return nblocks;
  } // create_cluster

  
  template <int BlockEdge, int Dimension=3>
  int generate(
        float const rsb
      , float const rtb
      , char const ref
      , int const echo=0
      , float const tolerance=1e-9
      , char const *filename="FD_problem.xml"
  ) {
      assert(BlockEdge > 0);
      assert(Dimension > 0 && Dimension < 4);
      int constexpr BS = BlockEdge * ((Dimension > 1)? BlockEdge : 1)
                                   * ((Dimension > 2)? BlockEdge : 1);
      BlockSparseOperator<BS, int32_t> A('A');
      BlockSparseOperator<BS,  int8_t> B('B');
      BlockSparseOperator<BS, std::complex<float>> X('X');

//            DENOM,          0,      1,      2,    3,    4,  5,  6,
// ==================================================================================================  
//                1,          0,      0,      0,    0,    0,  0,  0, & !  Laplacian switched off
//                1,         -2,      1,      0,    0,    0,  0,  0, & !  2nd order Laplacian(lowest)
//               12,        -30,     16,     -1,    0,    0,  0,  0, & !  4th order
//              180,       -490,    270,    -27,    2,    0,  0,  0, & !  6th order
//             5040,     -14350,   8064,  -1008,  128,   -9,  0,  0, & !  8th order
//            25200,     -73766,  42000,  -6000, 1000, -125,  8,  0, & ! 10th order
//           831600,   -2480478,1425600,-222750,44000,-7425,864,-50  & ! 12th order

#ifdef HIGHER_FINITE_DIFFERENCE_ORDER      
      int constexpr nFD = 6; // number of finite-difference neighbors
      int32_t const FDdenom = 831600;
      int32_t const FDcoeff[1 + nFD] = {2480478, -1425600, 222750, -44000, 7425, -864, 50}; // minus Laplacian (as appearing in electrostatics)
#else
      int constexpr nFD = 4; // number of finite-difference neighbors
      int32_t const FDdenom = 5040;
      int32_t const FDcoeff[1 + nFD] = {14350, -8064, 1008, -128, 9}; // minus Laplacian (as appearing in electrostatics)
#endif
      if (echo > 0) {
          int64_t checksum{0};
          std::cout << "# use " << nFD << " finite-difference neighbors with coefficients:" << std::endl;
          for(int iFD = 0; iFD <= nFD; ++iFD) {
              std::cout << "# " << iFD << "\t"  <<  FDcoeff[iFD] << "/" << FDdenom 
                        << "\t= " << FDcoeff[iFD]/double(FDdenom) << std::endl;
              checksum += FDcoeff[iFD] * (1 + (iFD > 0));
          } // iFD
          std::cout << std::endl;
          assert(0 == checksum);
      } // echo

      // create the stencil around each origin, max block is 1 + Dimension*nFD
      uint8_t constexpr StencilBlockZero = 255;
      uint8_t origin_blocks[StencilBlockZero][4];
      assert(1 + nFD*Dimension < StencilBlockZero);

      auto const origin_block_index = new uint8_t[32][32][32];
      for(size_t i = 0; i < 32*32*32; ++i) {
          origin_block_index[0][0][i] = StencilBlockZero; // init as non-existing
      } // i

      int const stencil_range = (nFD - 1)/BlockEdge + 1;
      assert(stencil_range >= 0);
      int iob{0};
      { // scope: create a finite-difference stencil in units of blocks
          int const sr = stencil_range;
          if (echo > 0) std::cout << "# stencil range " << sr << " blocks"
              " of " << BlockEdge << "^" << Dimension << " = " << BS << " grid points" << std::endl;
          assert(sr < 16 && "finite-difference order is too large for origin_block_index[32][32][32]");
          for(int isr = 0; isr <= sr; ++isr) {
              for(int ipm = 1; ipm >= -1; ipm -= 2) {
                  for(int dir = 0; dir < Dimension; ++dir) {
                      int xyz[] = {0, 0, 0};
                      xyz[dir] = isr*ipm;
                      int const x = xyz[0], y = xyz[1], z = xyz[2];
                      auto & ob_ind = origin_block_index[z & 0x1f][y & 0x1f][x & 0x1f];
                      if (StencilBlockZero == ob_ind) {
                          origin_blocks[iob][0] = x;
                          origin_blocks[iob][1] = y;
                          origin_blocks[iob][2] = z;
                          origin_blocks[iob][3] = pow2(isr);
                          ob_ind = iob;
                          ++iob;
                      } // ob_ind
                  } // dir
              } // ipm
          } // isr

          assert(iob < StencilBlockZero && "stencil extent is too large");
      } // scope
      int const nob = iob;
      if (echo > 0) std::cout << "# " << nob << " nonzero stencil blocks" << std::endl;

      // the stencil has integer coefficients if we do not apply the finite-difference denominator
      std::vector<DenseBlock<BS, int32_t>> Stencil(nob);

      // loop over all grid points inside one block
      for(int z = 0; z <= (BlockEdge - 1)*(Dimension > 2); ++z) {
      for(int y = 0; y <= (BlockEdge - 1)*(Dimension > 1); ++y) {
      for(int x = 0; x <   BlockEdge; ++x) {
          if (echo > 19) std::printf("# z=%d y=%d x=%d\n", z, y, x);
          int const ixyz[3] = {x, y, z}; // grid point coords
          int const ib = (z*BlockEdge + y)*BlockEdge + x;
          assert(0 <= ib); assert(ib < BS);
          for(int dir = 0; dir < Dimension; ++dir) {
              if (echo > 9) std::printf("# z=%d y=%d x=%d dir=%c\n", z, y, x, dir+'x');
              int xyz_m[3] = {x, y*(Dimension > 1), z*(Dimension > 2)}; // modified coords modulo block

              // something seems to produce unsymmetric stencils here
              // ToDo

              for(int iFD = -nFD; iFD <= nFD; ++iFD) {
//                if (echo > 99) std::printf("# z=%d y=%d x=%d dir=%c iFD=%i\n", z, y, x, dir+'x', iFD);
                  int const j_dir = ixyz[dir] + iFD; // shifted grid point
//                   int const shift_dir = j_dir / BlockEdge; // how many blocks shifted?
                  int const shift_dir = (j_dir + 99*BlockEdge) / BlockEdge - 99; // how many blocks shifted?
                  xyz_m[dir] = (99*BlockEdge + j_dir) % BlockEdge;
                  assert(xyz_m[dir] >= 0);
                  auto const jb = (xyz_m[2]*BlockEdge + xyz_m[1])*BlockEdge + xyz_m[0];
                  assert(0 <= jb); assert(jb < BS);
                  int sxyz[3] = {0, 0, 0}; // block shift in all directions
                  sxyz[dir] = shift_dir;
                  if (echo > 9) {
                      std::printf("# %i%i%i dir=%c iFD=%i p=%i m=%i s=%i\n", 
                                     z,y,x, dir+'x', iFD, j_dir, xyz_m[dir], shift_dir);
                  } // echo
                  uint8_t const jx = sxyz[0] & 0x1f,
                                jy = sxyz[1] & 0x1f,
                                jz = sxyz[2] & 0x1f;
                  auto const job = origin_block_index[jz][jy][jx];
                  assert(StencilBlockZero != job);
                  Stencil[job].data[ib][jb] += FDcoeff[std::abs(iFD)];
              } // iFD
          } // dir
      }}} // x y z
      delete[] origin_block_index;

      
      // create a cluster of source blocks (Right Hand Sides)
      if (echo > 0) std::cout << "# radius_source_blocks= " << rsb << std::endl;
      std::vector<uint32_t> source_block_index;
  //  float const source_cluster_center[3] = {0.125, 0.125, 0.125}; // can be adjusted, e.g. {.5,.5,.5} gives access do a different set of numbers of sources
      int8_t const source_cluster_center[3] = {0, 0, 0};
      auto const n_sources = create_cluster(source_block_index, source_cluster_center, rsb/BlockEdge, nullptr, echo);
      if (echo > 0) std::cout << "# " << n_sources << " source blocks" << std::endl;

      // create a cluster of target blocks around each source block
      if (echo > 0) std::cout << "# radius_target_blocks= " << rtb << std::endl;
      int constexpr nbits_xy = 8, nbits_z = 8;
      size_t n_all_targets{0};
      uint32_t const max_nonzero = 1ull << (2*nbits_xy + nbits_z); // 2^24
      std::vector<bool> nonzero(max_nonzero, false);
      std::vector<std::vector<uint32_t>> target_block_index(n_sources);
      for(int isrc = 0; isrc < n_sources; ++isrc) {
          index4_t const target_cluster_center(source_block_index[isrc]);
          auto const n_targets = create_cluster(target_block_index[isrc], target_cluster_center.i8, rtb/BlockEdge, &nonzero, echo - 9);
          if (echo > 7) std::cout << "# source " << isrc << " has " << n_targets << " target blocks" << std::endl;
          n_all_targets += n_targets;
      } // isrc
      auto const average_target_blocks_per_source = n_all_targets/double(n_sources);
      if (echo > 0) std::cout << "# " << n_all_targets*.001 << " k target blocks, "
                              << average_target_blocks_per_source << " per source block" << std::endl;

      // now enumerate the nonzero rows (memory intensive variant)
      auto const row_index = new int32_t[1u << nbits_z][1u << nbits_xy][1u << nbits_xy]; // 64 MiByte for 3D, nbits=8
      std::vector<uint32_t> row_coord(0);
      row_coord.reserve(average_target_blocks_per_source);
      for(uint32_t i = 0; i < max_nonzero; ++i) {
          row_index[0][0][i] = -1; // non-existing
          if (nonzero[i]) {
              int32_t const irow = row_coord.size();
              row_index[0][0][i] = irow; // store a valid row index in a 3D array
              row_coord.push_back(i);
          } // nonzero
      } // i
      auto const nrows = row_coord.size();
      if (echo > 0) std::cout << "# " << nrows << " nonzero rows" << std::endl;
      
      auto const average_target_blocks_per_row = n_all_targets/double(nrows);

      // translate the data from target_block_index[isrc][jtrg] --> X_column_index[irow][jnz] using row_index[][][]
      std::vector<std::vector<uint32_t>> X_column_index(nrows);
      for(int irow = 0; irow < nrows; ++irow) {
          X_column_index.reserve(average_target_blocks_per_row*2.0); // estimate
      } // irow

      for(uint32_t isrc = 0; isrc < n_sources; ++isrc) {
          for(int jtrg = 0; jtrg < target_block_index[isrc].size(); ++jtrg) {
              index4_t const j4(target_block_index[isrc][jtrg]);
              auto const irow = row_index[j4.ui8[2]][j4.ui8[1]][j4.ui8[0]];
              assert(irow >= 0 && "this row should exists");
              X_column_index[irow].push_back(isrc);
          } // jtrg
      } // isrc






      // ========= construct sparsity pattern for X

      // create the BSR structure for X (result of linear system solve)
      X.bsr.nRows = nrows;
      X.bsr.nCols = n_sources;
      X.bsr.nnzb  = n_all_targets;
      X.bsr.RowPtr.resize(X.bsr.nRows + 1, 0);
      X.bsr.ColInd.resize(X.bsr.nnzb);

  
      // fill the bsr structures with data
      size_t n_all_targets_prefix{0};
      for(uint32_t irow = 0; irow < nrows; ++irow) {
          index4_t const i4(row_coord[irow]);
          assert(irow == row_index[i4.ui8[2]][i4.ui8[1]][i4.ui8[0]] && "fatal error");
          for(uint32_t jcol = 0; jcol < X_column_index[irow].size(); ++jcol) {
              X.bsr.ColInd[n_all_targets_prefix] = X_column_index[irow][jcol];
              ++n_all_targets_prefix;
          } // jcol
          X.bsr.RowPtr[irow + 1] = n_all_targets_prefix;
      } // irow
      
      if (1) { // check that we have translated all entries
          if (echo > 0) std::cout << "# " << n_all_targets_prefix*.001 << " k target blocks" << std::endl;
          assert(n_all_targets == n_all_targets_prefix);
      } // scope: sanity checks






      // ========= construct sparsity pattern for B
      
      // create the BSR structure for B (identity operator)
      B.bsr.nRows = nrows;
      B.bsr.nCols = n_sources;
      B.bsr.nnzb  = n_sources; // identity operator has exatcly one nonzero block per row
      B.bsr.RowPtr.resize(B.bsr.nRows + 1, 0);
      B.bsr.ColInd.resize(B.bsr.nnzb);
      
      // translate the data from source_block_index[isrc] --> row_index
      std::vector<int32_t> isource_row(nrows, -1);
      for(int32_t isrc = 0; isrc < n_sources; ++isrc) {
          index4_t const i4(source_block_index[isrc]);
          auto const irow = row_index[i4.ui8[2]][i4.ui8[1]][i4.ui8[0]];
          if (echo > 6) {
              std::printf("# source block %i at %i %i %i (0x%6.6x) has row %i\n", 
                  isrc, i4.i8[0],i4.i8[1],i4.i8[2], source_block_index[isrc], irow);
          } // echo
          assert(irow >= 0 && "strange! source blocks are not included in the target area");
          isource_row[irow] = isrc;
      } // isrc

      size_t n_all_sources_prefix{0};
      for(uint32_t irow = 0; irow < nrows; ++irow) {
          auto const isrc = isource_row[irow];
          if (isrc >= 0) {
              B.bsr.ColInd[n_all_sources_prefix] = isrc;
              ++n_all_sources_prefix;
          } // diagonal entry
          B.bsr.RowPtr[irow + 1] = n_all_sources_prefix;
      } // irow

      if (1) { // check that we have translated all entries
          if (echo > 0) std::cout << "# " << n_all_sources_prefix << " source blocks" << std::endl;
          assert(n_sources == n_all_sources_prefix);
      } // scope: sanity checks

      // fill with data (all nonzero data blocks are unit matrices)
      bool constexpr fill_B_with_data = true;
      bool constexpr use_B_indirection = true;
      DenseBlock<BS, int8_t> unit_block(0);
      for(int i = 0; i < BS; ++i) unit_block.data[i][i] = 1;
      if (fill_B_with_data) {
          if (use_B_indirection) {
              B.blocks.resize(1);
              B.blocks[0] = &unit_block;
              B.indirection.resize(B.bsr.nnzb, 0); // all indirection indices are 0
          } else {
              B.blocks.resize(B.bsr.nnzb);
              for(unsigned inzb = 0; inzb < B.bsr.nnzb; ++inzb) {
                  B.blocks[inzb] = &unit_block;
              } // inzb
              B.indirection.clear();
          } // use_B_indirection
      } else {
          if (echo > 0) std::cout << "# data blocks of B are simple unit matrices" << std::endl;
      } // fill_B_with_data

      
      
      
      
      // now construct the A operator

      // ========= construct sparsity pattern for A
      
      // create the BSR structure for A (block sparse action)
      A.bsr.nRows = nrows;
      A.bsr.nCols = nrows; // logically square operator
      A.bsr.nnzb  = 0; // to be determined
      A.bsr.RowPtr.resize(A.bsr.nRows + 1, 0);
      A.bsr.ColInd.reserve(nrows*nob); // = std::vector<int>(A.bsr.nnzb);

      A.scale_data = 1./FDdenom; // store the global denominator instead of loading floating point values

      bool const use_A_indirection = true;
      if (use_A_indirection) {
          A.blocks.resize(nob);
          for(int iob = 0; iob < nob; ++iob) {
              A.blocks[iob] = &Stencil[iob];
          } // iob
          A.indirection.reserve(nrows*nob);
      } else {
          A.blocks.reserve(nrows*nob);
      } // use_A_indirection

      size_t A_hits_outside{0};
      std::vector<uint32_t> stencil_outside(1 + pow2(stencil_range), 0);
      for(uint32_t irow = 0; irow < nrows; ++irow) {
          index4_t const i4(row_coord[irow]);
          for(int iob = 0; iob < nob; ++iob) {
              uint16_t new_coords[3];
              for(int d = 0; d < 3; ++d) {
                  int16_t const shift = origin_blocks[iob][d];
                  int16_t const shifted_coordinate = shift + i4.i8[d];
                  new_coords[d] = shifted_coordinate & 0xff; // modulo 256
              } // d
              auto const jrow = row_index[new_coords[2]][new_coords[1]][new_coords[0]];
              if (jrow >= 0) {
                  A.bsr.ColInd.push_back(jrow);
                  if (use_A_indirection) {
                      A.indirection.push_back(iob);
                  } else {
                      A.blocks.push_back(&Stencil[iob]);
                  }
              } else {
                  assert(-1 == jrow);
                  ++A_hits_outside;
                  auto const sr = origin_blocks[iob][3];
                  ++stencil_outside[sr];
              } // jrow
          } // iob
          A.bsr.RowPtr[irow + 1] = A.bsr.ColInd.size();
      } // irow
      A.bsr.nnzb = A.bsr.ColInd.size();
      if (echo > 0) std::cout << "# operator has " << A.bsr.nnzb << " nonzero blocks and "
                              << A_hits_outside << " hits outside" << std::endl;
      for(int h = 0; h <= pow2(stencil_range); ++h) {
          if (echo > 6 && stencil_outside[h] > 0) {
              std::cout << "# " << std::sqrt(h) << " " << stencil_outside[h] << " hits outside" << std::endl;
          } // echo and histogram nonzero
      } // h

      if ('r' == (ref | 32)) {
          if (echo > 0) std::cout << "# create a reference solution" << std::endl;
//        ToDo:
//          loop over RHS blocks and solve the problem using a dense linear algebra method, e.g. zgesv
//          store the solution in X as reference
#ifdef  HAS_LAPACK
          
#endif // HAS_LAPACK
      } // create a reference solution using LAPACK
      
      // export the matrices into the format required for tfqmrgpu

      if (nullptr != filename) {
          auto const f = std::fopen(filename, "w");
          assert(nullptr != f && "failed to open existing file for writing!");
        
          std::fprintf(f, "<?xml version=\"%.1f\"?>\n", 1.0);
          std::fprintf(f, "<LinearProblem problem_kind=\"A*X==B\"\n"
                          "               generator_version=\"%.1f\" tolerance=\"%.3e\">\n", 0.0, tolerance);
          std::fprintf(f, "  <!-- input: radius_source_blocks=%g", rsb);
          std::fprintf(f,              " radius_target_blocks=%g", rtb);
          std::fprintf(f,              " block_edge=%d", BlockEdge);
          std::fprintf(f,              " dimensions=%d", Dimension);
          std::fprintf(f,              " finite_difference=%d", nFD);
          std::fprintf(f,              " -->\n");
          xml_export_operator(f, A);
          xml_export_operator(f, B);
          xml_export_operator(f, X);

          std::fprintf(f, "</LinearProblem>\n");
          std::fclose(f);
          
          if (echo > 0) std::cout << "# file \"" << filename << "\" written" << std::endl;
      } else {
          if (echo > 1) std::cout << "# filename empty, no file written" << std::endl;
      } // filename

      return 0;
  } // generate

  template <int Dimension>
  int generate(
        int const block_edge
      , float const rsb
      , float const rtb
      , char const ref
      , int const echo=0
  ) {
      // resolve non-type template parameter BlockEdge
      switch(block_edge) {
        case 1: return generate<1,Dimension>(rsb, rtb, ref, echo);
        case 2: return generate<2,Dimension>(rsb, rtb, ref, echo);
        case 4: return generate<4,Dimension>(rsb, rtb, ref, echo);
        case 8: return generate<8,Dimension>(rsb, rtb, ref, echo);
        default:
          int constexpr line_to_modify = __LINE__ - 3;
          assert(block_edge > 0 && "block_edge must be a positive integer!");
          std::cerr << std::endl 
                    << "ERROR: missing case for BlockEdge= " << block_edge << " in "
                    << __FILE__ << ":" << line_to_modify << std::endl << std::endl;
          return -1; // error
      } // switch block_edge
  } // generate

#ifdef __MAIN__
  int main(int argc, char *argv[]) {

      // parse command line arguments or set default values
      char const *executable =   (argc > 0) ?           argv[0]  : __FILE__;
      float const rsb = std::abs((argc > 1) ? std::atof(argv[1]) : 1.75); // in units of grid points
      float const rtb = std::abs((argc > 2) ? std::atof(argv[2]) : 6.75); // in units of grid points
      int const BlockEdge      = (argc > 3) ? std::atoi(argv[3]) : 2;
      int const dimension      = (argc > 4) ? std::atoi(argv[4]) : 3;
      char const ref           = (argc > 5) ?          *argv[5]  : 'n';
      int const echo           = (argc > 6) ? std::atoi(argv[6]) : 5;

      if (echo > 0) {
          std::cout << std::endl;
          std::cout << "# ========================================" << std::endl;
          std::cout << "# === tfQMRgpu example input generator ===" << std::endl;
          std::cout << "# ========================================" << std::endl;
          std::cout << "# " << executable
                    << "  radius_source_blocks= " << rsb
                    << "  radius_target_blocks= " << rtb
                    << std::endl << "#     "
                    << "  block_edge= " << BlockEdge
                    << "  dimension= " << dimension
                    << "  compute_reference= " << ref
                    << "  echo= " << echo
                    << std::endl << std::endl;
      } // echo

      switch (dimension) {
        case  1: return generate<1>(BlockEdge, rsb, rtb, ref, echo);
        case  2: return generate<2>(BlockEdge, rsb, rtb, ref, echo);
        case  3: return generate<3>(BlockEdge, rsb, rtb, ref, echo);
        default: std::cout << "Error, dimension must be in {1,2,3}!" << std::endl;
      }
      return 0;
  } // main
#endif // __MAIN__
