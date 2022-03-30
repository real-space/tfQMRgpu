#pragma once

// #define DEBUG
// #define DEBUGGPU

#include "tfqmrgpu_util.hxx" // ...
        // get_data_from_gpu, copy_data_to_gpu, FlopChar, take_gpu_memory, print_array
#include "tfqmrgpu_linalg.hxx" // ...
        // highestbit, ...
        // TFQMRGPU_MEMORY_ALIGNMENT, TFQMRGPU_STATUS_SUCCESS, ...
        // set_complex_value, clear_on_gpu, ...
        // dotp, nrm2, xpay, axpy, ...
        // tfQMRdec*, add_RHS, set_unit_blocks
        // create_random_numbers
        // debug_printf

namespace tfqmrgpu {

  template <class action_t>
  tfqmrgpuStatus_t solve(
        action_t & action
      , char* const gpu_memory_buffer=nullptr // pass in GPU memory here, memcount-mode if nullptr
      , double const tolerance=1e-6
      , int const MaxIterations=999
      , cudaStream_t streamId=0
      , bool const rhs_trivial=false // trivial: right hand side are columns of the unit matrix
  ) {
      PUSH_RANGE("tfQMR preparation"); // NVTX

      using real_t = typename action_t::real_t; // abbreviate
      auto constexpr LM = action_t::LM; // abbreviate block size

      auto const p = action.get_plan(); // get pointer to plan, plan gets modified
//    auto const precond = int(action.has_preconditioner());

      // abbreviations
      auto const nnzbB = p->subset.size(); // number of non-zero blocks in B
      auto const nnzbX = p->colindx.size(); // number of non-zero blocks in X
      auto const nCols = p->nCols; // number of block columns
      debug_printf("\n# nnzbB=%lu nnzbX=%lu nCols=%u\n\n", nnzbB, nnzbX, nCols);

      char* buffer{gpu_memory_buffer}; // memcount-mode computes the ...
      // ... total memory requirement and creates the memory window descriptors
      char* const buffer_start = buffer;

      auto const v1 = take_gpu_memory<real_t[2][LM][LM]>(buffer, nnzbX, &(p->matXwin), "X"); // the result X will be here
      // save position and length of the v1 section in the buffer
      auto const v4 = take_gpu_memory<real_t[2][LM][LM]>(buffer, nnzbX);
      auto const v5 = take_gpu_memory<real_t[2][LM][LM]>(buffer, nnzbX);
      auto const v6 = take_gpu_memory<real_t[2][LM][LM]>(buffer, nnzbX);
      auto const v7 = take_gpu_memory<real_t[2][LM][LM]>(buffer, nnzbX);
      auto const v8 = take_gpu_memory<real_t[2][LM][LM]>(buffer, nnzbX);
      auto const v9 = take_gpu_memory<real_t[2][LM][LM]>(buffer, nnzbX);
//    auto const vP = take_gpu_memory<real_t[2][LM][LM]>(buffer, nnzbX*precond);

      // random number vector, random values are generated outside of this routine
      auto const v3 = take_gpu_memory<float[2][LM][LM]>(buffer, nnzbX, &(p->vec3win), "v3"); // v3 is always of type float as this stores only random numbers
      debug_printf("# v3 has address %s%p\n", buffer_start?"":"buffer + ", (void*)v3);

      // matrix B
      auto const v2 = take_gpu_memory<real_t[2][LM][LM]>(buffer, nnzbB, &(p->matBwin), "B"); // usually small

      // complex scalars per RHS
      auto const rho  = take_gpu_memory<real_t[2][LM]>(buffer, nCols);
      auto const alfa = take_gpu_memory<real_t[2][LM]>(buffer, nCols);
      auto const beta = take_gpu_memory<real_t[2][LM]>(buffer, nCols);
      auto const c67  = take_gpu_memory<real_t[2][LM]>(buffer, nCols);
      auto const eta  = take_gpu_memory<real_t[2][LM]>(buffer, nCols);

      assert(nnzbX > 0);
      unsigned const l2nX = highestbit(nnzbX - 1) + 1; // number of reduction levels == ceiling(log2(nnzbX))
      auto const zvv  = take_gpu_memory<double[2][LM]>(buffer, (1ul << l2nX)*nCols);
      // real-valued scalars per RHS
      auto const dvv  = take_gpu_memory<double[1][LM]>(buffer, (1ul << l2nX)*nCols);
      auto const tau  = take_gpu_memory<double[LM]>(buffer, nCols);
      auto const var  = take_gpu_memory<double[LM]>(buffer, nCols);

      assert(nCols <= (1ul << 16)); // working with uint16_t as column indices, we hay at most have 65,536 block columns
      auto const colindx = take_gpu_memory<uint16_t>(buffer, nnzbX, &(p->colindxwin), "colindx");
      auto const subset  = take_gpu_memory<uint32_t>(buffer, nnzbB, &(p->subsetwin), "subset");
      auto const status  = take_gpu_memory<int8_t[LM]>(buffer, nCols);

      // transfer index lists into GPU memory, can be skipped if action_t uses managed memory

      if (nullptr != gpu_memory_buffer) {
          action.transfer(gpu_memory_buffer, streamId);
      }

      action.take_memory(buffer); // measure memory consumption and set memory windows or set GPU pointers into memory buffer

      if (nullptr == gpu_memory_buffer) { // memcount-mode
          p->gpu_mem = buffer - buffer_start + (1ull << TFQMRGPU_MEMORY_ALIGNMENT); // add safety
          debug_printf("# GPU memory requirement = %.6f GByte\n", p->gpu_mem*1e-9);
          return TFQMRGPU_STATUS_SUCCESS; // return early as we only counted the device memory requirement
      } // memcount
      
      // host array allocations

      int const nRHSs = nCols*LM; // total number of right-hand sides
      auto const status_h = new int8_t[nCols][LM]; // tfQMR status on host
      for(auto rhs = 0; rhs < nRHSs; ++rhs) { status_h[0][rhs] = 0; } // this needs to be done every time
      auto const resnrm2_h = new double[nCols][1][LM]; // residual norm squared on host
      auto const res_ub_h = new double[nCols][LM]; // residual upper bound squared on host
      auto const invBn2_h = new double[nCols][LM]; // inverse_norm2_of_B on host

      ////////////////////////////////////////////////
      // no GPU kernels are called before this line //
      ////////////////////////////////////////////////
      
      clear_on_gpu<real_t[2][LM][LM]>(v4, nnzbX, streamId);
      clear_on_gpu<real_t[2][LM][LM]>(v5, nnzbX, streamId);
      clear_on_gpu<real_t[2][LM][LM]>(v6, nnzbX, streamId);
      clear_on_gpu<real_t[2][LM][LM]>(v7, nnzbX, streamId);
      clear_on_gpu<real_t[2][LM][LM]>(v8, nnzbX, streamId);
      clear_on_gpu<real_t[2][LM][LM]>(v9, nnzbX, streamId);
      
      clear_on_gpu<real_t[2][LM]>(eta, nCols, streamId); // eta = 0; // needs to run every time again?
      set_complex_value<real_t,LM>(rho, nCols, 1., 0., streamId); // needs to run every time!
      clear_on_gpu<double[LM]>(var, nCols, streamId); // var = 0; // needs to run every time!

      clear_on_gpu<real_t[2][LM][LM]>(v1, nnzbX, streamId); // deletes the content of v, ToDo: only if setMatrix('X') has not been called
      
      clear_on_gpu<int8_t[LM]>(status, nCols, streamId); // set status = 0

      double const tol2 = tolerance*tolerance; // internally, the squares of all norms and thresholds are used
      double target_bound2{tol2*(100*100)}; // init with test_factor=100
      double residual2_reached{1e300};

      double nFlop{0}; // counter for floating point multiplications
#define DOTP(d,w,v) nFlop += dotp<real_t,LM>(d, v, w, colindx, nnzbX, nCols, l2nX, streamId)
#define NRM2(d,v)   nFlop += nrm2<real_t,LM>(d, v,    colindx, nnzbX, nCols, l2nX, streamId)
#define XPAY(y,a,x) nFlop += xpay<real_t,LM>(y, a, x, colindx, nnzbX, streamId)
#define AXPY(y,x,a) nFlop += axpy<real_t,LM>(y, x, a, colindx, nnzbX, streamId)
#define MULT(y,x)   nFlop += action.multiply(y, x,    colindx, nnzbX, nCols, l2nX, streamId)

      if (rhs_trivial) {
          clear_on_gpu<real_t[2][LM][LM]>(v2, nnzbB, streamId);
          set_unit_blocks<real_t,LM>(v2, nnzbB, streamId,  1,0  );
          for(auto rhs = 0; rhs < nRHSs; ++rhs) invBn2_h[0][rhs] = 1;
          // also, we probably called ::solve without much surrounding, so we need to regenerate the random numbers
          create_random_numbers(v3[0][0][0], nnzbX*2*LM*LM, streamId);
      } else {
          // rhs is non-trivial
          // ToDo: move this part into the tail of setMatrix('B')
          // v5 == 0
          add_RHS<real_t,LM>(v5, v2, 1, subset, nnzbB, streamId); // v5 := v5 + v2
          NRM2(dvv, v5); // dvv := ||v5||
          cudaMemcpy(tau, dvv, nCols*LM*sizeof(double), cudaMemcpyDeviceToDevice);
          // if we always have unit matrices in B, these 3 steps can be avoided and tau := 1
          // ToDo: check if we can call nrm2<real_t,LM>(dvv, v2, ColIndOfB, nnzbB, nCols, l2nX)

          // ToDo: split this part into two: allocation on CPU and transfer to the CPU, can be done when setMatrix('B')
          get_data_from_gpu<double[LM]>(invBn2_h, tau, nCols, streamId); // inverse_norm2_of_B
          for(auto rhs = 0; rhs < nRHSs; ++rhs) invBn2_h[0][rhs] = 1./invBn2_h[0][rhs]; // invert in-place on the host
      } // rhs_trivial

      tfqmrgpuStatus_t return_status{TFQMRGPU_STATUS_MAX_ITERATIONS}; // preliminary result
      p->iterations_needed = MaxIterations; // preliminary, will be changed if it converges
      p->flops_performed = 0;
      
      int iteration{0};

      POP_RANGE(); // end of NVTX range
      PUSH_RANGE("tfQMR iterations"); // NVTX

      while (iteration < MaxIterations) {
          ++iteration;
          debug_printf("# iteration %i of %d\n", iteration, MaxIterations);

          // =============================
          // tfQMR loop body:
          // first half-step

          DOTP(zvv, v3, v5); // zvv := v3.v5

          // decisions based on v3.v5 and rho
          tfQMRdec35<real_t,LM>
#ifndef HAS_NO_CUDA
              <<< nCols, LM, 0, streamId >>>
#endif // HAS_CUDA
              (status, rho, beta, zvv, nCols);

          XPAY(v6, beta, v5); // v6 := v5 + beta*v6

          XPAY(v4, beta, v8); // v4 := v8 + beta*v4 // can be executed in parallel to the next A call

          MULT(v9, v6); // v9 := A*v6

          XPAY(v4, beta, v9); // v4 := v9 + beta*v4

          DOTP(zvv, v3, v4); // zvv := v3.v4
          
          // decisions based on v3.v4 and rho
          tfQMRdec34<real_t,LM>
#ifndef HAS_NO_CUDA
              <<< nCols, LM, 0, streamId >>>
#endif // HAS_CUDA
              (status, c67, alfa, rho, eta, zvv, var, nCols);

          XPAY(v7, c67, v6); // v7 := v6 + c67*v7

          AXPY(v5, v9, alfa); // v5 := alfa*v9 + v5

          NRM2(dvv, v5); // dvv := ||v5||

          // decisions based on tau
          tfQMRdecT<real_t,LM> 
#ifndef HAS_NO_CUDA
              <<< nCols, LM, 0, streamId >>>
#endif // HAS_CUDA
              (status, c67, eta, var, tau, alfa, dvv, nCols);
      
          AXPY(v1, v7, eta); // v1 := eta*v7 + v1 // update solution vector
      
          AXPY(v6, v4, alfa); // v6 := alfa*v4 + v6
      
          XPAY(v7, c67, v6); // v7 := v6 + c67*v7 // can be executed in parallel to the next A call

          // second half-step

          MULT(v8, v6); // v8 := A*v6
      
          AXPY(v5, v8, alfa); // v5 := alfa*v8 + v5

          NRM2(dvv, v5); // dvv := ||v5||

          // decisions based on tau
          tfQMRdecT<real_t,LM>
#ifndef HAS_NO_CUDA
              <<< nCols, LM, 0, streamId >>>
#endif // HAS_CUDA
              (status, 0x0, eta, var, tau, alfa, dvv, nCols);

          AXPY(v1, v7, eta); // v1 := eta*v7 + v1 // update solution vector

          get_data_from_gpu<double[LM]>(res_ub_h, tau, nCols, streamId); // missing factor inverse_norm2_of_B
          get_data_from_gpu<int8_t[LM]>(status_h, status, nCols, streamId);
          // CCheck(cudaDeviceSynchronize()); // necessary?

          double max_bound2{0};
          double min_bound2{9e99}; // debug
          int breakdown5{0}, breakdown4{0};
          for(auto rhs = 0; rhs < nRHSs; ++rhs) {
              res_ub_h[0][rhs] *= invBn2_h[0][rhs]; // apply factor inverse_norm2_of_B    
              max_bound2 = std::max(max_bound2, res_ub_h[0][rhs]);
              min_bound2 = std::min(min_bound2, res_ub_h[0][rhs]);
              breakdown4 += (-2 == status_h[0][rhs]); // breakdown detected in dec34
              breakdown5 += (-1 == status_h[0][rhs]); // breakdown detected in dec35
          } // rhs
          if (0 == (iteration & 0xf)) { // every 16th iteration
              debug_printf("# in iteration %d, min_bound2 = %g, max_bound2 = %g * %d = %g, target_bound2 = %g\n", iteration, 
                  min_bound2, max_bound2, 2*iteration + 1, max_bound2*(2*iteration + 1), target_bound2);
          }
          max_bound2 *= (2*iteration + 1); // multiply with 2 times the iteration number

          bool probe{(iteration >= MaxIterations || max_bound2 <= target_bound2)};
          if (nRHSs == breakdown5 + breakdown4) {
              debug_printf("# in iteration %d, all %d+%d of %d components broke down!\n", iteration, breakdown5, breakdown4, nRHSs);
              iteration += MaxIterations; // stop the loop
              return_status = TFQMRGPU_STATUS_BREAKDOWN;
              probe = false;
          } // stop now

          if (probe) { // compute the residual
          
              MULT(v9, v1); // v9 := A*v1

//            add_RHS<real_t,LM> <<< nnzbB, { LM, LM, 1 }, 0, streamId >>> (v9, v2, -1, subset, nnzbB); // v9 := v9 - v2
              add_RHS<real_t,LM>(v9, v2, -1, subset, nnzbB, streamId); // v9 := v9 - v2

              NRM2(dvv, v9); // dvv := ||v9||

              get_data_from_gpu<double[1][LM]>(resnrm2_h, dvv, nCols, streamId, "resnrm2"); // missing factor inverse_norm2_of_B
              // CCheck(cudaDeviceSynchronize()); // necessary?

              double max_residual2{1e-99}, min_residual2{9e99};
              bool isDone{true}, status_modified{false};
              for(auto rhs = 0; rhs < nRHSs; ++rhs) {
                  resnrm2_h[0][0][rhs] *= invBn2_h[0][rhs]; // apply factor inverse_norm2_of_B
                  auto const res2 = resnrm2_h[0][0][rhs];
                  max_residual2 = std::max(max_residual2, res2);
                  min_residual2 = std::min(min_residual2, res2);
                  if (res2 > tol2) {
                      if (0 == status_h[0][rhs]) isDone = false; // no breakdown has occurred --> continue converging
                  } else if (res2 <= 0) {
                      status_h[0][rhs] = 1; // component converged
                      status_modified = true;
  //                  converged_at[0][rhs] = iteration; // store iteration in which this rhs-column converged
                  }
              } // rhs
              residual2_reached = max_residual2;
    
              target_bound2 = (max_bound2 / max_residual2) * tol2; // for the next iteration
              debug_printf("# in iteration %d, max_res2 = %g, min_res2 = %g, new target_bound2 = %g\n", 
                                 iteration,    max_residual2, min_residual2,     target_bound2);

              if (isDone) {
                  p->iterations_needed = iteration;
                  iteration += 2*MaxIterations; // stop the iteration loop
                  return_status = TFQMRGPU_STATUS_SUCCESS; // converged
              } // isDone

              if (status_modified) {
                  copy_data_to_gpu<int8_t[LM]>(status, status_h, nCols, streamId, "status");
              } // status_modified

          } // probe

      } // while

      // now, result can be retrieved by get_data_from_gpu, see getMatrix('X')

#pragma omp single
      { // single
          debug_printf("# ran %d iterations\n", (iteration - 1)%std::max(1, MaxIterations) + 1);
          debug_printf("# GPU performed %.6f T%clop\n", nFlop*1e-12, FlopChar<real_t>());
      } // single

#undef  MULT    
#undef  DOTP
#undef  NRM2
#undef  XPAY
#undef  AXPY

      // set the returned status according to converged due to residuum < threshold or iter > MaxIterations or breakdown
    
      p->flops_performed = nFlop;
      p->residuum_reached = std::sqrt(residual2_reached);

      delete[] status_h; // tfQMR status on host
      delete[] resnrm2_h; // residual norm squared on host
      delete[] res_ub_h; // residual upper bound squared on host
      delete[] invBn2_h; // inverse_norm2_of_B on host

      POP_RANGE(); // end of NVTX range

      return return_status;
  } // solve

} // tfqmrgpu

