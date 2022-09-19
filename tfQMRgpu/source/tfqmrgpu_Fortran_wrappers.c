/* This wrapper performs the following forwarding
 * The lowercase names with underscore on the left are 
 *      void functions with a status variable as last argument, 
 *      all arguments are passed by reference
 *      so they can be called from Fortran like subroutines.
 * The right hand side are C-interfaced functions 
 *      that return a status (int) as defined in "tfqmrgpu.h".
 *
 * tfqmrgpucreatehandle_        --> tfqmrgpuCreateHandle
 * tfqmrgpusetstream_           --> tfqmrgpuSetStream
 * tfqmrgpugetstream_           --> tfqmrgpuGetStream
 * tfqmrgpuprinterror_          --> tfqmrgpuPrintError
 * tfqmrgpu_bsrsv_createplan_   --> tfqmrgpu_bsrsv_createPlan
 * tfqmrgpu_bsrsv_buffersize_   --> tfqmrgpu_bsrsv_bufferSize
 * tfqmrgpucreateworkspace_     --> tfqmrgpuCreateWorkspace
 * tfqmrgpu_bsrsv_setbuffer_    --> tfqmrgpu_bsrsv_setBuffer
 * tfqmrgpu_bsrsv_getbuffer_    --> tfqmrgpu_bsrsv_getBuffer
 * tfqmrgpu_bsrsv_setmatrix_c_  --> tfqmrgpu_bsrsv_setMatrix
 * tfqmrgpu_bsrsv_setmatrix_z_  --> tfqmrgpu_bsrsv_setMatrix
 * tfqmrgpu_bsrsv_solve_        --> tfqmrgpu_bsrsv_solve
 * tfqmrgpu_bsrsv_getinfo_      --> tfqmrgpu_bsrsv_getInfo
 * tfqmrgpu_bsrsv_getmatrix_c_  --> tfqmrgpu_bsrsv_getMatrix
 * tfqmrgpu_bsrsv_getmatrix_z_  --> tfqmrgpu_bsrsv_getMatrix
 * tfqmrgpu_bsrsv_destroyplan_  --> tfqmrgpu_bsrsv_destroyPlan
 * tfqmrgpudestroyworkspace_    --> tfqmrgpuDestroyWorkspace
 * tfqmrgpudestroyhandle_       --> tfqmrgpuDestroyHandle
 * 
 * The order of listing roughly resembles the default workflow
 * of this library (except for GetStream and getBuffer).
 * The _c_ and _z_ suffixes are for 32bit and 64bit complex arrays, respectively.
 */

// #include <assert.h>
#include <stddef.h> // for size_t
#include <stdint.h> // int32_t, int64_t

// #define DEBUG

#ifdef DEBUG
	#include <stdio.h> // printf
	#include <stdlib.h>
#endif

typedef int64_t cudaStream_t; // workaround to test without cuda headers:

#include "tfqmrgpu.h" // the full C-API of the tfqmrgpu library
// type abbreviations
typedef tfqmrgpuBsrsvPlan_t plan_t; //
typedef tfqmrgpuHandle_t handle_t; //
typedef tfqmrgpuStatus_t stat_t; //
typedef tfqmrgpuDataLayout_t layout_t; //

  // For the Fortran interface, we generate a set of wrapper void functions, 
  // which can be called like subroutines in Fortran

  void tfqmrgpuprinterror_(stat_t const *status, stat_t *stat) {
       *stat = tfqmrgpuPrintError(*status);
  }

  void tfqmrgpucreatehandle_(handle_t *handle, stat_t *stat) {
      *handle = NULL;
      *stat = tfqmrgpuCreateHandle(handle); // here, handle is passed by reference
  }

  void tfqmrgpudestroyhandle_(handle_t *handle, stat_t *stat) {
      *stat = tfqmrgpuDestroyHandle(*handle);
      *handle = NULL;
  }

  void tfqmrgpusetstream_(handle_t const *handle, cudaStream_t const *streamId, stat_t *stat) {
      *stat = tfqmrgpuSetStream(*handle, *streamId);
  }

  void tfqmrgpugetstream_(handle_t const *handle, cudaStream_t *streamId, stat_t *stat) {
      *stat = tfqmrgpuGetStream(*handle, streamId);
  }

  void tfqmrgpu_bsrsv_createplan_(handle_t const *handle, plan_t *plan, int32_t const *mb, 
      int32_t const* bsrRowPtrA, int32_t const *nnzbA, int32_t const* bsrColIndA,
      int32_t const* bsrRowPtrX, int32_t const *nnzbX, int32_t const* bsrColIndX,
      int32_t const* bsrRowPtrB, int32_t const *nnzbB, int32_t const* bsrColIndB, 
      stat_t *stat) {
      int32_t const FortranIndexOffset = 1;
      *plan = NULL;
#ifdef  DEBUG
      printf("tfqmrgpu_bsrsv_createplan_(handle=%p, *plan=%p, mb=%d, \n"
               "         bsrRowPtrA=%p, nnzbA=%d, bsrColIndA=%p, \n"
               "         bsrRowPtrX=%p, nnzbX=%d, bsrColIndX=%p, \n"
               "         bsrRowPtrB=%p, nnzbB=%d, bsrColIndB=%p, indexOffset=%d)\n",
               *handle, *plan, *mb,            bsrRowPtrA, *nnzbA, bsrColIndA, 
               bsrRowPtrX, *nnzbX, bsrColIndX, bsrRowPtrB, *nnzbB, bsrColIndB, FortranIndexOffset);
#endif // DEBUG
      *stat = tfqmrgpu_bsrsv_createPlan(*handle, plan, *mb, // here, plan is passed by reference
               bsrRowPtrA, *nnzbA, bsrColIndA,
               bsrRowPtrX, *nnzbX, bsrColIndX,
               bsrRowPtrB, *nnzbB, bsrColIndB,
               FortranIndexOffset); // passed by value
      if (TFQMRGPU_STATUS_SUCCESS != *stat) tfqmrgpuPrintError(*stat);
#ifdef  DEBUG
      printf("done tfqmrgpu_bsrsv_createplan_(handle=%p, *plan=%p, ...)\n", *handle, *plan);      
#endif // DEBUG
  }

  void tfqmrgpu_bsrsv_destroyplan_(handle_t const *handle, plan_t *plan, stat_t *stat) {
      *stat = tfqmrgpu_bsrsv_destroyPlan(*handle, *plan);
      *plan = NULL;
  }

  void tfqmrgpu_bsrsv_buffersize_(handle_t const *handle, plan_t const *plan,
      int32_t const *ldA, int32_t const *blockDim, int32_t const *ldB, int32_t const *RhsBlockDim,
      char const *doublePrecision, size_t *pBufferSizeInBytes, stat_t *stat) {
      *stat = tfqmrgpu_bsrsv_bufferSize(*handle, *plan,
                *ldA, *blockDim, *ldB, *RhsBlockDim, *doublePrecision, 
                pBufferSizeInBytes); // here, pBufferSizeInBytes is passed by reference
  }

  void tfqmrgpucreateworkspace_(void* *pBuffer, size_t const *pBufferSizeInBytes, stat_t *stat) {
#ifdef  DEBUGGPU
      printf("# try to allocate %.6f MByte @device\n", 1e-6*(*pBufferSizeInBytes));
#endif // DEBUGGPU
      *stat = tfqmrgpuCreateWorkspace(pBuffer, *pBufferSizeInBytes, 'd'); // 'd':use device memory, 'm': use managed memory 
#ifdef  DEBUGGPU
      printf("# allocate %.6f MByte at %p @device\n", 1e-6*(*pBufferSizeInBytes), *pBuffer);
#endif // DEBUGGPU
  }

  void tfqmrgpudestroyworkspace_(void* *pBuffer, stat_t *stat) {
      *stat = tfqmrgpuDestroyWorkspace(*pBuffer);
  }

  void tfqmrgpu_bsrsv_setbuffer_(handle_t const *handle, plan_t const *plan, 
              void* const *pBuffer, stat_t *stat) {
#ifdef  DEBUG
      printf("# register device pointer %p @device in plan\n", *pBuffer);
#endif // DEBUG
      *stat = tfqmrgpu_bsrsv_setBuffer(*handle, *plan, *pBuffer);
#ifdef  DEBUG
      if (TFQMRGPU_STATUS_SUCCESS != *stat) tfqmrgpuPrintError(*stat);
#endif // DEBUG
  }

  void tfqmrgpu_bsrsv_getbuffer_(handle_t const *handle, plan_t const *plan,
              void* *pBuffer, stat_t *stat) {
#ifdef  DEBUG
      printf("# query device pointer registered in plan\n");
#endif // DEBUG
      *stat = tfqmrgpu_bsrsv_getBuffer(*handle, *plan, pBuffer); // here, pBuffer is passed by reference
#ifdef  DEBUG
      if (TFQMRGPU_STATUS_SUCCESS != *stat) tfqmrgpuPrintError(*stat);
      printf("# device pointer %p @device registered in plan\n", *pBuffer);
#endif // DEBUG
  }

  void tfqmrgpu_bsrsv_setmatrix_c_(handle_t const *handle, plan_t const *plan, char const *var, 
          float const*  val, int32_t const *ld, int32_t const *d2, char const *trans, layout_t const *layout, stat_t *stat) {
      *stat = tfqmrgpu_bsrsv_setMatrix(*handle, *plan, *var, (void*) val, 'C', *ld, *d2, *trans, *layout);
  }

  void tfqmrgpu_bsrsv_setmatrix_z_(handle_t const *handle, plan_t const *plan, char const *var, 
          double const* val, int32_t const *ld, int32_t const *d2, char const *trans, layout_t const *layout, stat_t *stat) {
      *stat = tfqmrgpu_bsrsv_setMatrix(*handle, *plan, *var, (void*) val, 'Z', *ld, *d2, *trans, *layout);
  }

  void tfqmrgpu_bsrsv_getmatrix_c_(handle_t const *handle, plan_t const *plan, char const *var,
          float*  val, int32_t const *ld, int32_t const *d2, char const *trans, layout_t const *layout, stat_t *stat) {
      *stat = tfqmrgpu_bsrsv_getMatrix(*handle, *plan, *var, (void*) val, 'C', *ld, *d2, *trans, *layout);
  }

  void tfqmrgpu_bsrsv_getmatrix_z_(handle_t const *handle, plan_t const *plan, char const *var,
          double* val, int32_t const *ld, int32_t const *d2, char const *trans, layout_t const *layout, stat_t *stat) {
      *stat = tfqmrgpu_bsrsv_getMatrix(*handle, *plan, *var, (void*) val, 'Z', *ld, *d2, *trans, *layout);
  }

  void tfqmrgpu_bsrsv_solve_(handle_t const *handle, plan_t const *plan,
          double const *threshold, int32_t const *maxIterations, stat_t *stat) {
      *stat = tfqmrgpu_bsrsv_solve(*handle, *plan, *threshold, *maxIterations);
  }

  void tfqmrgpu_bsrsv_getinfo_(handle_t const *handle, plan_t const *plan, double *residuum_reached,
        int32_t *iterations_needed, double *flops_performed, double *flops_performed_all, stat_t *stat) {
      *stat = tfqmrgpu_bsrsv_getInfo(*handle, *plan, residuum_reached, iterations_needed, flops_performed, flops_performed_all); // last 4 args by reference
#ifdef  DEBUG
      printf("# %s residuum_reached= %.1e  iterations_needed= %d\n", __func__, *residuum_reached, *iterations_needed);
#endif // DEBUG
  }
