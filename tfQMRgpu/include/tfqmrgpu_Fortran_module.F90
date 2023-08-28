!! This file is part of tfQMRgpu under MIT-License

#define DEBUG

!! The Fortran API is based on subroutines
!! compared to the C-API, the status is returned in the appended ierr argument
module tfqmrgpu
implicit none
  private !! default visibility for this namespace

  !! export these functions or interfaces
  public :: print_error
  public :: create, destroy, free
  public :: set, get
  public :: solve
  public :: tfqmrgpu_bsrsv_complete ! example will call this

  !! include the Fortran header file
  include "tfqmrgpu_Fortran.h"

  !! export these symbols from tfqmrgpu_Fortran.h
  public :: TFQMRGPU_HANDLE_KIND
  public :: TFQMRGPU_PLAN_KIND
  public :: cuda_stream_kind
  public :: TFQMRGPU_LAYOUT_RIRIRIRI

  interface create
    module procedure createHandle, bsrsv_createPlan, createWorkspace
  endinterface

  interface destroy
    module procedure destroyHandle, bsrsv_destroyPlan
  endinterface

  interface free
    module procedure destroyWorkspace
  endinterface

  interface set
    module procedure setStream, &
      bsrsv_setBuffer, &
      bsrsv_setMatrix_c, &
      bsrsv_setMatrix_z
  endinterface

  interface get
    module procedure getStream, &
      bsrsv_bufferSize,  &
      bsrsv_getBuffer,  &
      bsrsv_getMatrix_c, &
      bsrsv_getMatrix_z, &
      bsrsv_getInfo
  endinterface

  interface solve
    module procedure bsrsv_solve, &
          tfqmrgpu_bsrsv_complete, &
          tfqmrgpu_bsrsv_rectangular
  endinterface

  contains

  subroutine print_error(status, ierr)
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=4), intent(in) :: status
    external :: tfqmrgpuprinterror
    call tfqmrgpuprinterror(status, ierr)
  endsubroutine ! print_error

  subroutine createHandle(handle, ierr)
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(out) :: handle
    external :: tfqmrgpucreatehandle
    call tfqmrgpucreatehandle(handle, ierr)
  endsubroutine ! create

  subroutine destroyHandle(handle, ierr)
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(inout) :: handle
    external :: tfqmrgpudestroyhandle
    call tfqmrgpudestroyhandle(handle, ierr)
  endsubroutine ! destroy

  subroutine setStream(handle, streamId, ierr)
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=cuda_stream_kind), intent(in) :: streamId
    external :: tfqmrgpusetstream
    call tfqmrgpusetstream(handle, streamId, ierr)
  endsubroutine ! set

  subroutine getStream(handle, streamId, ierr)
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=cuda_stream_kind), intent(out) :: streamId
    external :: tfqmrgpugetstream
    call tfqmrgpugetstream(handle, streamId, ierr)
  endsubroutine ! get


#define DevPtrType integer(kind=8)

  subroutine createWorkspace(pBuffer, pBufferSizeInBytes, ierr)
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    DevPtrType, intent(inout) :: pBuffer
    integer(kind=8), intent(in) :: pBufferSizeInBytes
    external :: tfqmrgpucreateworkspace
    call tfqmrgpucreateworkspace(pBuffer, pBufferSizeInBytes, ierr)
#ifdef  DEBUG
    write(*, '(a,":",i0,a,z0)') __FILE__, &
        __LINE__," pBuffer = 0x",pBuffer
#endif
  endsubroutine ! create

  subroutine destroyWorkspace(pBuffer, ierr)
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    DevPtrType, intent(inout) :: pBuffer
    external :: tfqmrgpudestroyworkspace
    call tfqmrgpudestroyworkspace(pBuffer, ierr)
  endsubroutine ! free

    !!!!!!!!!!! bsrsv specific routines !!!!!!!!!!!!!!!
    !! bsrsv is a linear solve of A * X == B
    !! with A, X and B are BSR (block compressed sparse row) formatted operators.
    !! 
    !! the bsrsv_* routines are listed in the order of how they should be called in a default use case.

  subroutine bsrsv_createPlan(handle, plan, &
                  mb, bsrRowPtrA, nnzbA, bsrColIndA, &
                      bsrRowPtrX, nnzbX, bsrColIndX, &
                      bsrRowPtrB, nnzbB, bsrColIndB, echo, ierr)
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(out) :: plan
    integer(kind=4), intent(in) :: mb
    integer(kind=4), intent(in) :: bsrRowPtrA(*)
    integer(kind=4), intent(in) :: nnzbA
    integer(kind=4), intent(in) :: bsrColIndA(*)
    integer(kind=4), intent(in) :: bsrRowPtrX(*)
    integer(kind=4), intent(in) :: nnzbX
    integer(kind=4), intent(in) :: bsrColIndX(*)
    integer(kind=4), intent(in) :: bsrRowPtrB(*)
    integer(kind=4), intent(in) :: nnzbB
    integer(kind=4), intent(in) :: bsrColIndB(*)
!   integer, parameter :: indexOffset = 1 !! Fortran indices start from 1, this is fixed in the C-wrappers
    integer(kind=4), intent(in) :: echo
    external :: tfqmrgpu_bsrsv_createplan
    call tfqmrgpu_bsrsv_createplan(handle, plan, &
                  mb, bsrRowPtrA, nnzbA, bsrColIndA, &
                      bsrRowPtrX, nnzbX, bsrColIndX, &
                      bsrRowPtrB, nnzbB, bsrColIndB, echo, ierr)
  endsubroutine ! create

  subroutine bsrsv_destroyPlan(handle, plan, ierr)
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(out) :: plan
    external :: tfqmrgpu_bsrsv_destroyplan
    call tfqmrgpu_bsrsv_destroyplan(handle, plan, ierr)
  endsubroutine ! destroy

  subroutine bsrsv_bufferSize(handle, plan, &
                    ldA, blockDim, ldB, RhsBlockDim, &
                    prec, pBufferSizeInBytes, ierr)
    !! returns the computed size to be allocated by cudaMalloc
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(inout) :: plan
    integer(kind=4), intent(in) :: ldA
    integer(kind=4), intent(in) :: blockDim
    integer(kind=4), intent(in) :: ldB
    integer(kind=4), intent(in) :: RhsBlockDim
    character      , intent(in) :: prec
    integer(kind=8), intent(out) :: pBufferSizeInBytes
    external :: tfqmrgpu_bsrsv_buffersize
    call tfqmrgpu_bsrsv_buffersize(handle, plan, ldA, blockDim, ldB, &
            RhsBlockDim, prec, pBufferSizeInBytes, ierr)
  endsubroutine ! get

  subroutine bsrsv_setBuffer(handle, plan, pBuffer, ierr)
    !! registers the GPU memory buffer pointer in the plan.
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(inout) :: plan
    DevPtrType, intent(in) :: pBuffer
    external :: tfqmrgpu_bsrsv_setbuffer
#ifdef  DEBUG
    write(*, '(a,":",i0,a,z0)') __FILE__, &
        __LINE__, " set pBuffer = 0x",pBuffer
#endif
    call tfqmrgpu_bsrsv_setbuffer(handle, plan, pBuffer, ierr)
  endsubroutine ! set

  subroutine bsrsv_getBuffer(handle, plan, pBuffer, ierr)
    !! retrieves the GPU memory buffer registered in plan.
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(in) :: plan
    DevPtrType, intent(inout) :: pBuffer
    external :: tfqmrgpu_bsrsv_getbuffer
#ifdef  DEBUG
    write(*, '(a,":",i0,9a)')   __FILE__, &
        __LINE__," call tfqmrgpu_bsrsv_getbuffer_()"
#endif
    call tfqmrgpu_bsrsv_getbuffer(handle, plan, pBuffer, ierr)
#ifdef  DEBUG
    write(*, '(a,":",i0,a,z0)') __FILE__, &
        __LINE__," got pBuffer = 0x",pBuffer
#endif
  endsubroutine ! get

  subroutine bsrsv_setMatrix_c(handle, plan, var, val, ld, d2, trans, layout, ierr)
    !! retrieves the GPU memory buffer registered in plan.
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(inout) :: plan
    character, intent(in) :: var
    complex(kind=4), intent(in) :: val(*)
    integer(kind=4), intent(in) :: ld, d2
    character, intent(in) :: trans
    integer(kind=4), intent(in) :: layout
    external :: tfqmrgpu_bsrsv_setmatrix_c
    call tfqmrgpu_bsrsv_setmatrix_c(handle, plan, var, val, ld, d2, trans, layout, ierr)
  endsubroutine ! set

  subroutine bsrsv_setMatrix_z(handle, plan, var, val, ld, d2, trans, layout, ierr)
    !! retrieves the GPU memory buffer registered in plan.
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(inout) :: plan
    character, intent(in) :: var
    complex(kind=8), intent(in) :: val(*)
    integer(kind=4), intent(in) :: ld, d2
    character, intent(in) :: trans
    integer(kind=4), intent(in) :: layout
    external :: tfqmrgpu_bsrsv_setmatrix_z
    call tfqmrgpu_bsrsv_setmatrix_z(handle, plan, var, val, ld, d2, trans, layout, ierr)
  endsubroutine ! set

  subroutine bsrsv_getMatrix_c(handle, plan, var, val, ld, d2, trans, layout, ierr)
    !! retrieves the GPU memory buffer registered in plan.
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(in) :: plan !! or is the plan modified?
    character, intent(in) :: var
    complex(kind=4), intent(out) :: val(*)
    integer(kind=4), intent(in) :: ld, d2
    character, intent(in) :: trans
    integer(kind=4), intent(in) :: layout
    external :: tfqmrgpu_bsrsv_getmatrix_c
    call tfqmrgpu_bsrsv_getmatrix_c(handle, plan, var, val, ld, d2, trans, layout, ierr)
  endsubroutine ! get

  subroutine bsrsv_getMatrix_z(handle, plan, var, val, ld, d2, trans, layout, ierr)
    !! retrieves the GPU memory buffer registered in plan.
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(in) :: plan !! or is the plan modified?
    character, intent(in) :: var
    complex(kind=8), intent(out) :: val(*)
    integer(kind=4), intent(in) :: ld, d2
    character, intent(in) :: trans
    integer(kind=4), intent(in) :: layout
    external :: tfqmrgpu_bsrsv_getmatrix_z
    call tfqmrgpu_bsrsv_getmatrix_z(handle, plan, var, val, ld, d2, trans, layout, ierr)
  endsubroutine ! get

  subroutine bsrsv_solve(handle, plan, threshold, maxIterations, ierr)
    !! solves the problem prepared in plan
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(inout) :: plan
    real(kind=8), intent(in) :: threshold
    integer(kind=4), intent(in) :: maxIterations
    external :: tfqmrgpu_bsrsv_solve
    call tfqmrgpu_bsrsv_solve(handle, plan, threshold, maxIterations, ierr)
  endsubroutine ! solve

  subroutine bsrsv_getInfo(handle, plan, residual_reached, iterations_needed, &
                            flops_performed, flops_performed_all, ierr)
    !! returns the computed size to be allocated by cudaMalloc
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(inout) :: plan
    real   (kind=8), intent(out) :: residual_reached
    integer(kind=4), intent(out) :: iterations_needed
    real   (kind=8), intent(out) :: flops_performed
    real   (kind=8), intent(out) :: flops_performed_all
    external :: tfqmrgpu_bsrsv_getInfo
    call tfqmrgpu_bsrsv_getInfo(handle, plan, residual_reached, iterations_needed, &
                                 flops_performed, flops_performed_all, ierr)
  endsubroutine ! get

  subroutine tfqmrgpu_bsrsv_rectangular(mb, ldA, ldB, &
    rowPtrA, colIndA, Amat, transA, &
    rowPtrX, colIndX, Xmat, transX, &
    rowPtrB, colIndB, Bmat, transB, iterations, residual, o, ierr)
  !! this routine is meant to test the usability of the tfqmrgpu bsrsv functionality
  !! without an often complicated integration of the library into the target application
  implicit none
    integer(kind=4), intent(in) :: mb ! the number of rows
    integer(kind=4), intent(in) :: ldA ! the block size and leading dimension of A,       ldA must be precompiled in tfqmrgpu.cu as LM
    integer(kind=4), intent(in) :: ldB ! the block size and leading dimension of B and X, ldB must be precompiled in tfqmrgpu.cu as LN
    integer(kind=4), intent(in) :: rowPtrA(:), rowPtrX(:), rowPtrB(:) ! BSR rowPtr#(mb+1)  where #=A,X,B
    integer(kind=4), intent(in) :: colIndA(:), colIndX(:), colIndB(:) ! BSR colInd#(nnzb#) where #=A,X,B
    character,       intent(in) :: transA, transX, transB ! transposition of blocks, allowed states are {'n','t'}
    complex(kind=8), intent(in)  :: Amat(ldA,ldA,*) ! dim(nnzbA) ! BSR non-zero block values of A
    complex(kind=8), intent(out) :: Xmat(ldB,ldA,*) ! dim(nnzbX) ! BSR non-zero block values of X, the result
    complex(kind=8), intent(in)  :: Bmat(ldB,ldA,*) ! dim(nnzbB) ! BSR non-zero block values of B (the RHSs)
    integer(kind=4), intent(inout) :: iterations ! on entry: stop critertion against runtime explosion
                                                 ! on exit:  how many iterations did it run
    real   (kind=8), intent(inout) :: residual   ! on entry: stop criterion for convergence
                                                 ! on exit:  how far down did it converge
    integer(kind=4), intent(in) :: o ! I/O unit to write to, 6=stdout, 0=no write (mute)
                             ! other values <n> may end in files fort.<n> or file connected to this unit
    integer(kind=4), intent(inout) :: ierr ! if ierr /= 0 on entry, debugging is activated

    ! local variables
    integer(kind=TFQMRGPU_HANDLE_KIND) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND) :: plan
    integer(kind=8) :: memSize
    real(kind=8) :: resid, flops, flops_all ! numbers of performed floating point operations, last run and all runs
    DevPtrType :: memBuffer, memBufferCopy
    integer(kind=cuda_stream_kind) :: streamId = 0, streamIdCopy !! 0:default stream
    integer(kind=4) :: maxit
    integer(kind=4), parameter :: layout = TFQMRGPU_LAYOUT_RIRIRIRI
    integer :: ios, debug, u

    debug = 0 ; if (0 /= ierr) debug = 1
    u = 0 ; if (debug > 0) u = o ! u is the debug unit
    maxit = iterations ! entry copy
    resid = residual   ! entry copy

#define  CheckError(ierr, message) call print_error(ierr, ios) ; if (0 /= ierr) stop message

    !! create a library handle
    call create(handle, ierr) 
    CheckError(ierr, "Failed to create tfqmrgpu library handle")

    !! set the CUDA stream to work on
    call set(handle, streamId, ierr) 
    CheckError(ierr, "Failed to set the CUDA stream")

    if (debug > 0) then
!+sanity-check
      !! get the registered CUDA stream and test if it is still the same
      call get(handle, streamIdCopy, ierr) 
      CheckError(ierr, "Failed to get the CUDA stream")
      if (streamIdCopy /= streamId) then 
        if(u>0) write(u, "(9(a,i0))") "streamId = ", &
            streamId," but streamIdCopy = ",streamIdCopy
        stop "[DEBUG] Failed to verify the CUDA stream in use"
      endif
!-sanity-check
    endif ! debug

    !! analyze the sparse matrix patterns of A, X and B and 
    !! create a plan for the forward multiplication Y=A*X and
    !! an index list for the sparse subtraction Y-B
    call create(handle, plan, mb, &
           rowPtrA, size(colIndA), colIndA, &
           rowPtrX, size(colIndX), colIndX, &
           rowPtrB, size(colIndB), colIndB, 9*debug, ierr)
    CheckError(ierr, "Failed to create the bsrsv plan")

    !! compute the size of the required GPU memory buffer
    !! 'z' means solve in double precision.
    !! The library will fail here, if the required block sizes ldA and ldB have not been compiled
    call get(handle, plan, ldA, ldA, ldB, ldB, 'z', memSize, ierr) 
    CheckError(ierr, "Failed to compute GPU memory requirement")

    !! allocate GPU memory
    call create(memBuffer, memSize, ierr)
    CheckError(ierr, "Failed to allocate GPU memory")

    !! register the memory buffer in the plan
    call set(handle, plan, memBuffer, ierr) 
    CheckError(ierr, "Failed to register GPU memory buffer")

    if (debug > 0) then
!+sanity-check
      call get(handle, plan, memBufferCopy, ierr)
      CheckError(ierr, "Failed to get the registered GPU memory buffer address")
      if (memBufferCopy /= memBuffer) then
        write(*, "(9(a,z0))") "memBuffer = 0x",memBuffer, &
                         " but memBufferCopy = 0x",memBufferCopy
        stop "[DEBUG] Failed to verify the GPU memory buffer address"
      endif
!-sanity-check
    endif ! debug

    if(u>0) write(u,*) "Upload Matrices A and B"
    !! upload the nonzero blocks of the input array A
    call set(handle, plan, 'A', Amat(:,1,1), ldA, ldA, transA, layout, ierr) 
    CheckError(ierr, "Failed to upload matrix A")

    !! upload the nonzero blocks of the input array B
    call set(handle, plan, 'B', Bmat(:,1,1), ldB, ldA, transB, layout, ierr) 
    CheckError(ierr, "Failed to upload matrix B")

    if(u>0) write(u,*) "Solve A * X == B"
    !! solve A*X == B using the tfQMR algorithm on the GPU
    call solve(handle, plan, residual, iterations, ierr) 
    CheckError(ierr, "Failed in solver")

    !! get the convergence info
    call get(handle, plan, residual, iterations, flops, flops_all, ierr) 
    CheckError(ierr, "Failed to get info")

    if(u>0) write(u,*) "Download Matrix X"
    !! download the solution matrix X
    call get(handle, plan, 'X', Xmat(:,1,1), ldB, ldA, transX, layout, ierr) 
    CheckError(ierr, "Failed to download matrix X")

    !! clean up
    call free(memBuffer, ierr)
    CheckError(ierr, "Failed to free GPU memory")
    call destroy(handle, plan, ierr)
    CheckError(ierr, "Failed to destroy bsrsv plan")
    call destroy(handle, ierr)
    CheckError(ierr, "Failed to destroy library handle")

#undef  CheckError

    if(o>0) write(o,"(2(a,es8.1),9(a,i0))") &
        " tfqmrgpu_bsrsv converged to",residual, &
        " (max",resid,") in ",iterations," (max ",maxit,") iterations."

  endsubroutine ! tfqmrgpu_bsrsv_rectangular


  subroutine tfqmrgpu_bsrsv_complete(mb, ldA, &
    rowPtrA, colIndA, Amat, transA, &
    rowPtrX, colIndX, Xmat, transX, &
    rowPtrB, colIndB, Bmat, transB, iterations, residual, o, ierr)
  !! this routine is meant to test the usability of the tfqmrgpu bsrsv functionality
  !! without an often complicated integration of the library into the target application
  implicit none
    integer(kind=4), intent(in) :: mb ! the number of rows
    integer(kind=4), intent(in) :: ldA ! the square block size and leading dimension, ldA must be precompiled in tfqmrgpu.cu
    integer(kind=4), intent(in) :: rowPtrA(:), rowPtrX(:), rowPtrB(:) ! BSR rowPtr#(mb+1)  where #=A,X,B
    integer(kind=4), intent(in) :: colIndA(:), colIndX(:), colIndB(:) ! BSR colInd#(nnzb#) where #=A,X,B
    character,       intent(in) :: transA, transX, transB ! transposition of blocks, allowed states are {'n','t'}
    complex(kind=8), intent(in)  :: Amat(ldA,ldA,*) ! dim(nnzbA) ! BSR non-zero block values of A
    complex(kind=8), intent(out) :: Xmat(ldA,ldA,*) ! dim(nnzbX) ! BSR non-zero block values of X, the result
    complex(kind=8), intent(in)  :: Bmat(ldA,ldA,*) ! dim(nnzbB) ! BSR non-zero block values of B (the RHSs)
    integer(kind=4), intent(inout) :: iterations ! on entry: stop critertion against runtime explosion
                                                 ! on exit:  how many iterations did it run
    real   (kind=8), intent(inout) :: residual   ! on entry: stop criterion for convergence
                                              ! on exit:  how far down did it converge
    integer(kind=4), intent(in) :: o ! I/O unit to write to, 6=stdout, 0=no write (mute)
                             ! other values <n> may end in files fort.<n> or file connected to this unit
    integer(kind=4), intent(inout) :: ierr ! if ierr /= o on entry, debugging is activated

    ! delegate
    call tfqmrgpu_bsrsv_rectangular(mb, ldA, ldA, &
      rowPtrA, colIndA, Amat, transA, &
      rowPtrX, colIndX, Xmat, transX, &
      rowPtrB, colIndB, Bmat, transB, &
      iterations, residual, o, ierr)

  endsubroutine ! tfqmrgpu_bsrsv_complete

endmodule ! tfqmrgpu
