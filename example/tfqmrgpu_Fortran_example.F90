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
          tfqmrgpu_bsrsv_complete
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
                      bsrRowPtrB, nnzbB, bsrColIndB, ierr)
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
    external :: tfqmrgpu_bsrsv_createplan
    call tfqmrgpu_bsrsv_createplan(handle, plan, &
                  mb, bsrRowPtrA, nnzbA, bsrColIndA, &
                      bsrRowPtrX, nnzbX, bsrColIndX, &
                      bsrRowPtrB, nnzbB, bsrColIndB, ierr)
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
                    doublePrecision, pBufferSizeInBytes, ierr)
    !! returns the computed size to be allocated by cudaMalloc
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(inout) :: plan
    integer(kind=4), intent(in) :: ldA
    integer(kind=4), intent(in) :: blockDim
    integer(kind=4), intent(in) :: ldB
    integer(kind=4), intent(in) :: RhsBlockDim
    character      , intent(in) :: doublePrecision
    integer(kind=8), intent(out) :: pBufferSizeInBytes
    external :: tfqmrgpu_bsrsv_buffersize
    call tfqmrgpu_bsrsv_buffersize(handle, plan, ldA, blockDim, ldB, &
            RhsBlockDim, doublePrecision, pBufferSizeInBytes, ierr)
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

  subroutine bsrsv_setMatrix_c(handle, plan, var, val, ld, trans, layout, ierr)
    !! retrieves the GPU memory buffer registered in plan.
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(inout) :: plan
    character, intent(in) :: var
    complex(kind=4), intent(in) :: val(*)
    integer(kind=4), intent(in) :: ld
    character, intent(in) :: trans
    integer(kind=4), intent(in) :: layout
    external :: tfqmrgpu_bsrsv_setmatrix_c
    call tfqmrgpu_bsrsv_setmatrix_c(handle, plan, var, val, ld, trans, layout, ierr)
  endsubroutine ! set

  subroutine bsrsv_setMatrix_z(handle, plan, var, val, ld, trans, layout, ierr)
    !! retrieves the GPU memory buffer registered in plan.
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(inout) :: plan
    character, intent(in) :: var
    complex(kind=8), intent(in) :: val(*)
    integer(kind=4), intent(in) :: ld
    character, intent(in) :: trans
    integer(kind=4), intent(in) :: layout
    external :: tfqmrgpu_bsrsv_setmatrix_z
    call tfqmrgpu_bsrsv_setmatrix_z(handle, plan, var, val, ld, trans, layout, ierr)
  endsubroutine ! set
    
  subroutine bsrsv_getMatrix_c(handle, plan, var, val, ld, trans, layout, ierr)
    !! retrieves the GPU memory buffer registered in plan.
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(in) :: plan !! or is the plan modified?
    character, intent(in) :: var
    complex(kind=4), intent(out) :: val(*)
    integer(kind=4), intent(in) :: ld
    character, intent(in) :: trans
    integer(kind=4), intent(in) :: layout
    external :: tfqmrgpu_bsrsv_getmatrix_c
    call tfqmrgpu_bsrsv_getmatrix_c(handle, plan, var, val, ld, trans, layout, ierr)
  endsubroutine ! get

  subroutine bsrsv_getMatrix_z(handle, plan, var, val, ld, trans, layout, ierr)
    !! retrieves the GPU memory buffer registered in plan.
    integer(kind=4), intent(out) :: ierr ! this is the return value in the C-API
    integer(kind=TFQMRGPU_HANDLE_KIND), intent(in) :: handle
    integer(kind=TFQMRGPU_PLAN_KIND), intent(in) :: plan !! or is the plan modified?
    character, intent(in) :: var
    complex(kind=8), intent(out) :: val(*)
    integer(kind=4), intent(in) :: ld
    character, intent(in) :: trans
    integer(kind=4), intent(in) :: layout
    external :: tfqmrgpu_bsrsv_getmatrix_z
    call tfqmrgpu_bsrsv_getmatrix_z(handle, plan, var, val, ld, trans, layout, ierr)
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

  subroutine tfqmrgpu_bsrsv_complete(mb, ldA, &
    rowPtrA, colIndA, Amat, transA, &
    rowPtrX, colIndX, Xmat, transX, &
    rowPtrB, colIndB, Bmat, transB, iterations, residual, o, ierr)
  !! this routine is meant to test the usability of the tfqmrgpu bsrsv functionality
  !! without an often complicated integration of the library into the target application
  implicit none
    integer(kind=4), intent(in) :: mb ! the number of rows
    integer(kind=4), intent(in) :: ldA ! the block size and leading dimension, ldA must be precompiled
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
           rowPtrB, size(colIndB), colIndB, ierr) 
    CheckError(ierr, "Failed to create the bsrsv plan")

    !! compute the size of the required GPU memory buffer
    !! 'z' means solve in double precision.
    !! The library will fail here, if the required block size ldA has not been compiled
    call get(handle, plan, ldA, ldA, ldA, ldA, 'z', memSize, ierr) 
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
    call set(handle, plan, 'A', Amat(:,1,1), ldA, transA, layout, ierr) 
    CheckError(ierr, "Failed to upload matrix A")

    !! upload the nonzero blocks of the input array B
    call set(handle, plan, 'B', Bmat(:,1,1), ldA, transB, layout, ierr) 
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
    call get(handle, plan, 'X', Xmat(:,1,1), ldA, transX, layout, ierr) 
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
  endsubroutine ! tfqmrgpu_bsrsv_complete

endmodule ! tfqmrgpu


#ifdef __MAIN__
!+__main__

  program tfqmrgpu_run_example
  !! test driver to run the Fortran example delivered with the tfqmrgpu library
  use tfqmrgpu, only: tfqmrgpu_bsrsv_complete
  implicit none

    integer :: iarg, length, status, it
    character(len=96) :: arg
    real(kind=8) :: maxdev(3)

    do iarg = 0, command_argument_count()
        call get_command_argument(iarg, arg, length, status)
        write(*,*) iarg, length, status, trim(arg)
    enddo ! iarg

    !! test 1) in this example we have only one block
    maxdev(1) = test(32, rowPtr=[1, 2], colInd=[1])

    !! test 2) in this example use a full matrix as the reference which is generated by matmul 
    !! and which does not treat the sparsity constraint onto Y exactly like the solver
    !!   colInd =
    !!   1 2 3 4
    !!   1 2 3 4
    !!   1 2 3 4
    !!   1 2 3 4
    !!   rowPtr = 1 +4 +4 +4 +4
    maxdev(2) = test(16, rowPtr=[1, 5, 9, 13, 17], &
        colInd=[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4])

    !! test 3) in this example we only check if it runs through for a
    !! not fully occupied sparsity pattern: 4x4 blocks
    !!   colInd =
    !!   1 2
    !!   1 2 3
    !!     2 3 4
    !!       3 4
    !!   rowPtr = 1 +2 +3 +3 +2
    maxdev(3) = test(4, rowPtr=[1, 3, 6, 9, 11], &
           colInd=[1, 2, 1, 2, 3, 2, 3, 4, 3, 4])

    write(*,"(a,9es9.1)") " maxdev(:)=",maxdev(:)

  contains

#define DEBUG

  real(kind=8) function test(ldA, rowPtr, colInd) result(maxdev)
    integer(kind=4), intent(in) :: ldA ! block dimension
    integer(kind=4), intent(in) :: rowPtr(:)
    integer(kind=4), intent(in) :: colInd(:)

    ! local variables
    integer(kind=4) :: mb, nnzb, N
    complex(kind=8), allocatable :: Amat(:,:,:)   ! Amat(ldA,ldA,nnzb) ! list of blocks of A, later used for B^T and X
    real(kind=8),    allocatable :: Amar(:,:,:,:) ! Amar(ldA,ldA,nnzb,2) ! temp as random_number() is only defined for reals

#ifdef  DEBUG
    complex(kind=8), allocatable, dimension(:,:) :: Afull, Xfull, Bfull, Yfull
#endif

    integer(kind=4) :: iterations ! max number of iterations allowed
    real(kind=8) :: residual ! threshold for the requested residual
    integer, parameter :: o = 6 ! unit to log to
    integer(kind=4) :: ierr

    iterations = 999
    residual = 1.d-9
    ierr = 1 ! 1: do also sanity checks

    mb = size(rowPtr) - 1
    nnzb = size(colInd)
    write(*,*) ! empty line
    write(*,"(9(a,i0))") "tfqmrgpu_Fortran_example BS=",ldA," m=",mb," nnzb=",nnzb
    if (nnzb > mb*mb) stop "tfqmrgpu_Fortran_example too many blocks!"

    !!
    !! Test symbols:
    !!      mb      number of block rows in the problem
    !!      nnzb    number of non-zero blocks
    !!      rowPtr  index starts of a row in the BSR formar
    !!      CSR     compressed sparse row format
    !!      BSR     block-compressed sparse row format
    !!              similar to CSR but with block matrices instead of scalars
    !!      indCol  column indices in the BSR format
    !!      ldA     leading dimension == blocksize
    !!      #full   dense matrix representation, # in {A,B,X,Y=A*X}


    allocate( Amat(ldA,ldA,nnzb), Amar(ldA,ldA,nnzb,2) )
    call random_number(Amar) ! generate ldA*ldA*nnzb*2 pseudo-random doubles
    Amat = dcmplx(Amar(:,:,:,1), Amar(:,:,:,2)) ! interleave real and imaginary parts
    deallocate( Amar )

    N = mb*ldA
#ifdef  DEBUG
    allocate( Afull(N,N), Bfull(N,N), Xfull(N,N), Yfull(N,N) )
    call convert_bsr_to_full(Afull, rowPtr, colInd, Amat, 'n')
    call convert_bsr_to_full(Bfull, rowPtr, colInd, Amat, 't')
#endif

    !! solve A * X == B for X
    call tfqmrgpu_bsrsv_complete(mb, ldA, &
        rowPtr, colInd, Amat, 'n', & !! A (in)
        rowPtr, colInd, Amat, 'n', & !! X (out)
        rowPtr, colInd, Amat, 't', & !! B (in)
        iterations, residual, o, ierr)

   maxdev = -1

#ifdef  DEBUG
    call write_bsr_to_file("Xmat", "X", rowPtr, colInd, Amat, 'n')
    call convert_bsr_to_full(Xfull, rowPtr, colInd, Amat, 'n')
    Yfull = matmul(Xfull, Afull) ! works if block transpositions are A:'n', B:'t', X:'n'
    maxdev = maxval(abs(Yfull - Bfull))
    write(*, '(a,es9.1)') " Max. deviation |A*X - B| =", maxdev
    if (maxdev > 1e-8) then
      write(23,*) "# Yfull" ; write(23,'(2f10.6)') Yfull
      write(13,*) "# Bfull" ; write(13,'(2f10.6)') Bfull
      write(*,*) "Warning! Deviations are larger the 10^-8, B and A*X are written to fort.13 and fort.23, respectively."
    endif
    deallocate( Afull, Bfull, Xfull, Yfull )
#endif
    deallocate( Amat )

  endfunction test

#ifdef  DEBUG
!+debug

    subroutine convert_bsr_to_full(full, rowPtr, colInd, mat, trans)
      complex(kind=8), intent(out) :: full(:,:) ! mat(bd*nrow+,bd*nrow)
      integer(kind=4), intent(in) :: rowPtr(:) ! assume rowPtr(nrow+1)
      integer(kind=4), intent(in) :: colInd(:) ! assume colInd(nnzb)
      complex(kind=8), intent(in) :: mat(:,:,:) ! mat(bd+,bd,nnzb)
      character, intent(in) :: trans ! block transposition

      integer :: irow, jcol, ib, jb, inzb, bd, nrow, nnzb ! sparse matrix variables
      integer :: ir, jc ! global row and column indices into matrix full(jc,ir)
      
      bd = size(mat, 2) ! block dimension
      if (bd > size(mat, 1)) then
          write(*,*) "convert_bsr_to_full: ERROR: ld=",size(mat, 1)," < bd=",bd
          return ! leading dimension must not be smaller than the block dimension
      endif

      nrow = size(rowPtr) - 1 ! number of rows
      nnzb = size(colInd) ! number of non-zero blocks

      full(:,:) = 0 ! clear
      do irow = 1, nrow ! block row index
        do inzb = rowPtr(irow), rowPtr(irow+1)-1
          jcol = colInd(inzb) ! block column index
          !======================================================
          do ib = 1, bd      ! row index of the block (irow,jcol)
            ir = (irow - 1)*bd + ib ! global row index
            do jb = 1, bd ! column index of the block (irow,jcol)
              jc = (jcol - 1)*bd + jb ! global column index
              if ('n' == trans) then
                full(jc,ir) = mat(jb,ib,inzb)
              else  ! transpose
                full(jc,ir) = mat(ib,jb,inzb) ! transposed
              endif ! transpose
            enddo ! jb
          enddo ! ib
          !======================================================
        enddo ! inzb
      enddo ! irow

    endsubroutine ! convert_bsr_to_full

    subroutine write_bsr_to_file(filename, comment, rowPtr, colInd, mat, trans)
      character(len=*), intent(in) :: filename, comment
      integer(kind=4), intent(in) :: rowPtr(:) ! assume rowPtr(nrow+1)
      integer(kind=4), intent(in) :: colInd(:) ! assume colInd(nnzb)
      complex(kind=8), intent(in) :: mat(:,:,:) ! mat(bd+,bd,nnzb)
      character, intent(in) :: trans ! block transposition
      
      integer :: irow, jcol, ib, jb, inzb, bd, nrow, nnzb ! sparse matrix variables
      integer :: u, ios
      character(len=*), parameter :: frmt = "(2f12.6)" ! "(2es24.16)"

      bd = size(mat, 2) ! block dimension
      if (bd > size(mat, 1)) stop "Error: leading dimension must not be smaller than the block dimension!"

      nrow = size(rowPtr) - 1 ! number of rows
      nnzb = size(colInd) ! number of non-zero blocks

      u = 642
      open(unit=u, file=filename, action="write", status="unknown", iostat=ios)
      if (0 /= ios) return ! cannot write the file

      write(u, fmt="(9a)", iostat=ios) "# block-sparse row format: ",trim(comment)
      do irow = 1, nrow ! block row index
        write(u, fmt="(/,9(a,i0))", iostat=ios) "# row ",irow," has ",rowPtr(irow+1) - 1 - rowPtr(irow)," elements"
        do inzb = rowPtr(irow), rowPtr(irow+1)-1
          jcol = colInd(inzb) ! block column index
          write(u, fmt="(/,a,9(' ',i0))", iostat=ios) "# block i j  ",irow,jcol
          do ib = 1, bd      ! row index of the block (irow,jcol)
            do jb = 1, bd ! column index of the block (irow,jcol)
              if ('n' == trans) then
                write(u, frmt, iostat=ios) mat(jb,ib,inzb)
              else  ! transpose
                write(u, frmt, iostat=ios) mat(ib,jb,inzb) ! transposed
              endif ! transpose
            enddo ! jb
          enddo ! ib
        enddo ! inzb
      enddo ! irow
      close(unit=u, iostat=ios)
      
    endsubroutine ! write_bsr_to_file

!-debug   
#endif

  endprogram
!-__main__
#endif
