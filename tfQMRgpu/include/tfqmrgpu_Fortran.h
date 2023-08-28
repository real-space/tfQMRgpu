!! This file is part of tfQMRgpu under MIT-License

!     !! error codes are default kind integers
      integer(kind=4), parameter :: TFQMRGPU_STATUS_SUCCESS = 0
!     integer(kind=4), parameter :: TFQMRGPU_STATUS_MAX_ITERATIONS = 9
!     integer(kind=4), parameter :: TFQMRGPU_STATUS_ALLOCATION_FAILED = 4
!     integer(kind=4), parameter :: TFQMRGPU_POINTER_INVALID = 7
!     integer(kind=4), parameter :: TFQMRGPU_CODE_LINE             = 1000
!     integer(kind=4), parameter :: TFQMRGPU_NO_IMPLEMENTATION     = 19
!     integer(kind=4), parameter :: TFQMRGPU_BLOCKSIZE_MISSING     = 12
!     integer(kind=4), parameter :: TFQMRGPU_UNDOCUMENTED_ERROR    = 14
!     integer(kind=4), parameter :: TFQMRGPU_CODE_CHAR             = 1000000
!     integer(kind=4), parameter :: TFQMRGPU_TANSPOSITION_UNKNOWN  = 17
!     integer(kind=4), parameter :: TFQMRGPU_VARIABLENAME_UNKNOWN  = 18
!     integer(kind=4), parameter :: TFQMRGPU_DATALAYOUT_UNKNOWN    = 15
!     integer(kind=4), parameter :: TFQMRGPU_PRECISION_MISSMATCH   = 16

!     !! layout keys are default kind integers
      integer(kind=4), parameter :: TFQMRGPU_LAYOUT_RRRRIIII = 15 !! native layout for the GPU version, real and imag part of each block are separated.
      integer(kind=4), parameter :: TFQMRGPU_LAYOUT_RRIIRRII = 51 !! intermediate layout, not implemented
      integer(kind=4), parameter :: TFQMRGPU_LAYOUT_RIRIRIRI = 85 !! default host layout, real and imag parts are interleaved.
      integer(kind=4), parameter :: TFQMRGPU_LAYOUT_DEFAULT  = 85 !! default Fortran data layout for complex and double complex

!     !! pointer types require 64bit
      integer, parameter :: TFQMRGPU_HANDLE_KIND = 8 !! a pointer to an opaque handle
      integer, parameter :: TFQMRGPU_PLAN_KIND = 8 !! a pointer to an opaque plan object
      integer, parameter :: TFQMRGPU_PTR_KIND = 8 !! a pointer to data

!     !! dummy that makes us independent of CUDA headers:
      integer, parameter :: cuda_stream_kind = 8 ! = INT_PTR_KIND()
