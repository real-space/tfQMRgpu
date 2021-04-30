tfQMRgpu 
========

transpose-free Quasi Minimal Resdiual method on GPUs
----------------------------------------------------
    A CUDA implementation for graphical processors of the 
    transpose-free Quasi-Minimal Residual method (tfQMR) for
    the iterative solution of linear systems of equations

Purpose
-------
    The transpose-free quasi-minimal residual method (tfQMR)
    solves a linear systems of simultaneous equations.
    This particular implementation benefits from solving
    for several right hand sides at a time.
    The underlying operators used here are block sparse matrices 
    in BSR format, and tfqmrgpu has been developed for GPUs,
    however, the main algorithm is defined in the headers
    and can be instanciated with user-defined linear operators
    and is independent of the platform.

### Details on the tfQMR method
    In this implementation of the method proposed by
    Freund R W (1993) A transpose-free quasi-minimal residual algorithm 
        for non-Hermitian linear systems SIAM J. Sci. Comput. 14 470?482
    Freund R W and Nachtigal N (1991) QMR: a Quasi-Minimal Residual 
        Method for Non-Hermitian Linear Systems Numer. Math. 60 315?339
    we focus on solving
        A * X == B
    for the operators A, X and B.
    Here, the operators X and B (and in the standard case also A) consist of 
    inner complex-valued square block matrices 
    in single (32 bit float) or double precision (64 bit float)
    and an outer sparse structure where we use the BSR format.
    The tfQMR method is outlined for non-Hermitian operators A
    but should also converge for Hermitian operators.
    The number of iterations needed might increase with the
    condition number of A. The logical shape of A is square.
    Operator B contains the right-hand site vectors as columns.
    After solving, operator X contains the solution vectors as columns.
    The BSR shape of operator B may be sparser than that of X,
    in particular, if a block is non-zero in B, it must be non-zero 
    in X but not vice versa.

### Details on the Block-compressed Spare Row (BSR) format
    The BSR format is a varaiant of the Compressed Sparse Row 
    format (CSR) with matrix blocks replacing the scalar values.
    The dimension of the square matrix blocks is a compile time 
    constant which makes tuning simpler.
    C++ non-type template arguments allow to provide
    objects with more than one matrix dimension precompiled.
    A possible extension to runtime compilation can be thought of.
    Most BSR-related integer lists are of type signed 32-bit integer
    as we expect the block indices not to exceed 2,147,483,647
    except for the column indices. Current configuation forsees uint16_t,
    i.e. there can be at most 65,536 block columns.

Which files belong to the test environment?
    test_tfqmrgpu.cpp (driver code for miniKKR using tfqmrgpu)
    bench_tfqmrgpu.cu (miniKKR benchmark using the tfqmrgpu library)
    tfqmrgpu_util.hxx (utilities also used in the library)
    tfqmrgpu_example_reader.hxx (reads miniKKR input files formatted for C)
    bsr.hxx (defines the struct of a block compressed sparse row matrix)
    Makefile
    
### Which files belong to the library?
    tfqmrgpu.h (C header)
    tfqmrgpu.hxx (extern C)
    tfqmrgpu_plan.hxx (struct def for an opaque handle)
    tfqmrgpu_handle.hxx (struct def for an opaque handle)
    tfqmrgpu_memWindow.h (internal struct def)
    tfqmrgpu_util.hxx (utilities also used in the benchmarks)
    tfqmrgpu.cu (CUDA implementation)
    Makefile

### Which files belong to the Fortran interface?
    tfqmrgpu_Fortran_wrappers.c (C interface with Fortran-compilant call by reference)
    tfqmrgpu_Fortran.F90 (Fortran90 module, executable program if -D__MAIN__)
    tfqmrgpu_Fortran.h (Fortran constants)
    tfqmrgpu_Fortran.F (not used, maybe not working)
    linkit.sh (simple script to compile Fortran glue code and link it on JURON)

### How to get started with C (or C++)?
    For simplicity, C++ users must refer to the C-interface 
    defined in the header file tfqmrgpu.h.
    The function bench_tfQMRgpu_library in bench_tfqmrgpu.cu
    gives an example of how to use the library correctly.
    However, it only compiles together with the test environment.

### How to get started with Fortran90?
    There is a quick-starter subroutine
        use tfqmrgpu, only: tfqmrgpu_bsrsv_complete
    as you can see in the program tfqmrgpu_run_example which is included
    in tfqmrgpu_Fortran.F90 if the preprocessor flag -D__MAIN__ is defined.
    The routine tfqmrgpu_bsrsv_complete allows first tests:
    It helps the new user to find out if tfQMR is applicable to the problem
    at hand without having to invest larger coding efforts for the 
    integration of the tfqmrgpu library into the user's application code.

    For production, the full integration of the tfqmrgpu library 
    into the user's application code should be done since calling 
    tfqmrgpu_bsrsv_complete() more than once might do the job
    but will not be efficient in terms of various resources:
        Allocations/deallocations of host(CPU) and device(GPU) memory.
        Runs on the default stream on the GPU, no task overlapping possible.
        The analysis step will be performed every time, even if the
        BSR patterns did not change (which often is the case).

### What about Fortran77?
    Currently, the dummy subroutines in tfqmrgpu_Fortran.F are out of date!
    F77 users, however, can in principle simply call the external subroutine 
    names with the tailing underscore defined in tfqmrgpu_Fortran_wrappers.c

### Is there a full API documentation?
    Sorry, but no. Please refer to the comments in the code.
