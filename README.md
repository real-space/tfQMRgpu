tfQMRgpu 
========

The transpose-free Quasi Minimal Resdiual library for GPUs
----------------------------------------------------------
    A CUDA implementation for graphical processors of the 
    transpose-free Quasi-Minimal Residual method (tfQMR) for
    the iterative solution of linear systems of equations

Purpose
-------
    The transpose-free Quasi-Minimal Residual method (tfQMR)
    solves a systems of simultaneous linear equations.
    This particular implementation benefits from solving for
    several right hand sides (RHSs) at a time using *vectorization* over CUDA threads.
    The underlying operators used here are block sparse matrices 
    in BSR format, and tfQMRgpu has been developed for GPUs,
    however, the main algorithm is defined in the headers
    and can be instanciated with user-defined linear operators
    and is independent of the platform.

### Details on the tfQMR method
    In this implementation of the QMR method proposed by
    Freund R W (1993) A transpose-free quasi-minimal residual algorithm 
        for non-Hermitian linear systems SIAM J. Sci. Comput. 14, 470--482
    Freund R W and Nachtigal N (1991) QMR: a Quasi-Minimal Residual
        Method for Non-Hermitian Linear Systems Numer. Math. 60, 315--339
    we focus on solving
        A * X == B
    for the operators A, X and B.
    Here, the operators X and B (and in the standard case also A)
    consist of inner complex-valued square block matrices 
    in single (32 bit float) or double precision (64 bit float)
    and an outer sparse structure where we use the BSR format.
    The tfQMR method is outlined for non-Hermitian operators A
    but should also converge for Hermitian operators.
    The number of iterations needed to converge might increase with the
    condition number of A. The logical shape of A is square.
    Operator B contains the right hand site (RHS) vectors as columns.
    After solving, operator X contains the solution vectors as columns.
    The BSR shape of operator B may be sparser than that of X,
    however, if a block is non-zero in B, it must be non-zero 
    in X but not vice versa.

### How to build, compile, install and test
    tfQMRgpu uses CMake, it is recommended to create a build directory
    cd ~/tfQMRgpu
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=~/tfQMRgpu 
             -DCMAKE_PREFIX_PATH=~/tfQMRgpu
    make -j
    make install
    cd ../test/multiplication
    ../../bin/bench_tfqmrgpu multi plan_unordered.14-287-16
    cd ../test/full_solver
    ../../bin/generate_FD_example
    ../../bin/bench_tfqmrgpu tfQMR FD_example.xml

### Details on the Block-compressed Spare Row (BSR) format
    The BSR format is a variant of the Compressed Sparse Row 
    format (CSR) with matrix blocks replacing the scalar values.
    The dimension of the square matrix blocks is a compile time 
    constant which facilitates tuning for performance.
    C++ non-type template arguments allow to provide
    objects with more than one matrix dimension precompiled.
    A possible extension to runtime compilation can be thought of.
    Most BSR-related integer lists are of type signed 32-bit integer
    `int32_t` as we expect the block indices not to exceed 2,147,483,647
    except for the column indices. The current configuration forsees 
    `uint16_t`, i.e. there can be at most 65,536 block columns.

### How to get started with C or C++?
    You need read access to the C-header file `tfQMRgpu/include/tfqmrgpu.h`.
    The function `GPUbench::bench_tfQMRgpu_library` in `tfQMRgpu/source/bench_tfqmrgpu.cu`
    gives an example of how to use the library correctly.

### How to get started with Fortran90?
    There is a quick-starter subroutine
        `use tfqmrgpu, only: tfqmrgpu_bsrsv_complete`
    as you can see in the program tfqmrgpu_run_example which is included
    in `tfqmrgpu_Fortran_example.F90` if the preprocessor flag `-D__MAIN__` is defined.
    The routine `tfqmrgpu_bsrsv_complete` allows first tests:
    The quick-starter routine helps new users to find out if tfQMR 
    is applicable to their problem without having to invest larger 
    coding efforts for the integration of the tfQMRgpu library into
    the application code.

    For production, the full integration of the tfQMRgpu library 
    into the user's application code should be done since calling 
    `tfqmrgpu_bsrsv_complete()` more than once might do the job
    but will not be efficient in terms of various resources:
        Allocations/deallocations of host(CPU) and device(GPU) memory.
        Runs on the default stream on the GPU, no task overlapping possible.
        The analysis step will be performed every time, even if the
        BSR patterns did not change (which might often be the case).

### What about Fortran77?
    F77 users can call the external subroutine names with the trailing 
    underscore defined in `tfQMRgpu/source/tfqmrgpu_Fortran_wrappers.c`

### How to provide a user-defined linear operator?
    The current version of tfQMRgpu forsees that X and B are always
    block-sparse complex matrices, however, C++ users
    can implement a custom operator A.
    Inspect the default block-sparse operator A defined in
    `tfQMRgpu/include/tfqmrgpu_blocksparse.hxx` to see the 
    interfaces of the member functions that a user defined 
    action of A needs to provide.
