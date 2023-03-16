#!/usr/bin/env python3
##############################################
### tfqmrgpu_python_example.py
### Marc P Vandelle
##############################################

import ctypes
import numpy as np
import scipy.sparse as sp

def tfqmrgpu_python(A, B, R, C, iterations, residual, echo, path):
    """
    Solve a linear system using the tfQMRgpu library.

    Args:
        A (scipy.sparse.bsr_matrix): Left-hand side matrix of the linear system.
        B (scipy.sparse.bsr_matrix): Right-hand side vectors of the linear system.
        R (int): Row block size of the left-hand side matrix.
        C (int): Column block size of the right-hand side vectors.
        iterations (int): Maximum number of iterations to perform.
        residual (float): Target residual for the solver.
        echo (int): Verbosity level of the solver.
        path (str): Path to the tfQMR-GPU library file.

    Returns:
        numpy.ndarray: The solution vector of the linear system 
        (list of interleaved real and imaginary part of the element of the bsr matrix X).
    """

    max_iterations = iterations
    threshold = residual
    echo_in = echo
    
    # Load the tfQMRgpu library using ctypes
    tfQMRgpu = ctypes.CDLL(path)

    # Define the argument types of the bsrsv_z function
    tfQMRgpu.tfqmrgpu_bsrsv_z.argtypes = (
        ctypes.c_int,                     # Number of block rows
        ctypes.c_int,                     # Number of rows per block
        ctypes.c_int,                     # Number of cols per block
        
        ctypes.POINTER(ctypes.c_int32),   # Row offsets array (indptr) for the left-hand side matrix A
        ctypes.c_int,                     # Number of non-zero blocks in the left-hand side matrix A
        ctypes.POINTER(ctypes.c_int32),   # Column indices array (indices) for the left-hand side matrix A
        ctypes.POINTER(ctypes.c_double),  # Data array for the left-hand side matrix A
        ctypes.c_char,                    # Transpose flag for the left-hand side matrix A ('n' or 't')
        
        ctypes.POINTER(ctypes.c_int32),   # Row offsets array (indptr) for the solution vectors X
        ctypes.c_int,                     # Number of non-zero blocks in the solution vectors X
        ctypes.POINTER(ctypes.c_int32),   # Column indices array (indices) for the solution vectors X
        ctypes.POINTER(ctypes.c_double),  # Data array for the solution vectors X
        ctypes.c_char,                    # Transpose flag for the solution vectors X ('n' or 't')
        
        ctypes.POINTER(ctypes.c_int32),   # Row offsets array (indptr) for the right-hand side vectors B
        ctypes.c_int,                     # Number of non-zero blocks in the right-hand side vectors B
        ctypes.POINTER(ctypes.c_int32),   # Column indices array (indices) for the right-hand side vectors B
        ctypes.POINTER(ctypes.c_double),  # Data array for the the right-hand side vectors B
        ctypes.c_char,                    # Transpose flag for the right-hand side vectors B ('n' or 't')
        
        ctypes.POINTER(ctypes.c_int),     # Number of iterations performed by the solver (before:maximum, after:performed)
        ctypes.POINTER(ctypes.c_float),   # Final residual achieved by the solver      (before:threshold, after:achieved)
        ctypes.c_int,                     # Index offset for the column indices arrays (0 for C,C++,python, 1 for Julia,Fortran)
        ctypes.c_int                      # Verbosity level (0:none to 9:debug)
    )

    # Create a dense matrix with all elements set to 1 as the initial guess for the solution
    X = sp.bsr_matrix(np.ones((B.shape[0], B.shape[1])), blocksize=(R, C))

    # Interleave the real and imaginary components of the data arrays
    Adata = interleave_complex_number(A)
    Xdata = interleave_complex_number(X)
    Bdata = interleave_complex_number(B)

    # Initialization of all the variables for the call of tfQMRgpu's C-interface
    mb =  ctypes.c_int(int(A.shape[0]//A.blocksize[0]))
    lda = ctypes.c_int(R)
    ldb = ctypes.c_int(C)
    Ar = (ctypes.c_int32*A.indptr.size).from_buffer(A.indptr)
    Xr = (ctypes.c_int32*X.indptr.size).from_buffer(X.indptr)
    Br = (ctypes.c_int32*B.indptr.size).from_buffer(B.indptr)
    nnzbA = ctypes.c_int(int(A.nnz//(A.blocksize[0]*A.blocksize[1])))
    nnzbX = ctypes.c_int(int(X.nnz//(X.blocksize[0]*X.blocksize[1])))
    nnzbB = ctypes.c_int(int(B.nnz//(B.blocksize[0]*B.blocksize[1])))
    Ai = (ctypes.c_int32*A.indices.size).from_buffer(A.indices)
    Xi = (ctypes.c_int32*X.indices.size).from_buffer(X.indices)
    Bi = (ctypes.c_int32*B.indices.size).from_buffer(B.indices)
    Ad = (ctypes.c_double*Adata.size).from_buffer(Adata)
    Xd = (ctypes.c_double*Xdata.size).from_buffer(Xdata)
    Bd = (ctypes.c_double*Bdata.size).from_buffer(Bdata)
    trans = ctypes.c_char(b'n')
    iterations = ctypes.c_int32(iterations)
    iterations_ptr = ctypes.byref(iterations)
    residual = ctypes.c_float(residual)
    residual_ptr = ctypes.byref(residual)
    indexoffset = ctypes.c_int(0)
    echo = ctypes.c_int(echo)

    # Call to the tfQMRgpu library
    error = tfQMRgpu.tfqmrgpu_bsrsv_z(mb, lda, ldb, Ar, nnzbA, Ai, Ad, trans, Xr, nnzbX, Xi, Xd, trans, Br, nnzbB, Bi, Bd, trans, iterations_ptr, residual_ptr, indexoffset, echo)
    if 0 == error:
        if echo_in > 0:
            print("# tfQMRgpu needed ",iterations.value," to converge to ",residual.value)
        return Xdata
    else:
        if echo_in > 0:
            tfQMRgpu.tfqmrgpuPrintError(error)
        return None # failed


def interleave_complex_number(matrix) :
    """
    Interleave the real and imaginary part of a complex bsr matrix elements.

    Args:
        matrix(scipy.sparse.bsr_matrix): Left-hand side matrix of the linear system.

    Returns:
        numpy.ndarray: The list of real and imaginary part of the elements of the matrix. 
    """
    matrix_real = matrix.data.real
    matrix_imag = matrix.data.imag
    return np.ascontiguousarray(np.column_stack((matrix_real, matrix_imag)).reshape(-1))


if __name__ == "__main__":
    # Run example
    print("# tfQMRgpu python example")
    
    mb = 7 # number of block rows
    ldA = 4 # square block dimension of A
    ldB = 5 # number of columns per block of B and X

    # A_dense = np.zeros((mb*ldA, mb*ldA), dtype=complex)
    A_dense = np.random.random_sample((mb*ldA, mb*ldA))
    print("# A_dense="); print(A_dense)
    A = sp.bsr_matrix(A_dense, blocksize=(ldA, ldA), dtype=complex)

    B_dense = np.zeros((mb*ldA, 1*ldB), dtype=complex)
    for i in range(ldB):
        B_dense[i%ldA,i] = 1
    print("# B_dense="); print(B_dense)
    B = sp.bsr_matrix(B_dense, blocksize=(ldA, ldB))
    
    iterations=33
    residual=1e-7
    Xdata = tfqmrgpu_python(A, B, ldA, ldB, iterations, residual, echo=9, path='../lib64/libtfQMRgpu.so')

    print("# Xdata="); print(Xdata)
