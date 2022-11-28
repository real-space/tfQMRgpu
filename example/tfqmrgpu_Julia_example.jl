#!/usr/bin/env julia
##############################################
### tfqmrgpu_Julia_example.jl
##############################################

plot = true

mb = 7 # number of block rows
ldA = 4 # square block dimension of A
ldB = 5 # number of columns per block of B and X

nnzbA = 19 # number of nonzero blocks in A
nnzbX = mb # number of nonzero blocks in X
nnzbB = 1

const ComplexF00 = ComplexF32 # F32 or F64
# ToDo: seems like F32 does not work properly

# get memory for nonzero blocks in Complex{Float64}
Amat = zeros(ComplexF00, ldA, ldA, nnzbA) # nonzero blocks of the problem
Xmat = zeros(ComplexF00, ldB, ldA, nnzbX) # nonzero blocks of the solution
Bmat = zeros(ComplexF00, ldB, ldA, nnzbB) # nonzero blocks of the right-hand-sides

# block transpositions
transA = 'n'
transX = 'n'
transB = 'n'

# starts of the sparse row
rowPtrA = zeros(Int32, mb + 1)
rowPtrX = zeros(Int32, mb + 1)
rowPtrB = zeros(Int32, mb + 1)

# column indices in CompressSparseRow format
colIndA = zeros(Int32, nnzbA)
colIndX = zeros(Int32, nnzbX)
colIndB = zeros(Int32, nnzbB)

for i = 1:ldB
    j = mod(i - 1, ldA) + 1
    p = div(i - 1, ldA)
    # use the 1st column of unit matrix as B
    Bmat[i,j,1] = (0 + 1im)^p
    # if ldB > ldA the diagonal of Bmat will be [1,...,1,i,...,i,-1,...,-1,-i,...,-i] testing the phase invariance
    if plot && (1 == j) println("### Bmat[",i,",",j,"] = ",Bmat[i,j,1]) end
end # i

counts = [0, 0]
inzb = 0
for ib = 1:mb # block rows
    for jb = max(1, ib - 1):min(mb, ib + 1) # block columns
        global inzb += 1 # count up
        colIndA[inzb] = jb - 1
        diagonal = (ib == jb) ? 2 : -1 # create a 1D finite-difference operator
        for i = 1:ldA
            Amat[i,i,inzb] = diagonal
        end # i
        counts[(ib == jb) ? 2 : 1] += 1
    end # jb
    rowPtrA[ib + 1] = inzb
    rowPtrX[ib + 1] = ib
#   rowPtrB[ib + 1] = 0
end # ib
rowPtrB[mb + 1] = nnzbB
println("### rowPtrA = ", rowPtrA)
println("### rowPtrX = ", rowPtrX)
println("### colIndA = ", colIndA)
@assert (nnzbA == inzb) "Counting failed"

println("### operator A: ",counts[1]," offdiagonal and ",counts[2]," diagonal blocks")

if true
    println("### tfQMRgpu solves A * X == B")

    const tf = "../lib64/libtfQMRgpu.so"
    nAllowedBlockSizes = zeros(Int32, 1)
    AllowedBlockSizes = zeros(Int32, 2, 100)
    status = @ccall tf.tfqmrgpu_bsrsv_allowedBlockSizes(nAllowedBlockSizes::Ref{Int32}, AllowedBlockSizes::Ref{Int32}, 200::Cint)::Int32
    if (0 != status)
        @ccall tf.tfqmrgpuPrintError(status::Cint)::Cint
    else
        @show nAllowedBlockSizes[1]
        @show AllowedBlockSizes[1:2,1:nAllowedBlockSizes[1]]'
    end # status

    iterations = zeros(Int32, 1); iterations[1] = 210 # max number of iterations
    residual = zeros(Float32, 1)
    echo = 9 # 9:debug output
    if ComplexF64 == ComplexF00
        residual[1] = 1.2e-8 # threshold to converge
        status = @ccall tf.tfqmrgpu_bsrsv_z(
            mb::Cint # number of block rows and number of block columns in A, number of block rows in X and B
          , ldA::Cint # block dimension of blocks of A
          , ldB::Cint # leading dimension of blocks in X and B
          , rowPtrA::Ref{Int32}, nnzbA::Cint, colIndA::Ref{Int32}, Amat::Ref{ComplexF64}, transA::Cchar # assumed data layout Complex Amat[ldA,ldA,nnzbA]
          , rowPtrX::Ref{Int32}, nnzbX::Cint, colIndX::Ref{Int32}, Xmat::Ref{ComplexF64}, transX::Cchar # assumed data layout Complex Xmat[ldB,ldA,nnzbX]
          , rowPtrB::Ref{Int32}, nnzbB::Cint, colIndB::Ref{Int32}, Bmat::Ref{ComplexF64}, transB::Cchar # assumed data layout Complex Bmat[ldB,ldA,nnzbB]
          , iterations::Ref{Int32} # on entry *iterations holds the max number of iterations, on exit *iteration is the number of iterations needed to converge
          , residual::Ref{Float32} # on entry *residual holds the threshold, on exit *residual hold the residual that has been reached after the last iteration
          , 0::Cint # C,C++ indices start at 0, Fortran and Julia native indices start at 1
          , echo::Cint # verbosity level, 0:no output, .... , 9: debug output
          )::Int32
    else
        residual[1] = 1.2e-5 # threshold to converge
        status = @ccall tf.tfqmrgpu_bsrsv_c(mb::Cint, ldA::Cint, ldB::Cint
          , rowPtrA::Ref{Int32}, nnzbA::Cint, colIndA::Ref{Int32}, Amat::Ref{ComplexF32}, transA::Cchar
          , rowPtrX::Ref{Int32}, nnzbX::Cint, colIndX::Ref{Int32}, Xmat::Ref{ComplexF32}, transX::Cchar
          , rowPtrB::Ref{Int32}, nnzbB::Cint, colIndB::Ref{Int32}, Bmat::Ref{ComplexF32}, transB::Cchar
          , iterations::Ref{Int32}, residual::Ref{Float32}, 0::Cint, echo::Cint)::Int32
    end # complex
    if (0 != status)
        @ccall tf.tfqmrgpuPrintError(status::Cint)::Cint
    else
        println("### tfQMRgpu converged to ",residual[1]," in ",iterations[1]," iterations")
        # the solution of the 1D finite-difference operator should be a straight line [.125, .25, .375, .5, .625, .75, .875]
        Xint8 = Complex{Int16}.(round.(Xmat*8))
        maximum_deviation = maximum(abs.(Xint8 - Xmat*8))/8.
        @show maximum_deviation
    end # if status
end # if true
