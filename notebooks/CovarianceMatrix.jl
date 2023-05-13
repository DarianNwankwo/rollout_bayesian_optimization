using LinearAlgebra


struct PreallocatedSymmetricMatrix{T <: Number}
    K
    fK
    active_rows_cols::Vector{Int}
    m::Int # Number of function observations
    h::Int # Number of function and gradient observations
    d::Int # Dimensionality of feature vector
end

function PreallocatedSymmetricMatrix{T}(m::Int, h::Int, d::Int) where T <: Number
    n = m + (h + 1)*(d + 1)
    K = zeros(n, n)
    fK = LowerTriangular(zeros(n, n))
    active_rows_cols = Int[]
    
    return PreallocatedSymmetricMatrix{T}(Ref(K), Ref(fK), active_rows_cols, m, h, d)
end

function clear_fantasized!(PSM::PreallocatedSymmetricMatrix)
    m = PSM.m
    PSM.K[][m+1:end, :] .= 0
    PSM.K[][:, m+1:end] .= 0
    
    PSM.fK[][m+1:end, :] .= 0
    PSM.fK[][:, m+1:end] .= 0
    
    nothing
end

"""
We'll use the notation in function definitions of F to denote function observation covariances
and G to denote gradient observation covariances.
"""

# Subroutine 1: Covariance Matrix
function update_knowns!(PSM::PreallocatedSymmetricMatrix, Kupdate)
    @assert size(Kupdate, 1) == PSM.m "Covariance matrix dimension is incorrect" 
    PSM.K[][1:PSM.m, 1:PSM.m] = Kupdate
    
    [push!(PSM.active_rows_cols, rc) for rc in 1:PSM.m]
    
    nothing
end

# Subroutine 1: Cholesky Factorization
function cholesky_update_knowns!(PSM::PreallocatedSymmetricMatrix, fKUpdate)
    @assert size(Kupdate, 1) == PSM.m "Cholesky factorization dimension is incorrect"
    PSM.fK[][1:PSM.m, 1:PSM.m] = fKupdate
    
    nothing
end


function update_fantasized_vs_knowns!(PSM::PreallocatedSymmetricMatrix, Kvec_update, row_ndx)
    @assert length(Kvec_update) == PSM.m "Covariance vector length != PSM.m (m = $(PSM.m))"
    
    PSM.K[][row_ndx, 1:m] = Kvec_update
    PSM.K[][1:m, row_ndx] = Kvec_update
        
    nothing
end

function update_fantasized_vs_fantasized!(PSM::PreallocatedSymmetricMatrix, Kupdate, row_ndx)
    @assert (row_ndx - PSM.m) == size(Kupdate, 2)
    ustride = PSM.m+1:row_ndx
    
    PSM.K[][ustride, ustride] = Kupdate
    push!(PSM.active_rows_cols, row_ndx)
    
    nothing
end

function update_gradfantasized_vs_fantasized_and_known!(PSM::PreallocatedSymmetricMatrix, Kupdate, grad_ndx)
    m, h, d = PSM.m, PSM.h, PSM.d
    grad_start = m+h+2
    num_cols = PSM.active_rows_cols[end]
    row_ustride = grad_start+(grad_ndx - 1)*d:grad_start+grad_ndx*d-1
    col_ustride = 1:num_cols

    PSM.K[][row_ustride, col_ustride] = Kupdate
    PSM.K[][col_ustride, row_ustride] = Kupdate'
    
    nothing
end

function update_gradfantasized_vs_self!(PSM::PreallocatedSymmetricMatrix, Kupdate, grad_ndx)
    m, h, d = PSM.m, PSM.h, PSM.d
    grad_start = m+h+2
    ustride = grad_start+(grad_ndx - 1)*d:grad_start+grad_ndx*d-1
    
    PSM.K[][ustride, ustride] = Kupdate
    
    nothing
end

function update_gradfantasized_vs_gradfantasized!(PSM::PreallocatedSymmetricMatrix, Kupdate, grad_ndx)
    m, h, d = PSM.m, PSM.h, PSM.d
    if grad_ndx == 1 return nothing end
    
    grad_start = m+h+2
    row_ustride = grad_start+(grad_ndx - 1)*d:grad_start+grad_ndx*d-1
    col_ustride = grad_start:grad_start+(grad_ndx - 1)*d-1
    
    PSM.K[][row_ustride, col_ustride] = Kupdate
    PSM.K[][col_ustride, row_ustride] = Kupdate'
    
    nothing
end

function update_gradfantasized_vs_allprev_fantasized!(PSM::PreallocatedSymmetricMatrix, Kupdate, grad_ndx)
    m, h, d = PSM.m, PSM.h, PSM.d
    if grad_ndx == 1 return nothing end
    
    grad_start = m+h+2
    row_ustride = grad_start:grad_start+(grad_ndx - 1)*d-1
    col_num = PSM.active_rows_cols[end]
    col_ustride = col_num:col_num
    
    # println("Subroutine: update_gradfantasized_vs_allprev_fantasized")
    # println("Column Update Stride: $col_ustride\nRow Update Stride: $row_ustride")
    
    PSM.K[][row_ustride, col_ustride] = Kupdate
    PSM.K[][col_ustride, row_ustride] = Kupdate'
    
    nothing
end


function get_KXX(PSM::PreallocatedSymmetricMatrix, row_ndx)
    return @view PSM.K[][1:row_ndx, 1:row_ndx]
end