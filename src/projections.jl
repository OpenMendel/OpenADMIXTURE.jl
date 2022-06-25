function project_q!(b::AbstractMatrix{T}, idx::AbstractVector{Int}; pseudocount=T(1e-5)) where T
    I = size(b, 2)
    @inbounds for i in 1:I
        project_q!(@view(b[:, i]), idx)
    end
    b
end

"""
    project_q!(b, idx; pseudocount=1e-5)
Project the Q matrix onto the probability simplex.
"""
function project_q!(b::AbstractVector{T}, idx::AbstractVector{Int}; pseudocount=T(1e-5)) where T
    n = length(b)
    τ = one(T) - n * pseudocount
    b .-= pseudocount
    bget = false

    sortperm!(idx, b, rev=true) # this allocates something
    tsum = zero(T)

    @inbounds for i = 1:n-1
        tsum += b[idx[i]]
        tmax = (tsum - τ)/i
        if tmax ≥ b[idx[i+1]]
            bget = true
            break
        end
    end

    if !bget
        tmax = (tsum + b[idx[n]] - τ) / n
    end

    @inbounds for i = 1:n
        b[i] = max(b[i] - tmax, 0)
    end
    b.+= pseudocount
end

"""
    project_f!(b; pseudocount=1e-5)
Project the P matrix onto [0, 1].
"""
function project_f!(b::AbstractArray{T}; pseudocount=T(1e-5)) where T
    @turbo for i in 1:length(b)
        b[i] = min(max(b[i], pseudocount), one(T) - pseudocount)
    end
    # b .= min.(max.(b, pseudocount), one(T) - pseudocount)
end
