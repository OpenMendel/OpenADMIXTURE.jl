function project_q!(b::AbstractVector{T}, idx::AbstractVector{Int}; pseudocount=T(1e-5)) where T
    n = length(b)
    τ = one(T) - n * pseudocount
    b .-= pseudocount
    bget = false

    sortperm!(idx, b, rev=true)
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

function project_f!(b::AbstractArray{T}; pseudocount=T(1e-5)) where T
    b .= min.(max.(b, pseudocount), one(T) - pseudocount)
end
