using .CUDA
@inline function loglikelihood_kernel(out, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, p::AbstractArray{T, 2}, irange, jrange, joffset=0) where T
    K = size(q, 1)
    firsti, firstj = first(irange), first(jrange)
    lasti, lastj = last(irange), last(jrange)
    xindex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    xstride = blockDim().x * gridDim().x
    yindex = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ystride = blockDim().y * gridDim().y   
    oneT = one(T)
    twoT = 2one(T)
    acc = zero(Float64)
    @inbounds for j = (firstj + yindex - 1):ystride:lastj
        for i = (firsti + xindex - 1):xstride:lasti
            qp_local = zero(eltype(out))
            for k in 1:K
                qp_local += q[k, i] * p[k, j]
            end
            blk = g[(i - 1) >> 2 + 1, j - joffset]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = T(gij_pre > 0x01) * T(gij_pre - 0x01)
            nonmissing = T(gij_pre != 0x01)
            @inbounds acc += nonmissing * (gij * log(qp_local) + 
                (twoT - gij) * log(oneT - qp_local))
        end
    end
    CUDA.@atomic out[] += acc
    return nothing
end

@inline function em_q_kernel!(q_next, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, p::AbstractArray{T, 2}, irange, jrange, joffset=0) where T
    K = size(q, 1)
    firsti, firstj = first(irange), first(jrange)
    lasti, lastj = last(irange), last(jrange)
    xindex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    xstride = blockDim().x * gridDim().x
    yindex = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ystride = blockDim().y * gridDim().y   
    oneT = one(T)
    twoT = 2one(T)
    @inbounds for j = (firstj + yindex - 1):ystride:lastj
        for i = (firsti + xindex - 1):xstride:lasti
            qp_local = zero(eltype(q_next))
            for k in 1:K
                qp_local += q[k, i] * p[k, j]
            end
            blk = g[(i - 1) >> 2 + 1, j - joffset]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = T(gij_pre > 0x01) * T(gij_pre - 0x01)
            nonmissing = T(gij_pre != 0x01)
            for k in 1:K
                tmp = (gij * q[k, i] * p[k, j] / qp_local + 
                    (2 - gij) * q[k, i] * (1 - p[k, j]) / (1 - qp_local))
                CUDA.@atomic q_next[k, i] += nonmissing * tmp
            end
        end
    end

    return nothing
end

@inline function em_p_kernel!(p_next, p_tmp, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, p::AbstractArray{T, 2}, irange, jrange, joffset=0) where T
    K = size(q, 1)
    firsti, firstj = first(irange), first(jrange)
    lasti, lastj = last(irange), last(jrange)
    xindex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    xstride = blockDim().x * gridDim().x
    yindex = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ystride = blockDim().y * gridDim().y   
    oneT = one(T)
    twoT = 2one(T)
    @inbounds for j = (firstj + yindex - 1):ystride:lastj
        for i = (firsti + xindex - 1):xstride:lasti
            qp_local = zero(eltype(p_next))
            for k in 1:K
                qp_local += q[k, i] * p[k, j]
            end
            blk = g[(i - 1) >> 2 + 1, j - joffset]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = T(gij_pre > 0x01) * T(gij_pre - 0x01)
            nonmissing = T(gij_pre != 0x01)
            for k in 1:K
                CUDA.@atomic p_tmp[k, j] += nonmissing * (gij * q[k, i] * p[k, j] / qp_local)
                CUDA.@atomic p_next[k, j] += nonmissing * ((twoT - gij) * q[k, i] * 
                    (oneT - p[k, j]) / (oneT - qp_local))
            end
        end
    end

    return nothing
end

@inline function update_q_kernel!(XtX, Xtz, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, p::AbstractArray{T, 2}, irange, jrange, joffset=0) where T
    K = size(q, 1)
    firsti, firstj = first(irange), first(jrange)
    lasti, lastj = last(irange), last(jrange)
    xindex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    xstride = blockDim().x * gridDim().x
    yindex = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ystride = blockDim().y * gridDim().y   
    oneT = one(T)
    twoT = 2one(T)
    @inbounds for j = (firstj + yindex - 1):ystride:lastj
        for i = (firsti + xindex - 1):xstride:lasti
            qp_local = zero(eltype(XtX))
            for k in 1:K
                qp_local += q[k, i] * p[k, j]
            end
            blk = g[(i - 1) >> 2 + 1, j - joffset]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = T(gij_pre > 0x01) * T(gij_pre - 0x01)
            nonmissing = T(gij_pre != 0x01)
            for k in 1:K
                CUDA.@atomic Xtz[k, i] += nonmissing * (gij * p[k, j] / qp_local + 
                    (twoT - gij) * (oneT - p[k, j]) / (oneT - qp_local))
                for k2 in 1:K
                    CUDA.@atomic XtX[k2, k, i] += nonmissing * (gij / (qp_local) ^ 2 * p[k, j] * p[k2, j] + 
                        (twoT - gij) / (oneT - qp_local) ^ 2 * (oneT - p[k, j]) * (oneT - p[k2, j]))
                end
            end
        end
    end

    return nothing
end

@inline function update_p_kernel!(XtX, Xtz, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, p::AbstractArray{T, 2}, irange, jrange, joffset=0) where T
    K = size(q, 1)
    firsti, firstj = first(irange), first(jrange)
    lasti, lastj = last(irange), last(jrange)
    xindex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    xstride = blockDim().x * gridDim().x
    yindex = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ystride = blockDim().y * gridDim().y   
    oneT = one(T)
    twoT = 2one(T)
    @inbounds for j = (firstj + yindex - 1):ystride:lastj
        for i = (firsti + xindex - 1):xstride:lasti
            qp_local = zero(eltype(XtX))
            for k in 1:K
                qp_local += q[k, i] * p[k, j]
            end
            blk = g[(i - 1) >> 2 + 1, j - joffset]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = T(gij_pre > 0x01) * T(gij_pre - 0x01)
            nonmissing = T(gij_pre != 0x01)
            for k in 1:K
                CUDA.@atomic Xtz[k, j] += nonmissing * (gij * q[k, i] / qp_local - 
                        (twoT - gij) * q[k, i] / (oneT - qp_local))
                for k2 in 1:K
                    CUDA.@atomic XtX[k2, k, j] += nonmissing * (gij / (qp_local) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qp_local) ^ 2 * q[k, i] * q[k2, i])
                end
            end
        end
    end

    return nothing
end
