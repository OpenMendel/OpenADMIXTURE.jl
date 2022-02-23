using .CUDA
@inline function loglikelihood_kernel(out, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, f::AbstractArray{T, 2}, irange, jrange, joffset=0) where T
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
            qf_local = zero(eltype(out))
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            blk = g[(i - 1) >> 2 + 1, j - joffset]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = T(gij_pre > 0x01) * T(gij_pre - 0x01)
            nonmissing = T(gij_pre != 0x01)
            @inbounds acc += nonmissing * (gij * log(qf_local) + 
                (twoT - gij) * log(oneT - qf_local))
        end
    end
    CUDA.@atomic out[] += acc
    return nothing
end

@inline function em_q_kernel!(q_next, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, f::AbstractArray{T, 2}, irange, jrange, joffset=0) where T
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
            qf_local = zero(eltype(q_next))
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            blk = g[(i - 1) >> 2 + 1, j - joffset]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = T(gij_pre > 0x01) * T(gij_pre - 0x01)
            nonmissing = T(gij_pre != 0x01)
            for k in 1:K
                tmp = (gij * q[k, i] * f[k, j] / qf_local + 
                    (2 - gij) * q[k, i] * (1 - f[k, j]) / (1 - qf_local))
                CUDA.@atomic q_next[k, i] += nonmissing * tmp
            end
        end
    end

    return nothing
end

@inline function em_f_kernel!(f_next, f_tmp, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, f::AbstractArray{T, 2}, irange, jrange, joffset=0) where T
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
            qf_local = zero(eltype(f_next))
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            blk = g[(i - 1) >> 2 + 1, j - joffset]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = T(gij_pre > 0x01) * T(gij_pre - 0x01)
            nonmissing = T(gij_pre != 0x01)
            for k in 1:K
                CUDA.@atomic f_tmp[k, j] += nonmissing * (gij * q[k, i] * f[k, j] / qf_local)
                CUDA.@atomic f_next[k, j] += nonmissing * ((twoT - gij) * q[k, i] * 
                    (oneT - f[k, j]) / (oneT - qf_local))
            end
        end
    end

    return nothing
end

@inline function update_q_kernel!(XtX, Xtz, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, f::AbstractArray{T, 2}, irange, jrange, joffset=0) where T
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
            qf_local = zero(eltype(XtX))
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            blk = g[(i - 1) >> 2 + 1, j - joffset]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = T(gij_pre > 0x01) * T(gij_pre - 0x01)
            nonmissing = T(gij_pre != 0x01)
            for k in 1:K
                CUDA.@atomic Xtz[k, i] += nonmissing * (gij * f[k, j] / qf_local + 
                    (twoT - gij) * (oneT - f[k, j]) / (oneT - qf_local))
                for k2 in 1:K
                    CUDA.@atomic XtX[k2, k, i] += nonmissing * (gij / (qf_local) ^ 2 * f[k, j] * f[k2, j] + 
                        (twoT - gij) / (oneT - qf_local) ^ 2 * (oneT - f[k, j]) * (oneT - f[k2, j]))
                end
            end
        end
    end

    return nothing
end

@inline function update_f_kernel!(XtX, Xtz, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, f::AbstractArray{T, 2}, irange, jrange, joffset=0) where T
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
            qf_local = zero(eltype(XtX))
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            blk = g[(i - 1) >> 2 + 1, j - joffset]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = T(gij_pre > 0x01) * T(gij_pre - 0x01)
            nonmissing = T(gij_pre != 0x01)
            for k in 1:K
                CUDA.@atomic Xtz[k, j] += nonmissing * (gij * q[k, i] / qf_local - 
                        (twoT - gij) * q[k, i] / (oneT - qf_local))
                for k2 in 1:K
                    CUDA.@atomic XtX[k2, k, j] += nonmissing * (gij / (qf_local) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qf_local) ^ 2 * q[k, i] * q[k2, i])
                end
            end
        end
    end

    return nothing
end
