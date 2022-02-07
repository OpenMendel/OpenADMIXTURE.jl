const TILE = Ref(512) # this is now a length, in bytes!
const nonmissing_map = [1.0, 0.0, 1.0, 1.0]
const g_map = [0.0, 0.0, 1.0, 2.0]

function tile_maxiter(::Type{<:AbstractArray{T}}) where {T}
    isbitstype(T) || return TILE[] ÷ 8
    max(TILE[] ÷ sizeof(T), 4)
end
tile_maxiter(::Type{AT}) where {AT} = TILE[] ÷ 8 # treat anything unkown like Float64

@inline function findcleft(r::UnitRange, step::Int)
    if length(r) >= 2*step
        minimum(r) - 1 + step * div(length(r), step * 2)
    else
        # minimum(r) - 1 + div(length(r), 2, RoundNearest) # not in Julia 1.3
        minimum(r) - 1 + round(Int, length(r)/2)
    end
end

@inline function cleave(range::UnitRange, step::Int=4)
    cleft = findcleft(range, step)
    first(range):cleft, cleft+1:last(range)
end

function tiler!(ftn!::F, ::Type{T}, arrs::Tuple, irange, jrange, K; maxL = tile_maxiter(T)) where {F <: Function, T}
    ilen, jlen = length(irange), length(jrange)
    maxL = tile_maxiter(T)
    if ilen > maxL || jlen > maxL
        if ilen > jlen
            I1s, I2s = cleave(irange)
            tiler!(ftn!, T, arrs, I1s, jrange, K)
            tiler!(ftn!, T, arrs, I2s, jrange, K)
        else
            J1s, J2s = cleave(jrange)
            tiler!(ftn!, T, arrs, irange, J1s, K)
            tiler!(ftn!, T, arrs, irange, J2s, K)          
        end
    else
        ftn!(arrs..., irange, jrange, K)
    end
end

function tiler_1d!(ftn!::F, ::Type{T}, arrs::Tuple, irange, jrange, K; maxL = tile_maxiter(T)) where {F <: Function, T}
    ilen, jlen = length(irange), length(jrange)
    maxL = tile_maxiter(T)
    if jlen > maxL
        J1s, J2s = cleave(jrange)
        tiler!(ftn!, T, arrs, irange, J1s, K)
        tiler!(ftn!, T, arrs, irange, J2s, K)          
    else
        ftn!(arrs..., irange, jrange, K)
    end
end

function tiler_scalar(ftn::F, ::Type{T}, z::RT, arrs::Tuple, irange, jrange, K) where {F <: Function, T, RT}
    ilen, jlen = length(irange), length(jrange)
    maxL = tile_maxiter(T)
    r = z
    if ilen > maxL || jlen > maxL
        if ilen > jlen
            I1s, I2s = cleave(irange)
            r += tiler_scalar(ftn, T, z, arrs, I1s, jrange, K)
            r += tiler_scalar(ftn, T, z, arrs, I2s, jrange, K)
        else
            J1s, J2s = cleave(jrange)
            r += tiler_scalar(ftn, T, z, arrs, irange, J1s, K)
            r += tiler_scalar(ftn, T, z, arrs, irange, J2s, K)          
        end
        return r
    else
        return ftn(arrs..., irange, jrange, K)::RT
    end
end

function tiler_scalar_1d(ftn::F, ::Type{T}, z::RT, arrs::Tuple, irange, jrange, K) where {F <: Function, T, RT}
    ilen, jlen = length(irange), length(jrange)
    maxL = tile_maxiter(T)
    r = z
    if jlen > maxL
        J1s, J2s = cleave(jrange)
        r += tiler_scalar_1d(ftn, T, z, arrs, irange, J1s, K)
        r += tiler_scalar_1d(ftn, T, z, arrs, irange, J2s, K)          
        return r
    else
        return ftn(arrs..., irange, jrange, K)::RT
    end
end

"""
    qf!(qf, q, f)
Computes q * f, in-place.
"""
function qf!(qf, q, f)
    @tullio qf[i, j] = q[k, i] * f[k, j]
end

@inline function qf_block!(qf_small::AbstractArray{T}, q, f, irange, jrange, K) where T
    fill!(qf_small, zero(T))
    firsti, firstj = first(irange), first(jrange)
    @turbo for j in jrange
        for i in irange
            for k in 1:K
                qf_small[i-firsti+1, j-firstj+1] += q[k, i] * f[k, j]
            end
        end
    end  
end

@inline function loglikelihood_loop(g::AbstractArray{T}, qf, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    r = zero(T)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            r += (gij * log(qf[i, j]) + (twoT - gij) * log(oneT - qf[i, j]))
        end
    end
    # @tullio r = g[i, j] * log(qf[i, j]) + (2 - g[i, j]) * log(1 - qf[i, j])
    r
end

@inline function loglikelihood_loop_skipmissing(g::AbstractArray{T}, qf, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    r = zero(T)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            r += gij != gij ? zero(T) : (gij * log(qf[i, j]) + (twoT - gij) * log(oneT - qf[i, j]))
        end
    end
    # @tullio r = g[i, j] * log(qf[i, j]) + (2 - g[i, j]) * log(1 - qf[i, j])
    r
end

@inline function loglikelihood_loop(g::SnpLinAlg{T}, qf, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    rslt = zero(T)
    gmat = g.s.data
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            r = (i - 1) % 4
            blk_shifted  = blk >> (r << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01] + (gij_pre == 0x01) * g.μ[j]
            rslt += (gij * log(qf[i, j]) + (twoT - gij) * log(oneT - qf[i, j]))
        end
    end
    # @tullio r = g[i, j] * log(qf[i, j]) + (2 - g[i, j]) * log(1 - qf[i, j])
    rslt
end

@inline function loglikelihood_loop_skipmissing(g::SnpLinAlg{T}, qf, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    rslt = zero(T)
    gmat = g.s.data
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            r = (i - 1) % 4
            blk_shifted  = blk >> (r << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01]
            nonmissing = nonmissing_map[gij_pre + 0x01]
            rslt += nonmissing * (gij * log(qf[i, j]) + (twoT - gij) * log(oneT - qf[i, j]))
        end
    end
    # @tullio r = g[i, j] * log(qf[i, j]) + (2 - g[i, j]) * log(1 - qf[i, j])
    rslt
end

@inline function loglikelihood_loop(g::AbstractArray{T}, q, f, qf_small, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    # firsti, firstj = first(irange), first(jrange)
    r = zero(T)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            qf_local = zero(T)
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            r += gij * log(qf_local)
        end
    end
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            qf_local = zero(T)
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            r += (twoT - gij) * log(oneT - qf_local)
        end
    end
    r
end

@inline function loglikelihood_loop_skipmissing(g::AbstractArray{T}, q, f, qf_small, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    # firsti, firstj = first(irange), first(jrange)
    r = zero(T)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            isnan = gij != gij
            qf_local = zero(T)
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            r += isnan ? zero(T) : gij * log(qf_local) + 
                (twoT - gij) * log(oneT - qf_local)
        end
    end
    r
end

@inline function loglikelihood_loop(g::SnpLinAlg{T}, q, f, qf_small, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    # firsti, firstj = first(irange), first(jrange)
    # qf_block!(qf_small, q, f, irange, jrange, K)
    r = zero(T)
    gmat = g.s.data
    @turbo for j in jrange
        for i in irange
            qf_local = zero(T)
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01] + (gij_pre == 0x01) * g.μ[j]
            r +=  (gij * log(qf_local))
        end
    end
    @turbo for j in jrange
        for i in irange
            qf_local = zero(T)
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01] + (gij_pre == 0x01) * g.μ[j]
            r += ((twoT - gij) * log(oneT - qf_local))
        end
    end
    # @tullio r = g[i, j] * log(qf[i, j]) + (2 - g[i, j]) * log(1 - qf[i, j])
    r
end

@inline function loglikelihood_loop_skipmissing(g::SnpLinAlg{T}, q, f, qf_small, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    # firsti, firstj = first(irange), first(jrange)
    # qf_block!(qf_small, q, f, irange, jrange, K)
    r = zero(T)
    gmat = g.s.data
    @turbo for j in jrange
        for i in irange
            qf_local = zero(T)
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01]
            nonmissing = nonmissing_map[gij_pre + 0x01]
            r += nonmissing * ((gij * log(qf_local)))
        end
    end
    @turbo for j in jrange
        for i in irange
            qf_local = zero(T)
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01]
            nonmissing = nonmissing_map[gij_pre + 0x01]
            r += nonmissing * ((twoT - gij) * log(oneT - qf_local))
        end
    end
    # @tullio r = g[i, j] * log(qf[i, j]) + (2 - g[i, j]) * log(1 - qf[i, j])
    r
end

@inline function em_q_loop!(q_next, g::AbstractArray{T}, q, f, qf, irange, jrange, K) where T
    @turbo for j in jrange
        for i in irange
            for k in 1:K
                gij = g[i, j]
                q_next[k, i] += (gij * q[k, i] * f[k, j] / qf[i, j] + 
                    (2 - gij) * q[k, i] * (1 - f[k, j]) / (1 - qf[i, j]))
            end
        end
    end
end

@inline function em_q_loop_skipmissing!(q_next, g::AbstractArray{T}, q, f, qf, irange, jrange, K) where T
    @turbo for j in jrange
        for i in irange
            for k in 1:K
                gij = g[i, j]
                isnan = gij != gij
                tmp = (gij * q[k, i] * f[k, j] / qf[i, j] + 
                    (2 - gij) * q[k, i] * (1 - f[k, j]) / (1 - qf[i, j]))
                q_next[k, i] += isnan ? zero(T) : tmp
            end
        end
    end
end

@inline function em_f_loop!(f_next, g::AbstractArray{T}, q, f, f_tmp, qf, irange, jrange, K) where T
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            for k in 1:K
                f_tmp[k, j] += gij * q[k, i] * f[k, j] / qf[i, j]
                f_next[k, j] += (2 - gij) * q[k, i] * (one(T) - f[k, j]) / (one(T) - qf[i, j])
            end
        end
    end
end

@inline function em_f_loop_skipmissing!(f_next, g::AbstractArray{T}, q, f, f_tmp, qf, irange, jrange, K) where T
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            isnan = gij != gij
            for k in 1:K
                f_tmp[k, j] += isnan ? zero(T) : gij * q[k, i] * f[k, j] / qf[i, j]
                f_next[k, j] += isnan ? zero(T) : (2 - gij) * q[k, i] * (one(T) - f[k, j]) / (one(T) - qf[i, j])
            end
        end
    end
end

@inline function update_q_loop!(XtX, Xtz, g::AbstractArray{T}, q, f, qf_small, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            for k in 1:K
                Xtz[k, i] += gij * f[k, j] / qf_small[i-firsti+1, j-firstj+1] + 
                    (twoT - gij) * (oneT - f[k, j]) / (oneT - qf_small[i-firsti+1, j-firstj+1])
                for k2 in 1:K
                    XtX[k2, k, i] +=  gij / (qf_small[i-firsti+1, j-firstj+1]) ^ 2 * f[k, j] * f[k2, j] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1]) ^ 2 * (oneT - f[k, j]) * (oneT - f[k2, j])
                end
            end
        end
    end
end

@inline function update_q_loop_skipmissing!(XtX, Xtz, g::AbstractArray{T}, q, f, qf_small, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            isnan = gij != gij
            for k in 1:K
                Xtz[k, i] += isnan ? zero(T) : gij * f[k, j] / qf_small[i-firsti+1, j-firstj+1] + 
                    (twoT - gij) * (oneT - f[k, j]) / (oneT - qf_small[i-firsti+1, j-firstj+1])
                for k2 in 1:K
                    XtX[k2, k, i] += isnan ? zero(T) : gij / (qf_small[i-firsti+1, j-firstj+1]) ^ 2 * f[k, j] * f[k2, j] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1]) ^ 2 * (oneT - f[k, j]) * (oneT - f[k2, j])
                end
            end
        end
    end
end

@inline function update_f_loop!(XtX, Xtz, g::AbstractArray{T}, q, f, qf_small, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            for k in 1:K
                Xtz[k, j] += gij * q[k, i] / qf_small[i-firsti+1, j-firstj+1] - 
                        (twoT - gij) * q[k, i] / (oneT - qf_small[i-firsti+1, j-firstj+1])
                for k2 in 1:K
                    XtX[k2, k, j] += gij / (qf_small[i-firsti+1, j-firstj+1]) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1]) ^ 2 * q[k, i] * q[k2, i]
                end
            end
        end
    end
end

@inline function update_f_loop_skipmissing!(XtX, Xtz, g::AbstractArray{T}, q, f, qf_small, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            isnan = gij != gij
            for k in 1:K
                Xtz[k, j] += isnan ? zero(T) : gij * q[k, i] / qf_small[i-firsti+1, j-firstj+1] - 
                        (twoT - gij) * q[k, i] / (oneT - qf_small[i-firsti+1, j-firstj+1])
                for k2 in 1:K
                    XtX[k2, k, j] += isnan ? zero(T) : gij / (qf_small[i-firsti+1, j-firstj+1]) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1]) ^ 2 * q[k, i] * q[k2, i]
                end
            end
        end
    end
end

@inline function update_q_loop!(XtX, Xtz, g::SnpLinAlg{T}, q, f, qf_small, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    gmat = g.s.data
    firsti, firstj = first(irange), first(jrange)
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01] + (gij_pre == 0x01) * g.μ[j]
            for k in 1:K
                Xtz[k, i] += (gij * f[k, j] / qf_small[i-firsti+1, j-firstj+1] + 
                    (twoT - gij) * (oneT - f[k, j]) / (oneT - qf_small[i-firsti+1, j-firstj+1]))
                for k2 in 1:K
                    XtX[k2, k, i] +=  (gij / (qf_small[i-firsti+1, j-firstj+1]) ^ 2 * f[k, j] * f[k2, j] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1]) ^ 2 * (oneT - f[k, j]) * (oneT - f[k2, j]))
                end
            end
        end
    end
end

@inline function update_q_loop_skipmissing!(XtX, Xtz, g::SnpLinAlg{T}, q, f, qf_small, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    gmat = g.s.data
    firsti, firstj = first(irange), first(jrange)
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01]
            nonmissing = nonmissing_map[gij_pre + 0x01]
            for k in 1:K
                Xtz[k, i] += nonmissing * (gij * f[k, j] / qf_small[i-firsti+1, j-firstj+1] + 
                    (twoT - gij) * (oneT - f[k, j]) / (oneT - qf_small[i-firsti+1, j-firstj+1]))
                for k2 in 1:K
                    XtX[k2, k, i] += nonmissing * (gij / (qf_small[i-firsti+1, j-firstj+1]) ^ 2 * f[k, j] * f[k2, j] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1]) ^ 2 * (oneT - f[k, j]) * (oneT - f[k2, j]))
                end
            end
        end
    end
end

function update_f_loop!(XtX, Xtz, g::SnpLinAlg{T}, q, f, qf_small, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    gmat = g.s.data
    firsti, firstj = first(irange), first(jrange)
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01] + (gij_pre == 0x01) * g.μ[j]
            for k in 1:K
                Xtz[k, j] += (gij * q[k, i] / qf_small[i-firsti+1, j-firstj+1] - 
                        (twoT - gij) * q[k, i] / (oneT - qf_small[i-firsti+1, j-firstj+1]))
                for k2 in 1:K
                    XtX[k2, k, j] += (gij / (qf_small[i-firsti+1, j-firstj+1]) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1]) ^ 2 * q[k, i] * q[k2, i])
                end
            end
        end
    end
end

function update_f_loop_skipmissing!(XtX, Xtz, g::SnpLinAlg{T}, q, f, qf_small, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    gmat = g.s.data
    firsti, firstj = first(irange), first(jrange)
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01]
            nonmissing = nonmissing_map[gij_pre + 0x01]
            for k in 1:K
                Xtz[k, j] += nonmissing * (gij * q[k, i] / qf_small[i-firsti+1, j-firstj+1] - 
                        (twoT - gij) * q[k, i] / (oneT - qf_small[i-firsti+1, j-firstj+1]))
                for k2 in 1:K
                    XtX[k2, k, j] += nonmissing * (gij / (qf_small[i-firsti+1, j-firstj+1]) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1]) ^ 2 * q[k, i] * q[k2, i])
                end
            end
        end
    end
end
