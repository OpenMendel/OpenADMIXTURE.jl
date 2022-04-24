const TILE = Ref(512) # this is now a length, in bytes!
const nonmissing_map_Float64 = [1.0, 0.0, 1.0, 1.0]
const g_map_Float64 = [0.0, 0.0, 1.0, 2.0]
const nonmissing_map_Float32 = [1.0f0, 0.0f0, 1.0f0, 1.0f0]
const g_map_Float32 = [0.0f0, 0.0f0, 1.0f0, 2.0f0]

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

@inline maybe32divsize(::Type{<:AbstractArray{T}}) where T<:Number = max(1, 32 ÷ sizeof(T))
@inline maybe32divsize(::Type) = 4

@inline function findthree(r::UnitRange)
    d = div(length(r), 3)
    i0 = first(r)
    (i0 : i0+d-1), (i0+d : i0+2d-1), (i0+2d : i0+length(r)-1)
end

function threader!(ftn!::F, ::Type{T}, arrs::Tuple, irange, jrange, K, split_i::Bool; 
    maxL = tile_maxiter(T), threads=nthreads()) where {F <: Function, T}
    threads = min(threads, split_i ? length(irange) : length(jrange))
    # println("starting threader: $irange $jrange on $(threadid()), $threads threads")
    if threads == 1 
        # println("launch tiler: $irange $jrange on thread # $(threadid())")
        tiler!(ftn!, T, arrs, irange, jrange, K; maxL=maxL)
        return
    elseif threads > 2 && threads % 3 == 0
        if split_i
            I1, I2, I3 = findthree(irange)
            task1 = Threads.@spawn begin
                # println("threader: $I1 $jrange on $(threadid())")
                threader!(ftn!, T, arrs, I1, jrange, K, split_i; maxL=maxL, threads=threads÷3)
            end
            task2 = Threads.@spawn begin
                # println("threader: $I2 $jrange on $(threadid())")
                threader!(ftn!, T, arrs, I2, jrange, K, split_i; maxL=maxL, threads=threads÷3)
            end
            # println("threader: $I3 $jrange on $(threadid())")
            threader!(ftn!, T, arrs, I3, jrange, K, split_i; maxL=maxL, threads=threads÷3)
            wait(task1)
            wait(task2)
        else
            J1, J2, J3 = findthree(jrange)
            task1 = Threads.@spawn begin
                # println("threader: $irange $J1 on $(threadid())")
                threader!(ftn!, T, arrs, irange, J1, K, split_i; maxL=maxL, threads=threads÷3)
            end
            task2 = Threads.@spawn begin
                # println("threader: $irange $J2 on $(threadid())")
                threader!(ftn!, T, arrs, irange, J2, K, split_i; maxL=maxL, threads=threads÷3)
            end
            # println("threader: $irange $J3 on $(threadid())")
            threader!(ftn!, T, arrs, irange, J3, K, split_i; maxL=maxL, threads=threads÷3)
            wait(task1)
            wait(task2)
        end
    else
        if split_i
            I1, I2 = cleave(irange, maybe32divsize(T))
            task = Threads.@spawn begin
                # println("threader: $I1 $jrange on $(threadid())")
                threader!(ftn!, T, arrs, I1, jrange, K, split_i; maxL=maxL, threads=threads÷2)
            end
            # println("threader: $I2 $jrange on $(threadid())")
            threader!(ftn!, T, arrs, I2, jrange, K, split_i; maxL=maxL, threads=threads÷2)
            wait(task)
        else
            J1, J2 = cleave(jrange, maybe32divsize(T))
            task = Threads.@spawn begin
                # println("threader: $irange $J1 on $(threadid())")
                threader!(ftn!, T, arrs, irange, J1, K, split_i; maxL=maxL, threads=threads÷2)
            end
            # println("threader: $irange $J2 on $(threadid())")
            threader!(ftn!, T, arrs, irange, J2, K, split_i; maxL=maxL, threads=threads÷2)
            wait(task)
        end
    end
end

function tiler!(ftn!::F, ::Type{T}, arrs::Tuple, irange, jrange, K; maxL = tile_maxiter(T)) where {F <: Function, T}
    ilen, jlen = length(irange), length(jrange)
    maxL = tile_maxiter(T)
    if ilen > maxL || jlen > maxL
        if ilen > jlen
            I1s, I2s = cleave(irange, maybe32divsize(T))
            tiler!(ftn!, T, arrs, I1s, jrange, K; maxL=maxL)
            tiler!(ftn!, T, arrs, I2s, jrange, K; maxL=maxL)
        else
            J1s, J2s = cleave(jrange, maybe32divsize(T))
            tiler!(ftn!, T, arrs, irange, J1s, K; maxL=maxL)
            tiler!(ftn!, T, arrs, irange, J2s, K; maxL=maxL)          
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

# """
#     qf!(qf, q, f)
# Computes q * f, in-place.
# """
# function qf!(qf, q, f)
#     @tullio qf[i, j] = q[k, i] * f[k, j]
# end

@inline function qf_block!(qf_small::AbstractArray{T}, q, f, irange, jrange, K) where T
    tid = threadid()
    fill!(view(qf_small, :, :, tid), zero(T))
    firsti, firstj = first(irange), first(jrange)
    @turbo for j in jrange
        for i in irange
            for k in 1:K
                qf_small[i-firsti+1, j-firstj+1, tid] += q[k, i] * f[k, j]
            end
        end
    end  
end

@inline function loglikelihood_loop(g::AbstractArray{T}, qf, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    r = zero(Float64)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            r += (gij * log(qf[i, j]) + (twoT - gij) * log(oneT - qf[i, j]))
        end
    end
    r
end

@inline function loglikelihood_loop_skipmissing(g::AbstractArray{T}, qf, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    r = zero(Float64)
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
    rslt = zero(Float64)
    gmat = g.s.data
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
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
    rslt = zero(Float64)
    gmat = g.s.data
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    nonmissing_map = T == Float64 ? nonmissing_map_Float64 : nonmissing_map_Float32
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            r = (i - 1) % 4
            blk_shifted  = blk >> (r << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01]
            nonmissing = nonmissing_map[gij_pre + 0x01]
            rslt += nonmissing ? (gij * log(qf[i, j]) + (twoT - gij) * log(oneT - qf[i, j])) : zero(T)
        end
    end
    # @tullio r = g[i, j] * log(qf[i, j]) + (2 - g[i, j]) * log(1 - qf[i, j])
    rslt
end

@inline function loglikelihood_loop(g::AbstractArray{T}, q, f, qf_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    r = zero(Float64)
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

@inline function loglikelihood_loop_skipmissing(g::AbstractArray{T}, q, f, qf_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    # firsti, firstj = first(irange), first(jrange)
    r = zero(Float64)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            nonmissing = (gij == gij)
            qf_local = zero(T)
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            r += nonmissing ? gij * log(qf_local) : zero(T)
        end
    end
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            nonmissing = (gij == gij)
            qf_local = zero(T)
            for k in 1:K
                qf_local += q[k, i] * f[k, j]
            end
            r += nonmissing ? (twoT - gij) * log(oneT - qf_local) : zero(T)
        end
    end
    r
end

@inline function loglikelihood_loop(g::SnpLinAlg{T}, q, f, qf_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    r = zero(Float64)
    gmat = g.s.data
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01] + (gij_pre == 0x01) * g.μ[j]
            r +=  (gij * log(qf_small[i-firsti+1, j-firstj+1, tid])) + ((twoT - gij) * log(oneT - qf_small[i-firsti+1, j-firstj+1, tid]))
            
        end
    end
    r
end


@inline function loglikelihood_loop_skipmissing(g::SnpLinAlg{T}, q, f, qf_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    r = zero(Float64)
    gmat = g.s.data
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    nonmissing_map = T == Float64 ? nonmissing_map_Float64 : nonmissing_map_Float32
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01]
            nonmissing = nonmissing_map[gij_pre + 0x01]
            r += nonmissing * ((gij * log(qf_small[i-firsti+1, j-firstj+1, tid])) + ((twoT - gij) * log(oneT - qf_small[i-firsti+1, j-firstj+1, tid])))
        end
    end
    r
end

@inline function em_q_loop!(q_next, g::AbstractArray{T}, q, f, qf_small::AbstractArray{T}, 
    irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            for k in 1:K
                gij = g[i, j]
                q_next[k, i] += (gij * q[k, i] * f[k, j] / qf_small[i-firsti+1, j-firstj+1, tid] + 
                    (2 - gij) * q[k, i] * (1 - f[k, j]) / (1 - qf_small[i-firsti+1, j-firstj+1, tid]))
            end
        end
    end
end

@inline function em_q_loop_skipmissing!(q_next, g::AbstractArray{T}, q, f, qf_small::AbstractArray{T}, 
    irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            for k in 1:K
                gij = g[i, j]
                nonmissing = (gij == gij)
                tmp = (gij * q[k, i] * f[k, j] / qf_small[i-firsti+1, j-firstj+1, tid] + 
                    (2 - gij) * q[k, i] * (1 - f[k, j]) / (1 - qf_small[i-firsti+1, j-firstj+1, tid]))
                q_next[k, i] += nonmissing ? tmp : zero(T)
            end
        end
    end
end

@inline function em_q_loop!(q_next, g::SnpLinAlg{T}, q, f, qf_small, irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    gmat = g.s.data
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    @turbo for j in jrange
        for i in irange
            for k in 1:K
                blk = gmat[(i - 1) >> 2 + 1, j]
                re = (i - 1) % 4
                blk_shifted  = blk >> (re << 1)
                gij_pre = blk_shifted & 0x03
                gij = g_map[gij_pre + 0x01] + (gij_pre == 0x01) * g.μ[j]
                q_next[k, i] += (gij * q[k, i] * f[k, j] / qf_small[i-firsti+1, j-firstj+1, tid] + 
                    (2 - gij) * q[k, i] * (1 - f[k, j]) / (1 - qf_small[i-firsti+1, j-firstj+1, tid]))
            end
        end
    end
end

@inline function em_q_loop_skipmissing!(q_next, g::SnpLinAlg{T}, q, f, qf_small::AbstractArray{T}, irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    gmat = g.s.data
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    nonmissing_map = T == Float64 ? nonmissing_map_Float64 : nonmissing_map_Float32
    @turbo for j in jrange
        for i in irange
            for k in 1:K
                blk = gmat[(i - 1) >> 2 + 1, j]
                re = (i - 1) % 4
                blk_shifted  = blk >> (re << 1)
                gij_pre = blk_shifted & 0x03
                gij = g_map[gij_pre + 0x01]
                nonmissing = nonmissing_map[gij_pre + 0x01]
                tmp = (gij * q[k, i] * f[k, j] / qf_small[i-firsti+1, j-firstj+1, tid] + 
                    (2 - gij) * q[k, i] * (1 - f[k, j]) / (1 - qf_small[i-firsti+1, j-firstj+1, tid]))
                q_next[k, i] += nonmissing * tmp
            end
        end
    end
end

@inline function em_f_loop!(f_next, g::AbstractArray{T}, q, f, f_tmp, qf_small::AbstractArray{T}, irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            for k in 1:K
                f_tmp[k, j] += gij * q[k, i] * f[k, j] / qf_small[i-firsti+1, j-firstj+1, tid]
                f_next[k, j] += (2 - gij) * q[k, i] * (one(T) - f[k, j]) / (one(T) - qf_small[i-firsti+1, j-firstj+1, tid])
            end
        end
    end
end

@inline function em_f_loop_skipmissing!(f_next, g::AbstractArray{T}, q, f, f_tmp, qf_small::AbstractArray{T}, irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            nonmissing = (gij == gij)
            for k in 1:K
                f_tmp[k, j] += nonmissing ? (gij * q[k, i] * f[k, j] / qf_small[i-firsti+1, j-firstj+1, tid]) : zero(T)
                f_next[k, j] += nonmissing ? ((2 - gij) * q[k, i] * (one(T) - f[k, j]) / (one(T) - qf_small[i-firsti+1, j-firstj+1, tid])) : zero(T)
            end
        end
    end
end

@inline function em_f_loop!(f_next, g::SnpLinAlg{T}, q, f, f_tmp, qf_small::AbstractArray{T}, irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    gmat = g.s.data
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01] + (gij_pre == 0x01) * g.μ[j]
            for k in 1:K
                f_tmp[k, j] += gij * q[k, i] * f[k, j] / qf_small[i-firsti+1, j-firstj+1, tid]
                f_next[k, j] += (2 - gij) * q[k, i] * (one(T) - f[k, j]) / (one(T) - qf_small[i-firsti+1, j-firstj+1, tid])
            end
        end
    end
end

@inline function em_f_loop_skipmissing!(f_next, g::SnpLinAlg{T}, q, f, f_tmp, qf_small::AbstractArray{T}, irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    gmat = g.s.data
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    nonmissing_map = T == Float64 ? nonmissing_map_Float64 : nonmissing_map_Float32
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01]
            nonmissing = nonmissing_map[gij_pre + 0x01]
            for k in 1:K
                f_tmp[k, j] += nonmissing * (gij * q[k, i] * f[k, j] / qf_small[i-firsti+1, j-firstj+1, tid])
                f_next[k, j] += nonmissing * ((2 - gij) * q[k, i] * (one(T) - f[k, j]) / (one(T) - qf_small[i-firsti+1, j-firstj+1, tid]))
            end
        end
    end
end

@inline function update_q_loop!(XtX, Xtz, g::AbstractArray{T}, q, f, qf_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            for k in 1:K
                Xtz[k, i] += gij * f[k, j] / qf_small[i-firsti+1, j-firstj+1, tid] + 
                    (twoT - gij) * (oneT - f[k, j]) / (oneT - qf_small[i-firsti+1, j-firstj+1, tid])
                for k2 in 1:K
                    XtX[k2, k, i] +=  gij / (qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * f[k, j] * f[k2, j] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * (oneT - f[k, j]) * (oneT - f[k2, j])
                end
            end
        end
    end
end

@inline function update_q_loop_skipmissing!(XtX, Xtz, g::AbstractArray{T}, q, f, qf_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            nonmissing = (gij == gij)
            for k in 1:K
                Xtz[k, i] += nonmissing ? (gij * f[k, j] / qf_small[i-firsti+1, j-firstj+1, tid] + 
                    (twoT - gij) * (oneT - f[k, j]) / (oneT - qf_small[i-firsti+1, j-firstj+1, tid])) : zero(T)
                for k2 in 1:K
                    XtX[k2, k, i] += nonmissing ? (gij / (qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * f[k, j] * f[k2, j] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * (oneT - f[k, j]) * (oneT - f[k2, j])) : zero(T)
                end
            end
        end
    end
end

@inline function update_f_loop!(XtX, Xtz, g::AbstractArray{T}, q, f, qf_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            for k in 1:K
                Xtz[k, j] += gij * q[k, i] / qf_small[i-firsti+1, j-firstj+1, tid] - 
                        (twoT - gij) * q[k, i] / (oneT - qf_small[i-firsti+1, j-firstj+1, tid])
                for k2 in 1:K
                    XtX[k2, k, j] += gij / (qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i]
                end
            end
        end
    end
end

@inline function update_f_loop_skipmissing!(XtX, Xtz, g::AbstractArray{T}, q, f, qf_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            nonmissing =  (gij == gij)
            for k in 1:K
                Xtz[k, j] += nonmissing ? (gij * q[k, i] / qf_small[i-firsti+1, j-firstj+1, tid] - 
                        (twoT - gij) * q[k, i] / (oneT - qf_small[i-firsti+1, j-firstj+1, tid])) : zero(T)
                for k2 in 1:K
                    XtX[k2, k, j] += nonmissing ? (gij / (qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i]) : zero(T)
                end
            end
        end
    end
end

@inline function update_q_loop!(XtX, Xtz, g::SnpLinAlg{T}, q, f, qf_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    gmat = g.s.data
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01] + (gij_pre == 0x01) * g.μ[j]
            for k in 1:K
                Xtz[k, i] += (gij * f[k, j] / qf_small[i-firsti+1, j-firstj+1, tid] + 
                    (twoT - gij) * (oneT - f[k, j]) / (oneT - qf_small[i-firsti+1, j-firstj+1, tid]))
                for k2 in 1:K
                    XtX[k2, k, i] +=  (gij / (qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * f[k, j] * f[k2, j] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * (oneT - f[k, j]) * (oneT - f[k2, j]))
                end
            end
        end
    end
end

@inline function update_q_loop_skipmissing!(XtX, Xtz, g::SnpLinAlg{T}, q, f, qf_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    gmat = g.s.data
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    nonmissing_map = T == Float64 ? nonmissing_map_Float64 : nonmissing_map_Float32
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01]
            nonmissing = nonmissing_map[gij_pre + 0x01]
            for k in 1:K
                Xtz[k, i] += nonmissing * (gij * f[k, j] / qf_small[i-firsti+1, j-firstj+1, tid] + 
                    (twoT - gij) * (oneT - f[k, j]) / (oneT - qf_small[i-firsti+1, j-firstj+1, tid]))
                for k2 in 1:K
                    XtX[k2, k, i] += nonmissing * (gij / (qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * f[k, j] * f[k2, j] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * (oneT - f[k, j]) * (oneT - f[k2, j]))
                end
            end
        end
    end
end

function update_f_loop!(XtX, Xtz, g::SnpLinAlg{T}, q, f, qf_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    gmat = g.s.data
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01] + (gij_pre == 0x01) * g.μ[j]
            for k in 1:K
                Xtz[k, j] += (gij * q[k, i] / qf_small[i-firsti+1, j-firstj+1, tid] - 
                        (twoT - gij) * q[k, i] / (oneT - qf_small[i-firsti+1, j-firstj+1, tid]))
                for k2 in 1:K
                    XtX[k2, k, j] += (gij / (qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i])
                end
            end
        end
    end
end

function update_f_loop_skipmissing!(XtX, Xtz, g::SnpLinAlg{T}, q, f, qf_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    gmat = g.s.data
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qf_block!(qf_small, q, f, irange, jrange, K)
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    nonmissing_map = T == Float64 ? nonmissing_map_Float64 : nonmissing_map_Float32
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01]
            nonmissing = nonmissing_map[gij_pre + 0x01]
            for k in 1:K
                Xtz[k, j] += nonmissing * (gij * q[k, i] / qf_small[i-firsti+1, j-firstj+1, tid] - 
                        (twoT - gij) * q[k, i] / (oneT - qf_small[i-firsti+1, j-firstj+1, tid]))
                for k2 in 1:K
                    XtX[k2, k, j] += nonmissing * (gij / (qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qf_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i])
                end
            end
        end
    end
end
