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

"""
Split the index space into two pieces.
"""
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

"""
Split the index space into three.
"""
@inline function findthree(r::UnitRange)
    d = div(length(r), 3)
    i0 = first(r)
    (i0 : i0+d-1), (i0+d : i0+2d-1), (i0+2d : i0+length(r)-1)
end

"""
    threader!(ftn!, ::Type{T}, arrs, irange, jrange, K, split_i; 
    maxL=tile_maxiter(T), threads=nthreads())

Apply `ftn!` over the index space `irange` x `jrange` of `arrs` with multiple threads.
It splits the index space in one dimension over either i dimension or j dimension 
in a single call. 
It recursively calls `threader!()` without changing `split_i` or 
calls `tiler!()` to eventually split in both directions.

# Input
- `ftn!`: The function to be applied
- `T`: element type of updated `arrs`
- `arrs`: a tuple of arrays to be updated
- `irange`: an index range
- `jrange`: another index range
- `K`: number of clusters
- `split_i`: if the dimension to split along is the "i" dimension. "j" dimension is split if `false`.
- `maxL`: maximum length of tile.
- `threads`: number of threads to be utilized 
"""
function threader!(ftn!::F, ::Type{T}, arrs::Tuple, irange, jrange, K, split_i::Bool; 
    maxL = tile_maxiter(T), threads=nthreads()) where {F <: Function, T}
    threads = min(threads, split_i ? length(irange) : length(jrange))
    # println("starting threader: $irange $jrange on $(threadid()), $threads threads")
    if threads == 1 # single-thread
        # println("launch tiler: $irange $jrange on thread # $(threadid())")
        tiler!(ftn!, T, arrs, irange, jrange, K; maxL=maxL)
        return
    elseif threads > 2 && threads % 3 == 0 # split into three pieces
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
        else # split j.
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
    else # split into two pieces.
        if split_i
            I1, I2 = cleave(irange, maybe32divsize(T))
            task = Threads.@spawn begin
                # println("threader: $I1 $jrange on $(threadid())")
                threader!(ftn!, T, arrs, I1, jrange, K, split_i; maxL=maxL, threads=threads÷2)
            end
            # println("threader: $I2 $jrange on $(threadid())")
            threader!(ftn!, T, arrs, I2, jrange, K, split_i; maxL=maxL, threads=threads÷2)
            wait(task)
        else # split j.
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

"""
    tiler!(ftn!, ::Type{T}, arrs, irange, jrange, K; 
    maxL=tile_maxiter(T), threads=nthreads())

Apply `ftn!` over the index space `irange` x `jrange` of `arrs` with a single thread. 
It splits the index space in two dimensions in small tiles. 

# Input
- `ftn!`: The function to be applied
- `T`: element type of updated `arrs`
- `arrs`: a tuple of arrays to be updated
- `irange`: an index range
- `jrange`: another index range
- `K`: number of clusters
- `maxL`: maximum length of tile.
"""
function tiler!(ftn!::F, ::Type{T}, arrs::Tuple, irange, jrange, K; maxL = tile_maxiter(T)) where {F <: Function, T}
    ilen, jlen = length(irange), length(jrange)
    maxL = tile_maxiter(T)
    if ilen > maxL || jlen > maxL
        if ilen > jlen # split i dimension
            I1s, I2s = cleave(irange, maybe32divsize(T))
            tiler!(ftn!, T, arrs, I1s, jrange, K; maxL=maxL)
            tiler!(ftn!, T, arrs, I2s, jrange, K; maxL=maxL)
        else # split j dimension
            J1s, J2s = cleave(jrange, maybe32divsize(T))
            tiler!(ftn!, T, arrs, irange, J1s, K; maxL=maxL)
            tiler!(ftn!, T, arrs, irange, J2s, K; maxL=maxL)          
        end
    else # base case.
        ftn!(arrs..., irange, jrange, K)
    end
end

"""
    tiler_1d!(ftn!, ::Type{T}, arrs, irange, jrange, K; 
    maxL=tile_maxiter(T), threads=nthreads())

Apply `ftn!` over the index space `irange` x `jrange` of `arrs` with a single thread. 
It splits the index space only in j dimension. 

# Input
- `ftn!`: The function to be applied
- `T`: element type of updated `arrs`
- `arrs`: a tuple of arrays to be updated
- `irange`: an index range
- `jrange`: another index range
- `K`: number of clusters
- `maxL`: maximum length of tile.
"""
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

"""
    tiler_scalar(ftn, ::Type{T}, z::RT, arrs, irange, jrange, K; 
    maxL=tile_maxiter(T))

Apply a reduction `ftn` over the index space `irange` x `jrange` of `arrs` with a single thread. 
It splits the index space both dimensions.

# Input
- `ftn`: The function to be applied
- `T`: element type of updated `arrs`
- `z`: the "zero" value for the return. 
- `arrs`: a tuple of input arrays
- `irange`: an index range
- `jrange`: another index range
- `K`: number of clusters
- `maxL`: maximum length of tile.
"""
function tiler_scalar(ftn::F, ::Type{T}, z::RT, arrs::Tuple, irange, jrange, K; maxL=tile_maxiter(T)) where {F <: Function, T, RT}
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

"""
    tiler_scalar_1d(ftn, ::Type{T}, z::RT, arrs, irange, jrange, K; 
    maxL=tile_maxiter(T))

Apply a reduction `ftn` over the index space `irange` x `jrange` of `arrs` with a single thread. 
It splits the index space only in j dimensions.

# Input
- `ftn`: The function to be applied
- `T`: element type of updated `arrs`
- `z`: the "zero" value for the return. 
- `arrs`: a tuple of input arrays
- `irange`: an index range
- `jrange`: another index range
- `K`: number of clusters
- `maxL`: maximum length of tile.
"""
function tiler_scalar_1d(ftn::F, ::Type{T}, z::RT, arrs::Tuple, irange, jrange, K; maxL=tile_maxiter(T)) where {F <: Function, T, RT}
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
    qp_block!(qp_small, q, p, irange, jrange, K)
Compute a block of the matrix Q x P.

# Input
- `qp_small`: the output.
- `q`: Q matrix.
- `p`: P matrix.
- `irange`: i index range over which this function is applied
- `jrange`: j index range over which this function is applied
- `K`: number of clusters
"""
@inline function qp_block!(qp_small::AbstractArray{T}, q, p, irange, jrange, K) where T
    tid = threadid()
    fill!(view(qp_small, :, :, tid), zero(T))
    firsti, firstj = first(irange), first(jrange)
    @turbo for j in jrange
        for i in irange
            for k in 1:K
                qp_small[i-firsti+1, j-firstj+1, tid] += q[k, i] * p[k, j]
            end
        end
    end  
end

"""
    loglikelihood_loop(g, qp, irange, jrange, K)
Compute loglikelihood using `qp`. Always returns the double precision result.

# Input
- `g`: genotype matrix.
- `qp`: QP matrix.
- `irange`: i index range over which this function is applied
- `jrange`: j index range over which this function is applied
- `K`: number of clusters
"""
@inline function loglikelihood_loop(g::AbstractArray{T}, qp, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    r = zero(Float64)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            r += (gij * log(qp[i, j]) + (twoT - gij) * log(oneT - qp[i, j]))
        end
    end
    r
end

"""
    loglikelihood_loop_skipmissing(g, qp, irange, jrange, K)
Compute loglikelihood using `qp`. Always returns the double precision result.

# Input
- `g`: genotype matrix.
- `qp`: QP matrix.
- `irange`: i index range over which this function is applied
- `jrange`: j index range over which this function is applied
- `K`: number of clusters
"""
@inline function loglikelihood_loop_skipmissing(g::AbstractArray{T}, qp, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    r = zero(Float64)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            r += gij != gij ? zero(T) : (gij * log(qp[i, j]) + (twoT - gij) * log(oneT - qp[i, j]))
        end
    end
    # @tullio r = g[i, j] * log(qf[i, j]) + (2 - g[i, j]) * log(1 - qf[i, j])
    r
end

@inline function loglikelihood_loop(g::SnpLinAlg{T}, qp, irange, jrange, K) where T
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
            rslt += (gij * log(qp[i, j]) + (twoT - gij) * log(oneT - qp[i, j]))
        end
    end
    # @tullio r = g[i, j] * log(qf[i, j]) + (2 - g[i, j]) * log(1 - qf[i, j])
    rslt
end

@inline function loglikelihood_loop_skipmissing(g::SnpLinAlg{T}, qp, irange, jrange, K) where T
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

"""
    loglikelihood_loop(g, q, p, qp_small, irange, jrange, K)
Compute loglikelihood using `q` and `p`, compute local `qp` on-the-fly. 
Always returns the double precision result.

# Input
- `g`: genotype matrix.
- `q`: Q matrix.
- `p`: P matrix.
- `qp_small`: dummy. 
- `irange`: i index range over which this function is applied
- `jrange`: j index range over which this function is applied
- `K`: number of clusters
"""
@inline function loglikelihood_loop(g::AbstractArray{T}, q, p, qp_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    r = zero(Float64)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            qp_local = zero(T)
            for k in 1:K
                qp_local += q[k, i] * p[k, j]
            end
            r += gij * log(qp_local)
        end
    end
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            qp_local = zero(T)
            for k in 1:K
                qp_local += q[k, i] * p[k, j]
            end
            r += (twoT - gij) * log(oneT - qp_local)
        end
    end
    r
end

"""
    loglikelihood_loop_skipmissing(g, q, p, qp_small, irange, jrange, K)
Compute loglikelihood using `q` and `p`, compute local `qp` on-the-fly. 
Always returns the double precision result.

# Input
- `g`: genotype matrix.
- `q`: Q matrix.
- `p`: P matrix.
- `qp_small`: dummy. 
- `irange`: i index range over which this function is applied
- `jrange`: j index range over which this function is applied
- `K`: number of clusters
"""
@inline function loglikelihood_loop_skipmissing(g::AbstractArray{T}, q, p, qp_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    # firsti, firstj = first(irange), first(jrange)
    r = zero(Float64)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            nonmissing = (gij == gij)
            qp_local = zero(T)
            for k in 1:K
                qp_local += q[k, i] * p[k, j]
            end
            r += nonmissing ? gij * log(qp_local) : zero(T)
        end
    end
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            nonmissing = (gij == gij)
            qp_small = zero(T)
            for k in 1:K
                qp_small += q[k, i] * p[k, j]
            end
            r += nonmissing ? (twoT - gij) * log(oneT - qp_small) : zero(T)
        end
    end
    r
end

@inline function loglikelihood_loop(g::SnpLinAlg{T}, q, p, qp_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
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
            r +=  (gij * log(qp_small[i-firsti+1, j-firstj+1, tid])) + ((twoT - gij) * log(oneT - qp_small[i-firsti+1, j-firstj+1, tid]))
            
        end
    end
    r
end


@inline function loglikelihood_loop_skipmissing(g::SnpLinAlg{T}, q, p, qp_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
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
            r += nonmissing * ((gij * log(qp_small[i-firsti+1, j-firstj+1, tid])) + ((twoT - gij) * log(oneT - qp_small[i-firsti+1, j-firstj+1, tid])))
        end
    end
    r
end

@inline function em_q_loop!(q_next, g::AbstractArray{T}, q, p, qp_small::AbstractArray{T}, 
    irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            for k in 1:K
                gij = g[i, j]
                q_next[k, i] += (gij * q[k, i] * p[k, j] / qp_small[i-firsti+1, j-firstj+1, tid] + 
                    (2 - gij) * q[k, i] * (1 - p[k, j]) / (1 - qp_small[i-firsti+1, j-firstj+1, tid]))
            end
        end
    end
end

"""
    em_q_loop_skipmissing!(q_next, g, q, p, qp_small, irange, jrange, K)
Compute EM update for `q`, compute local `qp` on-the-fly.

# Input
- `q_next`: result to be updated inplace.
- `g`: genotype matrix.
- `q`: Q matrix.
- `p`: P matrix.
- `qp_small`: dummy. 
- `irange`: i index range over which this function is applied
- `jrange`: j index range over which this function is applied
- `K`: number of clusters
"""
@inline function em_q_loop_skipmissing!(q_next, g::AbstractArray{T}, q, p, qp_small::AbstractArray{T}, 
    irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            for k in 1:K
                gij = g[i, j]
                nonmissing = (gij == gij)
                tmp = (gij * q[k, i] * p[k, j] / qp_small[i-firsti+1, j-firstj+1, tid] + 
                    (2 - gij) * q[k, i] * (1 - p[k, j]) / (1 - qp_small[i-firsti+1, j-firstj+1, tid]))
                q_next[k, i] += nonmissing ? tmp : zero(T)
            end
        end
    end
end

@inline function em_q_loop!(q_next, g::SnpLinAlg{T}, q, p, qp_small, irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
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
                q_next[k, i] += (gij * q[k, i] * p[k, j] / qp_small[i-firsti+1, j-firstj+1, tid] + 
                    (2 - gij) * q[k, i] * (1 - p[k, j]) / (1 - qp_small[i-firsti+1, j-firstj+1, tid]))
            end
        end
    end
end

@inline function em_q_loop_skipmissing!(q_next, g::SnpLinAlg{T}, q, p, qp_small::AbstractArray{T}, irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
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
                tmp = (gij * q[k, i] * p[k, j] / qp_small[i-firsti+1, j-firstj+1, tid] + 
                    (2 - gij) * q[k, i] * (1 - p[k, j]) / (1 - qp_small[i-firsti+1, j-firstj+1, tid]))
                q_next[k, i] += nonmissing * tmp
            end
        end
    end
end

@inline function em_p_loop!(f_next, g::AbstractArray{T}, q, p, p_tmp, qp_small::AbstractArray{T}, irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            for k in 1:K
                p_tmp[k, j] += gij * q[k, i] * p[k, j] / qp_small[i-firsti+1, j-firstj+1, tid]
                f_next[k, j] += (2 - gij) * q[k, i] * (one(T) - p[k, j]) / (one(T) - qp_small[i-firsti+1, j-firstj+1, tid])
            end
        end
    end
end

"""
    em_p_loop_skipmissing!(p_next, g, q, p, qp_small, irange, jrange, K)
Compute EM update for `p`, compute local `qp` on-the-fly.

# Input
- `p_next`: result to be updated inplace
- `g`: genotype matrix.
- `q`: Q matrix.
- `p`: P matrix.
- `qp_small`: dummy. 
- `irange`: i index range over which this function is applied
- `jrange`: j index range over which this function is applied
- `K`: number of clusters
"""
@inline function em_p_loop_skipmissing!(p_next, g::AbstractArray{T}, q, p, p_tmp, qp_small::AbstractArray{T}, irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            nonmissing = (gij == gij)
            for k in 1:K
                p_tmp[k, j] += nonmissing ? (gij * q[k, i] * p[k, j] / qp_small[i-firsti+1, j-firstj+1, tid]) : zero(T)
                p_next[k, j] += nonmissing ? ((2 - gij) * q[k, i] * (one(T) - p[k, j]) / (one(T) - qp_small[i-firsti+1, j-firstj+1, tid])) : zero(T)
            end
        end
    end
end

@inline function em_p_loop!(p_next, g::SnpLinAlg{T}, q, p, p_tmp, qp_small::AbstractArray{T}, irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
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
                p_tmp[k, j] += gij * q[k, i] * p[k, j] / qp_small[i-firsti+1, j-firstj+1, tid]
                p_next[k, j] += (2 - gij) * q[k, i] * (one(T) - p[k, j]) / (one(T) - qp_small[i-firsti+1, j-firstj+1, tid])
            end
        end
    end
end

@inline function em_p_loop_skipmissing!(f_next, g::SnpLinAlg{T}, q, p, p_tmp, qp_small::AbstractArray{T}, irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
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
                p_tmp[k, j] += nonmissing * (gij * q[k, i] * p[k, j] / qp_small[i-firsti+1, j-firstj+1, tid])
                f_next[k, j] += nonmissing * ((2 - gij) * q[k, i] * (one(T) - p[k, j]) / (one(T) - qp_small[i-firsti+1, j-firstj+1, tid]))
            end
        end
    end
end


@inline function update_q_loop!(XtX, Xtz, g::AbstractArray{T}, q, p, qp_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            for k in 1:K
                Xtz[k, i] += gij * p[k, j] / qp_small[i-firsti+1, j-firstj+1, tid] + 
                    (twoT - gij) * (oneT - p[k, j]) / (oneT - qp_small[i-firsti+1, j-firstj+1, tid])
                for k2 in 1:K
                    XtX[k2, k, i] +=  gij / (qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * p[k, j] * p[k2, j] + 
                        (twoT - gij) / (oneT - qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * (oneT - p[k, j]) * (oneT - p[k2, j])
                end
            end
        end
    end
end

"""
    update_q_loop_skipmissing!(XtX, Xtz, g, q, p, qp_small, irange, jrange, K)
Compute gradient and hessian of loglikelihood w.r.t. `q`, compute local `qp` on-the-fly.

# Arguments
- `XtX`: Hessian of loglikelihood computed inplace
- `Xtz`: Gradient of loglikelihood computed inplace
- `g`: genotype matrix.
- `q`: Q matrix.
- `p`: P matrix.
- `qp_small`: dummy. 
- `irange`: i index range over which this function is applied
- `jrange`: j index range over which this function is applied
- `K`: number of clusters
"""
@inline function update_q_loop_skipmissing!(XtX, Xtz, g::AbstractArray{T}, q, p, qp_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            nonmissing = (gij == gij)
            for k in 1:K
                Xtz[k, i] += nonmissing ? (gij * p[k, j] / qp_small[i-firsti+1, j-firstj+1, tid] + 
                    (twoT - gij) * (oneT - p[k, j]) / (oneT - qp_small[i-firsti+1, j-firstj+1, tid])) : zero(T)
                for k2 in 1:K
                    XtX[k2, k, i] += nonmissing ? (gij / (qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * p[k, j] * p[k2, j] + 
                        (twoT - gij) / (oneT - qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * (oneT - p[k, j]) * (oneT - p[k2, j])) : zero(T)
                end
            end
        end
    end
end

@inline function update_p_loop!(XtX, Xtz, g::AbstractArray{T}, q, p, qp_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            for k in 1:K
                Xtz[k, j] += gij * q[k, i] / qp_small[i-firsti+1, j-firstj+1, tid] - 
                        (twoT - gij) * q[k, i] / (oneT - qp_small[i-firsti+1, j-firstj+1, tid])
                for k2 in 1:K
                    XtX[k2, k, j] += gij / (qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i]
                end
            end
        end
    end
end

"""
    update_p_loop_skipmissing!(XtX, Xtz, g, q, p, qp_small, irange, jrange, K)
Compute gradient and hessian of loglikelihood w.r.t. `p`, compute local `qp` on-the-fly.

# Arguments
- `XtX`: Hessian of loglikelihood computed inplace
- `Xtz`: Gradient of loglikelihood computed inplace
- `g`: genotype matrix.
- `q`: Q matrix.
- `p`: P matrix.
- `qp_small`: dummy. 
- `irange`: i index range over which this function is applied
- `jrange`: j index range over which this function is applied
- `K`: number of clusters
"""
@inline function update_p_loop_skipmissing!(XtX, Xtz, g::AbstractArray{T}, q, p, qp_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
    @turbo for j in jrange
        for i in irange
            gij = g[i, j]
            nonmissing =  (gij == gij)
            for k in 1:K
                Xtz[k, j] += nonmissing ? (gij * q[k, i] / qp_small[i-firsti+1, j-firstj+1, tid] - 
                        (twoT - gij) * q[k, i] / (oneT - qp_small[i-firsti+1, j-firstj+1, tid])) : zero(T)
                for k2 in 1:K
                    XtX[k2, k, j] += nonmissing ? (gij / (qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i]) : zero(T)
                end
            end
        end
    end
end

@inline function update_q_loop!(XtX, Xtz, g::SnpLinAlg{T}, q, p, qp_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    gmat = g.s.data
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01] + (gij_pre == 0x01) * g.μ[j]
            for k in 1:K
                Xtz[k, i] += (gij * p[k, j] / qp_small[i-firsti+1, j-firstj+1, tid] + 
                    (twoT - gij) * (oneT - p[k, j]) / (oneT - qp_small[i-firsti+1, j-firstj+1, tid]))
                for k2 in 1:K
                    XtX[k2, k, i] +=  (gij / (qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * p[k, j] * p[k2, j] + 
                        (twoT - gij) / (oneT - qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * (oneT - p[k, j]) * (oneT - p[k2, j]))
                end
            end
        end
    end
end

@inline function update_q_loop_skipmissing!(XtX, Xtz, g::SnpLinAlg{T}, q, p, qp_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    gmat = g.s.data
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
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
                Xtz[k, i] += nonmissing * (gij * p[k, j] / qp_small[i-firsti+1, j-firstj+1, tid] + 
                    (twoT - gij) * (oneT - p[k, j]) / (oneT - qp_small[i-firsti+1, j-firstj+1, tid]))
                for k2 in 1:K
                    XtX[k2, k, i] += nonmissing * (gij / (qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * p[k, j] * p[k2, j] + 
                        (twoT - gij) / (oneT - qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * (oneT - p[k, j]) * (oneT - p[k2, j]))
                end
            end
        end
    end
end

function update_p_loop!(XtX, Xtz, g::SnpLinAlg{T}, q, p, qp_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    gmat = g.s.data
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    @turbo for j in jrange
        for i in irange
            blk = gmat[(i - 1) >> 2 + 1, j]
            re = (i - 1) % 4
            blk_shifted  = blk >> (re << 1)
            gij_pre = blk_shifted & 0x03
            gij = g_map[gij_pre + 0x01] + (gij_pre == 0x01) * g.μ[j]
            for k in 1:K
                Xtz[k, j] += (gij * q[k, i] / qp_small[i-firsti+1, j-firstj+1, tid] - 
                        (twoT - gij) * q[k, i] / (oneT - qp_small[i-firsti+1, j-firstj+1, tid]))
                for k2 in 1:K
                    XtX[k2, k, j] += (gij / (qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i])
                end
            end
        end
    end
end

function update_p_loop_skipmissing!(XtX, Xtz, g::SnpLinAlg{T}, q, p, qp_small::AbstractArray{T}, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    gmat = g.s.data
    firsti, firstj = first(irange), first(jrange)
    tid = threadid()
    qp_block!(qp_small, q, p, irange, jrange, K)
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
                Xtz[k, j] += nonmissing * (gij * q[k, i] / qp_small[i-firsti+1, j-firstj+1, tid] - 
                        (twoT - gij) * q[k, i] / (oneT - qp_small[i-firsti+1, j-firstj+1, tid]))
                for k2 in 1:K
                    XtX[k2, k, j] += nonmissing * (gij / (qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i] + 
                        (twoT - gij) / (oneT - qp_small[i-firsti+1, j-firstj+1, tid]) ^ 2 * q[k, i] * q[k2, i])
                end
            end
        end
    end
end
