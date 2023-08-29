"""
    loglikelihood(g, q, p, qp_small, K, skipmissing)
Compute loglikelihood.

# Input 
- `g`: genotype matrix 
- `q`: Q matrix
- `p`: P matrix
- `qp_small`: cache memory for storing partial QP matrix.
- `K`: number of clusters
- `skipmissing`: must be `true` in most cases
"""
function loglikelihood(g::AbstractArray{T}, q, p, qp_small::AbstractArray{T, 3}, K, skipmissing) where T
    I = size(q, 2)
    J = size(p, 2)
    #K = 0 # dummy
    # r = tiler_scalar_1d(loglikelihood_loop, typeof(qp_small), zero(T), (g, q, p, qp_small), 1:I, 1:J, K)
    if !skipmissing
        r = loglikelihood_loop(g, q, p, nothing, 1:I, 1:J, K)
    else
        r = loglikelihood_loop_skipmissing(g, q, p, nothing, 1:I, 1:J, K)
    end
    # r = tiler_scalar(loglikelihood_loop, typeof(qp_small), zero(T), (g, q, p, qp_small), 1:I, 1:J, K)
    r
end

function loglikelihood(g::SnpLinAlg{T}, q, p, qp_small::AbstractArray{T, 3}, K, skipmissing) where T
    I = size(q, 2)
    J = size(p, 2)
    #K = 0 # dummy
    r = threader_scalar(skipmissing ? loglikelihood_loop_skipmissing : loglikelihood_loop, 
        typeof(qp_small), zero(Float64), (g, q, p, qp_small), 1:I, 1:J, K)
    # r = loglikelihood_loop(g, q, p, nothing, 1:I, 1:J, K)
    r
end

# function loglikelihood(g::AbstractArray{T}, q, p, qp_small, K, tmp) where T
#     I = size(q, 2)
#     J = size(p, 2)
#     #K = 0 # dummy
#     # r = tiler_scalar_1d(loglikelihood_loop, typeof(qp_small), zero(T), (g, q, p, qp_small), 1:I, 1:J, K)
#     r = loglikelihood_loop(g, q, p, nothing, 1:I, 1:J, K, tmp)
#     r
# end

# function loglikelihood(g::AbstractArray{T}, qf) where T
#     I = size(qp, 1)
#     J = size(qp, 2)
#     K = 0 # dummy
#     r = loglikelihood_loop(g, qp, 1:I, 1:J, K)
#     r
# end

"""
    em_q!(d, g, mode=:base, verbose=false)
FRAPPE EM algorithm for updaing Q.
Assumes qp to be pre-computed.

# Input
- `d`: an `AdmixData`.
- `g`: a genotype matrix.
- `mode`: selects which member of `d` is used.. 
    - `:base` or `:fallback`: `q_next` is updated based on  `q` and `p`.
    - `:fallback2`: `q_next2` is updated based on `q_next` and `p_next`.
- `verbose`: whether to print timings (boolean)
"""
# function em_q!(q_new, g::AbstractArray{T}, q, p, qf) where T # summation over j, block I side.
function em_q!(d::AdmixData{T}, g::AbstractArray{T}; mode=:base, verbose=false) where T
    I, J, K = d.I, d.J, d.K
    qp_small = d.qp_small
    if mode == :base || mode == :fallback
        q_next, q, p = d.q_next, d.q, d.p
    elseif mode == :fallback2
        q_next, q, p = d.q_next2, d.q_next, d.p_next
    end
    fill!(q_next, zero(T))
    if verbose
        @time threader!(d.skipmissing ? em_q_loop_skipmissing! : em_q_loop!,
                        typeof(q_next), (q_next, g, q, p, qp_small),
                        1:I, 1:J, K, true; maxL=64)
    else
        threader!(d.skipmissing ? em_q_loop_skipmissing! : em_q_loop!,
                        typeof(q_next), (q_next, g, q, p, qp_small),
                        1:I, 1:J, K, true; maxL=64)
    end
    # @time tiler!(d.skipmissing ? em_q_loop_skipmissing! : em_q_loop!, 
    #     typeof(q_next), (q_next, g, q, p, qp_small), 1:I, 1:J, K)
    # @time em_q_loop!(q_next, g, q, p, qp, 1:I, 1:J, K)
    q_next ./= 2J
    # @tullio q_new[k, i] = (g[i, j] * q[k, i] * f[k, j] / qf[i, j] + 
    #     (2 - g[i, j]) * q[k, i] * (1 - f[k, j]) / (1 - qf[i, j])) / 2J
end

"""
    em_p!(d, g, mode=:base, verbose=false)
FRAPPE EM algorithm for updaing P.
Assumes qp to be pre-computed.

# Input
- `d`: an `AdmixData`.
- `g`: a genotype matrix.
- `mode`: selects which member of `d` is used.. 
    - `:base`: `p_next` is updated based on  `q` and `p`.
    - `:fallback`: `p_next` is updated based on `q_next` and `p`.
    - `:fallback2`: `p_next2` is updated based on `q_next` and `p_next`.
- `verbose`: whether to print timings (boolean)
"""
function em_p!(d::AdmixData{T}, g::AbstractArray{T}; mode=:base, verbose=false) where T 
    # summation over i: block J side.
    I, J, K = d.I, d.J, d.K
    p_tmp, qp_small = d.p_tmp, d.qp_small
    if mode == :base
        p_next, p, q = d.p_next, d.p, d.q
    elseif mode == :fallback
        p_next, p, q = d.p_next, d.p, d.q_next
    elseif mode == :fallback2
        p_next, p, q = d.p_next2, d.p_next, d.q_next2
    end
    fill!(p_tmp, zero(T))
    fill!(p_next, zero(T))
    if verbose
        @time threader!(d.skipmissing ? em_p_loop_skipmissing! : em_p_loop!, 
                        typeof(p_next), (p_next, g, q, p, p_tmp, qp_small),
                        1:I, 1:J, K, false; maxL=64)
    else
        threader!(d.skipmissing ? em_p_loop_skipmissing! : em_p_loop!, 
                        typeof(p_next), (p_next, g, q, p, p_tmp, qp_small),
                        1:I, 1:J, K, false; maxL=64)
    end
    # @time tiler!(d.skipmissing ? em_f_loop_skipmissing! : em_f_loop!, 
    #     typeof(p_next), (p_next, g, q, p, f_tmp, qp_small), 1:I, 1:J, K)
    # @time em_f_loop!(p_next, g, q, p, f_tmp, qp_small, 1:I, 1:J, K)
    @turbo for j in 1:J
        for k in 1:K
            p_next[k, j] = p_tmp[k, j] / (p_tmp[k, j] + p_next[k, j])
        end
    end
    # @tullio f_tmp[k, j] = g[i, j] * q[k, i] * f[k, j] / qf[i, j]
    # @tullio f_new[k, j] = (2 - g[i, j]) * q[k, i] * (1 - f[k, j]) / (1 - qf[i, j])
    # @tullio f_new[k, j] = f_tmp[k, j] / (f_tmp[k, j] + f_new[k, j])
end

const tau_schedule = collect(0.7^i for i in 1:10)

function print_timed(timed_results, verbose::Bool)
    if verbose
        allocs = timed_results.gcstats.malloc + timed_results.gcstats.realloc +
                 timed_results.gcstats.poolalloc + timed_results.gcstats.bigalloc
        allocd = timed_results.gcstats.allocd
        seconds = timed_results.time
        println("$seconds seconds ($allocs allocations: $allocd bytes)")
    end
end

"""
    update_q!(d, g, update2=false; d_cu=nothing, g_cu=nothing, verbose=false)
Update Q using sequential quadratic programming.

# Input
- `d`: an `AdmixData`.
- `g`: a genotype matrix.
- `update2`: if running the second update for quasi-Newton step
- `d_cu`: a `CuAdmixData` if using GPU, `nothing` otherwise.
- `g_cu`: a `CuMatrix{UInt8}` corresponding to the data part of
- `verbose`: whether to print timings (boolean)
"""
function update_q!(d::AdmixData{T}, g::AbstractArray{T}, update2=false;
                   d_cu=nothing, g_cu=nothing, verbose=false) where T
# function update_q!(q_next, g::AbstractArray{T}, q, p, qdiff, XtX, Xtz, qf) where T
    I, J, K = d.I, d.J, d.K
    qdiff, XtX, Xtz, qp_small = d.q_tmp, d.XtX_q, d.Xtz_q, d.qp_small
    q_next = update2 ? d.q_next2 : d.q_next
    q      = update2 ? d.q_next  : d.q
    p      = update2 ? d.p_next  : d.p
    qv     = update2 ? d.q_nextv : d.qv
    qdiffv = d.q_tmpv
    XtXv   = d.XtX_qv
    Xtzv   = d.Xtz_qv

    # qf!(qp, q, f)
    d.ll_prev = d.ll_new # loglikelihood(g, qf)
    if verbose
        println(d.ll_prev)
    end
    a = @timed if d_cu === nothing # CPU operation
        fill!(XtX, zero(T))
        fill!(Xtz, zero(T))
        threader!(d.skipmissing ? update_q_loop_skipmissing! : update_q_loop!, 
            typeof(XtX), (XtX, Xtz, g, q, p, qp_small), 1:I, 1:J, K, true; maxL=16)
    else # GPU operation
        @assert d.skipmissing "`skipmissing`` must be true for GPU computation"
        copyto_sync!([d_cu.q, d_cu.p], [q, p])
        update_q_cuda!(d_cu, g_cu)
        copyto_sync!([XtX, Xtz], [d_cu.XtX_q, d_cu.Xtz_q])
    end

    print_timed(a, verbose)

    # Solve the quadratic programs
    b = @timed begin
        Xtz .*= -1 
        pmin = zeros(T, K)
        pmax = ones(T, K)

        @batch threadlocal=QPThreadLocal{T}(K) for i in 1:I
            # even the views are allocating something, so we use preallocated views.
            XtX_ = XtXv[i]
            Xtz_ = Xtzv[i]
            q_ = qv[i]
            qdiff_ = qdiffv[i]

            tableau_k2 = threadlocal.tableau_k2
            tmp_k = threadlocal.tmp_k
            tmp_k2 = threadlocal.tmp_k2
            tmp_k2_ = threadlocal.tmp_k2_
            swept = threadlocal.swept
            
            create_tableau!(tableau_k2, XtX_, Xtz_, q_, d.v_kk, tmp_k, true)
            quadratic_program!(qdiff_, tableau_k2, q_, pmin, pmax, K, 1, 
                tmp_k2, tmp_k2_, swept)
        end
        @turbo for i in 1:I
            for k in 1:K
                q_next[k, i] = q[k, i] + qdiff[k, i]
            end
        end
        project_q!(q_next, d.idxv[1])
    end

    print_timed(b, verbose)
end

"""
    update_p!(d, g, update2=false; d_cu=nothing, g_cu=nothing, verbose=false)
Update P with sequential quadratic programming.

# Input
- `d`: an `AdmixData`.
- `g`: a genotype matrix.
- `update2`: if running the second update for quasi-Newton step
- `d_cu`: a `CuAdmixData` if using GPU, `nothing` otherwise.
- `g_cu`: a `CuMatrix{UInt8}` corresponding to the data part of
- `verbose`: whether to print timings (boolean)
"""
function update_p!(d::AdmixData{T}, g::AbstractArray{T},
                   update2=false; d_cu=nothing, g_cu=nothing,
                   verbose=false) where T
# function update_p!(p_next, g::AbstractArray{T}, p, q, fdiff, XtX, Xtz, qf) where T
    I, J, K = d.I, d.J, d.K
    pdiff, XtX, Xtz, qp_small = d.p_tmp, d.XtX_p, d.Xtz_p, d.qp_small
    p_next = update2 ? d.p_next2 : d.p_next
    q      = update2 ? d.q_next2 : d.q_next
    p      = update2 ? d.p_next  : d.p
    pv     = update2 ? d.p_nextv : d.pv
    pdiffv = d.p_tmpv
    XtXv   = d.XtX_pv
    Xtzv   = d.Xtz_pv

    d.ll_prev = d.ll_new 
    a = @timed if d_cu === nothing # CPU operation
        fill!(XtX, zero(T))
        fill!(Xtz, zero(T))

        threader!(d.skipmissing ? update_p_loop_skipmissing! : update_p_loop!, 
            typeof(XtX), (XtX, Xtz, g, q, p, qp_small), 1:I, 1:J, K, false; maxL=16)
    else # GPU operation
        @assert d.skipmissing "`skipmissing`` must be true for GPU computation"
        copyto_sync!([d_cu.q, d_cu.p], [q, p])
        update_p_cuda!(d_cu, g_cu)
        copyto_sync!([XtX, Xtz], [d_cu.XtX_p, d_cu.Xtz_p])
    end

    print_timed(a, verbose)

    # solve quadratic programming problems.
    b = @timed begin       
        Xtz .*= -1 
        pmin = zeros(T, K)
        pmax = ones(T, K)
        
        @batch threadlocal=QPThreadLocal{T}(K) for j in 1:J
            # even the views are allocating something, so we use preallocated views.
            XtX_ = XtXv[j]
            Xtz_ = Xtzv[j]
            p_ = pv[j]
            pdiff_ = pdiffv[j]

            t = threadid()
            tableau_k1 = threadlocal.tableau_k1
            tmp_k = threadlocal.tmp_k
            tmp_k1 = threadlocal.tmp_k1
            tmp_k1_ = threadlocal.tmp_k1_
            swept = threadlocal.swept

            create_tableau!(tableau_k1, XtX_, Xtz_, p_, d.v_kk, tmp_k, false)
            quadratic_program!(pdiff_, tableau_k1, p_, pmin, pmax, K, 0,
                tmp_k1, tmp_k1_, swept)
        end
        @turbo for j in 1:J
            for k in 1:K
                p_next[k, j] = p[k, j] + pdiff[k, j]
            end
        end
        # p_next .= f .+ fdiff
        project_p!(p_next)
        # println(maximum(abs.(fdiff)))
    end

    print_timed(b, verbose)
end
