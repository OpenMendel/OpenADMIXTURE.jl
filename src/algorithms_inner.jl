"""
    loglikelihood(g, qf)
Compute loglikelihood.
"""
function loglikelihood(g::AbstractArray{T}, q, f, qf_small::AbstractArray{T, 3}, K, skipmissing) where T
    I = size(q, 2)
    J = size(f, 2)
    #K = 0 # dummy
    # r = tiler_scalar_1d(loglikelihood_loop, typeof(qf_small), zero(T), (g, q, f, qf_small), 1:I, 1:J, K)
    if !skipmissing
        r = loglikelihood_loop(g, q, f, nothing, 1:I, 1:J, K)
    else
        r = loglikelihood_loop_skipmissing(g, q, f, nothing, 1:I, 1:J, K)
    end
    # r = tiler_scalar(loglikelihood_loop, typeof(qf_small), zero(T), (g, q, f, qf_small), 1:I, 1:J, K)
    r
end

function loglikelihood(g::SnpLinAlg{T}, q, f, qf_small::AbstractArray{T, 3}, K, skipmissing) where T
    I = size(q, 2)
    J = size(f, 2)
    #K = 0 # dummy
    r = tiler_scalar(skipmissing ? loglikelihood_loop_skipmissing : loglikelihood_loop, 
        typeof(qf_small), zero(Float64), (g, q, f, qf_small), 1:I, 1:J, K)
    # r = loglikelihood_loop(g, q, f, nothing, 1:I, 1:J, K)
    r
end

# function loglikelihood(g::AbstractArray{T}, q, f, qf_small, K, tmp) where T
#     I = size(q, 2)
#     J = size(f, 2)
#     #K = 0 # dummy
#     # r = tiler_scalar_1d(loglikelihood_loop, typeof(qf_small), zero(T), (g, q, f, qf_small), 1:I, 1:J, K)
#     r = loglikelihood_loop(g, q, f, nothing, 1:I, 1:J, K, tmp)
#     r
# end

# function loglikelihood(g::AbstractArray{T}, qf) where T
#     I = size(qf, 1)
#     J = size(qf, 2)
#     K = 0 # dummy
#     r = loglikelihood_loop(g, qf, 1:I, 1:J, K)
#     r
# end

"""
FRAPPE EM algorithm
Assumes qf to be pre-computed.
"""
# function em_q!(q_new, g::AbstractArray{T}, q, f, qf) where T # summation over j, block I side.
function em_q!(d::AdmixData{T}, g::AbstractArray{T}, mode=:base) where T
    I, J, K = d.I, d.J, d.K
    qf_small = d.qf_small
    if mode == :base || mode == :fallback
        q_next, q, f = d.q_next, d.q, d.f
    elseif mode == :fallback2
        q_next, q, f = d.q_next2, d.q_next, d.f_next
    end
    fill!(q_next, zero(T))
    @time threader!(d.skipmissing ? em_q_loop_skipmissing! : em_q_loop!,
        typeof(q_next), (q_next, g, q, f, qf_small), 1:I, 1:J, K, true; maxL=64)
    # @time tiler!(d.skipmissing ? em_q_loop_skipmissing! : em_q_loop!, 
    #     typeof(q_next), (q_next, g, q, f, qf_small), 1:I, 1:J, K)
    # @time em_q_loop!(q_next, g, q, f, qf, 1:I, 1:J, K)
    q_next ./= 2J
    # @tullio q_new[k, i] = (g[i, j] * q[k, i] * f[k, j] / qf[i, j] + 
    #     (2 - g[i, j]) * q[k, i] * (1 - f[k, j]) / (1 - qf[i, j])) / 2J
end

function em_f!(d::AdmixData{T}, g::AbstractArray{T}, mode=:base) where T 
    # summation over i: block J side.
    I, J, K = d.I, d.J, d.K
    f_tmp, qf_small = d.f_tmp, d.qf_small
    if mode == :base
        f_next, f, q = d.f_next, d.f, d.q
    elseif mode == :fallback
        f_next, f, q = d.f_next, d.f, d.q_next
    elseif mode == :fallback2
        f_next, f, q = d.f_next2, d.f_next, d.q_next2
    end
    fill!(f_tmp, zero(T))
    fill!(f_next, zero(T))

    @time threader!(d.skipmissing ? em_f_loop_skipmissing! : em_f_loop!, 
        typeof(f_next), (f_next, g, q, f, f_tmp, qf_small), 1:I, 1:J, K, false; maxL=64)
    # @time tiler!(d.skipmissing ? em_f_loop_skipmissing! : em_f_loop!, 
    #     typeof(f_next), (f_next, g, q, f, f_tmp, qf_small), 1:I, 1:J, K)
    # @time em_f_loop!(f_next, g, q, f, f_tmp, qf_small, 1:I, 1:J, K)
    @turbo for j in 1:J
        for k in 1:K
            f_next[k, j] = f_tmp[k, j] / (f_tmp[k, j] + f_next[k, j])
        end
    end
    # @tullio f_tmp[k, j] = g[i, j] * q[k, i] * f[k, j] / qf[i, j]
    # @tullio f_new[k, j] = (2 - g[i, j]) * q[k, i] * (1 - f[k, j]) / (1 - qf[i, j])
    # @tullio f_new[k, j] = f_tmp[k, j] / (f_tmp[k, j] + f_new[k, j])
end

const tau_schedule = collect(0.7^i for i in 1:10)

function update_q!(d::AdmixData{T}, g::AbstractArray{T}, update2=false; d_cu=nothing, g_cu=nothing) where T
# function update_q!(q_next, g::AbstractArray{T}, q, f, qdiff, XtX, Xtz, qf) where T
    I, J, K = d.I, d.J, d.K
    qdiff, XtX, Xtz, qf_small = d.q_tmp, d.XtX_q, d.Xtz_q, d.qf_small
    q_next = update2 ? d.q_next2 : d.q_next
    q      = update2 ? d.q_next  : d.q
    f      = update2 ? d.f_next  : d.f
    qv     = update2 ? d.q_nextv : d.qv
    qdiffv = d.q_tmpv
    XtXv   = d.XtX_qv
    Xtzv   = d.Xtz_qv

    # qf!(qf, q, f)
    d.ll_prev = d.ll_new # loglikelihood(g, qf)
    println(d.ll_prev)
    @time if d_cu === nothing
        fill!(XtX, zero(T))
        fill!(Xtz, zero(T))
        threader!(d.skipmissing ? update_q_loop_skipmissing! : update_q_loop!, 
            typeof(XtX), (XtX, Xtz, g, q, f, qf_small), 1:I, 1:J, K, true; maxL=16)
        # tiler!(d.skipmissing ? update_q_loop_skipmissing! : update_q_loop!, 
        #     typeof(XtX), (XtX, Xtz, g, q, f, qf_small), 1:I, 1:J, K)
        #update_q_loop!(XtX, Xtz, g, f, qf, 1:I, 1:J, K)
    else
        @assert d.skipmissing "`skipmissing`` must be true for GPU computation"
        copyto_sync!([d_cu.q, d_cu.f], [q, f])
        update_q_cuda!(d_cu, g_cu)
        copyto_sync!([XtX, Xtz], [d_cu.XtX_q, d_cu.Xtz_q])
    end

    # println(Xtz)
    @time begin
        # for QP formulation
        Xtz .*= -1 
        # matrix_a = ones(T, 1, K)
        # b = [one(T)]
        pmin = zeros(T, K)
        pmax = ones(T, K)
        # XtX_SA = reinterpret(reshape, SMatrix{K,K,Float64, K*K}, reshape(XtX, K*K, :))
        # Xtz_SA = reinterpret(reshape, SVector{K, Float64}, Xtz)
        # q_SA   = reinterpret(reshape, SVector{K, Float64}, q)
        # qdiff_SA   = reinterpret(reshape, SVector{K, Float64}, qdiff)

        # XtX_ = @MMatrix zeros(T, K, K)
        # Xtz_ = @MVector zeros(T, K)
        # q_   = @MVector zeros(T, K)
        # qdiff_ = @MVector zeros(T, K)
        @threads for i in 1:I
            # XtX_ = unsafe_wrap(Array{T,2}, pointer(XtX, (i-1) * K * K + 1), (K,K))
            XtX_ = XtXv[i]#view(XtX, :, :, i)
            # Xtz_ = unsafe_wrap(Array{T, 1}, pointer(Xtz, (i-1) * K + 1), (K,))
            Xtz_ = Xtzv[i]#view(Xtz, :, i)
            # q_   = unsafe_wrap(Array{T, 1}, pointer(q, (i-1) * K + 1), (K,))
            q_ = qv[i]#view(q, :, i)
            # qdiff_ = unsafe_wrap(Array{T, 1}, pointer(qdiff, (i-1) * K + 1), (K,))
            qdiff_ = qdiffv[i]#view(qdiff, :, i)

            t = threadid()
            tableau_k2 = d.tableau_k2v[t]
            tmp_k = d.tmp_kv[t]
            tmp_k2 = d.tmp_k2v[t]
            tmp_k2_ = d.tmp_k2_v[t]
            swept = d.sweptv[t]
            
            create_tableau!(tableau_k2, XtX_, Xtz_, q_, d.v_kk, tmp_k, true)
            # tableau = create_tableau(XtX_, Xtz_, matrix_a, b, q_)
            quadratic_program!(qdiff_, tableau_k2, q_, pmin, pmax, K, 1, 
                tmp_k2, tmp_k2_, swept)
        end
        @turbo for i in 1:I
            for k in 1:K
                q_next[k, i] = q[k, i] + qdiff[k, i]
            end
        end
        # q_next .= q .+ qdiff
        project_q!(q_next, d.idxv[1])
    end

    # Line serach for step size

    # begin
    #     # qf!(qf, q_next, f)
    #     @time d.ll_new = loglikelihood(g, q_next, f, qf_small, K, d.skipmissing)
    #     println("update_q: ", d.ll_new)
    #     tau = 1.0
    #     for cnt in 1:length(tau_schedule)
    #         if d.ll_prev < d.ll_new
    #             return
    #         end
    #         println(cnt)
    #         tau = tau_schedule[cnt]
    #         q_next .= q .+ tau .* qdiff
    #         project_q!(q_next, d.idx)
    #         # qf!(qf, q_next, f)
    #         d.ll_new = loglikelihood(g, q_next, f, qf_small, K, d.skipmissing)
    #     end
    #     # println(maximum(abs.(qdiff)))
    #     println("Update failed. Falling back to EM update.")
    #     # qf!(qf, q, f)
    #     em_q!(d, g, update2 ? :fallback2 : :fallback)
    #     project_q!(q_next, d.idx)
    #     # qf!(qf, q_next, f)
    #     d.ll_new = loglikelihood(g, q_next, f, qf_small, K, d.skipmissing)
    #     return
    # end
end

function update_f!(d::AdmixData{T}, g::AbstractArray{T}, update2=false; d_cu=nothing, g_cu=nothing) where T
# function update_f!(f_next, g::AbstractArray{T}, f, q, fdiff, XtX, Xtz, qf) where T
    I, J, K = d.I, d.J, d.K
    fdiff, XtX, Xtz, qf_small = d.f_tmp, d.XtX_f, d.Xtz_f, d.qf_small
    f_next = update2 ? d.f_next2 : d.f_next
    q      = update2 ? d.q_next2 : d.q_next
    f      = update2 ? d.f_next  : d.f
    fv     = update2 ? d.f_nextv : d.fv
    fdiffv = d.f_tmpv
    XtXv   = d.XtX_fv
    Xtzv   = d.Xtz_fv


    # qf!(qf, q, f)
    d.ll_prev = d.ll_new # loglikelihood(g, qf)
    @time if d_cu === nothing
        fill!(XtX, zero(T))
        fill!(Xtz, zero(T))
        # @time tiler_1d!(update_f_loop!, typeof(XtX), (XtX, Xtz, g, q, f, qf_thin), 1:I, 1:J, K)
        threader!(d.skipmissing ? update_f_loop_skipmissing! : update_f_loop!, 
            typeof(XtX), (XtX, Xtz, g, q, f, qf_small), 1:I, 1:J, K, false; maxL=16)
        # tiler!(d.skipmissing ? update_f_loop_skipmissing! : update_f_loop!, 
        #     typeof(XtX), (XtX, Xtz, g, q, f, qf_small), 1:I, 1:J, K)
        # @time update_f_loop!(XtX, Xtz, g, q, qf, 1:I, 1:J, K)
    else
        @assert d.skipmissing "`skipmissing`` must be true for GPU computation"
        copyto_sync!([d_cu.q, d_cu.f], [q, f])
        update_f_cuda!(d_cu, g_cu)
        copyto_sync!([XtX, Xtz], [d_cu.XtX_f, d_cu.Xtz_f])
    end

    @time begin       
        Xtz .*= -1 
        # matrix_a = ones(T, 0, 0)
        # b = ones(T, 0)
        pmin = zeros(T, K)
        pmax = ones(T, K)
        
        @threads for j in 1:J
            # XtX_ = unsafe_wrap(Array{T, 2}, pointer(XtX, (j-1) * K * K + 1), (K,K))
            XtX_ = XtXv[j]# view(XtX, :, :, j)
            # Xtz_ = unsafe_wrap(Array{T, 1}, pointer(Xtz, (j-1) * K + 1), (K,))
            Xtz_ = Xtzv[j]#view(Xtz, :, j)
            # f_ = unsafe_wrap(Array{T, 1}, pointer(f, (j-1) * K + 1), (K,))
            f_ = fv[j] #view(f, :, j)
            # fdiff_ = unsafe_wrap(Array{T, 1}, pointer(fdiff, (j-1) * K + 1), (K,))
            fdiff_ = fdiffv[j] #view(fdiff, :, j)

            t = threadid()
            tableau_k1 = d.tableau_k1v[t]
            tmp_k = d.tmp_kv[t]
            tmp_k1 = d.tmp_k1v[t]
            tmp_k1_ = d.tmp_k1_v[t]
            swept = d.sweptv[t]

            create_tableau!(tableau_k1, XtX_, Xtz_, f_, d.v_kk, tmp_k, false)
            # tableau = create_tableau(XtX_, Xtz_, matrix_a, b, f_)
            # println(diag(tableau))
            quadratic_program!(fdiff_, tableau_k1, f_, pmin, pmax, K, 0,
                tmp_k1, tmp_k1_, swept) 
            # fdiff_ .= fd     
     
        end
        @turbo for j in 1:J
            for k in 1:K
                f_next[k, j] = f[k, j] + fdiff[k, j]
            end
        end
        # f_next .= f .+ fdiff
        project_f!(f_next)
        # println(maximum(abs.(fdiff)))
    end
    
    # Line serach for step size

    # begin
    #     tau = 1.0
    #     # qf!(qf, q, f_next)
    #     @time d.ll_new = loglikelihood(g, q, f_next, qf_small, K, d.skipmissing)
    #     println("update_f: ", d.ll_new)
    #     for cnt in 1:length(tau_schedule)
    #         if d.ll_prev < d.ll_new
    #             return
    #         end
    #         println(cnt)
    #         tau = tau_schedule[cnt]
    #         f_next .= f .+ tau .* fdiff
    #         project_f!(f_next)
    #         # qf!(qf, q, f_next)
    #         d.ll_new = loglikelihood(g, q, f_next, qf_small, K, d.skipmissing)
    #     end
    #     println(maximum(abs.(fdiff)))
    #     println("Update failed. Falling back to EM update.")
    #     # qf!(qf, q, f)
    #     em_f!(d, g, update2 ? :fallback2 : :fallback)
    #     project_f!(f_next)
    #     # qf!(qf, q, f_next)
    #     d.ll_new = loglikelihood(g, q, f_next, qf_small, K, d.skipmissing)
    #     return 
    # end
end
