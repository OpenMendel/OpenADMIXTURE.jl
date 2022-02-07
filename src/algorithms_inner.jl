"""
    loglikelihood(g, qf)
Compute loglikelihood.
"""
function loglikelihood(g::AbstractArray{T}, q, f, qf_small, K) where T
    I = size(q, 2)
    J = size(f, 2)
    #K = 0 # dummy
    # r = tiler_scalar_1d(loglikelihood_loop, typeof(qf_small), zero(T), (g, q, f, qf_small), 1:I, 1:J, K)
    r = loglikelihood_loop(g, q, f, nothing, 1:I, 1:J, K)
    r
end

function loglikelihood(g::AbstractArray{T}, q, f, qf_small, K, tmp) where T
    I = size(q, 2)
    J = size(f, 2)
    #K = 0 # dummy
    # r = tiler_scalar_1d(loglikelihood_loop, typeof(qf_small), zero(T), (g, q, f, qf_small), 1:I, 1:J, K)
    r = loglikelihood_loop(g, q, f, nothing, 1:I, 1:J, K, tmp)
    r
end

function loglikelihood(g::AbstractArray{T}, qf) where T
    I = size(qf, 1)
    J = size(qf, 2)
    K = 0 # dummy
    r = loglikelihood_loop(g, qf, 1:I, 1:J, K)
    r
end

"""
FRAPPE EM algorithm
Assumes qf to be pre-computed.
"""
# function em_q!(q_new, g::AbstractArray{T}, q, f, qf) where T # summation over j, block I side.
function em_q!(d::AdmixData{T}, g::AbstractArray{T}, mode=:base) where T
    I, J, K = d.I, d.J, d.K
    qf = d.qf
    if mode == :base || mode == :fallback
        q_next, q, f = d.q_next, d.q, d.f
    elseif mode == :fallback2
        q_next, q, f = d.q_next2, d.q_next, d.f_next
    end
    fill!(q_next, zero(T))
    @time tiler!(em_q_loop!, typeof(q_next), (q_next, g, q, f, qf), 1:I, 1:J, K)
    # @time em_q_loop!(q_next, g, q, f, qf, 1:I, 1:J, K)
    q_next ./= 2J
    # @tullio q_new[k, i] = (g[i, j] * q[k, i] * f[k, j] / qf[i, j] + 
    #     (2 - g[i, j]) * q[k, i] * (1 - f[k, j]) / (1 - qf[i, j])) / 2J
end

function em_f!(d::AdmixData{T}, g::AbstractArray{T}, mode=:base) where T 
    # summation over i: block J side.
    I, J, K = d.I, d.J, d.K
    f_tmp, qf = d.f_tmp, d.qf
    if mode == :base
        f_next, f, q = d.f_next, d.f, d.q
    elseif mode == :fallback
        f_next, f, q = d.f_next, d.f, d.q_next
    elseif mode == :fallback2
        f_next, f, q = d.f_next2, d.f_next, d.q_next2
    end
    fill!(f_tmp, zero(T))
    fill!(f_next, zero(T))

    # @time tiler!(em_f_loop!, T, (f_next, g, q, f, f_tmp, qf), 1:I, 1:J, K)
    @time em_f_loop!(f_next, g, q, f, f_tmp, qf, 1:I, 1:J, K)
    @turbo for j in 1:J
        for k in 1:K
            f_next[k, j] = f_tmp[k, j] / (f_tmp[k, j] + f_next[k, j])
        end
    end
    # @tullio f_tmp[k, j] = g[i, j] * q[k, i] * f[k, j] / qf[i, j]
    # @tullio f_new[k, j] = (2 - g[i, j]) * q[k, i] * (1 - f[k, j]) / (1 - qf[i, j])
    # @tullio f_new[k, j] = f_tmp[k, j] / (f_tmp[k, j] + f_new[k, j])
end

const tau_schedule = [collect(0.7^i for i in 1:10); 0.01; 0.001; 0.0001; 
    0.00001; 0.000001; 0.0000001; 0.00000001]

function update_q!(d::AdmixData{T}, g::AbstractArray{T}, update2=false) where T
# function update_q!(q_next, g::AbstractArray{T}, q, f, qdiff, XtX, Xtz, qf) where T
    I, J, K = d.I, d.J, d.K
    qdiff, XtX, Xtz, qf, qf_small = d.q_tmp, d.XtX_q, d.Xtz_q, d.qf, d.qf_small
    q_next = update2 ? d.q_next2 : d.q_next
    q      = update2 ? d.q_next  : d.q
    f      = update2 ? d.f_next  : d.f

    # qf!(qf, q, f)
    d.ll_prev = d.ll_new # loglikelihood(g, qf)
    println(d.ll_prev)
    fill!(XtX, zero(T))
    fill!(Xtz, zero(T))
    # @time tiler!(update_q_loop!, T, (XtX, Xtz, g, f, qf), 1:I, 1:J, K)
    @time tiler!(update_q_loop!, typeof(XtX), (XtX, Xtz, g, q, f, qf_small), 1:I, 1:J, K)
    #update_q_loop!(XtX, Xtz, g, f, qf, 1:I, 1:J, K)

    @time begin
        # for QP formulation
        Xtz .*= -1 
        matrix_a = ones(T, 1, K)
        b = [one(T)]
        pmin = zeros(T, K)
        pmax = ones(T, K)
        @inbounds  for i in 1:I
            XtX_ = view(XtX, :, :, i)
            Xtz_ = view(Xtz, :, i)
            q_ = view(q, :, i)
            qdiff_ = view(qdiff, :, i)
            
            tableau = create_tableau(XtX_, Xtz_, matrix_a, b, q_)
            itr, qd = quadratic_program!(qdiff_, tableau, q_, pmin, pmax, K, 1) 
            # qdiff_ .= qd
        end
        q_next .= q .+ qdiff
        project_q!(q_next, d.idx)
    end

    begin
        qf!(qf, q_next, f)
        @time d.ll_new = loglikelihood(g, qf)
        println("update_q: ", d.ll_new)
        tau = 1.0
        for cnt in 1:length(tau_schedule)
            if d.ll_prev < d.ll_new
                return
            end
            println(cnt)
            tau = tau_schedule[cnt]
            q_next .= q .+ tau .* qdiff
            project_q!(q_next, d.idx)
            qf!(qf, q_next, f)
            d.ll_new = loglikelihood(g, qf)
        end
        println(maximum(abs.(qdiff)))
        println("Update failed. Falling back to EM update.")
        qf!(qf, q, f)
        em_q!(d, g, update2 ? :fallback2 : :fallback)
        project_q!(q_next, d.idx)
        qf!(qf, q_next, f)
        d.ll_new = loglikelihood(g, qf)
        return
    end
end

function update_f!(d::AdmixData{T}, g::AbstractArray{T}, update2=false) where T
# function update_f!(f_next, g::AbstractArray{T}, f, q, fdiff, XtX, Xtz, qf) where T
    I, J, K = d.I, d.J, d.K
    fdiff, XtX, Xtz, qf, qf_small = d.f_tmp, d.XtX_f, d.Xtz_f, d.qf, d.qf_small
    f_next = update2 ? d.f_next2 : d.f_next
    q      = update2 ? d.q_next2 : d.q_next
    f      = update2 ? d.f_next  : d.f

    # qf!(qf, q, f)
    d.ll_prev = d.ll_new # loglikelihood(g, qf)
    fill!(XtX, zero(T))
    fill!(Xtz, zero(T))
    # @time tiler_1d!(update_f_loop!, typeof(XtX), (XtX, Xtz, g, q, f, qf_thin), 1:I, 1:J, K)
    @time tiler!(update_f_loop!, typeof(XtX), (XtX, Xtz, g, q, f, qf_small), 1:I, 1:J, K)
    # @time update_f_loop!(XtX, Xtz, g, q, qf, 1:I, 1:J, K)
    # not tiling f loop performs better... 
    # maybe memory access pattern for f update is pretty, and we have overhead for tiling.

    @time begin       
        Xtz .*= -1 
        matrix_a = ones(T, 0, 0)
        b = ones(T, 0)
        pmin = zeros(T, K)
        pmax = ones(T, K)
        
        @inbounds for j in 1:J
            XtX_ = view(XtX, :, :, j)
            Xtz_ = view(Xtz, :, j)
            f_ = view(f, :, j)
            fdiff_ = view(fdiff, :, j)
            
            tableau = create_tableau(XtX_, Xtz_, matrix_a, b, f_)
            # println(diag(tableau))
            itr, fd = quadratic_program!(fdiff_, tableau, f_, pmin, pmax, K, 0) 
            # fdiff_ .= fd     
     
        end
        f_next .= f .+ fdiff
        project_f!(f_next)
        println(maximum(abs.(fdiff)))
    end
    

    begin
        tau = 1.0
        qf!(qf, q, f_next)
        @time d.ll_new = loglikelihood(g, qf)
        println("update_f: ", d.ll_new)
        for cnt in 1:length(tau_schedule)
            if d.ll_prev < d.ll_new
                return
            end
            println(cnt)
            tau = tau_schedule[cnt]
            f_next .= f .+ tau .* fdiff
            project_f!(f_next)
            qf!(qf, q, f_next)
            d.ll_new = loglikelihood(g, qf)
        end
        println(maximum(abs.(fdiff)))
        println("Update failed. Falling back to EM update.")
        qf!(qf, q, f)
        em_f!(d, g, update2 ? :fallback2 : :fallback)
        project_f!(f_next)
        qf!(qf, q, f_next)
        d.ll_new = loglikelihood(g, qf)
        return 
    end
end
