"""
    init_em!(d, g, iter; d_cu=nothing, g_cu=nothing)
Initialize P and Q with the FRAPPE EM algorithm

# Input
- `d`: an `AdmixData`.
- `g`: a genotype matrix.
- `iter`: number of iterations.
- `d_cu`: a `CuAdmixData` if using GPU, `nothing` otherwise.
- `g_cu`: a `CuMatrix{UInt8}` corresponding to the data part of 
"""
function init_em!(d::AdmixData{T}, g::AbstractArray{T}, iter::Integer; d_cu=nothing, g_cu=nothing) where T
    # qf!(d.qf, d.q, d.f)
    if d_cu !== nothing
        copyto_sync!([d_cu.q, d_cu.p], [d.q, d.p])
    end
    d_ = (d_cu === nothing) ? d : d_cu
    g_ = (g_cu === nothing) ? g : g_cu
    for _ in 1:iter
        em_q!(d_, g_)
        em_p!(d_, g_)
        d_.p .= d_.p_next
        d_.q .= d_.q_next
    end
    if d_cu !== nothing
        copyto_sync!([d.p, d.q], [d_cu.p, d_cu.q])
        d.ll_new = loglikelihood(d_cu, g_cu)
    else
        d.ll_new = loglikelihood(g, d.q, d.p, d.qp_small, d.K, d.skipmissing)
    end
end

"""
    admixture_qn!(d, g, iter=1000, rtol=1e-7; d_cu=nothing, g_cu=nothing, 
        mode=:ZAL, iter_count_offset=0)
Initialize P and Q with the FRAPPE EM algorithm

# Input
- `d`: an `AdmixData`.
- `g`: a genotype matrix.
- `iter`: number of iterations.
- `rtol`: convergence tolerance in terms of relative change of loglikelihood.
- `d_cu`: a `CuAdmixData` if using GPU, `nothing` otherwise.
- `g_cu`: a `CuMatrix{UInt8}` corresponding to the data part of genotype matrix
- `mode`: `:ZAL` for Zhou-Alexander-Lange acceleration (2009), `:LBQN` for Agarwal-Xu (2020). 
"""
function admixture_qn!(d::AdmixData{T}, g::AbstractArray{T}, iter::Int=1000, 
    rtol= 1e-7; d_cu=nothing, g_cu=nothing, mode=:ZAL, iter_count_offset=0, fix_p=false, fix_q=false) where T
    # qf!(d.qf, d.q, d.f)
    # ll_prev = loglikelihood(g, d.q, d.f, d.qp_small, d.K, d.skipmissing)
    # d.ll_new = ll_prev
    
    if isnan(d.ll_new)
        if d_cu !== nothing
            copyto_sync!([d_cu.p, d_cu.q], [d.p, d.q])
            d.ll_new = loglikelihood(d_cu, g_cu)
        else
            d.ll_new = loglikelihood(g, d.q, d.p, d.qp_small, d.K, d.skipmissing)
        end
    end

    println("initial ll: ", d.ll_new)
    for i in (iter_count_offset + 1):iter
        @time begin
            # qf!(d.qf, d.q, d.f)
            # ll_prev = loglikelihood(g, d.qf)
            d.ll_prev = d.ll_new
            if !fix_q
                update_q!(d, g; d_cu=d_cu, g_cu=g_cu)
            end
            if !fix_p
                update_p!(d, g; d_cu=d_cu, g_cu=g_cu)
            end
            if !fix_q
                update_q!(d, g, true; d_cu=d_cu, g_cu=g_cu)
            end
            if !fix_p
                update_p!(d, g, true; d_cu=d_cu, g_cu=g_cu)
            end

            # qf!(d.qf, d.q_next2, d.f_next2)
            ll_basic = if d_cu !== nothing
                copyto_sync!([d_cu.p, d_cu.q], [d.p_next2, d.q_next2])
                loglikelihood(d_cu, g_cu)
            else
                loglikelihood(g, d.q_next2, d.p_next2, d.qp_small, d.K, d.skipmissing)
            end
            
            if mode == :ZAL
                update_UV!(d.U, d.V, d.x_flat, d.x_next_flat, d.x_next2_flat, i, d.Q)
                U_part = i < d.Q ? view(d.U, :, 1:i) : view(d.U, :, :)
                V_part = i < d.Q ? view(d.V, :, 1:i) : view(d.V, :, :)

                update_QN!(d.x_tmp_flat, d.x_next_flat, d.x_flat, U_part, V_part)
            elseif mode == :LBQN
                update_UV_LBQN!(d.U, d.V, d.x_flat, d.x_next_flat, d.x_next2_flat, i, d.Q)
                U_part = i < d.Q ? view(d.U, :, 1:i) : view(d.U, :, :)
                V_part = i < d.Q ? view(d.V, :, 1:i) : view(d.V, :, :)
                update_QN_LBQN!(d.x_tmp_flat, d.x_flat, d.x_qq, d.x_rr, U_part, V_part)
            else 
                @assert false "Invalid mode"
            end

            project_p!(d.p_tmp)
            project_q!(d.q_tmp, d.idxv[1])
            # qf!(d.qf, d.q_tmp, d.f_tmp)
            ll_qn = if d_cu !== nothing # GPU mode
                copyto_sync!([d_cu.p, d_cu.q], [d.p_tmp, d.q_tmp])
                loglikelihood(d_cu, g_cu)
            else # CPU mode
                loglikelihood(g, d.q_tmp, d.p_tmp, d.qp_small, d.K, d.skipmissing)
            end
            if d.ll_prev < ll_qn
                d.x .= d.x_tmp
                d.ll_new = ll_qn
            else
                d.x .= d.x_next2
                d.ll_new = ll_basic
            end
            println(d.ll_prev)
            println(ll_basic)
            println(ll_qn)
            reldiff = abs((d.ll_new - d.ll_prev) / d.ll_prev)
            println("Iteration $i: ll = $(d.ll_new), reldiff = $reldiff")
            if reldiff < rtol
                break
            end
        end
        println()
        println()
    end
end
