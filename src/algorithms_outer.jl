function init_em!(d::AdmixData{T}, g::AbstractArray{T}, iter::Integer; d_cu=nothing) where T
    # qf!(d.qf, d.q, d.f)
    if d_cu !== nothing
        copyto_sync!([d_cu.q, d_cu.f], [d.q, d.f])
    end
    d_ = (d_cu === nothing) ? d : d_cu
    for _ in 1:iter
        em_q!(d_, g)
        em_f!(d_, g)
        d_.f .= d_.f_next
        d_.q .= d_.q_next
    end
    if d_cu !== nothing
        copyto_sync!([d.f, d.q], [d_cu.f, d_cu.q])
        d.ll_new = loglikelihood(d_cu, g)
    else
        d.ll_new = loglikelihood(g, d.q, d.f, d.qf_small, d.K, d.skipmissing)
    end
end

function admixture_qn!(d::AdmixData{T}, g::AbstractArray{T}, iter::Int=30, 
    rtol= 1e-7; d_cu=nothing) where T
    # qf!(d.qf, d.q, d.f)
    # ll_prev = loglikelihood(g, d.q, d.f, d.qf_small, d.K, d.skipmissing)
    # d.ll_new = ll_prev
    
    if isnan(d.ll_new)
        if d_cu !== nothing
            copyto_sync!([d_cu.f, d_cu.q], [d.f, d.q])
            d.ll_new = loglikelihood(d_cu, g)
        else
            d.ll_new = loglikelihood(g, d.q, d.f, d.qf_small, d.K, d.skipmissing)
        end
    end

    println("initial ll: ", d.ll_new)
    for i in 1:iter
        @time begin
            # qf!(d.qf, d.q, d.f)
            # ll_prev = loglikelihood(g, d.qf)
            d.ll_prev = d.ll_new
            update_q!(d, g; d_cu=d_cu)
            update_f!(d, g; d_cu=d_cu)
            update_q!(d, g, true; d_cu=d_cu)
            update_f!(d, g, true; d_cu=d_cu)

            # qf!(d.qf, d.q_next2, d.f_next2)
            ll_basic = if d_cu !== nothing
                copyto_sync!([d_cu.f, d_cu.q], [d.f_next2, d.q_next2])
                loglikelihood(d_cu, g)
            else
                loglikelihood(g, d.q_next2, d.f_next2, d.qf_small, d.K, d.skipmissing)
            end
            
            update_UV!(d.U, d.V, d.x_flat, d.x_next_flat, d.x_next2_flat, i, d.Q)
            U_part = i < d.Q ? view(d.U, :, 1:i) : view(d.U, :, :)
            V_part = i < d.Q ? view(d.V, :, 1:i) : view(d.V, :, :)
            update_QN!(d.x_tmp_flat, d.x_next_flat, d.x_flat, U_part, V_part)
            project_f!(d.f_tmp)
            project_q!(d.q_tmp, d.idx)
            # qf!(d.qf, d.q_tmp, d.f_tmp)
            ll_qn = if d_cu !== nothing
                copyto_sync!([d_cu.f, d_cu.q], [d.f_tmp, d.q_tmp])
                loglikelihood(d_cu, g)
            else
                loglikelihood(g, d.q_tmp, d.f_tmp, d.qf_small, d.K, d.skipmissing)
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
