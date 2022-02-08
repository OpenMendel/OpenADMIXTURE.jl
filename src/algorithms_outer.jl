function init_em!(d::AdmixData{T}, g::AbstractArray{T}, iter::Integer) where T
    # qf!(d.qf, d.q, d.f)
    for _ in 1:iter
        # println(likelihood(g_numeric, f, q, qf))
        # qf!(d.qf, d.q, d.f)
        em_q!(d, g)
        em_f!(d, g)
        d.f .= d.f_next
        d.q .= d.q_next
        # qf!(d.qf, d.q, d.f)
        # println(loglikelihood(g, d.qf))
    end
end

function admixture_qn!(d::AdmixData{T}, g::AbstractArray{T}, iter::Int=30, rtol=1e-7) where T
    # qf!(d.qf, d.q, d.f)
    ll_prev = loglikelihood(g, d.q, d.f, d.qf_small, d.K)
    d.ll_new = ll_prev
    println(ll_prev)
    for i in 1:iter
        @time begin
            # qf!(d.qf, d.q, d.f)
            # ll_prev = loglikelihood(g, d.qf)
            ll_prev = d.ll_new
            update_q!(d, g)
            update_f!(d, g)
            update_q!(d, g, true)
            update_f!(d, g, true)

            # qf!(d.qf, d.q_next2, d.f_next2)
            ll_basic = d.ll_new #loglikelihood(g, d.qf)
            
            update_UV!(d.U, d.V, d.x_flat, d.x_next_flat, d.x_next2_flat, i, d.Q)
            U_part = i < d.Q ? view(d.U, :, 1:i) : view(d.U, :, :)
            V_part = i < d.Q ? view(d.V, :, 1:i) : view(d.V, :, :)
            update_QN!(d.x_tmp_flat, d.x_next_flat, d.x_flat, U_part, V_part)
            project_f!(d.f_tmp)
            project_q!(d.q_tmp, d.idx)
            # qf!(d.qf, d.q_tmp, d.f_tmp)
            ll_qn = loglikelihood(g, d.q_tmp, d.f_tmp, d.qf_small, d.K)
            if ll_prev < ll_qn
                d.x .= d.x_tmp
                d.ll_new = ll_qn
            else
                d.x .= d.x_next2
                d.ll_new = ll_basic
            end
            println(ll_prev)
            println(ll_basic)
            println(ll_qn)
            reldiff = abs((d.ll_new - ll_prev) / ll_prev)
            println("Iteration $i: ll = $(d.ll_new), reldiff = $reldiff")
            if reldiff < rtol
                break
            end
        end
        println()
        println()
    end
end
