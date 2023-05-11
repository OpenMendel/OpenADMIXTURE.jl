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
function init_em!(d::AdmixData{T}, g::AbstractArray{T}, iter::Integer;
                  d_cu=nothing, g_cu=nothing, verbose=false) where T
    # qf!(d.qf, d.q, d.f)
    if d_cu !== nothing
        copyto_sync!([d_cu.q, d_cu.p], [d.q, d.p])
    end
    d_ = (d_cu === nothing) ? d : d_cu
    g_ = (g_cu === nothing) ? g : g_cu
    println("Performing $iter EM steps for priming")
    for i in 1:iter
        if verbose
            println("Initialization EM Iteration $(i)")
        end
        t = @timed begin
            em_q!(d_, g_; verbose=verbose)
            em_p!(d_, g_; verbose=verbose)
            d_.p .= d_.p_next
            d_.q .= d_.q_next
        end
        if d_cu !== nothing
            copyto_sync!([d.p, d.q], [d_cu.p, d_cu.q])
            ll = loglikelihood(d_cu, g_cu)
        else
            ll = loglikelihood(g, d.q, d.p, d.qp_small, d.K, d.skipmissing)
        end
        println("EM Iteration $(i) ($(t.time) sec): Loglikelihood = $ll")
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
- `verbose`: Print verbose timing information like original script.
- `progress_bar`: Show progress bar while executing.
"""
function admixture_qn!(d::AdmixData{T}, g::AbstractArray{T}, iter::Int=1000, 
    rtol= 1e-7; d_cu=nothing, g_cu=nothing, mode=:ZAL, iter_count_offset=0,
    verbose=false, progress_bar=false) where T
    # qf!(d.qf, d.q, d.f)
    # ll_prev = loglikelihood(g, d.q, d.f, d.qp_small, d.K, d.skipmissing)
    # d.ll_new = ll_prev
    
    bgpartial = ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇']
    p = Progress(iter, dt=0.5, desc="Running main algorithm",
        barglyphs=BarGlyphs('|','█', bgpartial,' ','|'),
        barlen=50, showspeed=true, enabled=progress_bar);
    fspec = FormatSpec(".4e")

    if isnan(d.ll_new)
        if d_cu !== nothing
            copyto_sync!([d_cu.p, d_cu.q], [d.p, d.q])
            d.ll_new = loglikelihood(d_cu, g_cu)
        else
            d.ll_new = loglikelihood(g, d.q, d.p, d.qp_small, d.K, d.skipmissing)
        end
    end

    println("initial ll: ", d.ll_new)
    if !progress_bar
        println("Starting main algorithm")
    end
    llhist = [d.ll_new]
    converged = false
    i = 0
    loopstats = @timed for outer i = (iter_count_offset + 1):iter
        iterinfo = @timed begin
            # qf!(d.qf, d.q, d.f)
            # ll_prev = loglikelihood(g, d.qf)
            d.ll_prev = d.ll_new
            update_q!(d, g; d_cu=d_cu, g_cu=g_cu, verbose=verbose)
            update_p!(d, g; d_cu=d_cu, g_cu=g_cu, verbose=verbose)
            update_q!(d, g, true; d_cu=d_cu, g_cu=g_cu, verbose=verbose)
            update_p!(d, g, true; d_cu=d_cu, g_cu=g_cu, verbose=verbose)

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
            (prev = d.ll_prev, basic = ll_basic, qn = ll_qn, new = d.ll_new,
             reldiff = abs((d.ll_new - d.ll_prev) / d.ll_prev))
        end
        lls = iterinfo[1]
        println("Iteration $i ($(iterinfo.time) sec): " *
                "LogLikelihood = $(lls.new), reldiff = $(lls.reldiff)")
        ll_other = "Previous = $(lls.prev) QN = $(lls.qn), Basic = $(lls.basic)"
        println("    LogLikelihoods: $ll_other")
        if verbose
            println("\n")
        end
        last_vals = map((x) -> fmt(fspec, x),
                        llhist[end-min(length(llhist)-1, 5):end])
        ProgressMeter.next!(p;
            showvalues = [
                (:INFO,"Percent is for max $iter iterations. " *
                    "Likely to converge early at LL ↗ of $rtol."),
                (:Iteration,i),
                (Symbol("Execution time"),iterinfo.time),
                (Symbol("Initial LogLikelihood"),llhist[1]),
                (Symbol("Current LogLikelihood"),lls.new),
                (Symbol("Other LogLikelihoods"),ll_other),
                (Symbol("LogLikelihood ↗"),lls.reldiff),
                (Symbol("Past LogLikelihoods"),last_vals)])
        push!(llhist, lls.new)
        if lls.reldiff < rtol
            converged = true
            ProgressMeter.finish!(p)
            break
        end
    end
    if converged
        println("Main algorithm converged in $i iterations over $(loopstats.time) sec.")
    else
        println("Main algorithm failed to converge after $i iterations over $(loopstats.time) sec.")
    end
end
