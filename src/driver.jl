function run_admixture(filename, K; 
    rng=Random.GLOBAL_RNG, 
    sparsity=nothing, 
    prefix=filename[1:end-4],
    skfr_tries = 1, 
    skfr_max_inner_iter=50, 
    admix_n_iter=1000, 
    admix_rtol=1e-7, 
    admix_em_iters = 5, 
    T=Float64, 
    Q=3, 
    use_gpu=false)
    @assert endswith(filename, ".bed") "filename should end with .bed"
    if sparsity !== nothing
        admix_input, clusters, aims = _filter_SKFR(filename, K, sparsity; rng=rng, prefix=prefix, 
            tries=skfr_tries, max_inner_iter=skfr_max_inner_iter)
    else
        admix_input = filename
        clusters, aims = nothing, nothing
    end
    d = _admixture_base(admix_input, K; 
        n_iter=admix_n_iter, rtol=admix_rtol, rng=rng, em_iters=admix_em_iters, 
        T=T, Q=Q, use_gpu=use_gpu)
    d, clusters, aims
end

function _filter_SKFR(filename, K, sparsity::Integer; 
    rng=Random.GLOBAL_RNG,
    prefix=filename[1:end-4],
    tries = 10,
    max_inner_iter = 50)
    @assert endswith(filename, ".bed") "filename should end with .bed"
    g = SnpArray(filename)
    ISM = SKFR.ImputedSnpMatrix{Float64}(g, K; rng=rng)
    if tries == 1
        (clusters, _, aims, _, _) = SKFR.sparsekmeans1(ISM, sparsity; max_iter = max_inner_iter, squares=false)
    else
        (clusters, _, aims, _, _, _) = SKFR.sparsekmeans_repeat(ISM, sparsity; iter = tries, max_inner_iter=max_inner_iter)
    end
    I, J = size(g)
    aims_sorted = sort(aims)
    des = "$(prefix)_$(K)_$(sparsity)aims"
    println(des)
    SnpArrays.filter(filename[1:end-4], trues(I), aims_sorted; des=des)
    des * ".bed", clusters, aims
end

function _filter_SKFR(filename, K, sparsities::AbstractVector{<:Integer}; 
    rng=Random.GLOBAL_RNG,
    prefix=filename[1:end-4],
    tries = 10,
    max_inner_iter = 50)
    @assert endswith(filename, ".bed") "filename should end with .bed"
    g = SnpArray(filename)
    ISM = SKFR.ImputedSnpMatrix{Float64}(g, K; rng=rng)
    if typeof(sparsities) <: AbstractVector
        @assert issorted(sparsities; rev=true) "sparsities should be decreasing"
        (clusters, aims) = SKFR.sparsekmeans_path(ISM, sparsities; iter=tries, max_inner_iter=max_inner_iter)
    end
    I, J = size(g)
    outputfiles = String[]
    for (s, aimlist) in zip(sparsities, aims)
        aimlist_sorted = sort(aimlist)
        des = "$(prefix)_$(K)_$(s)aims"
        SnpArrays.filter(filename[1:end-4], trues(I), aimlist_sorted; des=des)
        push!(outputfiles, des)
    end
    outputfiles, clusters, aims
end

function _admixture_base(filename, K; 
    n_iter=1000, 
    rtol=1e-7, 
    rng=Random.GLOBAL_RNG,
    em_iters = 5, 
    T=Float64, 
    Q=3, 
    use_gpu=false)
    g = SnpArray(filename)
    g_la = SnpLinAlg{T}(g)
    I = size(g_la, 1)
    J = size(g_la, 2)
    d = AdmixData{T}(I, J, K, Q; skipmissing=true, rng=rng)
    if use_gpu
        d_cu = AdmiXpress.CuAdmixData(d, g_la)
        Ibytes = (I + 3) ÷ 4
        g_cu = CuArray{UInt8, 2}(undef, Ibytes, J)
        copyto!(g_cu, g.data)
    else 
        d_cu = nothing
        g_cu = nothing
    end
    @time init_em!(d, g_la, em_iters; d_cu = d_cu, g_cu=g_cu)
    @time admixture_qn!(d, g_la, n_iter, rtol; d_cu = d_cu, g_cu = g_cu, mode=:ZAL)
    d
end
