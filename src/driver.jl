"""
    run_admixture(filename, K;
        rng=Random.GLOBAL_RNG, 
        sparsity=nothing,
        prefix=filename[1:end-4],
        skfr_tries = 1,
        skfr_max_inner_iter = 50,
        admix_n_iter = 1000,
        admix_rtol=1e-7,
        admix_n_em_iter = 5,
        T = Float64,
        Q = 3,
        use_gpu=false,
        verbose=false)

The main runner function for admixture. 

# Input: 
- `filename``: the PLINK BED file name to analyze, including the extension.
- `K`: number of clusters.
- `rng`: random number generator.
- `sparsity`: number of AIMs to be utilized. `nothing` to not run the SKFR step.
- `prefix`: prefix used for the output PLINK file if SKFR is used.
- `skfr_tries`: number of repeats of SKFR with different initializations.
- `skfr_max_inner_iter`: maximum number of iterations for each call for SKFR.
- `admix_n_iter`: number of Admixture iterations
- `admix_rtol`: relative tolerance for Admixture
- `admix_n_em_iters`: number of iterations for EM initialization
- `T`: Internal type for floating-point numbers
- `Q`: number of steps used for quasi-Newton acceleration
- `use_gpu`: whether to use GPU for computation
"""
function run_admixture(filename, K; 
    rng=Random.GLOBAL_RNG, 
    sparsity=nothing, 
    prefix=filename[1:end-4],
    skfr_tries = 1, 
    skfr_max_inner_iter=50,
    skfr_mode=:global, 
    admix_n_iter=1000, 
    admix_rtol=1e-7, 
    admix_n_em_iters = 5, 
    T=Float64, 
    Q=3, 
    use_gpu=false,
    verbose=false)
    @assert endswith(filename, ".bed") "filename should end with .bed"
    if sparsity !== nothing
        ftn = if skfr_mode == :global
            SKFR.sparsekmeans1
        elseif skfr_mode == :local
            SKFR.sparsekmeans2
        else
            @assert false "skfr_mode can only be :global or :local"
        end
        admix_input, clusters, aims = _filter_SKFR(filename, K, sparsity; rng=rng, prefix=prefix, 
            tries=skfr_tries, max_inner_iter=skfr_max_inner_iter, ftn=ftn)
    else
        admix_input = filename
        clusters, aims = nothing, nothing
    end
    d = _admixture_base(admix_input, K; 
        n_iter=admix_n_iter, rtol=admix_rtol, rng=rng, em_iters=admix_n_em_iters, 
        T=T, Q=Q, use_gpu=use_gpu, verbose=verbose)
    d, clusters, aims
end

"""
Run SKFR then filter the PLINK file to only keep AIMs.
"""
function _filter_SKFR(filename, K, sparsity::Integer; 
    rng=Random.GLOBAL_RNG,
    prefix=filename[1:end-4],
    tries = 10,
    max_inner_iter = 50, 
    ftn = SKFR.sparsekmeans1)
    @assert endswith(filename, ".bed") "filename should end with .bed"
    g = SnpArray(filename)
    ISM = SKFR.ImputedSnpMatrix{Float64}(g, K; rng=rng)
    if tries == 1
        (clusters, _, aims, _, _) = ftn(ISM, sparsity; max_iter = max_inner_iter, squares=false)
    else
        (clusters, _, aims, _, _, _) = SKFR.sparsekmeans_repeat(ISM, sparsity; iter = tries, 
            max_inner_iter=max_inner_iter, ftn=ftn)
    end
    I, J = size(g)
    aims_sorted = sort(aims)
    des = "$(prefix)_$(K)_$(sparsity)aims"
    println(des)
    SnpArrays.filter(filename[1:end-4], trues(I), aims_sorted; des=des)
    des * ".bed", clusters, aims
end

"""
Run SKFR then filter the PLINK file to only keep AIMs. Run for multiple sparsities (in decreasing order)
"""
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
    use_gpu=false,
    verbose=false)
    g = SnpArray(filename)
    g_la = SnpLinAlg{T}(g)
    I = size(g_la, 1)
    J = size(g_la, 2)
    d = AdmixData{T}(I, J, K, Q; skipmissing=true, rng=rng)
    if use_gpu
        d_cu, g_cu = _cu_admixture_base(d, g_la, I, J)
    else 
        d_cu = nothing
        g_cu = nothing
    end
    if verbose
        @time init_em!(d, g_la, em_iters;
                       d_cu = d_cu, g_cu=g_cu, verbose=verbose)
        @time admixture_qn!(d, g_la, n_iter, rtol;
                            d_cu = d_cu, g_cu = g_cu, mode=:ZAL,
                            verbose=verbose)
    else
        init_em!(d, g_la, em_iters; d_cu = d_cu, g_cu=g_cu)
        admixture_qn!(d, g_la, n_iter, rtol;
                      d_cu = d_cu, g_cu = g_cu, mode=:ZAL)
    end
    d
end

function _cu_admixture_base(d, g_la, I, J)
    # dummy, main body defined inside CUDA portion.
end