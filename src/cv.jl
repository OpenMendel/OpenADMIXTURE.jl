# ADMIXTURE-compatible cross-validation (ADMIXTURE 1.3 manual, section 2.1.1;
# Alexander & Lange 2011, BMC Bioinformatics 12:246).
#
# The observed genotypes are split into `v` folds. Each fold in turn is masked
# (set to missing), the model is re-estimated on the masked data starting from
# the full-data estimates, and each masked genotype n_ij is predicted by
# μ_ij = 2 Σ_k q_ik p_kj. The CV error is the mean of d(n_ij, μ_ij) over all
# masked entries of all folds.

"""
    binomial_deviance(n, μ)

ADMIXTURE manual eq. (1): `d(n, μ) = n log(n/μ) + (2 - n) log((2 - n)/(2 - μ))`,
with the convention `0 log 0 = 0`.
"""
function binomial_deviance(n::Real, μ::Real)
    z = zero(float(μ))
    a = n == 0 ? z : n * log(n / μ)
    b = n == 2 ? z : (2 - n) * log((2 - n) / (2 - μ))
    a + b
end

@inline function _splitmix64(x::UInt64)
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    x ⊻ (x >> 31)
end

"""
    cv_fold(i, j, v, seed)

Fold (in `1:v`) of genotype `(i, j)`. A hash of `(seed, i, j)`, so the
assignment needs no memory and is reproducible.
"""
@inline function cv_fold(i::Integer, j::Integer, v::Integer, seed::Integer)
    h = _splitmix64(_splitmix64(UInt64(seed) ⊻ (UInt64(i) * 0x9e3779b97f4a7c15)) + UInt64(j))
    Int(h % UInt64(v)) + 1
end

"""
    mask_fold(g, fold, v, seed)

In-memory copy of `g` where the genotypes of `fold` are set to missing.
"""
function mask_fold(g::SnpArray, fold::Integer, v::Integer, seed::Integer)
    I, J = size(g)
    m = SnpArray(undef, I, J)
    copyto!(m.data, g.data)
    # one column per iteration: columns are stored in separate bytes
    Threads.@threads for j in 1:J
        for i in 1:I
            if cv_fold(i, j, v, seed) == fold
                m[i, j] = 0x01
            end
        end
    end
    m
end

# PLINK code -> count of the allele modelled by P (same coding as `g_map_Float64`)
@inline _allele_count(code::UInt8) = code == 0x00 ? 0 : code == 0x02 ? 1 : 2

"""
    cv_error(g, K, q0, p0; folds=5, seed=43, quiet=false, fit_kwargs...)

ADMIXTURE cross-validation error for `K` populations. `q0` (K x I) and `p0`
(K x J) are the full-data estimates, used as starting point of each fold's fit.
`fit_kwargs` are passed to the fit (e.g. `atol`, `n_iter`, `Q`).
"""
function cv_error(g::SnpArray, K::Integer, q0::AbstractMatrix, p0::AbstractMatrix;
                  folds::Integer=5, seed::Integer=43, quiet::Bool=false, fit_kwargs...)
    folds >= 2 || throw(ArgumentError("cross-validation needs at least 2 folds"))
    I, J = size(g)
    dev = zeros(Float64, J)  # per-SNP accumulators: no shared writes across threads
    cnt = zeros(Int, J)
    for f in 1:folds
        masked = mask_fold(g, f, folds, seed)
        fitfold() = _admixture_base(masked, K; init_q=copy(q0), init_p=copy(p0),
            rng=StableRNG(seed + f), progress_bar=false, fit_kwargs...)
        d = quiet ? (@suppress_out fitfold()) : fitfold()
        q, p = d.q, d.p
        Threads.@threads for j in 1:J
            s, c = 0.0, 0
            for i in 1:I
                cv_fold(i, j, folds, seed) == f || continue
                code = g[i, j]
                code == 0x01 && continue  # missing in the data: nothing to predict
                μ = 0.0
                for k in 1:K
                    μ += q[k, i] * p[k, j]
                end
                s += binomial_deviance(_allele_count(code), 2μ)
                c += 1
            end
            dev[j] += s
            cnt[j] += c
        end
    end
    sum(dev) / sum(cnt)
end
