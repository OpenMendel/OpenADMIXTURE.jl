# K = 1 (a single population, as accepted by ADMIXTURE): no admixture, so the
# maximum-likelihood estimates are analytic. Every q_i1 = 1 and p_j is the
# observed frequency of the modelled allele, clamped like `project_p!`.

"""
    OnePopFit

Result of [`fit_one_population`](@ref): `q` (1 x I, all ones), `p` (1 x J) and
the log-likelihood `ll_new`, the fields used from an `AdmixData`.
"""
struct OnePopFit
    q::Matrix{Float64}
    p::Matrix{Float64}
    ll_new::Float64
end

"""
    fit_one_population(g::SnpArray; pseudocount=1e-5)

Maximum-likelihood fit for K = 1. Missing genotypes are skipped; a SNP with no
observed genotype gets p = 0.5.
"""
function fit_one_population(g::SnpArray; pseudocount::Real=1e-5)
    I, J = size(g)
    p = zeros(Float64, 1, J)
    llj = zeros(Float64, J)
    Threads.@threads for j in 1:J
        a, n = 0, 0
        for i in 1:I
            code = g[i, j]
            code == 0x01 && continue
            a += _allele_count(code)
            n += 1
        end
        f = clamp(n == 0 ? 0.5 : a / 2n, pseudocount, 1 - pseudocount)
        p[1, j] = f
        llj[j] = a * log(f) + (2n - a) * log(1 - f)
    end
    OnePopFit(ones(Float64, 1, I), p, sum(llj))
end

# Fit used by the command line, cross-validation and bootstrap: analytic for K = 1
# (the iterative options do not apply), the quasi-Newton algorithm otherwise.
_fit(g::SnpArray, K::Integer; kwargs...) =
    K == 1 ? fit_one_population(g) : _admixture_base(g, K; kwargs...)
