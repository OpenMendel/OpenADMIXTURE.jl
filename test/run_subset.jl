# Helper run in a child Julia process so the test suite can control the thread layout
# (e.g. `-t 4` vs `-t 4,0`). Prints the final log-likelihood on the last line.
using OpenADMIXTURE, StableRNGs

bed, K = ARGS[1], parse(Int, ARGS[2])
d, _, _ = OpenADMIXTURE.run_admixture(bed, K; rng=StableRNG(1), progress_bar=false)
q = Array(d.q)
@assert all(isfinite, q) "non-finite Q"
@assert all(x -> isapprox(x, 1; atol=1e-6), sum(q; dims=1)) "Q columns do not sum to 1"
println("LL=", d.ll_new)
