using OpenAdmixture, SnpArrays, Random
using Test

@testset "OpenAdmixture.jl" begin
    EUR = SnpArrays.datadir("EUR_subset.bed")
    rng = MersenneTwister(7856)
    d, _, _ = OpenAdmixture.run_admixture(EUR, 4; rng=rng)
    @test d.ll_new ≈ -1.555219392972111e7
    rng = MersenneTwister(7856)
    d, _, _ = OpenAdmixture.run_admixture(EUR, 4; sparsity=1000, rng=rng, prefix="test")
    @test d.ll_new ≈ -216519.78193892565
end
