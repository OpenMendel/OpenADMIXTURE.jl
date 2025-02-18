using OpenADMIXTURE, SnpArrays, Random, StableRNGs
using Test

@testset "OpenADMIXTURE.jl" begin
    EUR = SnpArrays.datadir("EUR_subset.bed")
    rng = StableRNG(7856)
    d, _, _ = OpenADMIXTURE.run_admixture(EUR, 4; rng=rng)
    @test d.ll_new ≈ -1.555219448773069e7
    rng = StableRNG(7856)
    d, _, _ = OpenADMIXTURE.run_admixture(EUR, 4; sparsity=1000, rng=rng, prefix="test")
    @test d.ll_new ≈ -235919.4556341082
    rm("test_4_1000aims.bed", force=true)
    rm("test_4_1000aims.bim", force=true)
    rm("test_4_1000aims.fam", force=true)
end
