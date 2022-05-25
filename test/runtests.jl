using OpenAdmixture, SnpArrays
using Test

@testset "OpenAdmixture.jl" begin
    EUR = SnpArrays.datadir("EUR_subset.bed")
    d = OpenAdmixture.run_admixture(EUR, 4; sparsity=10000)
end
