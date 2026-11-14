using OpenADMIXTURE, SnpArrays, Random, StableRNGs
using Test

const EUR = SnpArrays.datadir("EUR_subset.bed")

# Write a small PLINK subset of EUR (first `nsnp` SNPs) to `dir`; return the .bed path.
function eur_subset(dir, nsnp; monomorphic_first=false)
    s = SnpArray(EUR)
    prefix = joinpath(dir, "sub")
    SnpArrays.filter(EUR[1:end-4], trues(size(s, 1)), (1:size(s, 2)) .<= nsnp; des=prefix)
    if monomorphic_first
        sub = SnpArray(prefix * ".bed", size(s, 1), "r+")
        sub[:, 1] .= 0x00  # every sample homozygous A1 -> monomorphic SNP (shared mmap)
        @assert all(==(0x00), SnpArray(prefix * ".bed")[:, 1])
    end
    prefix * ".bed"
end

# Run test/run_subset.jl in a child process with the given `-t` specification.
function run_child(bed, K, threads)
    script = joinpath(@__DIR__, "run_subset.jl")
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -t $threads $script $bed $K`
    out = read(pipeline(cmd; stderr=devnull), String)
    parse(Float64, split(last(split(strip(out), '\n')), "=")[2])
end

@testset "OpenADMIXTURE" begin
@testset "EUR reference values" begin
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

@testset "monomorphic SNPs give a finite likelihood" begin
    mktempdir() do dir
        bed = eur_subset(dir, 2000; monomorphic_first=true)
        d, _, _ = OpenADMIXTURE.run_admixture(bed, 3; rng=StableRNG(1), progress_bar=false)
        @test isfinite(d.ll_new)
        @test all(isfinite, d.q)
        @test all(isfinite, d.p)
    end
end

@testset "multithreading (incl. Julia >= 1.12 interactive thread)" begin
    mktempdir() do dir
        bed = eur_subset(dir, 2000)
        ll_1 = run_child(bed, 3, "1")
        # default layout: on Julia >= 1.12 this adds an interactive thread,
        # so threadid() can exceed nthreads()
        ll_4 = run_child(bed, 3, "4")
        ll_4_0 = run_child(bed, 3, "4,0")
        @test isfinite(ll_1)
        @test ll_4 ≈ ll_1 rtol = 1e-8
        @test ll_4_0 ≈ ll_1 rtol = 1e-8
    end
end
end # OpenADMIXTURE
