# Tests for ADMIXTURE-compatible cross-validation, bootstrap and command line.
using OpenADMIXTURE, SnpArrays, Random, StableRNGs, Test

const CODE = (0x00, 0x02, 0x03)  # PLINK codes for 0, 1, 2 copies of A2

"""
Simulate admixed genotypes: `K` populations, half of the individuals unadmixed,
half with uniform (Dirichlet(1)) ancestry. Returns (SnpArray, Q (N x K), chromosomes).
"""
function simulate(N, M, K; rng=StableRNG(2024), nchrom=2)
    anc = 0.1 .+ 0.8 .* rand(rng, M)
    P = clamp.(anc' .+ 0.6 .* (rand(rng, K, M) .- 0.5), 0.02, 0.98)  # K x M
    Q = zeros(N, K)
    for i in 1:N
        if isodd(i)
            Q[i, mod1(i, K)] = 1.0
        else
            e = -log.(rand(rng, K))
            Q[i, :] .= e ./ sum(e)
        end
    end
    s = SnpArray(undef, N, M)
    fill!(s.data, 0x00)
    H = Q * P
    for j in 1:M, i in 1:N
        s[i, j] = CODE[1 + (rand(rng) < H[i, j]) + (rand(rng) < H[i, j])]
    end
    chroms = [1 + (j - 1) * nchrom ÷ M for j in 1:M]
    s, Q, chroms
end

function write_plink(prefix, s, chroms)
    SnpArray(prefix * ".bed", s)
    open(prefix * ".bim", "w") do io
        for j in 1:size(s, 2)
            println(io, chroms[j], "\tsnp", j, "\t0\t", 1000j, "\tA\tG")
        end
    end
    open(prefix * ".fam", "w") do io
        for i in 1:size(s, 1)
            println(io, "fam", i, "\tind", i, "\t0\t0\t0\t-9")
        end
    end
end

fit(s, K; seed=1) = OpenADMIXTURE._admixture_base(s, K; rng=StableRNG(seed), progress_bar=false)

@testset "binomial deviance (ADMIXTURE manual eq. 1)" begin
    d = OpenADMIXTURE.binomial_deviance
    @test d(0, 0.5) ≈ 2 * log(2 / 1.5)
    @test d(2, 0.5) ≈ 2 * log(2 / 0.5)
    @test d(1, 0.5) ≈ log(1 / 0.5) + log(1 / 1.5)
    @test d(1, 1.0) ≈ 0 atol = 1e-12
    @test d(0, 0.0) == 0
    @test d(2, 2.0) == 0
end

@testset "cross-validation folds" begin
    v = 5
    f = [OpenADMIXTURE.cv_fold(i, j, v, 43) for i in 1:200, j in 1:100]
    @test all(in(1:v), f)
    @test f == [OpenADMIXTURE.cv_fold(i, j, v, 43) for i in 1:200, j in 1:100]  # deterministic
    @test f != [OpenADMIXTURE.cv_fold(i, j, v, 44) for i in 1:200, j in 1:100]  # seed matters
    counts = [count(==(k), f) for k in 1:v]
    @test all(c -> abs(c - length(f) / v) < 0.05 * length(f) / v, counts)    # balanced
end

@testset "cross-validation error picks the true K" begin
    s, _, _ = simulate(200, 3000, 3)
    errs = Dict(K => begin
        d = fit(s, K)
        OpenADMIXTURE.cv_error(s, K, d.q, d.p; folds=5, seed=43, quiet=true)
    end for K in 2:3)
    @test all(e -> 0 < e < 1.5, values(errs))
    @test errs[3] < errs[2]
end

@testset "moving-block bootstrap indices stay within chromosomes" begin
    chroms = [1, 1, 1, 1, 1, 2, 2, 2, 3]
    for seed in 1:20
        idx = OpenADMIXTURE.block_bootstrap_indices(chroms, 2, StableRNG(seed))
        @test length(idx) == length(chroms)
        @test all(in(1:5), idx[1:5])
        @test all(in(6:8), idx[6:8])
        @test idx[9] == 9
    end
    # block = 1 is the independence bootstrap within chromosomes
    idx = OpenADMIXTURE.block_bootstrap_indices(fill(1, 1000), 1, StableRNG(3))
    @test length(unique(idx)) < 1000
end

@testset "bootstrap standard errors" begin
    s, _, chroms = simulate(120, 2000, 3)
    d = fit(s, 3)
    se = OpenADMIXTURE.bootstrap_se(s, 3, d.q, d.p; B=4, chroms=chroms, block=20,
                                    rng=StableRNG(5), quiet=true)
    @test size(se) == (120, 3)
    @test all(isfinite, se)
    @test all(>=(0), se)
    @test any(>(0), se)
end

@testset "command-line parsing (ADMIXTURE syntax)" begin
    parse = OpenADMIXTURE.parse_cli
    o = parse(["data.bed", "3"])
    @test o.bedfile == "data.bed" && o.K == 3
    @test o.cv == 0 && o.bootstrap == 0 && o.seed == 43
    @test o.atol == 1e-4 && o.max_iter === nothing          # ADMIXTURE default -C
    @test o.minor_iter == 3 && o.minor_atol === nothing     # ADMIXTURE default -c
    @test parse(["--cv", "x.bed", "2"]).cv == 5
    @test parse(["--cv=10", "x.bed", "2"]).cv == 10
    @test parse(["-B", "x.bed", "2"]).bootstrap == 200
    @test parse(["-B2000", "x.bed", "2"]).bootstrap == 2000
    @test parse(["x.bed", "2", "-j4"]).threads == 4
    @test parse(["-j", "8", "x.bed", "2"]).threads == 8
    @test parse(["-s", "12345", "x.bed", "2"]).seed == 12345
    @test parse(["-s12345", "x.bed", "2"]).seed == 12345
    @test parse(["-s", "time", "x.bed", "2"]).seed isa Integer
    o = parse(["-C", "0.1", "x.bed", "2"])
    @test o.atol == 0.1 && o.max_iter === nothing
    o = parse(["-C", "10", "x.bed", "2"])
    @test o.max_iter == 10 && o.atol === nothing
    o = parse(["-c", "0.5", "x.bed", "2"])
    @test o.minor_atol == 0.5 && o.minor_iter === nothing
    @test parse(["-a", "qn2", "x.bed", "2"]).qn == 2
    @test parse(["--bootstrap-block=50", "-B10", "x.bed", "2"]).block == 50
    @test_throws ArgumentError parse(["x.bed"])
    @test_throws ArgumentError parse(["x.ped", "2"])           # only .bed input
    @test_throws ArgumentError parse(["--supervised", "x.bed", "2"])
    @test_throws ArgumentError parse(["--bogus", "x.bed", "2"])
end

@testset "command line end to end" begin
    s, _, chroms = simulate(90, 1500, 3)
    mktempdir() do dir
        datadir = mkpath(joinpath(dir, "data"))
        prefix = joinpath(datadir, "toy")
        write_plink(prefix, s, chroms)
        outdir = mkpath(joinpath(dir, "run"))
        out = cd(outdir) do
            pipe = Pipe()
            status = redirect_stdout(pipe) do
                OpenADMIXTURE.main(["--cv=3", "-B3", "-s", "7", "-C", "1e-2",
                                    prefix * ".bed", "3"])
            end
            close(pipe.in)
            @test status == 0
            read(pipe, String)
        end
        @test occursin(r"CV error \(K=3\): \d+\.\d{5}\n", out)
        @test occursin(r"Loglikelihood: -\d+\.\d+", out)
        # outputs go to the current directory, named <input base>.<K>.<ext>
        Qf, Pf, SEf = (joinpath(outdir, "toy.3." * e) for e in ("Q", "P", "Q_se"))
        @test isfile(Qf) && isfile(Pf) && isfile(SEf)
        Q = [parse.(Float64, split(l)) for l in eachline(Qf)]
        @test length(Q) == 90 && all(r -> length(r) == 3 && isapprox(sum(r), 1; atol=1e-5), Q)
        @test occursin(r"^\d\.\d{6} \d\.\d{6} \d\.\d{6}$", first(eachline(Qf)))
        @test countlines(Pf) == 1500
        @test countlines(SEf) == 90
    end
end
