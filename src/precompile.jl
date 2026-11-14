# Precompilation workload: run a small command-line analysis (point estimation,
# cross-validation, bootstrap, output) while the package is precompiled, so the
# native code is cached in the package image and `bin/openadmixture` starts fast.
#
# A PackageCompiler system image was tried first: LoopVectorization's `@turbo`
# kernels lose type inference inside system images (Julia 1.13), which made the
# compiled app ~27x slower. Package images do not have that problem.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    N, M, K = 40, 400, 2
    rng = StableRNG(1)
    P = 0.1 .+ 0.8 .* rand(rng, K, M)
    H = [P[mod1(i, K), j] for i in 1:N, j in 1:M]
    s = SnpArray(undef, N, M)
    fill!(s.data, 0x00)
    for j in 1:M, i in 1:N
        s[i, j] = (0x00, 0x02, 0x03)[1 + (rand(rng) < H[i, j]) + (rand(rng) < H[i, j])]
    end
    @compile_workload begin
        mktempdir() do dir
            prefix = joinpath(dir, "workload")
            SnpArray(prefix * ".bed", s)
            write(prefix * ".bim", join(("$(1 + (j > M ÷ 2))\tsnp$j\t0\t$(100j)\tA\tG\n" for j in 1:M)))
            write(prefix * ".fam", join(("f$i\ti$i\t0\t0\t0\t-9\n" for i in 1:N)))
            cd(dir) do
                redirect_stdout(devnull) do
                    redirect_stderr(devnull) do
                        main(["--cv=2", "-B2", "-s", "1", "-C", "1e-2", prefix * ".bed", "2"])
                        main(["-C", "3", prefix * ".bed", "2"])
                        main(["--help"])
                        main(["--bogus"])
                    end
                end
            end
        end
    end
end
