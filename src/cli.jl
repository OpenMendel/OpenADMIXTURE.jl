# ADMIXTURE 1.3-compatible command line:
#   openadmixture [options] input.bed K
# Outputs <input base>.<K>.Q, .P (and .Q_se with -B) in the current directory,
# and prints "Loglikelihood: ..." and "CV error (K=k): ..." like ADMIXTURE.

const USAGE = """
Usage: openadmixture [options] <input.bed> <K>

ADMIXTURE 1.3-compatible options:
  -jN, -j N        number of threads (read by the bin/openadmixture launcher)
  -s SEED|time     random seed (default 43)
  --cv[=v]         v-fold cross-validation (default v = 5)
  -B[n]            bootstrap standard errors with n replicates (default 200)
  -C ε|N           major termination: stop when the log-likelihood changes by
                   less than ε (default 1e-4), or after N iterations
  -c ε|N           minor termination for bootstrap replicates (default 3 iterations)
  -a qnN           quasi-Newton acceleration with N secant conditions (default 3)
Extensions:
  --bootstrap-block=L   bootstrap block length in SNPs (default cube root of the
                        number of SNPs; 1 = resample SNPs independently)
  -h, --help       show this help

Not supported yet: --supervised, -P (projection), -m, -a none, -l, -e, --haploid,
.ped/.geno input (convert with `plink --make-bed`).
"""

Base.@kwdef mutable struct CLIOptions
    bedfile::String = ""
    K::Int = 0
    cv::Int = 0
    bootstrap::Int = 0
    block::Union{Nothing, Int} = nothing
    seed::Int = 43
    threads::Union{Nothing, Int} = nothing
    atol::Union{Nothing, Float64} = 1e-4
    max_iter::Union{Nothing, Int} = nothing
    minor_atol::Union{Nothing, Float64} = nothing
    minor_iter::Union{Nothing, Int} = 3
    qn::Int = 3
    help::Bool = false
end

const UNSUPPORTED = ("--supervised", "-P", "-m", "-l", "-e")

# ADMIXTURE -C / -c: an integer is an iteration count, anything else a delta.
function _criterion(s::AbstractString, flag)
    n = tryparse(Int, s)
    if n !== nothing
        n >= 1 || throw(ArgumentError("$flag: iteration count must be ≥ 1"))
        return (nothing, n)
    end
    ε = tryparse(Float64, s)
    (ε === nothing || !(0 < ε < 1)) && throw(ArgumentError("$flag: expected 0 < ε < 1 or an iteration count, got \"$s\""))
    (ε, nothing)
end

function _int(s::AbstractString, flag; min=1)
    n = tryparse(Int, s)
    (n === nothing || n < min) && throw(ArgumentError("$flag: expected an integer ≥ $min, got \"$s\""))
    n
end

"""
    parse_cli(args) -> CLIOptions

Parse ADMIXTURE-style arguments. Throws `ArgumentError` on invalid input.
"""
function parse_cli(args::AbstractVector{<:AbstractString})
    o = CLIOptions()
    pos = String[]
    i = 1
    # value of an option given either attached (-s42) or as the next argument (-s 42)
    function value(a, flag)
        length(a) > length(flag) && return a[length(flag)+1:end]
        i < length(args) || throw(ArgumentError("$flag needs a value"))
        i += 1
        args[i]
    end
    while i <= length(args)
        a = args[i]
        if a in ("-h", "--help")
            o.help = true
        elseif a == "--cv"
            o.cv = 5
        elseif startswith(a, "--cv=")
            o.cv = _int(a[6:end], "--cv"; min=2)
        elseif startswith(a, "--bootstrap-block=")
            o.block = _int(a[19:end], "--bootstrap-block")
        elseif a == "-B"
            o.bootstrap = 200
        elseif occursin(r"^-B\d+$", a)
            o.bootstrap = _int(a[3:end], "-B"; min=2)
        elseif startswith(a, "-j")
            o.threads = _int(value(a, "-j"), "-j")
        elseif startswith(a, "-s")
            v = value(a, "-s")
            o.seed = v == "time" ? Int(round(time()) % typemax(Int32)) : _int(v, "-s"; min=0)
        elseif startswith(a, "-C")
            o.atol, o.max_iter = _criterion(value(a, "-C"), "-C")
        elseif startswith(a, "-c")
            o.minor_atol, o.minor_iter = _criterion(value(a, "-c"), "-c")
        elseif startswith(a, "-a")
            v = value(a, "-a")
            m = match(r"^qn(\d+)$", v)
            m === nothing && throw(ArgumentError("-a $v is not supported (only qnN)"))
            o.qn = _int(m[1], "-a")
        elseif a in UNSUPPORTED || startswith(a, "--haploid")
            throw(ArgumentError("ADMIXTURE option $a is not supported yet"))
        elseif startswith(a, "-") && length(a) > 1
            throw(ArgumentError("unknown option $a"))
        else
            push!(pos, a)
        end
        i += 1
    end
    o.help && return o
    length(pos) == 2 || throw(ArgumentError("expected <input.bed> <K>, got $(length(pos)) positional argument(s)"))
    endswith(pos[1], ".bed") || throw(ArgumentError("only PLINK .bed input is supported; convert with `plink --make-bed`"))
    o.bedfile = pos[1]
    o.K = _int(pos[2], "K"; min=2)
    o
end

function _write_matrix(path, M::AbstractMatrix)
    open(path, "w") do io
        for r in eachrow(M)
            join(io, (@sprintf("%.6f", x) for x in r), ' ')
            print(io, '\n')
        end
    end
end

# Chromosome of each SNP, from the first column of the .bim file.
_chromosomes(bim) = [first(split(l)) for l in eachline(bim) if !isempty(strip(l))]

"""
    main(args=ARGS) -> Int

Command-line entry point. Returns the process exit code.
"""
function main(args=ARGS)
    o = try
        parse_cli(args)
    catch e
        e isa ArgumentError || rethrow()
        println(stderr, "Error: ", e.msg, "\n\n", USAGE)
        return 1
    end
    if o.help
        print(USAGE)
        return 0
    end
    base = o.bedfile[1:end-4]
    for ext in (".bed", ".bim", ".fam")
        isfile(base * ext) || (println(stderr, "Error: missing file ", base * ext); return 1)
    end
    if o.threads !== nothing && o.threads != Threads.nthreads()
        println("Note: -j$(o.threads) requested but Julia runs $(Threads.nthreads()) thread(s); ",
                "start Julia with `-t $(o.threads)` (the bin/openadmixture launcher does this).")
    end
    println("OpenADMIXTURE (ADMIXTURE 1.3-compatible command line)")
    println("Random seed: ", o.seed, "    Threads: ", Threads.nthreads())

    # Major termination criterion (-C): a delta, or a fixed number of iterations.
    major = o.max_iter === nothing ? (n_iter=1000, rtol=1e-7, atol=o.atol) :
                                     (n_iter=o.max_iter, rtol=0.0, atol=nothing)
    g = SnpArray(o.bedfile)
    t = @elapsed d = _admixture_base(g, o.K; rng=StableRNG(o.seed), Q=o.qn,
                                     progress_bar=false, major...)
    @printf("Loglikelihood: %.6f\n", d.ll_new)
    @printf("Point estimation time: %.1f sec\n", t)

    out = basename(base) * ".$(o.K)."
    _write_matrix(out * "Q", permutedims(d.q))
    _write_matrix(out * "P", permutedims(d.p))

    if o.bootstrap > 0
        chroms = _chromosomes(base * ".bim")
        block = something(o.block, default_block(length(chroms)))
        println("Bootstrapping: $(o.bootstrap) replicates, blocks of $block SNP(s)")
        t = @elapsed se = bootstrap_se(g, o.K, d.q, d.p; B=o.bootstrap, chroms=chroms,
            block=block, rng=StableRNG(o.seed), minor_iter=o.minor_iter,
            minor_atol=o.minor_atol, quiet=true, Q=o.qn)
        _write_matrix(out * "Q_se", se)
        @printf("Bootstrap time: %.1f sec\n", t)
    end

    if o.cv > 0
        t = @elapsed err = cv_error(g, o.K, d.q, d.p; folds=o.cv, seed=o.seed,
                                    quiet=true, Q=o.qn, major...)
        @printf("CV error (K=%d): %.5f\n", o.K, err)
        @printf("Cross-validation time: %.1f sec\n", t)
    end
    println("Wrote ", out, "Q, ", out, "P", o.bootstrap > 0 ? ", $(out)Q_se" : "")
    return 0
end
