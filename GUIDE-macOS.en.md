# OpenADMIXTURE on Apple Silicon Macs — user guide

*Version française : [GUIDE-macOS.md](GUIDE-macOS.md).*

This guide explains how to run an ancestry analysis with the OpenADMIXTURE fork
(a native arm64 replacement for ADMIXTURE 1.3) on the team's M3 Ultra Mac
Studio. It also covers integration into a pipeline, in particular
[Moira](#9-integration-with-moira) (`/Users/loic/moira`).

Last checked: 15 November 2026, Julia 1.13.4, macOS 27, fork
`ltalignani/OpenADMIXTURE.jl`, branch `apple-silicon-fixes`.

---

## 1. Quick start

```bash
cd /output/directory
openadmixture -j8 --cv data.bed 3        # data.bed + data.bim + data.fam
# → data.3.Q, data.3.P in the current directory
# → stdout: "Loglikelihood: -15600872.569969" and "CV error (K=3): 0.43520"
```

- It uses the same syntax, writes the same output files and prints the same key
  stdout lines as ADMIXTURE 1.3. A script written for `admixture` works if you
  swap the binary name, except for the options listed in
  [§8](#8-differences-from-admixture-13-and-open-validation-points).
- It runs natively on arm64, without Rosetta. On 2,000 individuals × 200,000
  SNPs, K = 4, 28 threads, a run takes 33 s; fastmixture takes 55 s. The x86
  ADMIXTURE binary does not run on this Mac without Rosetta.
- Every launch has a fixed cost of about 6.5 s to load Julia and the packages.

---

## 2. What is installed on this Mac

| Item | Location |
|---|---|
| Julia 1.13.4 (Homebrew, arm64) | `/opt/homebrew/bin/julia` |
| Fork source code (git) | `/Users/loic/OpenADMIXTURE.jl` |
| Branch | `apple-silicon-fixes`, tracking `origin` (https://github.com/ltalignani/OpenADMIXTURE.jl) |
| Launcher (bash script) | `/Users/loic/OpenADMIXTURE.jl/bin/openadmixture` |
| Command on the PATH | `/opt/homebrew/bin/openadmixture` → symlink to the launcher above |
| Julia packages and precompiled cache | `~/.julia/` (default Julia depot) |
| Exact dependency versions | `/Users/loic/OpenADMIXTURE.jl/Manifest.toml` (not under version control, see §10) |

The symlink is in `/opt/homebrew/bin` rather than `/usr/local/bin`, which
requires `sudo`. Both directories are on the PATH. The command is also visible
from conda environments (checked with the `moira-test` environment).

Check the installation:

```bash
openadmixture --help     # prints the help, exit code 0, ~6.5 s
```

### How the launcher works

`bin/openadmixture` is a bash script that:

1. follows symlinks to find the repository directory;
2. reads `-jN` or `-j N` from the arguments (default 1, as in ADMIXTURE);
3. runs
   `julia --project=<repo> --startup-file=no -t N -e 'using OpenADMIXTURE; exit(OpenADMIXTURE.main(ARGS))' -- <arguments>`.

Julia fixes its thread count at start-up, so `-j` is read by the script, not by
the program. `JULIA_NUM_THREADS` is ignored, because the launcher always passes
`-t`. The launcher looks for `julia` on the PATH.

---

## 3. Installing on another Apple Silicon Mac

```bash
brew install julia                       # or juliaup; only tested with Julia 1.13
git clone -b apple-silicon-fixes https://github.com/ltalignani/OpenADMIXTURE.jl.git ~/OpenADMIXTURE.jl
cd ~/OpenADMIXTURE.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'   # a few minutes, once
ln -s ~/OpenADMIXTURE.jl/bin/openadmixture /opt/homebrew/bin/openadmixture
openadmixture --help
```

During `Pkg.precompile()`, the package runs a small complete analysis (fit,
cross-validation, bootstrap) and caches the native code. This is what makes
later launches fast. If you skip this step, the first analysis precompiles
everything itself, which takes several minutes.

To run the test suite (156 tests, ~40 s):

```bash
cd ~/OpenADMIXTURE.jl && julia --project=. -t 4 -e 'using Pkg; Pkg.test()'
```

---

## 4. Preparing the data

### Input format

The only accepted input is **binary PLINK 1**: `prefix.bed` + `prefix.bim` +
`prefix.fam` in the same directory. Pass the path to the `.bed` file. The
program exits with code 1 if any of the three files is missing.

Common conversions (plink2 is available in Moira's `cli-genomics` conda
environment):

```bash
# VCF/BCF → bed (biallelic SNPs only)
plink2 --vcf data.vcf.gz --allow-extra-chr --snps-only --max-alleles 2 \
       --set-all-var-ids '@:#' --make-bed --out data
# PED/MAP → bed
plink --file data --make-bed --out data
```

### What the program accepts (unlike ADMIXTURE)

- **Any chromosome codes** (`chr1`, `2L`, `scaffold_12`…). The `.bim` file is
  only read for the bootstrap (column 1 = chromosome). ADMIXTURE requires
  integer codes; with OpenADMIXTURE there is no need to recode them to `0`.
  Checked with a `.bim` using `chr17`.
- **Missing data**: handled (coded as missing in the `.bed`), as in ADMIXTURE.
- **Monomorphic SNPs**: accepted. The upstream version produced NaN on them;
  this is fixed and covered by a test.

### Recommendations (same as for ADMIXTURE)

- **Prune for linkage disequilibrium**: the model assumes independent SNPs.
  Example:
  `plink2 --bfile data --indep-pairwise 50 10 0.1 --out ld && plink2 --bfile data --extract ld.prune.in --make-bed --out data.pruned`.
- **Remove related individuals**: the model assumes unrelated individuals.
- **Filter** rare SNPs and SNPs with a high missing rate before the analysis.

---

## 5. Running an analysis

```
openadmixture [options] <input.bed> <K>
```

Options may come before or after the positional arguments. Output files are
written to the **current directory**, not next to the `.bed` file, so `cd` to
the output directory before launching.

### Options

| Option | Effect | Default |
|---|---|---|
| `-jN` or `-j N` | number of threads | 1 |
| `-s N` or `-s time` | random seed (integer ≥ 0, or the clock) | 43 |
| `--cv` / `--cv=v` | v-fold cross-validation (v ≥ 2) | off; `--cv` alone = 5 folds |
| `-B` / `-Bn` | bootstrap standard errors of Q, n replicates (n ≥ 2) | off; `-B` alone = 200 |
| `-C ε` or `-C N` | stopping rule for the main fit: absolute change in log-likelihood < ε (0 < ε < 1), or exactly N iterations (integer) | ε = 1e-4, at most 1,000 iterations |
| `-c ε` or `-c N` | same, for each bootstrap replicate | 3 iterations |
| `-a qnN` | quasi-Newton acceleration with N secant conditions | 3 |
| `--bootstrap-block=L` | bootstrap block length in SNPs (1 = resample SNPs independently) | cube root of the number of SNPs |
| `-h`, `--help` | help | |

Note that `-C 3` means "3 iterations" and `-C 0.001` means "stop when
Δ < 0.001": as in ADMIXTURE, the program decides by whether the value is an
integer.

### Examples

```bash
openadmixture data.bed 3                          # plain fit, 1 thread
openadmixture -j16 --cv=10 data.bed 4             # 10-fold CV
openadmixture -j16 -s 12345 -B500 data.bed 4      # 500 bootstrap replicates → data.4.Q_se
openadmixture -j8 -C 1e-6 data.bed 5              # stricter convergence
openadmixture -j8 -C 50 data.bed 5                # exactly 50 iterations (quick test)
```

### Output files

For `prefix.bed` and K, where `prefix` is the `.bed` name without directory or
extension:

| File | Content | Dimensions |
|---|---|---|
| `prefix.K.Q` | ancestry proportions; one row per individual, in `.fam` order; each row sums to 1 | N rows × K columns |
| `prefix.K.P` | allele frequencies per ancestral population; one row per SNP, in `.bim` order | M rows × K columns |
| `prefix.K.Q_se` | bootstrap standard deviation of each Q value (only with `-B`) | N rows × K columns |

Format: plain text, space-separated, 6 decimals (`0.370394 0.135791 0.493815`),
no header. Existing files are overwritten without warning.

Column (cluster) order is arbitrary and can change from one seed to another, as
with ADMIXTURE. To compare replicates, align them first (CLUMPP, pong, or a
column matching).

### Standard output and exit codes

The lines a script should parse have exactly ADMIXTURE's format:

```
Loglikelihood: -15600872.569969
CV error (K=3): 0.43520
```

These regular expressions work (they are Moira's):
`^Loglikelihood:\s*([0-9.eE+-]+)` (multiline) and `CV error \(K=(\d+)\):\s*([0-9.eE+-]+)`.

The other lines are informational only: `Random seed: … Threads: …`,
`Loaded N samples and M SNPs`, one line per EM iteration and then per main
iteration, `Point estimation time: … sec`, `Bootstrapping: …`,
`Bootstrap time: …`, `Cross-validation time: …`, and finally
`Wrote prefix.K.Q, prefix.K.P[, prefix.K.Q_se]`. `Loglikelihood:` is printed
only once, after the fit.

| Exit code | Situation |
|---|---|
| 0 | success (including `--help`) |
| 1 | invalid or unsupported option, K < 1, input not `.bed`, missing `.bed`/`.bim`/`.fam` file: `Error: …` on stderr, followed by the help |
| 1 | internal Julia error (stack trace on stderr) |

---

## 6. Choosing K by cross-validation

The procedure is the same as in the ADMIXTURE manual (§2.1.1): run each K with
`--cv` and keep the K with the lowest CV error (or at the elbow of the curve).

```bash
for K in 1 2 3 4 5 6 7 8; do
    openadmixture -j16 -s 43 --cv data.bed $K > log_K$K.out
done
grep -h "CV error" log_K*.out | sort -t= -k2n
```

How the error is computed:
- for each fold, 1/v of the non-missing genotypes are masked (assigned
  deterministically from the seed);
- the model is refitted starting from the full-data estimates;
- the error is the mean binomial deviance over the masked genotypes, pooled
  across all folds.

The formula is the one in the ADMIXTURE manual:
d(n, μ) = n log(n/μ) + (2−n) log((2−n)/(2−μ)).

Cost: CV refits the model v times. On the EUR data set, 5-fold CV took 32 s,
against 8 s for the fit alone.

Running several seeds per K (as Moira does) helps detect local optima. Keep the
replicate with the best log-likelihood or the lowest CV error.

---

## 7. Performance and resources

### Measurements on this Mac (M3 Ultra, 28 performance cores)

| Data set | Command | Total time |
|---|---|---|
| EUR (379 ind. × 54,051 SNPs) | `-j4 --cv data.bed 3` | 47 s (fit 8 s, CV 32 s, loading ~7 s) |
| EUR | `-j1 data.bed 3` | fit 40.5 s |
| EUR | `-j4 data.bed 3` | fit 10.7 s |
| Simulated 2,000 × 200,000 | `-j28 data.bed 4` | 32.7 s (main loop 22.7 s) |

For comparison, fastmixture takes 55 s on the large data set. x86 ADMIXTURE
does not run on this Mac without Rosetta.

### Threads

- For a single large analysis, use up to `-j28`.
- For many small analyses in parallel (several K, seeds or subsets), several
  processes with 4–8 threads each use the machine better than one 28-thread
  process. Keep the total number of threads at or below the number of cores.
- The system time (`sys`) reported by `time` is high (601 s sys for 228 s user
  in the EUR `-j4` example). This is Julia threads waiting for work. It is not
  an error, but it is a reason not to request more threads than needed.

### Memory

Genotypes stay in memory in the 2-bit `.bed` format; the program adds Q and P
in double precision. As a rough guide, a run uses about 650 MB of resident
memory (RSS) for EUR with CV, about half of it for Julia itself. Allow roughly
N × M / 4 bytes per copy of the genotypes; CV makes one masked in-memory copy
per fold, one fold at a time.

### Fixed start-up cost

Each launch takes about 6.5 s to load Julia and the packages, even when the
native code is already cached. Over 500 small jobs, that adds up to about
55 minutes of CPU time, spread over the parallel jobs.

A PackageCompiler app would start in 0.25 s, but it was abandoned: inside a
system image, LoopVectorization's `@turbo` kernels lose type inference under
Julia 1.13, and the app computed about 27 times more slowly. The scripts from
that attempt are kept, outside git, in `/Users/loic/OpenADMIXTURE.jl/build/`.

---

## 8. Differences from ADMIXTURE 1.3 and open validation points

### Reproducibility

- Same seed and same input give **bit-identical files**, whatever the number
  of threads. Checked on EUR, K = 3: `-j1` and `-j4` give the same `.Q`, and so
  do two `-j4` runs.
- The same seed does **not** give the same values as ADMIXTURE, because the
  random number generator and the initialisation differ. Solutions are
  comparable up to cluster permutation and local optima.

### Not supported (the program exits with code 1)

| ADMIXTURE feature | Status |
|---|---|
| `--supervised` (`.pop` file) | not available |
| `-P` (projection onto fixed P) | not available (the Julia library can fix P or Q, but this is not exposed on the command line) |
| `-m`, `-a none`, `-l`, `-e`, `--haploid` | not available |
| `.ped` / `.geno` input | convert with `plink --make-bed` |

### Choices specific to this implementation

- **K = 1**: supported, as in ADMIXTURE. The solution is analytic: Q = 1, and
  P = the observed allele frequency, bounded to [1e-5, 1 − 1e-5]. `-C`, `-a`
  and the seed therefore have no effect on the fit; the seed only changes the
  CV fold assignment. The bootstrap writes a `.Q_se` of zeros.
- **Bootstrap**: blocks of consecutive SNPs are resampled within each
  chromosome (column 1 of the `.bim`). The default block length (cube root of
  the number of SNPs) is our choice, because ADMIXTURE does not document its
  own. If all SNPs are coded on chromosome `0`, blocks can span the boundary
  between two real chromosomes.
- **CV stopping rule**: each fold starts from the full-data estimates, with the
  same stopping rule as the main fit (`-C`).

### Not yet validated against x86 ADMIXTURE

Validation needs reference outputs from ADMIXTURE 1.3 on an x86 Linux machine,
on the same data:

1. **Allele orientation in `.P`**: OpenADMIXTURE reports the frequency of
   allele A2 (column 6 of the `.bim`). Whether ADMIXTURE reports the same
   allele is still to be checked. If not, P_ADMIXTURE = 1 − P_OpenADMIXTURE.
2. **CV error values**: same formula, but the fold masking differs. The chosen
   K should be the same; the exact values will not be.
3. **`.Q_se` values**: they depend on the block length.

Until then, compare `.Q` up to a column permutation (correlation > 0.99
expected on well-structured data). Do not mix ADMIXTURE and OpenADMIXTURE CV
errors in the same curve.

---

## 9. Integration with Moira

Moira (`/Users/loic/moira`) calls ADMIXTURE as an external binary from
`workflow/lib/moira/admixture.py` (function `run`):

```
<binary> -s SEED -jN --cv <absolute path>/F{subset}.bed K      # cwd = run directory
```

It then reads `F{subset}.K.Q` and extracts `CV error` and `Loglikelihood` from
the output with `CV_RE` and `LL_RE`. OpenADMIXTURE accepts exactly this command
and produces the same files and the same lines.

### Configuration

In `config/local.yaml` (not under version control):

```yaml
tools:
  admixture: /opt/homebrew/bin/openadmixture
```

Both the symlink path and the repository path
(`/Users/loic/OpenADMIXTURE.jl/bin/openadmixture`) work. The launcher resolves
the symlink and needs `julia` on the PATH, which is the case from conda
environments (`/opt/homebrew/bin` stays on the PATH).

### Checked compatibility points

| Moira item | Behaviour with OpenADMIXTURE |
|---|---|
| `-s SEED` with `derive_seed` ∈ [1, 2³¹−1] | accepted (any integer seed ≥ 0) |
| `-j{threads}` (`threads: 4` in rule `structure_admixture_run`) | Julia starts with 4 threads |
| `--cv` (5 folds) | identical |
| `cwd=outdir` | outputs written to `outdir`, as with ADMIXTURE |
| `CV_RE`, `LL_RE` | identical format (a single `Loglikelihood:` line) |
| `{bed.stem}.{k}.Q`, row order = `.fam` | identical |
| `proc.returncode != 0` → retry | exit code 1 on error |
| `mem_mb = 4000 * attempt` | sufficient (~0.7 GB for 379 ind. × 54 k SNPs; revisit for very large N) |
| `_set_unplaced_chromosomes` (chromosomes → `0`) | unnecessary for OpenADMIXTURE but harmless, since Moira does not use `-B` |
| `admixture.log` (stdout + stderr) | longer than with ADMIXTURE (one line per iteration) |

### Problems found in the first trials (fixed)

- **K = 1 rejected**: versions before the K = 1 commit exited with code 1,
  while Moira has `DEFAULTS["k_min"] = 1`. With 20 % of runs failing, the
  `gather` rule stopped (`min_success_fraction = 0.9`). K = 1 is now supported,
  so `k_min: 1` works.
- **`--cv` crashed** at random depending on seed and K, with
  `ArgumentError: matrix contains Infs or NaNs` in `inv()`, `update_QN!`,
  called from `cv_error`.
  - Cause: `d.q` and `d.p` were `unsafe_wrap` views into the memory of the fit
    result. The garbage collector could free that memory during CV, and the
    folds then started from garbage values.
  - Fix: the root cause is fixed. In addition, the quasi-Newton step now falls
    back to the plain EM step when the matrix to invert is not finite or is
    singular.
  - Check: 379 individuals × 2,000 SNPs, K = 1 to 3, seeds 12345, 1, 2 and 3,
    with `--cv`: 12 of 12 runs succeeded.

### Resources for Snakemake scheduling

- Each job pays about 6.5 s of start-up. This matters more for subsets of
  100,000 SNPs than for large data sets.
- Several OpenADMIXTURE jobs can run at the same time: they share the
  precompiled cache read-only.
- **After any update** (`git pull` of the fork, `brew upgrade julia`, a
  dependency change), run `openadmixture --help` once before Snakemake.
  Otherwise every job started in parallel recompiles the package on its own,
  taking several minutes each and competing for the same cache.

### Moira tests

Tests that run the real binary can use `openadmixture` on the synthetic data
set (3 groups × 10 diploid individuals). On small data sets the ~6.5 s start-up
dominates. For a quick test, use `-C 3` (3 iterations).

---

## 10. Updating and maintenance

```bash
cd /Users/loic/OpenADMIXTURE.jl
git pull                                                 # branch apple-silicon-fixes
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
openadmixture --help                                     # checks that everything starts
```

- **Julia update** (`brew upgrade julia`): a new minor version (1.13 → 1.14)
  invalidates the precompiled cache, so rerun `Pkg.precompile()`. The Apple
  Silicon fixes were validated under Julia 1.13; run `Pkg.test()` after a
  minor-version change.
- **`Manifest.toml`** is not under version control, so another machine may
  resolve different dependency versions. To pin the same versions across the
  team, it would have to be committed to the fork's branch. This has not been
  done yet.
- **Sensitive dependencies**: LoopVectorization (deprecated, but it makes the
  main loop 60 % faster) and HostCPUFeatures. An update to either is the first
  thing to suspect after a performance regression.
- **Upstream code**: https://github.com/OpenMendel/OpenADMIXTURE.jl. No pull
  request has been sent there.

---

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| The first launch takes several minutes | The precompiled cache is missing or invalidated. Wait, or run `Pkg.precompile()` (§10). Expected after an update |
| `Note: -jN requested but Julia runs 1 thread(s)` | Julia was called directly, without the launcher. Use `openadmixture`, or `julia -t N` |
| `matrix contains Infs or NaNs` (in `update_QN!` / `cv_error`) | Fork version older than the fix: `git pull` (§10) |
| `Error: only PLINK .bed input is supported` | The first positional argument must end in `.bed` |
| `Error: missing file prefix.bim` | The three PLINK files must sit side by side |
| `Error: ADMIXTURE option --supervised is not supported yet` | See §8 |
| `julia: command not found` in a job | `/opt/homebrew/bin` is not on the job's PATH. Add it, or symlink `julia` elsewhere |
| `cfunction: closures are not supported` | Old OpenADMIXTURE version (Polyester). Make sure the `apple-silicon-fixes` branch is used |
| `BoundsError` in threads | Old version under Julia ≥ 1.12. Same fix |
| NaN in `.Q` or `.P` | Old version with monomorphic SNPs. Same fix |
| `LLVM ERROR: Cannot select … vscale` | Only with PackageCompiler (§7). Do not build an app |

---

## 12. Using it from Julia

Use this to run several K in a single process (without paying the start-up
cost each time) or to work with the matrices directly:

```julia
# julia --project=/Users/loic/OpenADMIXTURE.jl -t 16
using OpenADMIXTURE, SnpArrays, StableRNGs

# 1. Exactly like the command line (files written to pwd())
for K in 1:8
    OpenADMIXTURE.main(["-s", "43", "--cv", "data.bed", string(K)])
end

# 2. Direct access
g = SnpArray("data.bed")
d = OpenADMIXTURE._admixture_base(g, 3; rng=StableRNG(43), progress_bar=false,
                                  n_iter=1000, rtol=1e-7, atol=1e-4)
Q = permutedims(d.q)        # N × K
P = permutedims(d.p)        # M × K
err = OpenADMIXTURE.cv_error(g, 3, d.q, d.p; folds=5, seed=43, quiet=true)
se  = OpenADMIXTURE.bootstrap_se(g, 3, d.q, d.p; B=200, quiet=true)   # N × K
```

`_admixture_base` is an internal function (K ≥ 2 only; K = 1 uses
`OpenADMIXTURE.fit_one_population(g)`), and its interface may change. For
long-term use, call `main`.

---

## 13. References

- ADMIXTURE 1.3 manual: https://dalexander.github.io/admixture/admixture-manual.pdf
- OpenADMIXTURE paper: Ko S. *et al.* (2023), *Am J Hum Genet* 110:314–325,
  https://doi.org/10.1016/j.ajhg.2022.12.008. Cite it together with ADMIXTURE
  (Alexander, Novembre & Lange 2009, *Genome Res* 19:1655–1664).
- Fork: https://github.com/ltalignani/OpenADMIXTURE.jl/tree/apple-silicon-fixes
- Fix history: `git log 401ac5c..apple-silicon-fixes` in the repository.
