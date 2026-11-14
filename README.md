# OpenADMIXTURE.jl
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://OpenMendel.github.io/OpenADMIXTURE.jl/dev)

This software package is an open-source Julia reimplementation of the [ADMIXTURE](https://dalexander.github.io/admixture/) package. It estimates ancestry with maximum-likelihood method for a large SNP genotype datasets, where individuals are assumed to be unrelated. The input is binary PLINK 1 BED-formatted file (`.bed`). Also, you will need an idea of $K$, the number of ancestral populations. If the number of SNPs is too large, you may choose to run on a subset of SNPs selected by their information content, using the [sparse K-means via feature ranking](https://github.com/kose-y/SKFR.jl) (SKFR) method.

With more efficient multi-threading scheme, it is 8 times faster than the original software on a 16-threaded machine. It also supports computation on an Nvidia CUDA GPU. By directly using the data format of the PLINK BED file, the memory usage is 16x-32x smaller than using `Float32` or `Float64` type.

## Installation

This package requires Julia v1.9 or later, which can be obtained from
<https://julialang.org/downloads/> or by building Julia from the sources in the
<https://github.com/JuliaLang/julia> repository.

The package can be installed by running the following code:
```julia
using Pkg
pkg"add https://github.com/kose-y/SKFR.jl"
pkg"add https://github.com/OpenMendel/OpenADMIXTURE.jl"
```
For running the examples in our documentation, the following are also necessary. 
```julia
pkg"add SnpArrays CSV DelimitedFiles"
```

For GPU support, an Nvidia GPU is required. Also, the following package has to be installed:
```julia
pkg"add CUDA"
```

## Command line (ADMIXTURE 1.3-compatible)

`bin/openadmixture` accepts ADMIXTURE's syntax and writes the same files in the
current directory:

```bash
bin/openadmixture -j8 --cv=10 -B500 -s 12345 data.bed 3
# -> data.3.Q, data.3.P, data.3.Q_se
# stdout: "Loglikelihood: ...", "CV error (K=3): 0.47835"
```

| Option | Meaning |
|---|---|
| `-jN` | threads (passed to Julia as `-t N`; default 1, as in ADMIXTURE) |
| `-s SEED\|time` | random seed (default 43) |
| `--cv[=v]` | v-fold cross-validation (default 5), ADMIXTURE manual §2.1.1 |
| `-B[n]` | bootstrap standard errors of Q, n replicates (default 200) |
| `-C ε\|N` | stop when the log-likelihood changes by less than ε (default 1e-4, absolute) or after N iterations |
| `-c ε\|N` | same, for bootstrap replicates (default 3 iterations) |
| `-a qnN` | quasi-Newton acceleration with N secant conditions (default 3) |
| `--bootstrap-block=L` | bootstrap block length in SNPs (default: cube root of the number of SNPs) |

Bootstrap resamples blocks of consecutive SNPs within each chromosome (read from
the `.bim` file). ADMIXTURE does not document its block length, so the default
here is a choice of this implementation.

Not supported yet: `--supervised`, `-P`, `-m`, `-a none`, `-l`, `-e`,
`--haploid`, and `.ped`/`.geno` input (convert with `plink --make-bed`).

The package precompiles a small command-line run (PrecompileTools), so the
first analysis of a session does not pay for compilation: a run starts in about
7 s, most of it loading packages. A PackageCompiler app would start faster but
is not provided: LoopVectorization's `@turbo` kernels lose type inference inside
system images (Julia 1.13), which made the compiled app about 27 times slower.

The same features are available from Julia: `OpenADMIXTURE.cv_error`,
`OpenADMIXTURE.bootstrap_se` and `OpenADMIXTURE.main(args)`.

## Citation
The methods and applications of this software package are detailed in the following publication:

_Ko S, Chu BB, Peterson D, Okenwa C, Papp JC, Alexander DH, Sobel EM, Zhou H, Lange K. Unsupervised Discovery of Ancestry Informative Markers and Genetic Admixture Proportions in Biobank-Scale Data Sets. Am J Hum Genet. 110, pp. 314–325. [[DOI](https://doi.org/10.1016/j.ajhg.2022.12.008)]_

If you use OpenMendel analysis packages in your research, please cite the following reference in the resulting publications:

_Zhou H, Sinsheimer JS, Bates DM, Chu BB, German CA, Ji SS, Keys KL, Kim J, Ko S, Mosher GD, Papp JC, Sobel EM, Zhai J, Zhou JJ, Lange K. OPENMENDEL: a cooperative programming project for statistical genetics. Hum Genet. 2020 Jan;139(1):61-71. doi: 10.1007/s00439-019-02001-z. Epub 2019 Mar 26. PMID: 30915546; PMCID: [PMC6763373](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6763373/)._

