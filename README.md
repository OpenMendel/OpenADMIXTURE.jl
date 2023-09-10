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

## Citation
The methods and applications of this software package are detailed in the following publication:

_Ko S, Chu BB, Peterson D, Okenwa C, Papp JC, Alexander DH, Sobel EM, Zhou H, Lange K. Unsupervised Discovery of Ancestry Informative Markers and Genetic Admixture Proportions in Biobank-Scale Data Sets. Am J Hum Genet. 110, pp. 314–325. [[DOI](https://doi.org/10.1016/j.ajhg.2022.12.008)]_

If you use OpenMendel analysis packages in your research, please cite the following reference in the resulting publications:

_Zhou H, Sinsheimer JS, Bates DM, Chu BB, German CA, Ji SS, Keys KL, Kim J, Ko S, Mosher GD, Papp JC, Sobel EM, Zhai J, Zhou JJ, Lange K. OPENMENDEL: a cooperative programming project for statistical genetics. Hum Genet. 2020 Jan;139(1):61-71. doi: 10.1007/s00439-019-02001-z. Epub 2019 Mar 26. PMID: 30915546; PMCID: [PMC6763373](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6763373/)._

