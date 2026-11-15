# OpenADMIXTURE sur Mac Apple Silicon — guide d'utilisation

*English version: [GUIDE-macOS.en.md](GUIDE-macOS.en.md).*

Ce guide explique comment lancer une analyse d'ascendance avec le fork
OpenADMIXTURE (remplaçant natif arm64 d'ADMIXTURE 1.3) sur le Mac Studio M3
Ultra de l'équipe, et comment l'intégrer dans un pipeline, en particulier
[Moira](#9-intégration-dans-moira) (`/Users/loic/moira`).

Dernière vérification : 15 novembre 2026, Julia 1.13.4, macOS 27, fork
`ltalignani/OpenADMIXTURE.jl`, branche `apple-silicon-fixes`.

---

## 1. En bref

```bash
cd /dossier/de/sortie
openadmixture -j8 --cv data.bed 3        # data.bed + data.bim + data.fam
# → data.3.Q, data.3.P dans le dossier courant
# → stdout : "Loglikelihood: -15600872.569969" et "CV error (K=3): 0.43520"
```

- Même syntaxe, mêmes fichiers de sortie et mêmes lignes clés sur la sortie
  standard qu'ADMIXTURE 1.3 : un script écrit pour `admixture` fonctionne en
  remplaçant le nom du binaire, sauf pour les options listées en
  [§8](#8-différences-avec-admixture-13-et-points-non-validés).
- Natif arm64, sans Rosetta. Sur 2 000 individus × 200 000 SNP, K = 4,
  28 threads : 33 s, contre 55 s pour fastmixture. Le binaire ADMIXTURE x86 ne
  tourne pas sur ce Mac sans Rosetta.
- Chaque lancement paie environ 6,5 s fixes de chargement de Julia et des
  paquets.

---

## 2. Ce qui est installé sur ce Mac

| Élément | Emplacement |
|---|---|
| Julia 1.13.4 (Homebrew, arm64) | `/opt/homebrew/bin/julia` |
| Code source du fork (git) | `/Users/loic/OpenADMIXTURE.jl` |
| Branche | `apple-silicon-fixes`, suivie sur `origin` (https://github.com/ltalignani/OpenADMIXTURE.jl) |
| Lanceur (script bash) | `/Users/loic/OpenADMIXTURE.jl/bin/openadmixture` |
| Commande sur le PATH | `/opt/homebrew/bin/openadmixture` → lien vers le lanceur ci-dessus |
| Paquets Julia et cache précompilé | `~/.julia/` (dépôt Julia par défaut) |
| Versions exactes des dépendances | `/Users/loic/OpenADMIXTURE.jl/Manifest.toml` (non versionné, voir §10) |

Le lien est dans `/opt/homebrew/bin`, et non dans `/usr/local/bin` (qui
demande `sudo`) ; les deux dossiers sont sur le PATH. Il est aussi visible
depuis les environnements conda (vérifié avec l'environnement `moira-test`).

Vérifier l'installation :

```bash
openadmixture --help     # affiche l'aide, code de sortie 0, ~6,5 s
```

### Fonctionnement du lanceur

`bin/openadmixture` est un script bash qui :

1. suit les liens symboliques pour retrouver le dossier du dépôt ;
2. lit `-jN` ou `-j N` dans les arguments (1 par défaut, comme ADMIXTURE) ;
3. exécute
   `julia --project=<dépôt> --startup-file=no -t N -e 'using OpenADMIXTURE; exit(OpenADMIXTURE.main(ARGS))' -- <arguments>`.

Julia fixe son nombre de threads au démarrage : c'est pour cela que `-j` est lu
par le script et non par le programme. `JULIA_NUM_THREADS` est ignoré (le
lanceur passe toujours `-t`). Le lanceur cherche `julia` dans le PATH.

---

## 3. Installer sur un autre Mac Apple Silicon

```bash
brew install julia                       # ou juliaup ; testé uniquement en Julia 1.13
git clone -b apple-silicon-fixes https://github.com/ltalignani/OpenADMIXTURE.jl.git ~/OpenADMIXTURE.jl
cd ~/OpenADMIXTURE.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'   # quelques minutes, une seule fois
ln -s ~/OpenADMIXTURE.jl/bin/openadmixture /opt/homebrew/bin/openadmixture
openadmixture --help
```

Pendant `Pkg.precompile()`, le paquet exécute une petite analyse complète
(ajustement, validation croisée, bootstrap) pour mettre en cache le code natif ;
c'est ce qui rend les lancements suivants rapides. Sans cette étape, la première
analyse précompile elle-même, ce qui prend plusieurs minutes.

Pour exécuter la suite de tests (156 tests, ~40 s) :

```bash
cd ~/OpenADMIXTURE.jl && julia --project=. -t 4 -e 'using Pkg; Pkg.test()'
```

---

## 4. Préparer les données

### Format d'entrée

Uniquement **PLINK 1 binaire** : `prefixe.bed` + `prefixe.bim` + `prefixe.fam`
dans le même dossier. Le programme reçoit le chemin du `.bed` ; il s'arrête avec
le code 1 si l'un des trois fichiers manque.

Conversions courantes (plink2 est dans l'environnement conda `cli-genomics` de
Moira) :

```bash
# VCF/BCF → bed (SNP bialléliques uniquement)
plink2 --vcf data.vcf.gz --allow-extra-chr --snps-only --max-alleles 2 \
       --set-all-var-ids '@:#' --make-bed --out data
# PED/MAP → bed
plink --file data --make-bed --out data
```

### Ce que le programme accepte (contrairement à ADMIXTURE)

- **Codes de chromosome quelconques** (`chr1`, `2L`, `scaffold_12`…). Le `.bim`
  n'est lu que pour le bootstrap (colonne 1 = chromosome). ADMIXTURE exige des
  entiers ; pour OpenADMIXTURE, il est inutile de remplacer les codes par `0`.
  Vérifié avec un `.bim` en `chr17`.
- **Données manquantes** : gérées (codées manquantes dans le `.bed`), comme
  ADMIXTURE.
- **SNP monomorphes** : acceptés. Ils faisaient produire des NaN à la version
  d'origine ; c'est corrigé et couvert par un test.

### Recommandations (identiques à ADMIXTURE)

- **Élaguer le déséquilibre de liaison** : le modèle suppose les SNP
  indépendants. Exemple :
  `plink2 --bfile data --indep-pairwise 50 10 0.1 --out ld && plink2 --bfile data --extract ld.prune.in --make-bed --out data.pruned`.
- **Retirer les individus apparentés** : le modèle suppose les individus non
  apparentés.
- **Filtrer** les SNP rares et à fort taux de données manquantes avant l'analyse.

---

## 5. Lancer une analyse

```
openadmixture [options] <entrée.bed> <K>
```

Les options peuvent précéder ou suivre les arguments positionnels. Les fichiers
de sortie sont écrits dans le **dossier courant**, et non à côté du `.bed` : il
faut faire un `cd` vers le dossier de sortie avant le lancement.

### Options

| Option | Effet | Défaut |
|---|---|---|
| `-jN` ou `-j N` | nombre de threads | 1 |
| `-s N` ou `-s time` | graine aléatoire (entier ≥ 0, ou l'horloge) | 43 |
| `--cv` / `--cv=v` | validation croisée à v plis (v ≥ 2) | désactivée ; `--cv` seul = 5 plis |
| `-B` / `-Bn` | erreurs standard de Q par bootstrap, n réplicats (n ≥ 2) | désactivé ; `-B` seul = 200 |
| `-C ε` ou `-C N` | arrêt de l'ajustement principal : variation absolue de log-vraisemblance < ε (0 < ε < 1), ou exactement N itérations (entier) | ε = 1e-4, au plus 1 000 itérations |
| `-c ε` ou `-c N` | même chose pour chaque réplicat du bootstrap | 3 itérations |
| `-a qnN` | accélération quasi-Newton à N conditions de sécante | 3 |
| `--bootstrap-block=L` | longueur des blocs du bootstrap, en SNP (1 = SNP rééchantillonnés indépendamment) | racine cubique du nombre de SNP |
| `-h`, `--help` | aide | |

Attention, `-C 3` signifie « 3 itérations » et `-C 0.001` « arrêt quand
Δ < 0,001 » : le programme décide selon que la valeur est entière ou non, comme
ADMIXTURE.

### Exemples

```bash
openadmixture data.bed 3                          # ajustement simple, 1 thread
openadmixture -j16 --cv=10 data.bed 4             # CV à 10 plis
openadmixture -j16 -s 12345 -B500 data.bed 4      # 500 réplicats bootstrap → data.4.Q_se
openadmixture -j8 -C 1e-6 data.bed 5              # convergence plus stricte
openadmixture -j8 -C 50 data.bed 5                # exactement 50 itérations (test rapide)
```

### Fichiers produits

Pour `prefixe.bed` et K, avec `prefixe` = nom du `.bed` sans dossier ni
extension :

| Fichier | Contenu | Dimensions |
|---|---|---|
| `prefixe.K.Q` | proportions d'ascendance ; une ligne par individu, dans l'ordre du `.fam` ; chaque ligne somme à 1 | N lignes × K colonnes |
| `prefixe.K.P` | fréquences alléliques par population ancestrale ; une ligne par SNP, dans l'ordre du `.bim` | M lignes × K colonnes |
| `prefixe.K.Q_se` | écart-type bootstrap de chaque valeur de Q (seulement avec `-B`) | N lignes × K colonnes |

Format : texte, valeurs séparées par une espace, 6 décimales (`0.370394 0.135791 0.493815`),
sans en-tête. Les fichiers existants sont écrasés sans avertissement.

L'ordre des colonnes (clusters) est arbitraire et peut changer d'une graine à
l'autre, comme avec ADMIXTURE. Pour comparer des réplicats, il faut les
aligner (CLUMPP, pong, ou une correspondance de colonnes).

### Sortie standard et codes de sortie

Les lignes à analyser dans un script ont exactement le format d'ADMIXTURE :

```
Loglikelihood: -15600872.569969
CV error (K=3): 0.43520
```

Expressions régulières qui fonctionnent (ce sont celles de Moira) :
`^Loglikelihood:\s*([0-9.eE+-]+)` (multiligne) et `CV error \(K=(\d+)\):\s*([0-9.eE+-]+)`.

Lignes supplémentaires, purement informatives : `Random seed: … Threads: …`,
`Loaded N samples and M SNPs`, une ligne par itération EM puis par itération
principale, `Point estimation time: … sec`, `Bootstrapping: …`,
`Bootstrap time: …`, `Cross-validation time: …`, et enfin
`Wrote prefixe.K.Q, prefixe.K.P[, prefixe.K.Q_se]`. `Loglikelihood:` n'est
imprimée qu'une fois, après l'ajustement.

| Code de sortie | Situation |
|---|---|
| 0 | succès (y compris `--help`) |
| 1 | option invalide ou non prise en charge, K < 1, entrée non `.bed`, fichier `.bed`/`.bim`/`.fam` manquant : message `Error: …` sur stderr, suivi de l'aide |
| 1 | erreur interne Julia (trace d'exécution sur stderr) |

---

## 6. Choisir K par validation croisée

Même procédure que dans le manuel d'ADMIXTURE (§2.1.1) : on lance chaque K avec
`--cv` et on retient le K à l'erreur de CV minimale (ou au coude de la courbe).

```bash
for K in 1 2 3 4 5 6 7 8; do
    openadmixture -j16 -s 43 --cv data.bed $K > log_K$K.out
done
grep -h "CV error" log_K*.out | sort -t= -k2n
```

Comment l'erreur est calculée : pour chaque pli, 1/v des génotypes non
manquants est masqué (affectation déterministe à partir de la graine) ; le
modèle est réajusté en partant de l'estimation complète, et l'erreur est la
déviance binomiale moyenne sur les génotypes masqués, tous plis confondus. La
formule est celle du manuel ADMIXTURE :
d(n, μ) = n log(n/μ) + (2−n) log((2−n)/(2−μ)).

Coût : la CV réajuste le modèle v fois. Sur le jeu EUR, la CV à 5 plis prenait
32 s contre 8 s pour l'ajustement seul.

Plusieurs graines par K (comme le fait Moira) aident à repérer les optimums
locaux : on garde le réplicat à la meilleure log-vraisemblance ou à la plus
faible erreur de CV.

---

## 7. Performances et ressources

### Mesures sur ce Mac (M3 Ultra, 28 cœurs de performance)

| Jeu de données | Commande | Temps total |
|---|---|---|
| EUR (379 ind. × 54 051 SNP) | `-j4 --cv data.bed 3` | 47 s (ajustement 8 s, CV 32 s, chargement ~7 s) |
| EUR | `-j1 data.bed 3` | ajustement 40,5 s |
| EUR | `-j4 data.bed 3` | ajustement 10,7 s |
| Simulé 2 000 × 200 000 | `-j28 data.bed 4` | 32,7 s (boucle principale 22,7 s) |

Pour comparaison : fastmixture 55 s sur le grand jeu. ADMIXTURE x86 ne tourne
pas sur ce Mac sans Rosetta.

### Threads

- Pour une seule grosse analyse, utiliser jusqu'à `-j28`.
- Pour beaucoup de petites analyses en parallèle (plusieurs K, graines ou
  sous-échantillons), plusieurs processus à 4–8 threads chacun utilisent mieux
  la machine qu'un processus à 28 threads. Le total des threads ne doit pas
  dépasser le nombre de cœurs.
- Le temps système (`sys`) mesuré par `time` est élevé (601 s de sys pour 228 s
  de user sur l'exemple EUR `-j4`) : ce sont les threads Julia qui attendent du
  travail. Ce n'est pas une erreur, mais c'est une raison de ne pas demander plus
  de threads que nécessaire.

### Mémoire

Les génotypes restent en mémoire au format 2 bits du `.bed`, et le programme y
ajoute Q et P en double précision. Ordre de grandeur : environ 650 Mo de
mémoire résidente (RSS) pour EUR avec CV, dont la moitié environ pour Julia
lui-même. Prévoir à peu près N × M / 4 octets par copie des génotypes ; la CV
fait une copie masquée en mémoire par pli, l'une après l'autre.

### Coût fixe au démarrage

Environ 6,5 s par lancement, pour charger Julia et les paquets, même si le code
natif est déjà en cache. Pour 500 petits jobs, cela fait environ 55 minutes de
temps CPU cumulé, réparti entre les jobs parallèles. Une application
PackageCompiler démarrerait en 0,25 s, mais elle a été abandonnée : dans une
image système, les noyaux `@turbo` de LoopVectorization perdent l'inférence de
type sous Julia 1.13, et l'application calculait environ 27 fois plus lentement.
Les scripts de cette tentative sont conservés dans
`/Users/loic/OpenADMIXTURE.jl/build/`, hors git.

---

## 8. Différences avec ADMIXTURE 1.3 et points non validés

### Reproductibilité

- Même graine, même entrée → **mêmes fichiers au bit près**, quel que soit le
  nombre de threads. Vérifié sur EUR, K = 3 : `-j1` et `-j4` donnent un `.Q`
  identique, et deux lancements `-j4` aussi.
- Une même graine ne donne **pas** les mêmes valeurs qu'ADMIXTURE : générateur
  aléatoire et initialisation différents. Les solutions sont comparables à
  la permutation des clusters et aux optimums locaux près.

### Non pris en charge (le programme sort avec le code 1)

| Fonction ADMIXTURE | État |
|---|---|
| `--supervised` (fichier `.pop`) | non disponible |
| `-P` (projection sur des P fixés) | non disponible (la bibliothèque Julia sait fixer P ou Q, mais rien n'est exposé en ligne de commande) |
| `-m`, `-a none`, `-l`, `-e`, `--haploid` | non disponibles |
| Entrée `.ped` / `.geno` | convertir avec `plink --make-bed` |

### Choix propres à cette implémentation

- **K = 1** : pris en charge, comme dans ADMIXTURE. La solution est analytique
  (Q = 1, P = fréquence allélique observée, bornée à [1e-5, 1 − 1e-5]), donc
  `-C`, `-a` et la graine n'ont pas d'effet sur l'ajustement. La graine change
  seulement la répartition des plis de CV. Le bootstrap écrit un `.Q_se` nul.

- **Bootstrap** : blocs de SNP consécutifs rééchantillonnés à l'intérieur de
  chaque chromosome (colonne 1 du `.bim`). La longueur par défaut (racine cubique
  du nombre de SNP) est notre choix : ADMIXTURE ne documente pas la sienne. Si
  tous les SNP sont codés sur le chromosome `0`, les blocs peuvent chevaucher la
  frontière entre deux chromosomes réels.
- **Arrêt de la CV** : chaque pli repart de l'estimation complète, avec le même
  critère d'arrêt que l'ajustement principal (`-C`).

### Pas encore validé contre ADMIXTURE x86

Il faut des sorties de référence produites par ADMIXTURE 1.3 sur une machine
Linux x86, sur les mêmes données :

1. **Orientation des allèles dans `.P`** : OpenADMIXTURE donne la fréquence de
   l'allèle A2 (colonne 6 du `.bim`). Il reste à vérifier qu'ADMIXTURE donne la
   même. Si ce n'est pas le cas, P_ADMIXTURE = 1 − P_OpenADMIXTURE.
2. **Valeurs de l'erreur de CV** : même formule, mais le masquage des plis
   diffère. Le K choisi devrait être le même ; les valeurs exactes non.
3. **Valeurs de `.Q_se`** : elles dépendent de la longueur des blocs.

En attendant, comparer `.Q` à une permutation des colonnes près (corrélation
> 0,99 attendue sur des données bien structurées), et ne pas mélanger des
erreurs de CV d'ADMIXTURE et d'OpenADMIXTURE dans une même courbe.

---

## 9. Intégration dans Moira

Moira (`/Users/loic/moira`) lance ADMIXTURE comme binaire externe dans
`workflow/lib/moira/admixture.py` (fonction `run`), avec :

```
<binaire> -s SEED -jN --cv <chemin absolu>/F{subset}.bed K      # cwd = dossier du run
```

puis lit `F{subset}.K.Q` et extrait `CV error` et `Loglikelihood` de la sortie
avec `CV_RE` et `LL_RE`. OpenADMIXTURE accepte exactement cette commande et
produit ces mêmes fichiers et ces mêmes lignes.

### Configuration

Dans `config/local.yaml` (non versionné) :

```yaml
tools:
  admixture: /opt/homebrew/bin/openadmixture
```

Le chemin du lien et celui du dépôt (`/Users/loic/OpenADMIXTURE.jl/bin/openadmixture`)
fonctionnent tous les deux. Le lanceur résout le lien, et il a besoin de
trouver `julia` dans le PATH : c'est le cas depuis les environnements conda
(`/opt/homebrew/bin` reste dans le PATH).

### Points de compatibilité vérifiés

| Élément de Moira | Comportement avec OpenADMIXTURE |
|---|---|
| `-s SEED` avec `derive_seed` ∈ [1, 2³¹−1] | accepté (toute graine entière ≥ 0) |
| `-j{threads}` (`threads: 4` dans la règle `structure_admixture_run`) | Julia est lancé avec 4 threads |
| `--cv` (5 plis) | identique |
| `cwd=outdir` | sorties écrites dans `outdir`, comme ADMIXTURE |
| `CV_RE`, `LL_RE` | format identique (une seule ligne `Loglikelihood:`) |
| `{bed.stem}.{k}.Q`, ordre des lignes = `.fam` | identique |
| `proc.returncode != 0` → nouvelle tentative | code 1 en cas d'erreur |
| `mem_mb = 4000 * attempt` | suffisant (~0,7 Go pour 379 ind. × 54 k SNP ; à revoir pour de très grands N) |
| `_set_unplaced_chromosomes` (chromosomes → `0`) | inutile pour OpenADMIXTURE mais sans effet, puisque Moira n'utilise pas `-B` |
| `admixture.log` (stdout + stderr) | plus long qu'avec ADMIXTURE (une ligne par itération) |

### Problèmes rencontrés lors des premiers essais (corrigés)

- **K = 1 refusé** : les versions antérieures au commit de K = 1 sortaient avec
  le code 1, alors que `DEFAULTS["k_min"] = 1` dans Moira. Avec 20 % des runs en
  échec, la règle `gather` s'arrêtait (`min_success_fraction = 0.9`). K = 1 est
  maintenant pris en charge ; `k_min: 1` fonctionne.
- **`--cv` plantait** (`ArgumentError: matrix contains Infs or NaNs` dans
  `inv()`, `update_QN!`, appelé depuis `cv_error`), au hasard selon la graine et
  K. Cause : `d.q` et `d.p` étaient des vues `unsafe_wrap` sur la mémoire du
  résultat de l'ajustement. Le ramasse-miettes pouvait libérer cette mémoire
  pendant la CV, et les plis repartaient alors de valeurs aléatoires. C'est
  corrigé à la racine. De plus, le pas quasi-Newton revient au pas EM simple
  quand la matrice à inverser n'est pas finie ou est singulière. Vérification :
  379 individus × 2 000 SNP, K = 1 à 3, graines 12345, 1, 2 et 3, avec `--cv` :
  12 runs sur 12 réussis.

### Ressources pour la planification Snakemake

- Chaque job paie environ 6,5 s de démarrage : ce coût compte davantage avec des
  sous-échantillons de 100 000 SNP qu'avec de gros jeux.
- Plusieurs jobs OpenADMIXTURE simultanés fonctionnent : le cache précompilé est
  partagé en lecture seule.
- **Après toute mise à jour** (`git pull` du fork, `brew upgrade julia`,
  changement de dépendances), lancer `openadmixture --help` une fois avant
  Snakemake. Sinon, chaque job démarré en parallèle recompile le paquet de son
  côté (plusieurs minutes chacun, en concurrence pour le même cache).

### Tests de Moira

Les tests qui lancent le vrai binaire peuvent utiliser `openadmixture` sur le
jeu synthétique (3 groupes × 10 individus diploïdes). Les petits jeux sont
dominés par les ~6,5 s de démarrage. Pour un test rapide, utiliser `-C 3`
(3 itérations).

---

## 10. Mise à jour et maintenance

```bash
cd /Users/loic/OpenADMIXTURE.jl
git pull                                                 # branche apple-silicon-fixes
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
openadmixture --help                                     # vérifie que tout démarre
```

- **Mise à jour de Julia** (`brew upgrade julia`) : une nouvelle version mineure
  (1.13 → 1.14) invalide le cache précompilé. Relancer `Pkg.precompile()`. Les
  correctifs Apple Silicon ont été validés sous Julia 1.13 ; lancer
  `Pkg.test()` après un changement de version mineure.
- **`Manifest.toml`** n'est pas versionné : une autre machine peut résoudre des
  versions de dépendances différentes. Pour figer les mêmes versions dans toute
  l'équipe, il faudrait le committer sur la branche du fork. Ce n'est pas encore
  fait.
- **Dépendances sensibles** : LoopVectorization (déprécié, mais il apporte 60 %
  de vitesse sur la boucle principale) et HostCPUFeatures. Une mise à jour de
  l'un d'eux est la première chose à suspecter en cas de régression de
  performance.
- **Code d'origine** : https://github.com/OpenMendel/OpenADMIXTURE.jl. Aucune
  pull request n'a été envoyée vers ce dépôt.

---

## 11. Dépannage

| Symptôme | Cause et solution |
|---|---|
| Le premier lancement prend plusieurs minutes | Le cache précompilé est absent ou invalidé. Attendre la fin, ou lancer `Pkg.precompile()` (§10). Normal après une mise à jour |
| `Note: -jN requested but Julia runs 1 thread(s)` | Julia a été appelé directement, sans le lanceur. Utiliser `openadmixture`, ou `julia -t N` |
| `matrix contains Infs or NaNs` (dans `update_QN!` / `cv_error`) | Version du fork antérieure au correctif : faire `git pull` (§10) |
| `Error: only PLINK .bed input is supported` | Le premier argument positionnel doit se terminer par `.bed` |
| `Error: missing file prefixe.bim` | Les trois fichiers PLINK doivent être côte à côte |
| `Error: ADMIXTURE option --supervised is not supported yet` | Voir §8 |
| `julia: command not found` dans un job | `/opt/homebrew/bin` n'est pas dans le PATH du job. L'ajouter, ou mettre un lien vers `julia` ailleurs |
| `cfunction: closures are not supported` | Ancienne version d'OpenADMIXTURE (Polyester). Vérifier que la branche `apple-silicon-fixes` est utilisée |
| `BoundsError` dans les threads | Ancienne version sous Julia ≥ 1.12. Même solution |
| NaN dans `.Q` ou `.P` | Ancienne version avec des SNP monomorphes. Même solution |
| `LLVM ERROR: Cannot select … vscale` | Uniquement avec PackageCompiler (§7). Ne pas construire d'application |

---

## 12. Utilisation depuis Julia

Pour enchaîner plusieurs K dans un seul processus (sans payer le démarrage à
chaque fois) ou accéder aux matrices directement :

```julia
# julia --project=/Users/loic/OpenADMIXTURE.jl -t 16
using OpenADMIXTURE, SnpArrays, StableRNGs

# 1. Exactement comme la ligne de commande (fichiers écrits dans pwd())
for K in 1:8
    OpenADMIXTURE.main(["-s", "43", "--cv", "data.bed", string(K)])
end

# 2. Accès direct
g = SnpArray("data.bed")
d = OpenADMIXTURE._admixture_base(g, 3; rng=StableRNG(43), progress_bar=false,
                                  n_iter=1000, rtol=1e-7, atol=1e-4)
Q = permutedims(d.q)        # N × K
P = permutedims(d.p)        # M × K
err = OpenADMIXTURE.cv_error(g, 3, d.q, d.p; folds=5, seed=43, quiet=true)
se  = OpenADMIXTURE.bootstrap_se(g, 3, d.q, d.p; B=200, quiet=true)   # N × K
```

`_admixture_base` est une fonction interne, réservée à K ≥ 2 ; pour K = 1,
utiliser `OpenADMIXTURE.fit_one_population(g)`. Son interface peut changer :
pour un usage durable, utiliser `main`.

---

## 13. Références

- Manuel ADMIXTURE 1.3 : https://dalexander.github.io/admixture/admixture-manual.pdf
- Article OpenADMIXTURE : Ko S. *et al.* (2023), *Am J Hum Genet* 110:314–325,
  https://doi.org/10.1016/j.ajhg.2022.12.008 — à citer avec ADMIXTURE
  (Alexander, Novembre & Lange 2009, *Genome Res* 19:1655–1664).
- Fork : https://github.com/ltalignani/OpenADMIXTURE.jl/tree/apple-silicon-fixes
- Historique des correctifs : `git log 401ac5c..apple-silicon-fixes` dans le dépôt.
