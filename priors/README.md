# priors/ : regulatory priors

`priors/` is the single location for all regulatory priors. scLATCH reads them relative to the
repo root. The curated TF lists and the TSS tables ship in this folder; the large ChIP-Atlas
binding cache is downloaded here.

scLATCH is multiome: the mask is ChIP-Atlas binding (this folder) intersected with scATAC
accessibility over a curated TF list. Accessibility comes from the dataset's own scATAC
(`data/<name>/atac.h5ad`) or a matched-tissue reference atlas passed to `build_prior`, not from
this folder. See `data/README.md`.

## What is here

```
priors/
├── allTFs_hg38.txt         # human TF list, SCENIC+ (ships in repo)
├── allTFs_mm.txt           # mouse TF list, SCENIC+ (ships in repo)
├── hs_hgnc_tfs.txt         # human TF list, pySCENIC (ships in repo)
├── mm_mgi_tfs.txt          # mouse TF list, pySCENIC (ships in repo)
├── tss/
│   ├── TSS_extend_1M.txt   # human (hg38) TSS table (ships in repo)
│   └── mm10_gene_tss.csv   # mouse (mm10) TSS table (ships in repo)
└── chip_atlas/
    ├── hg38/{TF}.10.tsv    # ChIP-Atlas human binding cache (DOWNLOAD)
    └── mm10/{TF}.10.tsv    # ChIP-Atlas mouse binding cache (DOWNLOAD)
```

## TF lists (shipped)

The candidate regulators come from one of two curated TF lists of the Aerts laboratory. Both
ship in this folder, so `build_prior` reads them with no download. Neither list depends on any
evaluation ground truth. Choose the list with `tf_list=` (CLI `--tf_list`).

| `tf_list` | human | mouse | source |
|---|---|---|---|
| `scenicplus` (default) | `allTFs_hg38.txt` | `allTFs_mm.txt` | SCENIC+ (Bravo González-Blas et al. 2023), https://resources.aertslab.org/cistarget/tf_lists/ |
| `pyscenic` | `hs_hgnc_tfs.txt` | `mm_mgi_tfs.txt` | pySCENIC (Van de Sande et al. 2020), https://github.com/aertslab/pySCENIC (resources/) |

The paper benchmark masks (PBMC, BMMC, SNARE-seq, mouse skin; `data/pbmc_10k` here) used the
SCENIC+ lists. The paper biology masks (Huang T cell, LCMV, ovarian TIL, melanoma;
`data/huang_tcell` here) used the pySCENIC lists. To rebuild a paper mask, pass the list it was
built with.

## TSS tables (shipped)

`tss/TSS_extend_1M.txt` (human) and `tss/mm10_gene_tss.csv` (mouse) give each gene's
transcription start site. With these present, `build_prior` maps peaks to genes offline, with
no Ensembl call, reproducing the paper TSS snapshot.

## ChIP-Atlas cache (download)

The per-TF ChIP-Atlas binding tables are large (1,681 files, 1.7 GB unpacked) and are not in
the repo. Download `scLATCH_priors_chip_atlas.tar.gz` (335 MB) from Google Drive `<DRIVE_LINK>`
(placeholder) and, from the repository root, run

```bash
tar -xzf scLATCH_priors_chip_atlas.tar.gz -C priors/
sha256sum -c --quiet priors/chip_atlas/SHA256SUMS   # optional check
```

The files land in `priors/chip_atlas/hg38/` (1,183 human TFs) and `priors/chip_atlas/mm10/`
(498 mouse TFs) as `{TF}.10.tsv`, which is where `build_prior` looks; no path setting is needed.
The same cache is in Zenodo Record 1 as `priors/chip_atlas_cache/{hg38,mm10}/`; copy those two
folders to `priors/chip_atlas/` if you use Zenodo instead.

The ChIP-Atlas cache is only needed to rebuild a mask from scratch. The paper masks were built
from this cache. Pass `allow_download=False` (CLI `--no_download`) to use only the cached files;
by default a TF missing from the cache is fetched from ChIP-Atlas, whose data change over time.
Precomputed masks are
provided per dataset and `build_prior` loads them when present, so the cache is optional for
running the model. If the cache is missing when you rebuild, `build_prior` raises a clear error
pointing here. Override the location with `SCLATCH_CHIPATLAS_DIR` or `cache_dir` / `--cache_dir`.

## Installing via pip without cloning

The priors are not bundled in the pip wheel. If you `pip install` scLATCH without cloning the
repo, obtain the `priors/` folder separately (from the repo or the Zenodo priors bundle) and set
`SCLATCH_DATA_ROOT` to the directory that contains it so the code can resolve `priors/`.
