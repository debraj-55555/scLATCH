# scLATCH

scLATCH infers pseudotime- and condition-resolved gene regulatory networks (GRNs) from
single-cell multiome data. It trains a conditional denoising diffusion model whose denoiser
embeds a learnable gene by gene adjacency A. The converged A is the GRN; there is no reverse
sampling at inference. A low-rank, pseudotime- and condition-gated term produces a dynamic
network A(tau, k).

## Method

- **A at the output.** Per-gene embeddings are mixed by `(I - A)` at the output, so the
  adjacency is the inferred network. The static adjacency is
  `A_static = sign(W) relu(|W| - tau_thr)`, masked and with the diagonal zeroed.
- **Multiome mask.** Edges are restricted to ChIP-Atlas TF binding within 10 kb of the TSS
  intersected with accessible scATAC peaks, over a curated TF list (`priors/README.md`). The mask is
  soft: `effective_mask = mask + 0.4 (1 - mask)`.
- **Dynamic adjacency.** `A(tau, k) = A_static + U diag(gate(c)) V` (rank 32). The gate reads a
  condition vector built from diffusion time, pseudotime tau, and an optional categorical
  condition k, each L2-normalized before summation.
- **Training.** Single-stage joint optimization; the final converged checkpoint is used;
  ground truth never enters model selection. Every hyperparameter is `FINAL_DYNAMIC_CONFIG`.

## Install

scLATCH trains on a CUDA GPU (the notebooks were run on a 24 GB GPU). Install a GPU build of
PyTorch **before** scLATCH; otherwise pip installs a default torch wheel that may not match your
CUDA driver.

```bash
git clone https://github.com/<org>/sclatch.git
cd sclatch

# 1. PyTorch >= 2.0 for your GPU. Check the CUDA version your driver supports with `nvidia-smi`,
#    then copy the matching command from https://pytorch.org/get-started/locally/
#    Example (CUDA 12.6 build):
pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu126

# 2. scLATCH
pip install -e ".[viz]"        # package + matplotlib/networkx for the plots
```

Python 3.9+ and PyTorch 2.0+. Core dependencies (installed automatically): torch, numpy,
pandas, scipy, scikit-learn, scikit-misc, scanpy, anndata, leidenalg. Verify that PyTorch sees
the GPU and that scLATCH is installed:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
python -c "import sclatch; print(sclatch.__version__)"
sclatch-train --help
```

If `torch.cuda.is_available()` prints `False`, reinstall PyTorch with the CUDA build that
matches your driver. Without a GPU, install the CPU build
(`--index-url https://download.pytorch.org/whl/cpu`); preprocessing, mask building and the
analyses run on CPU, but training on all genes is very slow.

## Download the priors (only to rebuild a mask)

The TF lists and TSS tables ship in `priors/`. The ChIP-Atlas binding cache (1,681 files,
1.7 GB unpacked) does not. You need it only to rebuild a mask from scATAC; the two bundled
examples load their precomputed masks without it.

1. Download `scLATCH_priors_chip_atlas.tar.gz` (335 MB) from Google Drive `<DRIVE_LINK>`.
2. From the repository root, extract it into `priors/`:

```bash
tar -xzf scLATCH_priors_chip_atlas.tar.gz -C priors/
sha256sum -c --quiet priors/chip_atlas/SHA256SUMS   # optional check (macOS: shasum -a 256 -c)
ls priors/chip_atlas/hg38/*.10.tsv | wc -l          # 1183 human TFs
ls priors/chip_atlas/mm10/*.10.tsv | wc -l          # 498 mouse TFs
```

The files land in `priors/chip_atlas/hg38/{TF}.10.tsv` and `priors/chip_atlas/mm10/{TF}.10.tsv`.
`build_prior` and `sclatch-mask` read this folder by default; no path setting is needed. The
cache is the ChIP-Atlas snapshot used for the paper (downloaded 30 March to 10 April 2026). Pass
`allow_download=False` (CLI `--no_download`) to build only from it; otherwise a TF missing from
the cache is fetched from ChIP-Atlas, whose data change over time.

## Quickstart (Python API)

Put a dataset in `data/<name>/` (see [`data/README.md`](data/README.md)) and priors in
`priors/` (see [`priors/README.md`](priors/README.md)):

```python
import sclatch

ds    = sclatch.load_dataset("huang_tcell")                  # reads data/huang_tcell/
ds    = sclatch.preprocess(ds, root_gene="Tcf7")            # normalize, embed, pseudotime
mask  = sclatch.build_prior(ds, organism="mouse")           # ChIP-Atlas ∩ ATAC ∩ TF list
model = sclatch.train(ds, mask, condition_key="condition")  # exact paper params
# train() saves runs/<name>/dynamic/model.pt; reload it later without retraining:
# model = sclatch.load_model("runs/<name>/dynamic/model.pt", dataset=ds)

conds = model.conditions()                                  # read from the data
model.static_network().plot()
model.reconfiguration_curve(condition=conds[0]).plot()
model.commitment_boundary(condition=conds[0])
model.plot_network(condition=conds[0], pseudotime=0.5)
model.delta_S(conds[0], conds[1]).plot()
model.knockout("Tox", condition=conds[0]).plot()
model.save_results("outputs/my_results")
```

The lower-level model is `from sclatch.model import scLATCH`. For scripted or headless
use (HPC, batch jobs), the same pipeline runs from the command line (below).

## Quickstart (command line)

The CLI runs the same steps without a notebook. `sclatch-preprocess` reads a raw
`data/<name>/` folder and writes a `processed/<name>/` folder (`data.npy`, `pseudotime.npy`,
`cell_metadata.csv`, `adata_atac.h5ad`); `sclatch-mask` and `sclatch-train` then read that
processed folder. Unlike the Python API, which keeps the full gene set, the CLI QC also drops
genes detected in fewer than `--min_cells` cells (default 10), so it builds its own mask rather
than using the bundled one.

```bash
# 1. preprocess: QC, normalize, embed, pseudotime -> processed/huang_tcell/
sclatch-preprocess --data_dir data/huang_tcell --organism mouse --root_gene Tcf7

# 2. build the ChIP-Atlas ∩ ATAC mask from the local cache (needs priors/chip_atlas/,
#    see "Download the priors", and data/huang_tcell/atac.h5ad)
sclatch-mask --data_dir processed/huang_tcell --organism mouse --tf_list pyscenic --no_download

# 3. train the dynamic model with pseudotime + condition + mask, and write the GRN
sclatch-train --data_dir processed/huang_tcell \
    --continuous pseudotime --categorical condition --atac_mask
```

The `condition` categories come from `metadata.csv`; drop `--categorical condition` for
pseudotime-only data. Results land in `results/<dataset>/<run>/`: `adjacency.npz` (full
signed weight matrix), `GRN_inference_result.tsv` (ranked edge list), `model.pt`, and
`training_config.json`. Run `sclatch-train --help` for the full flag list;
`sclatch-ground-truth` fetches the CollecTRI network for benchmark scoring.

## Accessibility (multiome)

The mask requires chromatin accessibility. Provide it in this order of preference:

1. **Own scATAC (default).** Put `atac.h5ad` in `data/<name>/`; `build_prior` intersects
   ChIP-Atlas binding with your accessible peaks.
2. **Reference atlas.** For a cohort with no matched ATAC, pass a matched-tissue reference:
   `build_prior(ds, reference_atac="atlas.h5ad")`. Fig 7 used the Satpathy TIL scATAC atlas
   (GSE129785).
3. **Mask-free (last resort).** `train(ds, mask=None)` drops the prior and runs RNA-only
   de-novo inference at expression-baseline level (the no-prior decomposition, AUPRC ~1.24 on
   PBMC). No benchmark uses it.

## No-condition (pseudotime-only) mode

If you do not provide a condition (`condition_key=None`, or no condition column), scLATCH
trains pseudotime-only A(tau); `model.conditions()` returns an empty list. The tau-dynamics all
apply: static and tau-only networks, `reconfiguration_curve`, `commitment_boundary`,
`gene_strength_heatmap`, `network_snapshots`, `edge_trajectory`, `knockout`, `quasi_potential`,
`plot_network`, and `save_results` (which writes STATIC plus tau-only DYNAMIC files). Compare
two pseudotimes with `top_changing_edges(tau_a=..., tau_b=...)`. The condition-only analyses
(`delta_S`, `plot_differential_network` between conditions) need at least two conditions and
raise a clear error otherwise. See `notebooks/sclatch_pbmc.ipynb`.

## Reading out and saving results

Each analysis is a method on the trained model returning values and a focused `.plot()`:
`static_network`, `dynamic_network(pseudotime, condition)`, `regulatory_output(condition)`,
`reconfiguration_curve`, `commitment_boundary`, `delta_S(a, b)`, `knockout(tf_or_list)`,
`quasi_potential`, `gene_strength_heatmap`, `plot_network`, `plot_differential_network`,
`network_snapshots`, `edge_trajectory(tf, target)`, `top_changing_edges`.

Export with documented, repo-relative formats:

```python
model.export_network("outputs/net.csv", condition=model.conditions()[0])
model.save_results("outputs/my_results")
```

`export_network` writes an edge-list CSV (`regulator_TF, target_gene, weight, sign`; weight is
the grn-normalized `A(tau,k)[target, regulator]`, sign +1 activation, -1 repression), loadable
in pandas, networkx, or Cytoscape. `save_results` writes a `static/` folder (static edge list,
static per-TF output S) and a `dynamic/` folder (per-condition edge lists, S by condition,
`delta_S`, reconfiguration curve + boundary, gate-entropy curve, top-changing edges, A(tau)
snapshots `.npz`), plus a `RESULTS_README.md` that labels each file static or dynamic,
documents every column, and gives code to reload and use each file without the model.

## Data and priors layout

```
data/<name>/       expression.h5ad (required), atac.h5ad (multiome), metadata.csv, ground_truth.csv
priors/            TF lists (*.txt), tss/*, chip_atlas/{hg38,mm10}/{TF}.10.tsv
```

`priors/` is the single location for all priors. `data/README.md` and `priors/README.md`
document the file formats. The TF lists and the TSS tables ship in the repo `priors/`;
download the ChIP-Atlas cache into `priors/chip_atlas/` only to rebuild masks
(`build_prior(rebuild=True)`; see "Download the priors"). `build_prior` reads these locations by default; override with
`SCLATCH_CHIPATLAS_DIR` / `cache_dir` / `tss_file`. The priors are not bundled in the pip wheel:
installing via pip without cloning requires obtaining the `priors/` folder (from the repo or the
Zenodo priors bundle) and pointing `SCLATCH_DATA_ROOT` at the directory that holds it.

Two curated TF lists ship in `priors/`: the SCENIC+ lists (`tf_list="scenicplus"`, the default) and
the pySCENIC lists (`tf_list="pyscenic"`). The bundled `pbmc_10k` mask was built with the SCENIC+
human list and the bundled `huang_tcell` mask with the pySCENIC mouse list.

## Notebooks

- [`notebooks/sclatch_tutorial.ipynb`](notebooks/sclatch_tutorial.ipynb): full worked example
  on huang_tcell (Fig 4, three conditions), all cells executed with outputs and figures inline.
- [`notebooks/sclatch_pbmc.ipynb`](notebooks/sclatch_pbmc.ipynb): the same pipeline on human
  PBMC with no condition (pseudotime-only), executed with outputs inline.

Both run offline on the bundled quick-run data in `data/huang_tcell/` and `data/pbmc_10k/`
(1200 cells each, all genes, plus the precomputed mask). These are labeled subsamples for a
fast run, not the paper runs. Each notebook has one optional cell that rebuilds the mask from
scATAC. It runs when `priors/chip_atlas/` (see "Download the priors") and
`data/<name>/atac.h5ad` (Google Drive `example_atac/<name>/atac.h5ad`) are present, and
reproduces the bundled mask (Jaccard 0.958 huang_tcell, 0.982 pbmc_10k). Without them it prints
a skip line and the notebook continues with the bundled mask.

## Data availability

All datasets are public. huang_tcell (Fig 4): GSE255042 (Huang et al., Sci. Immunol. 2025).
PBMC: 10x Genomics `pbmc_granulocyte_sorted_10k` multiome. Reference TIL scATAC atlas (Fig 7):
GSE129785 (Satpathy et al. 2019).

- **Google Drive** `<DRIVE_LINK>`: `scLATCH_priors_chip_atlas.tar.gz` (ChIP-Atlas cache;
  extract into `priors/`, see "Download the priors") and `example_atac/huang_tcell/atac.h5ad`,
  `example_atac/pbmc_10k/atac.h5ad` (full scATAC of the two examples; copy each to
  `data/<name>/atac.h5ad`).
- **Zenodo** (DOI on acceptance): Record 1 holds the processed RNA matrices, pseudotime,
  metadata and masks of every dataset, the same ChIP-Atlas cache
  (`priors/chip_atlas_cache/{hg38,mm10}/`), the ground-truth networks, the inferred networks and
  the figure source data. Record 2 holds the trained model checkpoints.

## License

MIT. See [`LICENSE`](LICENSE).
