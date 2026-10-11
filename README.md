# scLATCH

scLATCH infers dynamic gene regulatory networks (GRNs) from single-cell multiome data
(paired scRNA-seq and scATAC-seq). It trains a conditional denoising diffusion model whose
denoiser contains a learnable gene-by-gene adjacency matrix A; the trained A is the GRN. A
low-rank term gated by pseudotime τ and an optional experimental condition k lets the network
change along a trajectory and between conditions, A(τ, k). Candidate edges are limited to
ChIP-Atlas transcription factor (TF) binding near each gene's transcription start site (TSS)
that falls in accessible chromatin.

## Installation

Training is designed for a CUDA GPU (we used a 24 GB GPU); it runs on CPU but is very slow.
Install a GPU build of PyTorch **before** scLATCH, so that pip does not install a default build
that may not match your CUDA driver.

```bash
git clone https://github.com/debraj-55555/scLATCH.git
cd scLATCH

# 1. PyTorch >= 2.0 for your GPU: check your CUDA version with `nvidia-smi`, then copy the
#    matching command from https://pytorch.org/get-started/locally/ (example: CUDA 12.6)
pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu126

# 2. scLATCH
pip install -e ".[viz]"
```

Check the installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # must print True
python -c "import sclatch; print(sclatch.__version__)"
```

If the first command prints `False`, reinstall PyTorch with the build that matches your driver.
Requirements: Python 3.9+, PyTorch 2.0+; other dependencies install automatically.

## Quick start

Two executed notebooks run the full workflow offline on small bundled datasets
(1,200 cells each, all genes, precomputed mask):

| Notebook | Dataset | Mode |
|---|---|---|
| [`notebooks/sclatch_tutorial.ipynb`](notebooks/sclatch_tutorial.ipynb) | `data/huang_tcell`: mouse CD8 T cells; naive, exhausted and TOX-knockout | pseudotime + condition |
| [`notebooks/sclatch_pbmc.ipynb`](notebooks/sclatch_pbmc.ipynb) | `data/pbmc_10k`: human PBMC | pseudotime only |

## Usage

### Python

```python
import sclatch

ds    = sclatch.load_dataset("huang_tcell", organism="mouse")   # reads data/huang_tcell/
ds    = sclatch.preprocess(ds, root_gene="Tcf7")                # normalization, pseudotime
mask  = sclatch.build_prior(ds, organism="mouse")               # regulatory mask
model = sclatch.train(ds, mask, condition_key="condition")      # saves runs/huang_tcell/dynamic/model.pt
# reload later: model = sclatch.load_model("runs/huang_tcell/dynamic/model.pt", dataset=ds)

conds = model.conditions()
model.static_network().plot()                                   # static GRN
model.dynamic_network(pseudotime=0.5, condition=conds[0])       # A(τ, k)
model.reconfiguration_curve(condition=conds[0]).plot()          # network change along τ
model.commitment_boundary(condition=conds[0])                   # τ of the largest change
model.delta_S(conds[0], conds[1]).plot()                        # regulator output between conditions
model.knockout("Tox", condition=conds[0]).plot()                # in-silico knockout
model.save_results("outputs/huang_tcell")                       # all results, documented
```

Each analysis returns its values and has a `.plot()` method. Other analyses:
`regulatory_output`, `quasi_potential`, `gene_strength_heatmap`, `plot_network`,
`plot_differential_network`, `network_snapshots`, `edge_trajectory`, `top_changing_edges`.
Without a condition (`condition_key=None`) scLATCH learns A(τ) only, and the
between-condition analyses are not available.

### Command line

```bash
sclatch-preprocess --data_dir data/huang_tcell --organism mouse --root_gene Tcf7
sclatch-mask       --data_dir processed/huang_tcell --organism mouse --tf_list pyscenic --no_download
sclatch-train      --data_dir processed/huang_tcell --continuous pseudotime \
                   --categorical condition --atac_mask
```

`sclatch-preprocess` writes `processed/<name>/`, and the next two commands read it. Its QC
removes genes detected in fewer than `--min_cells` cells (default 10), so `sclatch-mask` builds
a new mask; this step needs `data/<name>/atac.h5ad` and the ChIP-Atlas cache (see
[Priors](#priors)). Omit `--categorical condition` for data without conditions. Run any command
with `--help` for all options.

## Input data

Put each dataset in `data/<name>/`: `expression.h5ad` (raw counts, required), `atac.h5ad`
(scATAC, to build a mask), `metadata.csv` (cell barcodes, optional `condition` column) and an
optional precomputed `atac_mask.npz`. If your data have no matched scATAC, use a reference
scATAC atlas from a matched tissue: `build_prior(ds, reference_atac="atlas.h5ad")`. See
[`data/README.md`](data/README.md) for the file formats.

## Priors

The TF lists and TSS tables are in `priors/`. The ChIP-Atlas binding cache is needed only to
build a new mask; the bundled examples include their masks. To install it, download
`scLATCH_priors_chip_atlas.tar.gz` (335 MB) from
[Google Drive](https://drive.google.com/drive/folders/1Uj2zvCKHfi9ET7Ud1NzR39VKzNwefF-p?usp=sharing)
and extract it from the repository root:

```bash
tar -xzf scLATCH_priors_chip_atlas.tar.gz -C priors/
ls priors/chip_atlas/hg38/*.10.tsv | wc -l    # 1183 human TFs
ls priors/chip_atlas/mm10/*.10.tsv | wc -l    # 498 mouse TFs
```

scLATCH reads `priors/chip_atlas/` automatically. See [`priors/README.md`](priors/README.md)
for the TF lists and options.

## Outputs

`model.save_results(dir)` writes a `static/` and a `dynamic/` folder of CSV and NPZ files (edge
lists, regulator output, network change along τ, network snapshots) and a `RESULTS_README.md`
that describes every file and column. `model.export_network(path, condition=...)` writes one
edge list (`regulator_TF, target_gene, weight, sign`) for pandas, networkx or Cytoscape.
`sclatch-train` writes `results/<dataset>/<run>/` with `adjacency.npz`,
`GRN_inference_result.tsv` (ranked edges), `model.pt` and `training_config.json`.

## Data availability

- **Example data.** huang_tcell: GEO [GSE255042](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE255042).
  pbmc_10k: 10x Genomics multiome `pbmc_granulocyte_sorted_10k`.
- **Google Drive** ([folder](https://drive.google.com/drive/folders/1Uj2zvCKHfi9ET7Ud1NzR39VKzNwefF-p?usp=sharing)):
  the ChIP-Atlas cache (`scLATCH_priors_chip_atlas.tar.gz`) and the full scATAC of the two
  examples (`example_atac/<name>/atac.h5ad`).
- **Zenodo** ([doi:<ZENODO_DOI>](https://doi.org/<ZENODO_DOI>)): processed datasets, inferred
  networks and trained models.

## Citation

<CITATION>

## License

MIT. See [`LICENSE`](LICENSE).
