# Input data

Each dataset is a folder `data/<name>/`. `sclatch.load_dataset("<name>")` reads it.

| File | Required | Content |
|---|---|---|
| `expression.h5ad` | yes | AnnData, cells x genes. Raw counts in `.X` or `layers["counts"]`; `obs_names` are cell barcodes, `var_names` are gene symbols. |
| `atac.h5ad` | to build a mask | AnnData, cells x peaks; `var_names` such as `chr1:100-900`. |
| `metadata.csv` | no | Indexed by cell barcode. A `condition` column gives condition-resolved networks; without it scLATCH learns a pseudotime-only network. An optional `cell_type` column is used for labels. |
| `atac_mask.npz` | no | Precomputed binary mask, sparse genes x genes, rows = targets, columns = TFs, in the gene order of `expression.h5ad`. `build_prior` loads it when present. |
| `ground_truth.csv` | no | Reference edges for evaluation, columns `Gene1` (TF) and `Gene2` (target). |

## Chromatin accessibility

The regulatory mask needs accessibility from one of two sources:

1. the dataset's own scATAC: `data/<name>/atac.h5ad`;
2. a reference scATAC atlas from a matched tissue: `build_prior(ds, reference_atac="atlas.h5ad")`.

Without either, `build_prior` stops with an error. `train(ds, mask=None)` trains without a
prior (RNA only), with lower accuracy.

## Bundled examples

| Folder | Organism | Cells | Genes | Conditions |
|---|---|---|---|---|
| `huang_tcell/` | mouse | 1,200 (400 per condition) | 21,731 | naive, exhausted_wt, tox_ko |
| `pbmc_10k/` | human | 1,200 | 21,960 | none |

Both are subsamples for a quick offline run and include their precomputed masks. To build the
masks again, copy `example_atac/<name>/atac.h5ad` from
[Google Drive](https://drive.google.com/drive/folders/1Uj2zvCKHfi9ET7Ud1NzR39VKzNwefF-p?usp=sharing)
to `data/<name>/atac.h5ad`, install the ChIP-Atlas cache (see `priors/README.md`) and run

```python
mask = sclatch.build_prior(ds, rebuild=True, tf_list="pyscenic", allow_download=False)   # huang_tcell
mask = sclatch.build_prior(ds, rebuild=True, allow_download=False)                       # pbmc_10k
```

The rebuilt masks match the bundled ones (Jaccard 0.96 and 0.98). The new mask is written to
`outputs/masks/<name>/atac_mask.npz`.

Git tracks only the bundled example files in this folder; anything else you add is ignored.
