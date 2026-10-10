# data/ : your datasets

Put each dataset in its own folder `data/<dataset_name>/`. You choose `<dataset_name>` and
pass it to the API, for example `sclatch.load_dataset("huang_tcell")` reads
`data/huang_tcell/`. Rename your downloaded files to the exact names below.

**Bundled quick-run.** This repo ships two small datasets for an offline run, each with
`expression.h5ad`, `metadata.csv` and the precomputed paper mask (`atac_mask.npz`):

| Folder | Organism | Cells | Genes | Conditions | Notebook |
|---|---|---|---|---|---|
| `data/huang_tcell/` | mouse | 1200 (400 per condition, fixed seed) | 21,731 | naive, exhausted_wt, tox_ko | `sclatch_tutorial.ipynb` |
| `data/pbmc_10k/` | human | 1200 | 21,960 | none (pseudotime only) | `sclatch_pbmc.ipynb` |

They are labeled subsamples so a fresh clone runs the notebooks with no downloads; they are not
the paper runs. For the full datasets (huang_tcell 35,423 cells, pbmc_10k 11,617 cells), take
`processed_data/<name>/adata_rna.h5ad` and `cell_metadata.csv` from Zenodo Record 1 and rename
them as below; the full scATAC is on Google Drive as `example_atac/<name>/atac.h5ad`.

**scLATCH is multiomic.** The per-cell model input is RNA, but the regulatory mask requires
chromatin accessibility. You must supply accessibility either as your own scATAC
(`atac.h5ad`) or, for a cohort with no matched ATAC, as a matched-tissue REFERENCE scATAC
atlas passed to `build_prior(reference_atac=<path>)` (Fig 7 used the Satpathy TIL scATAC
atlas, GSE129785, for the melanoma cohort). Without accessibility, `build_prior` raises an
error.

## Files in data/<dataset_name>/

| File | Required | Format | Notes |
|---|---|---|---|
| `expression.h5ad` | required | AnnData, cells x genes | Raw counts in `.X`, or in a `counts` layer. `.obs_names` are cell barcodes, `.var_names` are gene symbols. |
| `atac.h5ad` | required (own scATAC) or supply a reference | AnnData, cells x peaks | scATAC accessibility, `.var_names` like `chr1:100-900`. Required to build the mask. If your cohort has no matched ATAC, omit this file and pass `reference_atac=<path>` to `build_prior` instead. |
| `metadata.csv` | required for conditions | CSV, `index_col=0` = cell barcode | A `condition` column (the categorical the dynamic model is resolved over); without it scLATCH trains pseudotime-only A(tau), as for `pbmc_10k`. An optional `cell_type` column is used for labels. Rows are matched to cells by barcode. |
| `ground_truth.csv` | optional | CSV with columns `Gene1,Gene2` | TF (`Gene1`) to target (`Gene2`) edges, for evaluation. Analysis-only datasets have none. |

## Rename your files

Downloaded a dataset as, say, `adata_rna.h5ad`, `cell_metadata.csv`, `adata_atac.h5ad`?
Rename them:

```
data/huang_tcell/
├── expression.h5ad     (was adata_rna.h5ad)
├── atac.h5ad           (was adata_atac.h5ad; required for the mask, or use a reference)
├── metadata.csv        (was cell_metadata.csv; keep the condition column)
└── ground_truth.csv    (optional; none for huang_tcell)
```

`sclatch.load_dataset("huang_tcell")` then reads this folder. Missing or misnamed files give
a clear error pointing back here.

## Recompute the mask (optional)

The default `build_prior(ds)` loads the bundled precomputed `atac_mask.npz` offline. To recompute
the mask from raw inputs with `build_prior(ds, rebuild=True)`, place the dataset's scATAC at
`data/<dataset_name>/atac.h5ad` (here; Google Drive `example_atac/<name>/atac.h5ad` for the two
examples) and extract the ChIP-Atlas cache into `priors/chip_atlas/<genome>/` (see "Download the
priors" in the main README). The TF lists and TSS tables already ship in `priors/`. The rebuilt
mask reproduces the bundled one:

| Dataset | `tf_list` | Rebuilt edges | Bundled edges | Jaccard |
|---|---|---|---|---|
| huang_tcell | `pyscenic` | 920,542 | 917,427 | 0.958 |
| pbmc_10k | `scenicplus` (default) | 3,283,896 | 3,237,289 | 0.982 |

The large files live on Google Drive `<DRIVE_LINK>` (placeholder) and the Zenodo record (DOI
on acceptance). Git tracks only this README, `.gitkeep`, and the `expression.h5ad`,
`metadata.csv` and `atac_mask.npz` of the two bundled folders; `atac.h5ad` and any dataset you
add here are ignored.
