"""High-level scLATCH API: load a dataset, preprocess it, build the prior, train, analyze.

The functions here wrap the canonical trainer and model so a user works with named datasets
and a trained-model object instead of the CLI. The folder conventions are documented in
data/README.md (datasets) and priors/README.md (priors).

    import sclatch
    ds    = sclatch.load_dataset("huang_tcell")            # reads data/huang_tcell/
    ds    = sclatch.preprocess(ds, root_gene="Tcf7")       # normalize, embed, pseudotime
    mask  = sclatch.build_prior(ds, organism="mouse")      # ChIP-Atlas ∩ ATAC ∩ TF list (offline)
    model = sclatch.train(ds, mask, condition_key="condition")   # exact paper params
    # or: model = sclatch.load_model("runs/<name>/dynamic/model.pt", dataset=ds)

    conds = model.conditions()                    # read from the data, never hardcoded
    curve = model.reconfiguration_curve(condition=conds[0]); curve.plot()
    dS    = model.delta_S()                        # defaults to the first two of conditions()

Normalization matches the paper: preprocessing applies library-size + log1p; the model then
applies per-cell min-max followed by per-gene z-score internally at train time.
"""

import os
import sys
import json
import logging
import subprocess

import numpy as np
from sclatch._verbose import log as _log, milestone as _ms
import pandas as pd
import torch

from sclatch.model import scLATCH
from sclatch.training import FINAL_DYNAMIC_CONFIG, build_command


def _disp(p):
    """Repo-relative display for a path (never changes the path used on disk)."""
    try:
        rp = os.path.relpath(str(p))
        return rp if not rp.startswith("..") else str(p)
    except Exception:
        return str(p)


# ---------------------------------------------------------------------------
# Small result object: values as attributes, plus an optional focused .plot()
# ---------------------------------------------------------------------------

class AnalysisResult:
    """Return value of an analysis method: named data attributes plus .plot(ax)."""

    def __init__(self, _plot=None, **data):
        self.__dict__.update(data)
        self._plot = _plot

    def plot(self, ax=None):
        if self._plot is None:
            raise NotImplementedError("this result has no plot")
        return self._plot(ax)

    def __repr__(self):
        keys = [k for k in self.__dict__ if not k.startswith("_")]
        return f"AnalysisResult({', '.join(keys)})"


_PALETTE = ["#c1663f", "#4a7ba6", "#8a6ba6", "#6a9a5b", "#b0902f", "#7a7a7a"]


def _ax(ax, figsize=(4.2, 3.0)):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset:
    """A named dataset following the data/<name>/ convention.

    Attributes
    ----------
    name : str
    adata : AnnData                 RNA expression (raw counts kept in .layers['counts'])
    atac : AnnData or None          scATAC peaks (for mask building)
    gene_names : list[str]
    metadata : DataFrame or None    per-cell metadata (has a 'condition' column if present)
    pseudotime : np.ndarray or None (n_cells,) in [0, 1], set by preprocess()
    ground_truth : DataFrame or None  columns Gene1, Gene2
    organism : str
    """

    def __init__(self, name, adata, atac=None, metadata=None, ground_truth=None,
                 organism="mouse", data_dir="data"):
        self.name = name
        self.adata = adata
        self.atac = atac
        self.metadata = metadata
        self.ground_truth = ground_truth
        self.organism = organism
        self.data_dir = data_dir
        self.gene_names = list(adata.var_names)
        self.pseudotime = None
        self._lognorm = None      # (n_cells, n_genes) log-normalized expression
        self.mask = None

    @property
    def n_cells(self):
        return self.adata.n_obs

    @property
    def n_genes(self):
        return self.adata.n_vars

    def conditions(self, key="condition"):
        if self.metadata is None or key not in self.metadata.columns:
            return None
        return self.metadata[key]

    def __repr__(self):
        return (f"Dataset('{self.name}', {self.n_cells} cells x {self.n_genes} genes, "
                f"organism={self.organism}, atac={'yes' if self.atac is not None else 'no'}, "
                f"conditions={'yes' if self.conditions() is not None else 'no'})")


def load_dataset(name, data_dir="data", organism="mouse"):
    """Load data/<name>/ per the data/README.md convention.

    Reads expression.h5ad (required), metadata.csv (required for conditions), atac.h5ad
    (optional), ground_truth.csv (optional). Raises a clear error pointing to data/README.md
    if the required files are missing.
    """
    import anndata as ad
    import scanpy as sc

    folder = os.path.join(data_dir, name)
    expr = os.path.join(folder, "expression.h5ad")
    if not os.path.exists(expr):
        raise FileNotFoundError(
            f"{expr} not found. Put your dataset in {folder}/ and rename the expression file "
            f"to expression.h5ad (raw counts in .X or a 'counts' layer). See data/README.md.")
    _log(f"[1/2] loading expression: {expr}")
    adata = sc.read_h5ad(expr)
    adata.var_names_make_unique()
    _log(f"      {adata.n_obs} cells x {adata.n_vars} genes; "
          f"layers={list(adata.layers.keys())}")

    metadata = None
    meta_path = os.path.join(folder, "metadata.csv")
    if os.path.exists(meta_path):
        metadata = pd.read_csv(meta_path, index_col=0)
        # align to cells by barcode where possible
        common = adata.obs_names.intersection(metadata.index)
        if len(common) >= 0.5 * adata.n_obs and len(common) > 0:
            metadata = metadata.reindex(adata.obs_names)
        cond = "condition" in metadata.columns
        _log(f"[2/2] metadata: {meta_path} ({metadata.shape[1]} columns, "
              f"condition={'yes' if cond else 'no'})")
        if cond:
            _log(f"      conditions: {dict(metadata['condition'].value_counts())}")
    else:
        _log(f"[2/2] metadata: none (no {meta_path}); conditions unavailable")

    atac = None
    atac_path = os.path.join(folder, "atac.h5ad")
    if os.path.exists(atac_path):
        atac = sc.read_h5ad(atac_path)
        _log(f"      atac: {atac_path} ({atac.n_obs} cells x {atac.n_vars} peaks)")
    else:
        _log("      atac: none (scLATCH is multiomic; the mask needs accessibility). Add "
              "atac.h5ad or pass build_prior(reference_atac=<path>). See data/README.md.")

    gt = None
    gt_path = os.path.join(folder, "ground_truth.csv")
    if os.path.exists(gt_path):
        gt = pd.read_csv(gt_path)
        _log(f"      ground_truth: {gt_path} ({len(gt)} edges)")

    return Dataset(name, adata, atac=atac, metadata=metadata, ground_truth=gt,
                   organism=organism, data_dir=data_dir)


# ---------------------------------------------------------------------------
# Preprocess (normalize, embed, pseudotime with an explicit root)
# ---------------------------------------------------------------------------

def preprocess(dataset_or_name, root_gene=None, root_cell_type=None, organism=None,
               data_dir="data", n_pcs=50, n_neighbors=30):
    """Normalize, embed, and compute diffusion pseudotime for a Dataset.

    Steps (printed): load, metadata/conditions, QC counts, normalization (library-size +
    log1p), embedding (PCA + kNN graph), diffusion pseudotime with a chosen root and a
    direction check. The provided gene set is preserved (no gene dropping), so it stays
    aligned with the precomputed mask and released checkpoints. The model's own per-cell
    min-max then per-gene z-score normalization is applied later, at train time.

    Root cell selection (pseudotime tau = 0):
      1. root_gene given: the cell with the highest expression of that gene.
      2. root_cell_type given: that cell type at the diffusion-component-1 extreme.
      3. neither: a gene-count (CytoTRACE-like) heuristic, with a printed root suggestion.
    Direction is validated: with a root_gene, tau is inverted to 1-tau if the root gene
    increases along tau (Spearman rho > +0.05); the root marker should decrease along tau.
    """
    import scanpy as sc
    from sclatch.preprocessing.pseudotime import compute_pseudotime

    ds = (load_dataset(dataset_or_name, data_dir=data_dir, organism=organism or "mouse")
          if isinstance(dataset_or_name, str) else dataset_or_name)
    if organism:
        ds.organism = organism
    adata = ds.adata

    _log(f"[1/6] input: {adata.n_obs} cells x {adata.n_vars} genes")

    # [2/6] conditions
    if ds.conditions() is not None:
        _log(f"[2/6] conditions ('condition'): "
              f"{dict(ds.conditions().value_counts())}")
    else:
        _log("[2/6] conditions: none")

    # raw counts source
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    n_before_c, n_before_g = adata.n_obs, adata.n_vars
    _log(f"[3/6] QC: preserving the provided analysis gene set "
          f"(before {n_before_c} x {n_before_g}); genes are not dropped so the set stays "
          f"aligned with the precomputed mask and checkpoints")

    # [4/6] normalization: library-size + log1p (model min-max->z happens at train time)
    adata.layers["counts"] = adata.X.copy()
    adata.uns.pop("log1p", None)   # clear any stale flag so log1p re-applies cleanly
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    _log(f"[4/6] normalized: library-size(1e4) + log1p; "
          f"per-cell min-max -> per-gene z is applied by the model at train time")

    # [5/6] embedding for the kNN graph (PCA on HVGs, neighbors; no UMAP)
    n_hvg = min(3000, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3", layer="counts")
    sub = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(sub, max_value=10)
    sc.tl.pca(sub, n_comps=min(n_pcs, sub.n_vars - 1, sub.n_obs - 1))
    adata.obsm["X_pca"] = sub.obsm["X_pca"]
    sc.pp.neighbors(adata, n_neighbors=n_neighbors,
                    n_pcs=min(n_pcs, adata.obsm["X_pca"].shape[1]), use_rep="X_pca")
    _log(f"[5/6] embedding: {adata.obsm['X_pca'].shape[1]} PCs, kNN graph "
          f"({n_neighbors} neighbors on {n_hvg} HVGs)")

    # [6/6] pseudotime with the chosen root + direction validation
    if root_gene is None and root_cell_type is None:
        suggestion = _suggest_root_gene(adata, ds.organism)
        _log(f"[6/6] pseudotime: no root given -> gene-count heuristic (may be wrong for "
              f"immune activation/exhaustion). Suggested root_gene='{suggestion}'; pass it "
              f"for a reliable direction.")
    # surface the root-cell selection and direction-validation log from compute_pseudotime
    import sys as _sys
    _ptlog = logging.getLogger("sclatch.preprocessing.pseudotime")
    _h = logging.StreamHandler(_sys.stdout)
    _h.setFormatter(logging.Formatter("[6/6] %(message)s"))
    from sclatch._verbose import get_verbosity as _gv
    _show_ptlog = _gv() >= 2
    if _show_ptlog:
        _ptlog.addHandler(_h)
    _prev_level, _prev_prop = _ptlog.level, _ptlog.propagate
    _ptlog.setLevel(logging.INFO if _show_ptlog else logging.WARNING)
    _ptlog.propagate = False
    try:
        adata = compute_pseudotime(adata, root_gene=root_gene, root_cell_type=root_cell_type)
    finally:
        if _show_ptlog: _ptlog.removeHandler(_h)
        _ptlog.setLevel(_prev_level); _ptlog.propagate = _prev_prop
    tau = adata.obs["pseudotime"].values.astype(np.float32)
    _log(f"[6/6] pseudotime: range [{tau.min():.3f}, {tau.max():.3f}], "
          f"median {np.median(tau):.3f}")

    ds.adata = adata
    ds.pseudotime = tau
    X = adata.X
    ds._lognorm = (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(np.float32)
    return ds


def _suggest_root_gene(adata, organism):
    """Suggest a stem/progenitor root marker present in the data."""
    candidates = ["Tcf7", "TCF7", "Cd34", "CD34", "Sox2", "SOX2", "Krt14", "KRT14",
                  "Lgr5", "LGR5", "Il7r", "IL7R"]
    for g in candidates:
        if g in adata.var_names:
            return g
    return str(adata.var_names[0])


# ---------------------------------------------------------------------------
# Build prior (mask)
# ---------------------------------------------------------------------------

def build_prior(dataset, organism=None, priors_dir="priors", rebuild=False,
                precomputed=None, reference_atac=None, skip_atac=False,
                tf_list="scenicplus", allow_download=True):
    """Return the regulatory mask for a dataset.

    scLATCH is a multiome method: the mask is ChIP-Atlas TF binding within 10 kb of the TSS
    INTERSECTED with accessible scATAC peaks, over a curated TF list (`tf_list`). Every
    benchmarked result in the paper uses this prior. Accessibility comes from, in order of
    preference:
      1. the dataset's own scATAC (`data/<name>/atac.h5ad`) -- the default multiome path;
      2. a matched-tissue REFERENCE scATAC atlas via `reference_atac=<path or AnnData>` --
         this is exactly what Fig 7 did (the melanoma cohort had no matched ATAC, so the
         Satpathy TIL scATAC atlas, GSE129785, supplied accessibility).

    If you have neither, you have two fallbacks, both worse and neither used for any
    benchmark:
      - `skip_atac=True` builds a ChIP-only mask (binding without the ATAC intersection):
        less specific;
      - mask-free inference: skip `build_prior` entirely and call `train(dataset, mask=None)`.
        This drops the prior, is RNA-only de-novo inference, and performs at expression-
        baseline level (the paper's no-prior decomposition, AUPRC ~1.24 on PBMC). Use it only
        as a last resort when you have neither your own scATAC nor a reference atlas.

    Loads a precomputed mask when available (fast, offline). Pass `rebuild=True` to build it
    from `priors/` with accessibility. The mask is binary M[target, TF], applied softly during
    training as effective_mask = M + 0.4 (1 - M). Reports the edge count, TFs with binding,
    density, and (if present) the mask-vs-ground-truth comparison.

    When rebuilding, `tf_list` picks the candidate-regulator universe: 'scenicplus' (SCENIC+
    allTFs, default) or 'pyscenic' (pySCENIC TF lists); use the list a deposited mask was built
    with to reproduce it. `allow_download=False` builds strictly from the local ChIP-Atlas
    cache instead of fetching missing TFs from chip-atlas.dbcls.jp.
    """
    from scipy.sparse import load_npz
    import anndata as ad
    organism = organism or dataset.organism

    # 1) explicit or conventional precomputed mask (already ATAC-derived)
    candidates = []
    if precomputed:
        candidates.append(precomputed)
    candidates += [
        os.path.join(dataset.data_dir, dataset.name, "mask.npz"),
        os.path.join(dataset.data_dir, dataset.name, "atac_mask.npz"),
    ]
    mask = None
    if not rebuild:
        for c in candidates:
            if c and os.path.exists(c):
                mask = load_npz(c).toarray().astype(np.float32)
                _log(f"loaded precomputed mask (ChIP-Atlas ∩ ATAC ∩ TF list): {c}")
                break

    # 2) build from priors/ with accessibility (offline)
    if mask is None:
        from sclatch.preprocessing.atac_mask import build_atac_mask
        atac = dataset.atac
        src = "own scATAC (data/%s/atac.h5ad)" % dataset.name
        if atac is None and reference_atac is not None:
            atac = reference_atac if not isinstance(reference_atac, str) else ad.read_h5ad(reference_atac)
            src = ("reference scATAC atlas: %s" %
                   (reference_atac if isinstance(reference_atac, str) else "AnnData"))
        if atac is None and not skip_atac:
            raise FileNotFoundError(
                "build_prior needs chromatin accessibility to build the mask (scLATCH is a "
                "multiome method). Choose one, in order of preference:\n"
                "  1. own scATAC: put it at data/%s/atac.h5ad (the default multiome path);\n"
                "  2. reference atlas: build_prior(..., reference_atac=<path>) with a "
                "matched-tissue scATAC atlas (as in Fig 7: Satpathy TIL scATAC, GSE129785);\n"
                "  3. no scATAC at all: run mask-free with train(dataset, mask=None) "
                "(RNA-only de-novo, expression-baseline level; last resort, not a benchmark), "
                "or build_prior(..., skip_atac=True) for a ChIP-only mask.\n"
                "See data/README.md and priors/README.md." % dataset.name)
        _log(f"building mask from priors/ (ChIP-Atlas within 10kb TSS ∩ ATAC ∩ {tf_list} TF list, "
              f"{organism}); accessibility: {'SKIPPED (degraded, ChIP-only)' if skip_atac else src}")
        work = _dataset_workdir(dataset, need_atac=False)
        out = os.path.abspath(os.path.join("outputs", "masks", dataset.name))  # repo-relative
        os.makedirs(out, exist_ok=True)
        _amlog = logging.getLogger("sclatch.preprocessing.atac_mask")
        _h = logging.StreamHandler(sys.stdout)
        _h.setFormatter(logging.Formatter("  %(message)s"))
        _amlog.addHandler(_h)
        _prev, _prop = _amlog.level, _amlog.propagate
        _amlog.setLevel(logging.INFO); _amlog.propagate = False
        try:
            mask = build_atac_mask(data_dir=work, organism=organism, out_dir=out,
                                   adata_atac=atac, skip_atac_peaks=skip_atac,
                                   tf_list=tf_list, allow_download=allow_download)
        finally:
            _amlog.removeHandler(_h); _amlog.setLevel(_prev); _amlog.propagate = _prop

    # report
    n = mask.shape[0]
    n_edges = int((mask > 0).sum())
    n_tfs = int((mask.sum(axis=0) > 0).sum())
    n_targets = int((mask.sum(axis=1) > 0).sum())
    _log(f"mask: {mask.shape}, {n_edges:,} candidate edges "
          f"({n_edges / (n * n) * 100:.3f}% density), "
          f"{n_tfs} TFs with binding, {n_targets} targets")
    if dataset.ground_truth is not None and len(dataset.ground_truth):
        _log(f"ground truth: {len(dataset.ground_truth):,} edges "
              f"(mask has {n_edges / max(len(dataset.ground_truth),1):.0f}x more; the mask is "
              f"a prior, not the ground truth)")
    else:
        _log("ground truth: none (analysis-only dataset); the mask is an experimental prior")
    dataset.mask = mask
    return mask


# ---------------------------------------------------------------------------
# Train / load
# ---------------------------------------------------------------------------

def _dataset_workdir(dataset, need_atac=False):
    """Materialize the trainer's expected files for a Dataset in a temp dir (one source of
    truth: the canonical trainer reads data.npy/gene_names.csv/pseudotime.npy/atac_mask.npz/
    cell_metadata.csv)."""
    from scipy.sparse import csr_matrix, save_npz
    work = getattr(dataset, "_workdir", None)
    if work and os.path.isdir(work):
        return work
    # repo-relative work dir (gitignored), not a system temp dir
    work = os.path.abspath(os.path.join("outputs", "work", dataset.name))
    os.makedirs(work, exist_ok=True)
    # expression: log-normalized (the trainer applies min-max -> z on top)
    X = dataset._lognorm
    if X is None:
        A = dataset.adata.X
        X = (A.toarray() if hasattr(A, "toarray") else np.asarray(A)).astype(np.float32)
    np.save(os.path.join(work, "data.npy"), X)
    pd.Series(dataset.gene_names).to_csv(os.path.join(work, "gene_names.csv"),
                                         index=False, header=False)
    if dataset.pseudotime is not None:
        np.save(os.path.join(work, "pseudotime.npy"),
                dataset.pseudotime.astype(np.float32))
    if dataset.metadata is not None:
        dataset.metadata.to_csv(os.path.join(work, "cell_metadata.csv"))
    if dataset.mask is not None:
        save_npz(os.path.join(work, "atac_mask.npz"),
                 csr_matrix(dataset.mask.astype(np.float32)))
    if need_atac and dataset.atac is not None:
        dataset.atac.write(os.path.join(work, "adata_atac.h5ad"))
    dataset._workdir = work
    return work


def train(dataset, mask=None, condition_key="condition", save_dir=None, seed=42,
          device=None, python=None, **overrides):
    """Train scLATCH with the exact paper configuration (FINAL_DYNAMIC_CONFIG).

    Uses the dataset's pseudotime as the continuous covariate and condition_key (if present
    in the metadata) as the categorical condition, so the trained network is A(tau, k). The
    canonical trainer runs as a subprocess and streams its loss / convergence log. Returns a
    TrainedModel. Override any hyperparameter via keyword, but the defaults reproduce the
    paper.

    scLATCH is a multiome method; every benchmarked result uses the regulatory prior, so pass
    the `mask` from `build_prior`. `mask=None` (and no `dataset.mask`) is the mask-free
    fallback: it drops the prior and does RNA-only de-novo inference at expression-baseline
    level (the paper's no-prior decomposition, AUPRC ~1.24 on PBMC). Use it only when you have
    neither your own scATAC nor a reference atlas; it is not the recommended path.
    """
    if mask is not None:
        dataset.mask = mask
    mask_free = dataset.mask is None
    if mask_free:
        _log("WARNING: training MASK-FREE (no regulatory prior). This is RNA-only de-novo "
              "inference at expression-baseline level (paper no-prior decomposition, AUPRC "
              "~1.24 on PBMC), a last-resort fallback, not the multiome method. Prefer "
              "build_prior with your own scATAC or a reference atlas. See data/README.md.")
    work = _dataset_workdir(dataset)
    # repo-relative run dir (gitignored, not a system temp dir); trainer writes <save_dir>/<name>/<run>/
    save_dir = save_dir or os.path.abspath("runs")
    os.makedirs(save_dir, exist_ok=True)
    has_cond = (dataset.metadata is not None and condition_key in dataset.metadata.columns)
    ov = dict(overrides)
    if has_cond:
        ov["categorical"] = [condition_key]
    if mask_free:
        ov["atac_mask"] = False           # no prior; do not pass --atac_mask
        ov["no_mask_anneal"] = False
    if device:
        ov["device"] = device
    _log("Training with FINAL_DYNAMIC_CONFIG (paper) "
          f"{'+ categorical=' + condition_key if has_cond else '(no condition)'}:")
    _log("  " + ", ".join(f"{k}={v}" for k, v in FINAL_DYNAMIC_CONFIG.items()))
    cmd = build_command(work, save_dir, run_name="dynamic", seed=seed, python=python, **ov)
    # Stream the canonical trainer's stdout through Python so the loss/convergence log is
    # captured (e.g. in a notebook cell), not lost to the raw file descriptor.
    proc = subprocess.Popen(cmd, cwd=os.getcwd(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    _last_bar = None
    for line in proc.stdout:
        # collapse repeated progress-bar redraws so a notebook shows a few
        # updates rather than one output per step
        if '|' in line and '%' in line and 'training' in line:
            key = line.split('|')[0]
            if key == _last_bar:
                continue
            _last_bar = key
        sys.stdout.write(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"training failed (exit {proc.returncode})")
    run_dir = os.path.join(save_dir, os.path.basename(work.rstrip("/")), "dynamic")
    ckpt = os.path.join(run_dir, "model.pt")
    return load_model(ckpt, dataset=dataset, condition_key=condition_key)


def load_model(path, dataset=None, condition_key="condition"):
    """Load a trained checkpoint into a TrainedModel for analysis.

    Reads the training_config.json next to the checkpoint to rebuild the exact architecture
    (gene count, categorical condition, dynamic rank), then loads the state dict. Pass the
    Dataset so the analyses can use its gene names, pseudotime, conditions, and mask.
    """
    run_dir = os.path.dirname(path)
    cfg = {}
    cfg_path = os.path.join(run_dir, "training_config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path)).get("args", {})

    if dataset is not None:
        gene_names = dataset.gene_names
        mask = dataset.mask
    else:
        gn = os.path.join(run_dir, "gene_names.csv")
        gene_names = ([l.strip() for l in open(gn)] if os.path.exists(gn) else None)
        mask = None
    if gene_names is None:
        raise ValueError("gene names unavailable; pass dataset=... to load_model")
    G = len(gene_names)

    cond_map = None
    categorical_cond_keys = None
    cats = cfg.get("categorical") or ([condition_key] if (dataset is not None
            and dataset.metadata is not None and condition_key in dataset.metadata.columns)
            else [])
    if cats:
        key = cats[0]
        vals = sorted(dataset.metadata[key].dropna().unique()) if dataset is not None else []
        cond_map = {v: i for i, v in enumerate(vals)}
        categorical_cond_keys = {key: max(len(vals), 1)}
        condition_key = key

    atac = mask if mask is not None else None
    model = scLATCH(
        n_genes=G,
        hidden_dims=cfg.get("hidden_dims", [16, 16, 16]),
        cond_dim=cfg.get("cond_dim", 64),
        # an empty list is a run without pseudotime (e.g. static, rank 0); keep it as trained
        continuous_cond_keys=cfg["continuous"] if "continuous" in cfg else ["pseudotime"],
        categorical_cond_keys=categorical_cond_keys,
        adj_dropout=cfg.get("adj_dropout", 0.3),
        sparse_threshold_coef=cfg.get("sparse_threshold_coef", 0.2),
        atac_mask=atac,
        mask_leak=cfg.get("mask_leak", 0.4),
        dynamic_rank=cfg.get("dynamic_rank", 32),
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    state = torch.load(path, map_location=dev)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.unfreeze_dynamic()
    model.eval()
    _log(f"loaded checkpoint {_disp(path)} on {dev} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return TrainedModel(model, gene_names, dataset=dataset,
                        condition_key=condition_key, condition_map=cond_map)


# ---------------------------------------------------------------------------
# TrainedModel: one analysis per method, each with its formula and a focused plot
# ---------------------------------------------------------------------------

class TrainedModel:
    """A trained scLATCH network with one analysis per method.

    The inferred GRN is the adjacency A[target, regulator]. The dynamic network is
    A(tau, k) = A_static + U diag(gate(c)) V, where c encodes diffusion time, pseudotime tau,
    and the categorical condition k.
    """

    def __init__(self, model, gene_names, dataset=None, condition_key="condition",
                 condition_map=None):
        self.model = model
        self.gene_names = list(gene_names)
        self.g2i = {g: i for i, g in enumerate(self.gene_names)}
        self.dataset = dataset
        self.condition_key = condition_key
        self.condition_map = condition_map or {}
        self.pseudotime = None if dataset is None else dataset.pseudotime
        self._grn_norm = 1.0 / (model.n_genes - 1)
        # low-rank cache
        d = model.dynamic_adj
        self._U = d.U.detach().float().cpu().numpy() if d is not None else None
        self._V = d.V.detach().float().cpu().numpy() if d is not None else None
        self._M = (self._U.T @ self._U) * (self._V @ self._V.T) if d is not None else None

    # ---- condition / cond-vector helpers ----
    def _cat(self, condition):
        if condition is None or not self.condition_map:
            return None
        if condition not in self.condition_map:
            raise KeyError(f"unknown condition '{condition}'; known: "
                           f"{list(self.condition_map)}")
        idx = self.condition_map[condition]
        dev = self.model.adj_A.device
        return {self.condition_key: torch.tensor([idx], device=dev)}

    @torch.no_grad()
    def _gate(self, tau, condition=None):
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float32))
        dev = self.model.adj_A.device
        cat = self._cat(condition)
        if cat is not None and len(tau) > 1:
            cat = {self.condition_key: cat[self.condition_key].repeat(len(tau))}
        cond = self.model.build_cond_vector(torch.tensor(tau, device=dev), categorical=cat)
        return self.model.dynamic_adj.gate_net(cond).float().cpu().numpy()

    @torch.no_grad()
    def _static_column(self, j):
        """Clean static A column for regulator j (soft-threshold + mask + zero self)."""
        col = self.model.soft_thresholding(self.model.adj_A[:, j].float(),
                                           self.model.sparse_threshold)
        col = col.clone()
        col[j] = 0.0
        if getattr(self.model, "atac_mask", None) is not None:
            col = col * self.model.atac_mask[:, j].float()
        return col.cpu().numpy()

    def _tf_indices(self, tfs):
        return [(t, self.g2i[t]) for t in tfs if t in self.g2i]

    # ---- 1. static network ----
    @torch.no_grad()
    def static_network(self, top_n_tfs=25):
        """Static backbone A_static[target, regulator] = sign(x) relu(|x|-tau_thr), masked,
        diagonal zeroed. Returns per-TF out-strength S_j = sum_i |A[i, j]| and the top TFs.
        """
        d = self.model
        cols_all = d.get_adj().detach().float().cpu().numpy()  # dense; fine for analysis
        S = np.abs(cols_all).sum(axis=0) / self._grn_norm
        strength = pd.Series(S, index=self.gene_names).sort_values(ascending=False)
        top = strength.head(top_n_tfs)

        def _plot(ax):
            a = _ax(ax, (4.4, 0.28 * len(top) + 0.5))
            a.barh(np.arange(len(top))[::-1], top.values, color=_PALETTE[1])
            a.set_yticks(np.arange(len(top))[::-1]); a.set_yticklabels(top.index, fontsize=7)
            a.set_xlabel("static out-strength  S_j = sum_i |A[i,j]|")
            a.set_title("static network: strongest regulators", fontsize=9)
            return a
        return AnalysisResult(_plot=_plot, adjacency=cols_all, strength=strength, top=top)

    # ---- 2. dynamic network ----
    @torch.no_grad()
    def dynamic_network(self, pseudotime, condition=None):
        """A(tau, k) = A_static + U diag(gate(c)) V, with c encoding diffusion time,
        pseudotime tau, and condition k. Returns the dense A(tau, k)."""
        dev = self.model.adj_A.device
        cond = self.model.build_cond_vector(float(pseudotime), categorical=self._cat(condition))
        A = self.model.get_adj_dynamic(cond).detach().float().cpu().numpy()
        if A.ndim == 3:
            A = A[0]
        return AnalysisResult(adjacency=A / self._grn_norm, pseudotime=float(pseudotime),
                              condition=condition)

    # ---- 3. regulatory output ----
    @torch.no_grad()
    def regulatory_output(self, condition=None, pseudotime=None, tfs=None, top_n=25):
        """Per-TF regulatory output S_j = sum_i |A(tau, k)[i, j]| at a representative tau for
        condition k (default: mean pseudotime of that condition's cells). Returns S over TFs.
        """
        pseudotime = self._repr_tau(condition) if pseudotime is None else float(pseudotime)
        tfs = tfs or self._default_tfs()
        idx = self._tf_indices(tfs)
        gate = self._gate(pseudotime, condition)[0] if self._U is not None else None
        S = {}
        for name, j in idx:
            col = self._static_column(j)
            if gate is not None:
                col = col + self._U @ (gate * self._V[:, j])
                if getattr(self.model, "atac_mask", None) is not None:
                    col = col * self.model.atac_mask[:, j].float().cpu().numpy()
                col[j] = 0.0
            S[name] = float(np.abs(col).sum() / self._grn_norm)
        strength = pd.Series(S).sort_values(ascending=False)
        top = strength.head(top_n)

        def _plot(ax):
            a = _ax(ax, (4.4, 0.28 * len(top) + 0.5))
            a.barh(np.arange(len(top))[::-1], top.values, color=_PALETTE[3])
            a.set_yticks(np.arange(len(top))[::-1]); a.set_yticklabels(top.index, fontsize=7)
            a.set_xlabel("regulatory output  S_j = sum_i |A(tau,k)[i,j]|")
            a.set_title(f"regulatory output ({condition or 'no condition'}, "
                        f"tau={pseudotime:.2f})", fontsize=9)
            return a
        return AnalysisResult(_plot=_plot, strength=strength, top=top,
                              condition=condition, pseudotime=pseudotime)

    # ---- 4. reconfiguration curve ----
    @torch.no_grad()
    def reconfiguration_curve(self, condition=None, tau_grid=None, ref_tau=0.0):
        """Network rewiring ||A(tau) - A(ref)||_F across pseudotime, exact from the low-rank
        term: ||U diag(d) V||_F^2 = d^T ((U^T U) . (V V^T)) d, d = gate(tau) - gate(ref)."""
        if self._U is None:
            raise ValueError("model has no dynamic term (dynamic_rank=0)")
        tau_grid = np.linspace(0, 1, 51) if tau_grid is None else np.asarray(tau_grid)
        gates = self._gate(tau_grid, condition)
        gref = self._gate(ref_tau, condition)[0]
        frob = np.array([np.sqrt(max(float((g - gref) @ self._M @ (g - gref)), 0.0))
                         for g in gates])

        def _plot(ax):
            a = _ax(ax)
            a.plot(tau_grid, frob, color=_PALETTE[0], lw=2,
                   label=condition or None)
            a.set_xlabel("pseudotime tau"); a.set_ylabel(r"$\|A(\tau)-A(\tau_0)\|_F$")
            a.set_title("network reconfiguration", fontsize=9)
            if condition:
                a.legend(frameon=False, fontsize=8)
            return a
        return AnalysisResult(_plot=_plot, tau=tau_grid, frob=frob, condition=condition)

    # ---- 5. commitment boundary ----
    def commitment_boundary(self, condition=None, tau_grid=None):
        """The pseudotime where rewiring is fastest: argmax_tau |d/dtau ||A(tau)-A(0)||_F|.
        Also returns the first inflection of the reconfiguration curve."""
        rc = self.reconfiguration_curve(condition=condition, tau_grid=tau_grid)
        tau, frob = rc.tau, rc.frob
        d1 = np.gradient(frob, tau)
        d2 = np.gradient(d1, tau)
        boundary = float(tau[int(np.argmax(np.abs(d1)))])
        infl = None
        for i in range(len(d2) - 1):
            if d2[i] * d2[i + 1] < 0:
                t0, t1, f0, f1 = tau[i], tau[i + 1], d2[i], d2[i + 1]
                infl = float(t0 - f0 * (t1 - t0) / (f1 - f0)); break

        def _plot(ax):
            a = rc.plot(ax)
            a.axvline(boundary, color="#333", ls="--", lw=1.1)
            a.set_title(f"commitment boundary tau={boundary:.2f}", fontsize=9)
            return a
        return AnalysisResult(_plot=_plot, boundary=boundary, inflection=infl,
                              tau=tau, frob=frob, condition=condition)

    # ---- 6. delta S across conditions ----
    @torch.no_grad()
    def delta_S(self, cond_a=None, cond_b=None, pseudotime=None, tfs=None, top_n=15):
        """Per-TF regulatory-output shift between two conditions: delta_S_j = S_j(a) - S_j(b),
        S_j = sum_i |A(tau, k)[i, j]|. Positive = stronger in a; negative = down-weighted in a.
        cond_a and cond_b are user-supplied; if omitted they default to the first two values of
        model.conditions()."""
        conds = self.conditions()
        if cond_a is None:
            cond_a = conds[0] if conds else None
        if cond_b is None:
            cond_b = conds[1] if len(conds) > 1 else None
        if not cond_a or not cond_b or cond_a == cond_b:
            raise ValueError(
                "delta_S compares two different conditions, but this model has "
                f"{len(conds)} condition(s): {conds or 'none'}. This looks like a "
                "pseudotime-only model; compare two pseudotimes with "
                "top_changing_edges(tau_a=..., tau_b=...) or follow one edge with "
                "edge_trajectory(tf, target).")
        tau = 0.5 if pseudotime is None else float(pseudotime)
        tfs = tfs or self._default_tfs()
        Sa = self.regulatory_output(cond_a, pseudotime=tau, tfs=tfs, top_n=10**9).strength
        Sb = self.regulatory_output(cond_b, pseudotime=tau, tfs=tfs, top_n=10**9).strength
        dS = (Sa - Sb).dropna().sort_values()
        pick = pd.concat([dS.head(top_n), dS.tail(top_n)])

        def _plot(ax):
            a = _ax(ax, (4.6, 0.26 * len(pick) + 0.6))
            colors = [_PALETTE[0] if v < 0 else _PALETTE[3] for v in pick.values]
            a.barh(np.arange(len(pick)), pick.values, color=colors)
            a.set_yticks(np.arange(len(pick))); a.set_yticklabels(pick.index, fontsize=7)
            a.axvline(0, color="#333", lw=0.8)
            a.set_xlabel(f"delta_S = S({cond_a}) - S({cond_b})")
            a.set_title("condition shift in regulatory output", fontsize=9)
            return a
        return AnalysisResult(_plot=_plot, delta_S=dS, top=pick, cond_a=cond_a, cond_b=cond_b)

    # ---- 7. in-silico knockout ----
    @torch.no_grad()
    def knockout(self, tf, condition=None, pseudotime=None, method="direct", top_n=15,
                 return_network=False):
        """In-silico knockout, computed PURELY from the trained network (no external
        perturbation data). Removes one TF or a list of TFs and predicts the expression
        signature:
          direct  : immediate targets, signature = -sum_over_TFs A(tau,k)[:, TF].
          leontief: propagated steady state, signature = -sum_over_TFs (I - A)^{-1}[:, TF].
        A[target, TF] is the effect of the TF on each target. With return_network=True, also
        returns the removed edges as an edge list (the network change). condition and
        pseudotime are user-supplied."""
        tfs = [tf] if isinstance(tf, str) else list(tf)
        for t in tfs:
            if t not in self.g2i:
                raise KeyError(f"{t} not in gene set")
        js = [self.g2i[t] for t in tfs]

        def _col(j):
            if condition is None and pseudotime is None:
                return self._static_column(j)
            tau = self._repr_tau(condition) if pseudotime is None else float(pseudotime)
            gate = self._gate(tau, condition)[0]
            c = self._static_column(j) + self._U @ (gate * self._V[:, j])
            if getattr(self.model, "atac_mask", None) is not None:
                c = c * self.model.atac_mask[:, j].float().cpu().numpy()
            c[j] = 0.0
            return c

        if method == "direct":
            sig = -np.sum([_col(j) for j in js], axis=0)
            sig = pd.Series(sig, index=self.gene_names)
        elif method == "leontief":
            A = (self.dynamic_network(pseudotime or 0.5, condition).adjacency
                 if (condition or pseudotime) else self.static_network().adjacency)
            inv = np.linalg.inv(np.eye(A.shape[0]) - A)
            sig = pd.Series(-inv[:, js].sum(axis=1), index=self.gene_names)
        else:
            raise ValueError("method must be 'direct' or 'leontief'")
        sig = sig.drop(labels=tfs, errors="ignore")
        top = sig.reindex(sig.abs().sort_values(ascending=False).index[:top_n])[::-1]
        label = tfs[0] if len(tfs) == 1 else f"{len(tfs)} TFs"

        def _plot(ax):
            a = _ax(ax, (4.2, 0.28 * len(top) + 0.6))
            colors = [_PALETTE[3] if v > 0 else _PALETTE[0] for v in top.values]
            a.barh(np.arange(len(top)), top.values, color=colors)
            a.set_yticks(np.arange(len(top))); a.set_yticklabels(top.index, fontsize=7)
            a.axvline(0, color="#333", lw=0.8)
            a.set_xlabel("predicted delta-expression")
            a.set_title(f"in-silico knockout: {label}", fontsize=9)
            return a
        res = AnalysisResult(_plot=_plot, signature=sig, top=top, tfs=tfs, condition=condition)
        if return_network:
            res.removed_edges = self._edge_list(tfs=tfs, condition=condition,
                                                pseudotime=pseudotime, top_n=200)
        return res

    # ---- 8. quasi-potential ----
    @torch.no_grad()
    def quasi_potential(self, condition=None, n_grid=200):
        """Empirical quasi-potential U(s) = -log rho(s) of a 1-D network-state coordinate
        (top principal component of the per-cell scaled gate). Two wells = a switch."""
        if self.pseudotime is None:
            raise ValueError("dataset pseudotime required for the landscape")
        gates = self._gate(self.pseudotime, condition)
        scale = np.sqrt((self._U ** 2).sum(0) * (self._V ** 2).sum(1) + 1e-12)
        Xg = gates * scale[None, :]; Xg = Xg - Xg.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xg, full_matrices=False)
        s = Xg @ Vt[0]
        if np.corrcoef(s, self.pseudotime)[0, 1] < 0:
            s = -s
        from scipy.stats import gaussian_kde
        grid = np.linspace(s.min(), s.max(), n_grid)
        U = -np.log(gaussian_kde(s)(grid) + 1e-4); U -= U.min()

        def _plot(ax):
            a = _ax(ax)
            a.plot(grid, U, color=_PALETTE[1], lw=2)
            a.set_xlabel("network-state coordinate s"); a.set_ylabel("U(s) = -log rho")
            a.set_title("quasi-potential landscape", fontsize=9)
            return a
        return AnalysisResult(_plot=_plot, grid=grid, potential=U, coord=s, condition=condition)

    # ---- 9. gene x tau strength heatmap ----
    @torch.no_grad()
    def gene_strength_heatmap(self, tfs=None, condition=None, tau_grid=None, row_zscore=True):
        """Per-TF regulatory output S_j(tau) = sum_i |A(tau, k)[i, j]| over a pseudotime grid,
        one row per TF. Returns the TF-by-tau matrix."""
        tau_grid = np.linspace(0, 1, 21) if tau_grid is None else np.asarray(tau_grid)
        tfs = tfs or self._default_tfs()
        idx = self._tf_indices(tfs)
        gates = self._gate(tau_grid, condition)
        M = np.zeros((len(idx), len(tau_grid)))
        static_cols = {j: self._static_column(j) for _, j in idx}
        atac = getattr(self.model, "atac_mask", None)
        atac = atac.float().cpu().numpy() if atac is not None else None
        for ti in range(len(tau_grid)):
            for k, (name, j) in enumerate(idx):
                col = static_cols[j] + self._U @ (gates[ti] * self._V[:, j])
                if atac is not None:
                    col = col * atac[:, j]
                col[j] = 0.0
                M[k, ti] = np.abs(col).sum() / self._grn_norm
        df = pd.DataFrame(M, index=[n for n, _ in idx],
                          columns=[round(float(t), 3) for t in tau_grid])

        def _plot(ax):
            a = _ax(ax, (5.0, max(2.5, 0.2 * len(df))))
            Z = df.values.astype(float)
            if row_zscore:
                Z = (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-9)
                im = a.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
                              extent=[tau_grid[0], tau_grid[-1], len(df), 0])
            else:
                im = a.imshow(Z, aspect="auto", cmap="magma",
                              extent=[tau_grid[0], tau_grid[-1], len(df), 0])
            a.set_yticks(np.arange(len(df)) + 0.5); a.set_yticklabels(df.index, fontsize=6)
            a.set_xlabel("pseudotime tau")
            a.figure.colorbar(im, ax=a, fraction=0.046, pad=0.04)
            a.set_title(f"regulon strength ({condition or 'no condition'})", fontsize=9)
            return a
        return AnalysisResult(_plot=_plot, matrix=df, condition=condition)

    # ---- 10. edge lists / network export ----
    @torch.no_grad()
    def _tf_columns(self, tfs, condition=None, pseudotime=None):
        """A(tau, k) columns for a list of regulator TFs, shape (n_genes, n_tf), normalized.
        Uses the static column plus the low-rank dynamic term when a condition/pseudotime is
        given; applies the ATAC mask and zeros the self-edge."""
        idx = self._tf_indices(tfs)
        gate = None
        if self._U is not None and (condition is not None or pseudotime is not None):
            tau = self._repr_tau(condition) if pseudotime is None else float(pseudotime)
            gate = self._gate(tau, condition)[0]
        atac = getattr(self.model, "atac_mask", None)
        atac = atac.float().cpu().numpy() if atac is not None else None
        cols = np.zeros((self.model.n_genes, len(idx)), dtype=np.float64)
        names, jidx = [], []
        for k, (name, j) in enumerate(idx):
            col = self._static_column(j)
            if gate is not None:
                col = col + self._U @ (gate * self._V[:, j])
                if atac is not None:
                    col = col * atac[:, j]
                col[j] = 0.0
            cols[:, k] = col / self._grn_norm
            names.append(name); jidx.append(j)
        return cols, names, jidx

    def edges(self, condition=None, pseudotime=None, top_n=200, tfs=None):
        """Top regulatory edges of A(tau, k) as a DataFrame [regulator_TF, target_gene, weight,
        sign]. condition and pseudotime are user-supplied."""
        return self._edge_list(tfs=tfs, condition=condition, pseudotime=pseudotime, top_n=top_n)

    def _edge_list(self, tfs=None, condition=None, pseudotime=None, top_n=200):
        """Top |weight| edges as a DataFrame [regulator_TF, target_gene, weight, sign]."""
        tfs = tfs or self._default_tfs(60)
        cols, names, _ = self._tf_columns(tfs, condition, pseudotime)
        flat = np.abs(cols).ravel()
        n = int(min(top_n * 3, flat.size))
        cand = np.argpartition(-flat, n - 1)[:n] if n < flat.size else np.arange(flat.size)
        rows = []
        ncol = cols.shape[1]
        for fi in cand:
            i, k = divmod(int(fi), ncol)
            w = cols[i, k]
            if w == 0 or self.gene_names[i] == names[k]:
                continue
            rows.append((names[k], self.gene_names[i], float(w), int(np.sign(w))))
        df = pd.DataFrame(rows, columns=["regulator_TF", "target_gene", "weight", "sign"])
        df = df.reindex(df["weight"].abs().sort_values(ascending=False).index)
        return df.head(top_n).reset_index(drop=True)

    def plot_network(self, condition=None, pseudotime=None, top_n_edges=60, tfs=None, ax=None):
        """Node-link diagram of the top regulatory edges of A(tau, k) at a given condition and
        pseudotime. TFs are hub nodes (source), targets are leaf nodes; node size scales with
        out-degree, edge width with |weight|, edge color with sign (activation vs repression).
        Interpretation: large hubs are TFs that regulate many targets in that state; thick
        edges are the strongest interactions. condition and pseudotime are user-supplied."""
        df = self._edge_list(tfs=tfs, condition=condition, pseudotime=pseudotime,
                             top_n=top_n_edges)
        from sclatch.plotting import plot_network as _pn
        bits = []
        if condition is not None:
            bits.append(str(condition))
        if pseudotime is not None:
            bits.append(f"tau={float(pseudotime):.2f}")
        title = "GRN top edges" + (f" ({', '.join(bits)})" if bits else " (static)")
        return _pn(df, title=title, ax=ax)

    def plot_differential_network(self, cond_a=None, cond_b=None, tau_a=None, tau_b=None,
                                  top_n_edges=60, tfs=None, ax=None):
        """Node-link diagram of the edges that change most between two conditions (cond_a vs
        cond_b) or two pseudotimes (tau_a vs tau_b). Edge color marks gained (stronger in a)
        vs lost (weaker in a); width scales with |change|. Conditions/pseudotimes are
        user-supplied."""
        chg = self.top_changing_edges(cond_a=cond_a, cond_b=cond_b, tau_a=tau_a, tau_b=tau_b,
                                      top_n=top_n_edges, tfs=tfs)
        from sclatch.plotting import plot_differential_network as _pdn
        lab = (f"{cond_a} vs {cond_b}" if cond_a is not None
               else f"tau {tau_a} vs {tau_b}")
        return _pdn(chg, title=f"differential GRN ({lab})", ax=ax)

    def export_network(self, path="outputs/network.csv", condition=None, pseudotime=None,
                       top_n=1000, tfs=None):
        """Write the top regulatory edges to a CSV with columns regulator_TF, target_gene,
        weight, sign. weight is the grn-normalized A(tau, k)[target, regulator]; sign is +1
        for activation, -1 for repression. Defaults to a repo-relative path.

        Load and use it later without the model:
            import pandas as pd, networkx as nx
            e = pd.read_csv(path)
            e.reindex(e.weight.abs().sort_values(ascending=False).index).head(20)   # top edges
            e[e.sign > 0]                                                            # activators
            G = nx.from_pandas_edgelist(e, "regulator_TF", "target_gene",
                                        edge_attr=["weight", "sign"], create_using=nx.DiGraph)
            # Cytoscape: import the CSV as a network (source=regulator_TF, target=target_gene)."""
        tfs = tfs or self._binding_tfs()
        df = self._edge_list(tfs=tfs, condition=condition, pseudotime=pseudotime, top_n=top_n)
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        _log(f"exported {len(df)} edges -> {_disp(path)}")
        return df

    # ---- 11. tau utilities ----
    @torch.no_grad()
    def network_snapshots(self, tau_grid=None, condition=None, top_n=500, out_dir=None):
        """A(tau, k) at several pseudotime points, returned as {tau: edge-list DataFrame}. If
        out_dir is given, also writes one CSV per tau (repo-relative). condition is
        user-supplied."""
        tau_grid = np.linspace(0, 1, 5) if tau_grid is None else np.asarray(tau_grid)
        tfs = self._binding_tfs()
        snaps = {}
        for t in tau_grid:
            df = self._edge_list(tfs=tfs, condition=condition, pseudotime=float(t), top_n=top_n)
            snaps[round(float(t), 3)] = df
            if out_dir:
                out_dir = os.path.abspath(out_dir); os.makedirs(out_dir, exist_ok=True)
                df.to_csv(os.path.join(out_dir, f"network_tau{float(t):.2f}.csv"), index=False)
        return snaps

    @torch.no_grad()
    def edge_trajectory(self, tf, target, condition=None, tau_grid=None):
        """Weight of a single TF -> target edge, A(tau, k)[target, tf], across pseudotime.
        Returns tau and weight; .plot() draws the trajectory. condition is user-supplied."""
        if tf not in self.g2i or target not in self.g2i:
            raise KeyError(f"{tf} or {target} not in gene set")
        j, i = self.g2i[tf], self.g2i[target]
        tau_grid = np.linspace(0, 1, 51) if tau_grid is None else np.asarray(tau_grid)
        static_ti = float(self._static_column(j)[i]) / self._grn_norm
        atac_ij = 1.0
        atac = getattr(self.model, "atac_mask", None)
        if atac is not None:
            atac_ij = float(atac[i, j])
        w = np.empty(len(tau_grid))
        if self._U is not None:
            gates = self._gate(tau_grid, condition)
            urow = self._U[i, :]
            vcol = self._V[:, j]
            for k, g in enumerate(gates):
                dyn = float(urow @ (g * vcol)) * atac_ij / self._grn_norm
                w[k] = static_ti + dyn
        else:
            w[:] = static_ti

        def _plot(ax):
            a = _ax(ax)
            a.plot(tau_grid, w, color=_PALETTE[2], lw=2)
            a.axhline(0, color="#999", lw=0.8)
            a.set_xlabel("pseudotime tau")
            a.set_ylabel(f"edge weight {tf} -> {target}")
            a.set_title(f"edge trajectory ({condition or 'no condition'})", fontsize=9)
            return a
        return AnalysisResult(_plot=_plot, tau=tau_grid, weight=w, tf=tf, target=target,
                              condition=condition)

    @torch.no_grad()
    def top_changing_edges(self, cond_a=None, cond_b=None, tau_a=None, tau_b=None,
                           top_n=30, tfs=None, condition=None):
        """Edges that change most between two conditions (cond_a, cond_b) or two pseudotimes
        (tau_a, tau_b, for a fixed condition). Returns a DataFrame [regulator_TF, target_gene,
        weight_a, weight_b, change]. All condition/pseudotime values are user-supplied. For a
        pseudotime-only (no-condition) model, use tau_a and tau_b."""
        tfs = tfs or self._binding_tfs()
        if cond_a is not None and cond_b is not None:
            ca = self._tf_columns(tfs, condition=cond_a)[0]
            cb = self._tf_columns(tfs, condition=cond_b)[0]
        elif tau_a is not None and tau_b is not None:
            ca = self._tf_columns(tfs, condition=condition, pseudotime=tau_a)[0]
            cb = self._tf_columns(tfs, condition=condition, pseudotime=tau_b)[0]
        else:
            raise ValueError(
                "top_changing_edges needs a pair to compare: two conditions (cond_a, cond_b) "
                "or two pseudotimes (tau_a, tau_b). For a pseudotime-only model use, e.g., "
                "top_changing_edges(tau_a=0.2, tau_b=0.8).")
        _, names, _ = self._tf_columns(tfs)   # names order matches columns
        delta = ca - cb
        flat = np.abs(delta).ravel()
        n = int(min(top_n * 3, flat.size))
        cand = np.argpartition(-flat, n - 1)[:n] if n < flat.size else np.arange(flat.size)
        rows = []
        ncol = delta.shape[1]
        for fi in cand:
            i, k = divmod(int(fi), ncol)
            if self.gene_names[i] == names[k] or delta[i, k] == 0:
                continue
            rows.append((names[k], self.gene_names[i], float(ca[i, k]), float(cb[i, k]),
                         float(delta[i, k])))
        df = pd.DataFrame(rows, columns=["regulator_TF", "target_gene",
                                         "weight_a", "weight_b", "change"])
        df = df.reindex(df["change"].abs().sort_values(ascending=False).index)
        return df.head(top_n).reset_index(drop=True)

    # ---- 12. save everything ----
    @torch.no_grad()
    def _save_tau_snapshot_npz(self, path, condition=None, tau_grid=None, top_n=200):
        """Save how a fixed set of top edges changes across pseudotime, as a compact .npz
        (tau grid, weights [n_edges, n_tau], regulator, target). No pickle needed on load."""
        tau_grid = np.linspace(0, 1, 11) if tau_grid is None else np.asarray(tau_grid)
        ref = self.edges(condition=condition, pseudotime=float(np.median(tau_grid)), top_n=top_n)
        tg = [self.g2i[t] for t in ref["target_gene"]]
        tj = [self.g2i[t] for t in ref["regulator_TF"]]
        W = np.zeros((len(ref), len(tau_grid)))
        atac = getattr(self.model, "atac_mask", None)
        atac = atac.float().cpu().numpy() if atac is not None else None
        static_cols = {j: self._static_column(j) for j in set(tj)}
        if self._U is not None:
            gates = self._gate(tau_grid, condition)
            for e in range(len(ref)):
                i, j = tg[e], tj[e]
                base = static_cols[j][i] / self._grn_norm
                a = (atac[i, j] if atac is not None else 1.0)
                urow, vcol = self._U[i, :], self._V[:, j]
                W[e, :] = [base + float(urow @ (g * vcol)) * a / self._grn_norm for g in gates]
        else:
            for e in range(len(ref)):
                W[e, :] = static_cols[tj[e]][tg[e]] / self._grn_norm
        np.savez(path, tau=tau_grid, weights=W,
                 regulator=np.array(ref["regulator_TF"].tolist()),
                 target=np.array(ref["target_gene"].tolist()),
                 condition=np.array(str(condition)))

    def save_results(self, out_dir="outputs/results"):
        """Write all key read-outs to out_dir/ (repo-relative), split into static/ and
        dynamic/ subfolders, with a RESULTS_README.md that labels each file static or dynamic,
        documents every column, and gives copy-pasteable code to load and use each file later
        without the model.

        static/ : the static network edge list and the static per-TF regulatory output.
        dynamic/: per-condition edge lists, the regulatory output S by condition, delta_S
                  between the first two conditions, the reconfiguration curve + commitment
                  boundary, the gate-entropy curve, the top-changing edges, and the A(tau)
                  snapshots (.npz)."""
        out_dir = os.path.abspath(out_dir)
        sdir = os.path.join(out_dir, "static"); ddir = os.path.join(out_dir, "dynamic")
        os.makedirs(sdir, exist_ok=True); os.makedirs(ddir, exist_ok=True)
        conds = self.conditions()
        static_files, dyn_files = [], []

        # ---- STATIC ----
        self.export_network(os.path.join(sdir, "static_network.csv"))
        static_files.append(("static/static_network.csv",
                             "Static backbone A_static top edges. Columns: regulator_TF, "
                             "target_gene, weight (grn-normalized A_static[target,regulator]), "
                             "sign (+1 activation, -1 repression)."))
        Sstat = self.static_network(top_n_tfs=10**9).strength.rename("S")
        Sstat.index.name = "TF"
        Sstat.to_frame().to_csv(os.path.join(sdir, "static_regulatory_output.csv"))
        static_files.append(("static/static_regulatory_output.csv",
                             "Static per-TF out-strength S_j = sum_i |A_static[i,j]|. Columns: "
                             "TF, S. Rank by S for the strongest regulators."))

        # ---- DYNAMIC ----
        skipped = []
        if conds:
            for c in conds:
                self.export_network(os.path.join(ddir, f"dynamic_network_{c}.csv"), condition=c)
                dyn_files.append((f"dynamic/dynamic_network_{c}.csv",
                                  f"A(tau, k='{c}') top edges at the mean pseudotime of '{c}'. "
                                  "Same columns as the static edge list."))
            S = pd.DataFrame({c: self.regulatory_output(condition=c, top_n=10**9).strength
                              for c in conds})
            S.index.name = "TF"
            S.to_csv(os.path.join(ddir, "regulatory_output_by_condition.csv"))
            dyn_files.append(("dynamic/regulatory_output_by_condition.csv",
                              "Per-TF regulatory output S_j = sum_i |A(tau,k)[i,j]|, one column "
                              "per condition. Rank a column for that condition's regulators."))
        else:
            # pseudotime-only (no condition): tau dynamic network + S over tau
            self.export_network(os.path.join(ddir, "dynamic_network_tau.csv"), pseudotime=0.5)
            dyn_files.append(("dynamic/dynamic_network_tau.csv",
                              "A(tau) top edges at tau=0.5 (pseudotime-only model, no condition). "
                              "Same columns as the static edge list."))
            gh = self.gene_strength_heatmap(tau_grid=np.linspace(0, 1, 11))
            gh.matrix.to_csv(os.path.join(ddir, "regulatory_output_over_tau.csv"))
            dyn_files.append(("dynamic/regulatory_output_over_tau.csv",
                              "Per-TF regulatory output S_j(tau) over a pseudotime grid (rows = "
                              "TFs, columns = tau). Pseudotime-only model."))
        if len(conds) >= 2:
            a, b = conds[0], conds[1]
            self.delta_S(a, b, top_n=10**9).delta_S.to_csv(
                os.path.join(ddir, f"delta_S_{a}_vs_{b}.csv"), header=["delta_S"])
            dyn_files.append((f"dynamic/delta_S_{a}_vs_{b}.csv",
                              f"delta_S_j = S_j('{a}') - S_j('{b}') per TF. Columns: TF, "
                              f"delta_S. Sort ascending for TFs down-weighted in '{a}'."))
            self.top_changing_edges(cond_a=a, cond_b=b, top_n=200).to_csv(
                os.path.join(ddir, f"top_changing_edges_{a}_vs_{b}.csv"), index=False)
            dyn_files.append((f"dynamic/top_changing_edges_{a}_vs_{b}.csv",
                              f"Edges changing most between '{a}' and '{b}'. Columns: "
                              "regulator_TF, target_gene, weight_a, weight_b, change."))
        else:
            skipped = ["delta_S (needs two conditions)",
                       "per-condition differential network and top_changing_edges by condition"]
        cc = conds[0] if conds else None
        if self._U is not None:
            rc = self.reconfiguration_curve(condition=cc)
            bnd = self.commitment_boundary(condition=cc)
            pd.DataFrame({"tau": rc.tau, "reconfiguration_frob": rc.frob}).to_csv(
                os.path.join(ddir, "reconfiguration_curve.csv"), index=False)
            dyn_files.append(("dynamic/reconfiguration_curve.csv",
                              "Network rewiring ||A(tau)-A(0)||_F vs pseudotime (condition "
                              f"'{cc}'). Columns: tau, reconfiguration_frob. Commitment boundary "
                              f"tau={bnd.boundary:.3f} (argmax of the rewiring rate)."))
            g = self._gate(rc.tau, cc); p = np.abs(g); p = p / (p.sum(1, keepdims=True) + 1e-10)
            entropy = -(p * np.log(p + 1e-10)).sum(1)
            pd.DataFrame({"tau": rc.tau, "gate_entropy": entropy}).to_csv(
                os.path.join(ddir, "gate_entropy_curve.csv"), index=False)
            dyn_files.append(("dynamic/gate_entropy_curve.csv",
                              "Entropy of the rank-mode gate over pseudotime (a Delta-S proxy). "
                              "Columns: tau, gate_entropy."))
            self._save_tau_snapshot_npz(os.path.join(ddir, "tau_snapshots.npz"), condition=cc)
            dyn_files.append(("dynamic/tau_snapshots.npz",
                              "How the top edges change across pseudotime. Arrays: tau "
                              "(n_tau,), weights (n_edges, n_tau), regulator (n_edges,), target "
                              f"(n_edges,), condition. Condition '{cc}'."))

        # ---- RESULTS_README ----
        multi = len(conds) >= 2
        L = ["# scLATCH results", "",
             f"Trained-network read-outs. Conditions: {conds or 'none (pseudotime-only model)'}. "
             "A[target, regulator]; weights are grn-normalized. Files are split into `static/` "
             "and `dynamic/`.", ""]
        L += ["## STATIC files", ""]
        for name, desc in static_files:
            L.append(f"- `{name}`: {desc}")
        L += ["", "## DYNAMIC files", ""]
        for name, desc in dyn_files:
            L.append(f"- `{name}`: {desc}")
        if skipped:
            L += ["", "## Skipped (no conditions)", "",
                  "This is a pseudotime-only model (no categorical condition), so these "
                  "condition-specific outputs were not written:"]
            L += [f"- {s}" for s in skipped]
            L += ["Compare two pseudotimes instead: `tau_snapshots.npz`, or "
                  "`top_changing_edges(tau_a=..., tau_b=...)`."]
        L += ["", "## How to load and use (no model needed)", "", "```python",
              "import pandas as pd, numpy as np", "",
              "# STATIC network edge list: rank edges, filter activators, import to networkx/Cytoscape",
              'edges = pd.read_csv("static/static_network.csv")',
              "edges = edges.reindex(edges.weight.abs().sort_values(ascending=False).index)",
              "top20 = edges.head(20)                       # strongest regulatory edges",
              "activators = edges[edges.sign > 0]           # activating edges only",
              "import networkx as nx",
              'G = nx.from_pandas_edgelist(edges, "regulator_TF", "target_gene",',
              '                            edge_attr=["weight", "sign"], create_using=nx.DiGraph)',
              "# Cytoscape: import static_network.csv as a network table",
              "#   (source = regulator_TF, target = target_gene, edge attrs = weight, sign).", "",
              "# STATIC per-TF output S: rank the strongest regulators",
              'S = pd.read_csv("static/static_regulatory_output.csv", index_col="TF")',
              'S.sort_values("S", ascending=False).head(15)', ""]
        if multi:
            a, b = conds[0], conds[1]
            L += ["# DYNAMIC per-condition edge lists: compare two conditions",
                  f'ea = pd.read_csv("dynamic/dynamic_network_{a}.csv")',
                  f'eb = pd.read_csv("dynamic/dynamic_network_{b}.csv")',
                  'm = ea.merge(eb, on=["regulator_TF", "target_gene"], suffixes=("_a", "_b"))',
                  'm["change"] = m["weight_a"] - m["weight_b"]  # stronger/weaker between conditions',
                  ""]
        else:
            L += ["# DYNAMIC tau network + S over tau (pseudotime-only model)",
                  'edges_tau = pd.read_csv("dynamic/dynamic_network_tau.csv")   # A(tau=0.5) edges',
                  'S_tau = pd.read_csv("dynamic/regulatory_output_over_tau.csv", index_col=0)  # TF x tau',
                  ""]
        L += ["# DYNAMIC A(tau) snapshots: how an edge changes across pseudotime",
              'snap = np.load("dynamic/tau_snapshots.npz")',
              'tau, W = snap["tau"], snap["weights"]         # W: (n_edges, n_tau)',
              'reg, tgt = snap["regulator"], snap["target"]',
              'e = 0; _log(reg[e], "->", tgt[e], "weight vs tau:", W[e])', ""]
        if multi:
            a, b = conds[0], conds[1]
            L += ["# DYNAMIC delta_S: rank TFs shifting between conditions",
                  f'dS = pd.read_csv("dynamic/delta_S_{a}_vs_{b}.csv", index_col=0)',
                  f'dS.sort_values("delta_S").head(10)           # most down-weighted in {a}']
        L += ["```", ""]
        open(os.path.join(out_dir, "RESULTS_README.md"), "w").write("\n".join(L) + "\n")
        n = len(static_files) + len(dyn_files)
        _ms(f"saved {n} result files ({len(static_files)} static, {len(dyn_files)} dynamic) "
              f"+ RESULTS_README.md -> {_disp(out_dir)}")
        return out_dir

    # ---- helpers ----
    def _repr_tau(self, condition):
        if (condition is not None and self.dataset is not None
                and self.dataset.metadata is not None and self.pseudotime is not None
                and self.condition_key in self.dataset.metadata.columns):
            m = (self.dataset.metadata[self.condition_key].values == condition)
            if m.any():
                return float(np.mean(self.pseudotime[m]))
        return 0.5

    def _default_tfs(self, n=40):
        if self.dataset is not None and self.dataset.ground_truth is not None \
                and len(self.dataset.ground_truth):
            tfs = [t for t in self.dataset.ground_truth["Gene1"].unique() if t in self.g2i]
            if tfs:
                return tfs[:n]
        # else the strongest static regulators
        return list(self.static_network(top_n_tfs=n).top.index)

    @torch.no_grad()
    def _binding_tfs(self, limit=300):
        """TF names that have mask binding (columns with any accessible edge), capped for cost;
        falls back to the strongest static regulators when there is no mask."""
        atac = getattr(self.model, "atac_mask", None)
        if atac is not None:
            col_sum = (atac.float() > 0).sum(0).cpu().numpy()
            tf_idx = np.where(col_sum > 0)[0]
            tfs = [self.gene_names[j] for j in tf_idx]
            if len(tfs) > limit:
                # keep the strongest by static out-strength
                strength = self.static_network(top_n_tfs=limit).top.index
                tfs = [t for t in strength if t in set(tfs)][:limit]
            return tfs
        return self._default_tfs(limit)

    def conditions(self):
        """List of condition values the model is resolved over (read from the data)."""
        return list(self.condition_map)
