"""Data loading — auto-detects format from directory contents.

Supports:
  - 10x Multiome h5 (RNA + ATAC in one file)
  - 10x CellRanger MTX triplet (matrix.mtx.gz + features.tsv.gz + barcodes.tsv.gz)
  - Text counts (genes x cells .txt.gz / .csv.gz)
  - Multiple samples in one directory (auto-concat with sample labels)
"""

import os
import glob
import logging

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import issparse, csr_matrix

logger = logging.getLogger(__name__)


def load_data(data_dir: str, organism: str = "human"):
    """Load raw data from data_dir. Auto-detects format.

    Parameters
    ----------
    data_dir : str
        Path to directory containing raw data files.
    organism : str
        'human' or 'mouse'.

    Returns
    -------
    adata_rna : AnnData
        RNA expression (raw counts).
    adata_atac : AnnData or None
        ATAC peak matrix if available.
    """
    data_dir = os.path.abspath(data_dir)
    fmt = _detect_format(data_dir)
    logger.info(f"Detected format: {fmt}")

    if fmt == "h5_multiome":
        adata_rna, adata_atac = _load_h5_multiome(data_dir, organism)
    elif fmt == "h5_multiome_multi":
        adata_rna, adata_atac = _load_h5_multiome_multi(data_dir, organism)
    elif fmt == "mtx":
        adata_rna, adata_atac = _load_mtx(data_dir, organism)
    elif fmt == "mtx_multi":
        adata_rna, adata_atac = _load_mtx_multi(data_dir, organism)
    elif fmt == "text":
        adata_rna, adata_atac = _load_text(data_dir, organism)
    elif fmt == "h5ad":
        adata_rna, adata_atac = _load_h5ad(data_dir, organism)
    else:
        raise ValueError(
            f"Cannot detect data format in {data_dir}. "
            "Expected: .h5 files, .h5ad files, matrix.mtx.gz triplets, "
            "or .txt.gz/.csv count files."
        )

    adata_rna = _merge_sidecar_metadata(data_dir, adata_rna)
    return adata_rna, adata_atac


def _merge_sidecar_metadata(data_dir, adata_rna):
    """Merge a sidecar ``metadata.csv`` (barcodes x annotations, e.g. condition,
    cell_type) into ``adata_rna.obs``, aligned by barcode. Columns already in obs
    are kept. No-op if the file is absent or the barcodes do not match."""
    meta_path = os.path.join(data_dir, "metadata.csv")
    if adata_rna is None or not os.path.exists(meta_path):
        return adata_rna
    try:
        meta = pd.read_csv(meta_path, index_col=0)
    except Exception as e:
        logger.warning(f"Could not read metadata.csv: {e}")
        return adata_rna
    common = adata_rna.obs_names.intersection(meta.index)
    if len(common) == 0:
        logger.warning("metadata.csv barcodes do not match RNA cells; skipping merge")
        return adata_rna
    meta = meta.reindex(adata_rna.obs_names)
    new_cols = [c for c in meta.columns if c not in adata_rna.obs.columns]
    for c in new_cols:
        adata_rna.obs[c] = meta[c].values
    logger.info(f"Merged metadata.csv: added obs columns {new_cols} "
                f"({len(common)}/{adata_rna.n_obs} cells matched)")
    return adata_rna


def _detect_format(data_dir: str) -> str:
    """Detect data format from file patterns in directory."""
    h5_files = glob.glob(os.path.join(data_dir, "*.h5"))
    # Filter out index files
    h5_files = [f for f in h5_files if not f.endswith(".tbi")]

    mtx_files = glob.glob(os.path.join(data_dir, "*matrix.mtx.gz")) + \
                glob.glob(os.path.join(data_dir, "*matrix.mtx"))

    txt_files = glob.glob(os.path.join(data_dir, "*counts*.txt.gz")) + \
                glob.glob(os.path.join(data_dir, "*counts*.csv.gz")) + \
                glob.glob(os.path.join(data_dir, "*counts*.tsv.gz")) + \
                glob.glob(os.path.join(data_dir, "*counts*.txt")) + \
                glob.glob(os.path.join(data_dir, "*counts*.csv")) + \
                glob.glob(os.path.join(data_dir, "*counts*.tsv"))

    # Also check for h5ad (already preprocessed — e.g. BMMC)
    h5ad_files = glob.glob(os.path.join(data_dir, "*.h5ad"))

    if len(h5_files) == 1:
        return "h5_multiome"
    elif len(h5_files) > 1:
        return "h5_multiome_multi"
    elif len(mtx_files) == 1:
        return "mtx"
    elif len(mtx_files) > 1:
        return "mtx_multi"
    elif txt_files:
        return "text"
    elif h5ad_files:
        return "h5ad"
    else:
        # Check subdirectories for CellRanger-style output
        subdirs = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
        for sub in subdirs:
            sub_h5 = glob.glob(os.path.join(data_dir, sub, "*.h5"))
            if sub_h5:
                return "h5_multiome_multi"
        return "unknown"


# ── h5ad (pre-processed AnnData) ─────────────────────────────

def _load_h5ad(data_dir: str, organism: str):
    """Load from h5ad files. Handles combined RNA+ATAC or separate files.

    Looks for:
      - adata_rna.h5ad / adata_atac.h5ad (separate, preferred)
      - *_rna_*.h5ad / *_atac_*.h5ad (named pattern)
      - Single .h5ad with feature_types column (combined multiome)

    Uses .layers['counts'] as raw counts if available, otherwise .X.
    Preserves obs columns (cell_type, batch, pseudotime, etc.).
    """
    h5ad_files = sorted(glob.glob(os.path.join(data_dir, "*.h5ad")))

    # Case 1: separate RNA + ATAC files
    rna_file = None
    atac_file = None
    for f in h5ad_files:
        bn = os.path.basename(f).lower()
        if 'rna' in bn or 'gex' in bn or 'cdna' in bn or 'expression' in bn or 'expr' in bn:
            rna_file = f
        elif 'atac' in bn or 'chromatin' in bn:
            atac_file = f

    if rna_file:
        logger.info(f"Loading h5ad RNA: {os.path.basename(rna_file)}")
        adata_rna = sc.read_h5ad(rna_file)
        # Use raw counts if available
        if 'counts' in adata_rna.layers:
            adata_rna.X = adata_rna.layers['counts']
            logger.info("  Using .layers['counts'] as raw counts")
        logger.info(f"  RNA: {adata_rna.n_obs} cells x {adata_rna.n_vars} genes")

        adata_atac = None
        if atac_file:
            logger.info(f"Loading h5ad ATAC: {os.path.basename(atac_file)}")
            adata_atac = sc.read_h5ad(atac_file)
            if 'counts' in adata_atac.layers:
                adata_atac.X = adata_atac.layers['counts']
            logger.info(f"  ATAC: {adata_atac.n_obs} cells x {adata_atac.n_vars} peaks")

        return adata_rna, adata_atac

    # Case 2: single combined h5ad with feature_types
    if len(h5ad_files) == 1:
        logger.info(f"Loading combined h5ad: {os.path.basename(h5ad_files[0])}")
        adata = sc.read_h5ad(h5ad_files[0])

        # Use raw counts
        if 'counts' in adata.layers:
            adata.X = adata.layers['counts']
            logger.info("  Using .layers['counts'] as raw counts")

        # Split if multiome (has feature_types)
        if 'feature_types' in adata.var.columns:
            ft = adata.var['feature_types']
            # Handle various naming conventions
            rna_vals = {'Gene Expression', 'GEX'}
            atac_vals = {'Peaks', 'ATAC'}
            rna_mask = ft.isin(rna_vals)
            atac_mask = ft.isin(atac_vals)

            if rna_mask.any() and atac_mask.any():
                adata_rna = adata[:, rna_mask].copy()
                adata_atac = adata[:, atac_mask].copy()
                logger.info(f"  Split: {adata_rna.n_vars} RNA + {adata_atac.n_vars} ATAC")
                return adata_rna, adata_atac

        # No split needed — treat entire file as RNA
        logger.info(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} features")
        return adata, None

    raise ValueError(f"Cannot determine RNA/ATAC from h5ad files in {data_dir}: {h5ad_files}")


# ── 10x Multiome h5 ──────────────────────────────────────────

def _split_multiome(adata):
    """Split a 10x Multiome AnnData into RNA and ATAC."""
    adata.var_names_make_unique()

    if "feature_types" in adata.var.columns:
        rna_mask = adata.var["feature_types"] == "Gene Expression"
        atac_mask = adata.var["feature_types"] == "Peaks"
    else:
        # Fallback: peaks have chr:start-end format
        atac_mask = adata.var_names.str.match(r"chr[\dXYMmt]+[:\-]")
        rna_mask = ~atac_mask

    rna = adata[:, rna_mask].copy()
    atac = adata[:, atac_mask].copy() if atac_mask.sum() > 0 else None

    return rna, atac


def _load_h5_multiome(data_dir: str, organism: str):
    """Single h5 file containing RNA + ATAC."""
    h5_files = glob.glob(os.path.join(data_dir, "*.h5"))
    h5_files = [f for f in h5_files if not f.endswith(".tbi")]
    h5_path = h5_files[0]
    logger.info(f"Loading {os.path.basename(h5_path)}")

    adata = sc.read_10x_h5(h5_path, gex_only=False)
    rna, atac = _split_multiome(adata)

    logger.info(f"  RNA:  {rna.n_obs} cells x {rna.n_vars} genes")
    if atac is not None:
        logger.info(f"  ATAC: {atac.n_obs} cells x {atac.n_vars} peaks")

    return rna, atac


def _load_h5_multiome_multi(data_dir: str, organism: str):
    """Multiple h5 files = multiple samples. Concat with sample labels."""
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*filtered_feature_bc_matrix.h5")))
    if not h5_files:
        h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        h5_files = [f for f in h5_files if not f.endswith(".tbi")]

    if not h5_files:
        # Check subdirectories
        h5_files = sorted(glob.glob(os.path.join(data_dir, "*", "*.h5")))

    rna_list, atac_list = [], []
    sample_names = []

    for h5_path in h5_files:
        # Derive sample name from filename
        base = os.path.basename(h5_path)
        sample = base.replace("_filtered_feature_bc_matrix.h5", "").replace(".h5", "")
        # Strip common prefixes like GSM...
        parts = sample.split("_")
        if parts[0].startswith("GSM"):
            sample = "_".join(parts[1:])
        sample_names.append(sample)

        logger.info(f"Loading sample: {sample}")
        adata = sc.read_10x_h5(h5_path, gex_only=False)
        rna, atac = _split_multiome(adata)

        # Tag cells with sample identity
        rna.obs["sample"] = sample
        rna.obs_names = [f"{sample}__{bc}" for bc in rna.obs_names]
        if atac is not None:
            atac.obs["sample"] = sample
            atac.obs_names = [f"{sample}__{bc}" for bc in atac.obs_names]

        rna_list.append(rna)
        if atac is not None:
            atac_list.append(atac)

        logger.info(f"  RNA:  {rna.n_obs} x {rna.n_vars}")

    # Concatenate
    adata_rna = ad.concat(rna_list, join="inner")
    # ATAC peaks differ across samples (different peak calling) — use outer join
    adata_atac = ad.concat(atac_list, join="outer", fill_value=0) if atac_list else None

    logger.info(f"Merged: {adata_rna.n_obs} cells x {adata_rna.n_vars} genes "
                f"({len(sample_names)} samples)")

    return adata_rna, adata_atac


# ── 10x MTX triplet ──────────────────────────────────────────

def _find_mtx_triplet(data_dir: str):
    """Find matrix.mtx.gz + features.tsv.gz + barcodes.tsv.gz."""
    mtx = glob.glob(os.path.join(data_dir, "*matrix.mtx.gz"))
    feat = glob.glob(os.path.join(data_dir, "*features.tsv.gz"))
    barc = glob.glob(os.path.join(data_dir, "*barcodes.tsv.gz"))
    if mtx and feat and barc:
        return mtx[0], feat[0], barc[0]
    return None, None, None


def _load_mtx(data_dir: str, organism: str):
    """Single MTX triplet."""
    mtx_path, feat_path, barc_path = _find_mtx_triplet(data_dir)
    if mtx_path is None:
        raise FileNotFoundError(f"No MTX triplet found in {data_dir}")

    logger.info(f"Loading MTX: {os.path.basename(mtx_path)}")
    adata = sc.read_10x_mtx(data_dir, var_names="gene_symbols", make_unique=True)
    return adata, None


def _load_mtx_multi(data_dir: str, organism: str):
    """Multiple MTX triplets — group by filename prefix."""
    mtx_files = sorted(glob.glob(os.path.join(data_dir, "*matrix.mtx.gz")))
    rna_list = []

    for mtx_path in mtx_files:
        base = os.path.basename(mtx_path)
        # Prefix = everything before _matrix.mtx.gz
        prefix = base.replace("_matrix.mtx.gz", "").replace("matrix.mtx.gz", "sample")

        # Find matching features and barcodes
        feat = mtx_path.replace("matrix.mtx.gz", "features.tsv.gz")
        barc = mtx_path.replace("matrix.mtx.gz", "barcodes.tsv.gz")
        if not os.path.exists(feat) or not os.path.exists(barc):
            continue

        logger.info(f"Loading sample: {prefix}")
        # Read using scanpy's MTX reader
        from scipy.io import mmread
        import gzip

        mat = mmread(mtx_path).T.tocsr()  # genes x cells -> cells x genes

        # Read features
        with gzip.open(feat, "rt") as f:
            features = pd.read_csv(f, sep="\t", header=None)
        gene_names = features[1].values if features.shape[1] > 1 else features[0].values

        # Read barcodes
        with gzip.open(barc, "rt") as f:
            barcodes = pd.read_csv(f, sep="\t", header=None)[0].values

        adata = ad.AnnData(
            X=mat[:len(barcodes), :len(gene_names)],
            obs=pd.DataFrame(index=[f"{prefix}__{bc}" for bc in barcodes]),
            var=pd.DataFrame(index=gene_names),
        )
        adata.var_names_make_unique()
        adata.obs["sample"] = prefix

        # If features has a 3rd column (feature_types), split RNA/ATAC
        if features.shape[1] >= 3:
            adata.var["feature_types"] = features[2].values[:len(gene_names)]

        rna_list.append(adata)
        logger.info(f"  {adata.n_obs} cells x {adata.n_vars} features")

    if not rna_list:
        raise FileNotFoundError(f"No valid MTX triplets in {data_dir}")

    # Split RNA/ATAC per sample BEFORE concat (peak names differ across samples)
    if rna_list and "feature_types" in rna_list[0].var.columns:
        rna_parts, atac_parts = [], []
        for adata in rna_list:
            r, a = _split_multiome(adata)
            rna_parts.append(r)
            if a is not None:
                atac_parts.append(a)
        merged_rna = ad.concat(rna_parts, join="inner")
        merged_atac = ad.concat(atac_parts, join="outer", fill_value=0) if atac_parts else None
        rna, atac = merged_rna, merged_atac
    else:
        merged = ad.concat(rna_list, join="inner")
        rna, atac = merged, None

    logger.info(f"Merged: {rna.n_obs} cells x {rna.n_vars} genes")
    return rna, atac


# ── Text counts ──────────────────────────────────────────────

def _find_text_files(data_dir: str):
    """Find count files, grouped by type (RNA vs ATAC)."""
    all_files = []
    for ext in ("*.txt.gz", "*.csv.gz", "*.tsv.gz", "*.txt", "*.csv", "*.tsv"):
        all_files.extend(glob.glob(os.path.join(data_dir, f"*counts*{ext[1:]}")))

    if not all_files:
        return None, None

    # Separate RNA vs ATAC files by name heuristics
    rna_file = None
    atac_file = None
    for f in all_files:
        fname = os.path.basename(f).lower()
        if any(k in fname for k in ("cdna", "rna", "gene_expression")):
            rna_file = f
        elif any(k in fname for k in ("chromatin", "atac", "peak")):
            atac_file = f

    # If no clear RNA/ATAC split, use the first file as RNA
    if rna_file is None:
        rna_file = all_files[0]

    return rna_file, atac_file


def _read_text_matrix(path: str):
    """Read a text count matrix file into AnnData.

    Auto-detects: separator (tab vs comma), orientation (genes×cells vs cells×genes).
    """
    logger.info(f"Loading text: {os.path.basename(path)}")

    sep = "\t" if (".tsv" in path or ".txt" in path) else ","
    df = pd.read_csv(path, sep=sep, index_col=0)

    # Detect orientation: genes×cells vs cells×genes
    # Heuristic: check if row names look like gene symbols (start with letter, no barcode pattern)
    # or if column names look like gene symbols
    import re
    sample_rows = [str(x) for x in df.index[:5]]
    sample_cols = [str(x) for x in df.columns[:5]]
    # Gene names: start with letter or digit, short (e.g., "TP53", "0610007P14Rik")
    # Barcodes: contain repeated separators (e.g., "R1.01,R2.01,R3.06,P1.55", "AACG-1")
    barcode_pattern = re.compile(r'[,.].*[,.].*[,.]')  # 3+ separators = barcode
    row_is_barcode = sum(1 for r in sample_rows if barcode_pattern.search(r)) > len(sample_rows) // 2
    col_is_barcode = sum(1 for c in sample_cols if barcode_pattern.search(c)) > len(sample_cols) // 2

    if col_is_barcode and not row_is_barcode:
        # Columns are barcodes (cells), rows are genes → transpose
        logger.info("  Detected genes x cells → transposing")
        df = df.T
    elif row_is_barcode and not col_is_barcode:
        # Rows are barcodes (cells), columns are genes → correct orientation
        pass
    elif df.shape[0] > df.shape[1]:
        # Fallback: more rows than columns → likely genes x cells
        logger.info("  Detected genes x cells → transposing")
        df = df.T

    adata = ad.AnnData(
        X=csr_matrix(df.values.astype(np.float32)),
        obs=pd.DataFrame(index=df.index),
        var=pd.DataFrame(index=df.columns),
    )
    adata.var_names_make_unique()
    logger.info(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} features")
    return adata


def _load_text(data_dir: str, organism: str):
    """Load text-format count matrix(es).

    Handles:
      - Single file: RNA only
      - Paired files: RNA (*cDNA*/*rna*) + ATAC (*chromatin*/*atac*)
        e.g. SNARE-seq GSE126074
    """
    rna_file, atac_file = _find_text_files(data_dir)

    if rna_file is None:
        raise FileNotFoundError(f"No count files found in {data_dir}")

    adata_rna = _read_text_matrix(rna_file)
    adata_atac = None

    if atac_file is not None:
        adata_atac = _read_text_matrix(atac_file)
        # Intersect barcodes
        common = sorted(set(adata_rna.obs_names) & set(adata_atac.obs_names))
        if len(common) < adata_rna.n_obs:
            logger.info(f"  Intersecting barcodes: {adata_rna.n_obs} RNA ∩ "
                        f"{adata_atac.n_obs} ATAC → {len(common)} common")
            adata_rna = adata_rna[common].copy()
            adata_atac = adata_atac[common].copy()

    # Try to load cell type annotations if present
    ct_files = glob.glob(os.path.join(data_dir, "*celltype*")) + \
               glob.glob(os.path.join(data_dir, "*cell_type*")) + \
               glob.glob(os.path.join(data_dir, "*metadata*"))
    if ct_files:
        ct_path = ct_files[0]
        logger.info(f"  Loading cell types: {os.path.basename(ct_path)}")
        ct_df = pd.read_csv(ct_path,
                            sep="\t" if (".tsv" in ct_path or ".txt" in ct_path) else ",")
        for bc_col in ["rna.bc", "barcode", "cell_barcode", "cell_id"]:
            if bc_col in ct_df.columns:
                for ct_col in ["celltype", "cell_type", "CellType", "cluster"]:
                    if ct_col in ct_df.columns:
                        bc_to_ct = dict(zip(ct_df[bc_col], ct_df[ct_col]))
                        adata_rna.obs["cell_type"] = [
                            bc_to_ct.get(bc, bc_to_ct.get(
                                bc.replace(".", ","), "unknown"))
                            for bc in adata_rna.obs_names
                        ]
                        break
                break

    return adata_rna, adata_atac
