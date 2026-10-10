"""Export processed data in scLATCH standard format.

Output directory structure:
    out_dir/
    ├── data.npy             — (cells x genes) float32 array (fast loading)
    ├── gene_names.csv       — gene names (one per line, for data.npy column mapping)
    ├── data.csv             — (cells x genes) CSV with headers (for baselines/inspection)
    ├── pseudotime.npy       — (n_cells,) float32 in [0, 1]
    ├── cell_metadata.csv    — cell barcodes, leiden, pseudotime, [cell_type, sample]
    └── adata_rna.h5ad       — full AnnData (for downstream / baselines)
"""

import os
import logging

import numpy as np
import pandas as pd
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


def export(adata, out_dir: str):
    """Save processed AnnData to standard scLATCH format.

    Saves expression as both .npy (fast) and .csv (portable).

    Parameters
    ----------
    adata : AnnData
        Preprocessed: log-normalized .X, .obs['leiden'], .obs['pseudotime'].
    out_dir : str
        Output directory (created if needed).
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Expression matrix (log-normalized, all genes)
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # Fast binary format
    np.save(os.path.join(out_dir, "data.npy"), X)
    logger.info(f"Saved data.npy: {X.shape[0]} cells x {X.shape[1]} genes "
                f"({X.nbytes / 1e6:.0f} MB)")

    # Gene names (needed to map data.npy columns)
    gene_names = list(adata.var_names)
    pd.Series(gene_names).to_csv(
        os.path.join(out_dir, "gene_names.csv"), index=False, header=False)
    logger.info(f"Saved gene_names.csv: {len(gene_names)} genes")

    # CSV (for baselines / manual inspection)
    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=gene_names)
    expr_df.to_csv(os.path.join(out_dir, "data.csv"))
    logger.info(f"Saved data.csv")

    # 2. Pseudotime
    if "pseudotime" in adata.obs.columns:
        tau = adata.obs["pseudotime"].values.astype(np.float32)
        np.save(os.path.join(out_dir, "pseudotime.npy"), tau)
        logger.info(f"Saved pseudotime.npy: {len(tau)} cells")

    # 3. Cell metadata
    meta_cols = ["leiden"]
    if "pseudotime" in adata.obs.columns:
        meta_cols.append("pseudotime")
    for extra in ("condition", "cell_type", "sample"):
        if extra in adata.obs.columns and extra not in meta_cols:
            meta_cols.append(extra)

    meta = adata.obs[meta_cols].copy()
    meta.to_csv(os.path.join(out_dir, "cell_metadata.csv"))
    logger.info(f"Saved cell_metadata.csv: {len(meta)} cells, columns={meta_cols}")

    # 4. Full AnnData
    adata.write(os.path.join(out_dir, "adata_rna.h5ad"))
    logger.info(f"Saved adata_rna.h5ad")

    logger.info(f"Export complete: {out_dir}/")
