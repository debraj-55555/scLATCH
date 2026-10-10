"""Normalization: library-size normalize + log1p."""

import logging

import scanpy as sc

logger = logging.getLogger(__name__)


def normalize(adata, target_sum: float = 1e4):
    """Normalize counts and log-transform.

    Stores raw counts in adata.layers['counts'] before transforming.

    Parameters
    ----------
    adata : AnnData
        QC-filtered raw counts.
    target_sum : float
        Target library size for normalization.

    Returns
    -------
    AnnData
        Log-normalized expression in .X, raw counts in .layers['counts'].
    """
    # Preserve raw counts
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    logger.info(f"Normalized: {adata.n_obs} cells x {adata.n_vars} genes (log1p)")

    return adata
