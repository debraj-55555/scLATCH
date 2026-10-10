"""Quality control: filter cells and genes."""

import logging

import numpy as np
import scanpy as sc

logger = logging.getLogger(__name__)

# Mitochondrial gene prefixes by organism
MT_PREFIX = {
    "human": "MT-",
    "mouse": "mt-",
}


def run_qc(
    adata,
    organism: str = "human",
    min_genes: int = 200,
    max_genes: int = 6000,
    max_pct_mt: float = 20.0,
    min_cells: int = 10,
):
    """Filter cells and genes based on QC metrics.

    Parameters
    ----------
    adata : AnnData
        Raw RNA counts (cells x genes).
    organism : str
        'human' or 'mouse' — determines MT prefix.
    min_genes : int
        Remove cells with fewer genes.
    max_genes : int
        Remove cells with more genes.
    max_pct_mt : float
        Remove cells with higher mitochondrial %.
    min_cells : int
        Remove genes expressed in fewer cells.

    Returns
    -------
    AnnData
        Filtered copy.
    """
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars

    # Compute QC metrics
    mt_prefix = MT_PREFIX.get(organism, "MT-")
    adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # Filter cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs["n_genes_by_counts"] < max_genes].copy()
    adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt].copy()

    # Filter genes
    sc.pp.filter_genes(adata, min_cells=min_cells)

    logger.info(
        f"QC: {n_cells_before} -> {adata.n_obs} cells, "
        f"{n_genes_before} -> {adata.n_vars} genes"
    )

    return adata
