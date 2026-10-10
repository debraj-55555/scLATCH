"""Preprocessing pipeline orchestrator."""

import logging

from sclatch.preprocessing.io import load_data
from sclatch.preprocessing.qc import run_qc
from sclatch.preprocessing.normalize import normalize
from sclatch.preprocessing.embed import embed
from sclatch.preprocessing.pseudotime import compute_pseudotime
from sclatch.preprocessing.export import export

logger = logging.getLogger(__name__)


def preprocess(
    data_dir: str,
    organism: str = "human",
    out_dir: str = None,
    # QC params
    min_genes: int = 200,
    max_genes: int = 6000,
    max_pct_mt: float = 20.0,
    min_cells: int = 10,
    # Embed params
    n_pcs: int = 50,
    n_neighbors: int = 30,
    resolution: float = 0.8,
    n_hvg_for_pca: int = 3000,
    # Pseudotime params
    compute_pseudotime_flag: bool = True,
    root_gene: str = None,
    root_cell_type: str = None,
):
    """Run full preprocessing pipeline.

    Parameters
    ----------
    data_dir : str
        Path to raw data directory.
    organism : str
        'human' or 'mouse'.
    out_dir : str, optional
        Output directory. Defaults to 'processed/{basename(data_dir)}'.
    min_genes, max_genes, max_pct_mt, min_cells
        QC thresholds.
    n_pcs, n_neighbors, resolution, n_hvg_for_pca
        Embedding parameters.
    compute_pseudotime_flag : bool
        Whether to compute diffusion pseudotime. Set False for
        non-trajectory data (e.g. cell line mixtures).
    root_gene, root_cell_type
        Pseudotime root cell selection (ignored if compute_pseudotime_flag=False).

    Returns
    -------
    adata : AnnData
        Fully preprocessed.
    adata_atac : AnnData or None
        ATAC peak matrix (raw, for mask building later).
    """
    import os

    if out_dir is None:
        out_dir = os.path.join("processed", os.path.basename(data_dir.rstrip("/")))

    logger.info(f"{'='*60}")
    logger.info(f"scLATCH preprocessing: {data_dir}")
    logger.info(f"  organism: {organism}")
    logger.info(f"  output:   {out_dir}")
    logger.info(f"{'='*60}")

    # Step 1: Load
    logger.info("\n[Step 1/5] Loading raw data...")
    adata_rna, adata_atac = load_data(data_dir, organism)

    # Step 2: QC
    logger.info("\n[Step 2/5] Quality control...")
    adata_rna = run_qc(
        adata_rna, organism=organism,
        min_genes=min_genes, max_genes=max_genes,
        max_pct_mt=max_pct_mt, min_cells=min_cells,
    )

    # Step 3: Normalize
    logger.info("\n[Step 3/5] Normalizing...")
    adata_rna = normalize(adata_rna)

    # Step 4: Embed
    logger.info("\n[Step 4/5] Embedding...")
    adata_rna = embed(
        adata_rna, n_pcs=n_pcs, n_neighbors=n_neighbors,
        resolution=resolution, n_hvg_for_pca=n_hvg_for_pca,
    )

    # Step 5: Pseudotime (optional — skip for non-trajectory data)
    if compute_pseudotime_flag:
        logger.info("\n[Step 5/5] Pseudotime...")
        adata_rna = compute_pseudotime(
            adata_rna, root_gene=root_gene, root_cell_type=root_cell_type,
        )
    else:
        logger.info("\n[Step 5/5] Pseudotime — SKIPPED (no trajectory data)")

    # Export
    logger.info("\nExporting...")
    export(adata_rna, out_dir)

    # Save ATAC AnnData separately (for mask building later)
    if adata_atac is not None:
        import os
        atac_path = os.path.join(out_dir, "adata_atac.h5ad")
        # Subset ATAC to cells that passed QC
        common = adata_rna.obs_names.intersection(adata_atac.obs_names)
        if len(common) > 0:
            adata_atac = adata_atac[common].copy()
            adata_atac.write(atac_path)
            logger.info(f"Saved adata_atac.h5ad: {adata_atac.n_obs} cells x {adata_atac.n_vars} peaks")
        else:
            logger.warning("No overlapping cells between RNA and ATAC — ATAC not saved")

    logger.info(f"\nDone! {adata_rna.n_obs} cells x {adata_rna.n_vars} genes")

    return adata_rna, adata_atac
