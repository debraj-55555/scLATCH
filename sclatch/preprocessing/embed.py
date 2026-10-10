"""Embedding: PCA, neighbors, UMAP, Leiden clustering.

Scaling is done on a temporary copy so that adata.X remains log-normalized
(needed for export). PCA coordinates are stored in adata.obsm['X_pca'].
"""

import logging

import scanpy as sc

logger = logging.getLogger(__name__)


def embed(
    adata,
    n_pcs: int = 50,
    n_neighbors: int = 30,
    resolution: float = 0.8,
    n_hvg_for_pca: int = 3000,
):
    """Compute PCA, UMAP, and Leiden clusters.

    For all-genes mode, PCA is computed on a temporary HVG subset
    (for efficiency), but the full gene set is preserved.

    Parameters
    ----------
    adata : AnnData
        Log-normalized expression (all genes).
    n_pcs : int
        Number of PCA components.
    n_neighbors : int
        Number of neighbors for kNN graph.
    resolution : float
        Leiden resolution.
    n_hvg_for_pca : int
        Number of HVGs to use for PCA computation (efficiency only,
        does NOT filter genes from output). Set to 0 to use all genes.

    Returns
    -------
    AnnData
        With .obsm['X_pca'], .obsm['X_umap'], .obs['leiden'].
    """
    # Compute HVGs for efficient PCA (all genes stay in output)
    if n_hvg_for_pca > 0 and adata.n_vars > n_hvg_for_pca:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_hvg_for_pca, flavor="seurat_v3",
            layer="counts",
        )
        hvg_genes = adata.var_names[adata.var["highly_variable"]]
        logger.info(f"Using {len(hvg_genes)} HVGs for PCA (all {adata.n_vars} genes kept)")
        adata_pca = adata[:, hvg_genes].copy()
    else:
        logger.info(f"Using all {adata.n_vars} genes for PCA")
        adata_pca = adata.copy()

    # Scale the PCA subset (destructive — that's why we use a copy)
    sc.pp.scale(adata_pca, max_value=10)

    # PCA
    sc.tl.pca(adata_pca, n_comps=min(n_pcs, adata_pca.n_vars - 1, adata_pca.n_obs - 1))

    # Transfer PCA back to full AnnData
    adata.obsm["X_pca"] = adata_pca.obsm["X_pca"]
    if "PCs" in adata_pca.varm:
        # Store for reference but shape won't match full var
        adata.uns["pca_variance_ratio"] = adata_pca.uns.get("pca", {}).get(
            "variance_ratio", None
        )

    # Neighbors + UMAP + Leiden
    sc.pp.neighbors(adata, n_neighbors=n_neighbors,
                    n_pcs=min(n_pcs, adata.obsm["X_pca"].shape[1]),
                    use_rep="X_pca")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution)

    n_clusters = adata.obs["leiden"].nunique()
    logger.info(f"Embedding: {n_pcs} PCs, {n_clusters} Leiden clusters")

    return adata
