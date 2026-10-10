"""Pseudotime computation via diffusion pseudotime (DPT).

Direction validation:
    DPT computes diffusion distance from a root cell — root gets τ=0,
    farthest cell gets τ=1. The direction is deterministic given the root.

    However, root cell selection can go wrong:
    - argmax(root_gene) may pick a cell in the wrong condition (e.g.,
      highest Tcf7 in WT exhausted progenitors, not in naive cells)
    - The kNN graph topology can make biologically "early" cells
      appear distant from the root

    We handle this with a two-layer strategy:
    1. If root_gene provided: validate that root_gene DECREASES along τ.
       If not (ρ > +0.05), invert pseudotime (1-τ).
    2. If no root_gene: use gene-count heuristic (CytoTRACE-inspired)
       as a soft signal, but with a WARNING that it may fail for
       immune activation / T cell exhaustion where activated cells
       express MORE genes than quiescent progenitors.

    The gene-count heuristic (CytoTRACE, Gulati et al. 2020 Science)
    assumes stem/progenitor cells express more genes. This holds for
    hematopoiesis and embryonic development but FAILS for:
    - T cell activation/exhaustion (exhausted > naive in gene count)
    - Immune activation (activated > resting)
    - Some cancers (proliferating > quiescent)

    Therefore: always provide root_gene for dynamic GRN analysis.
    Gene-count fallback is a heuristic, not a guarantee.
"""

import logging

import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def _compute_gene_count_score(X):
    """CytoTRACE-inspired stemness score: # genes expressed per cell.

    Higher gene count = more transcriptionally active = assumed more
    potent/stem-like. Normalized to [0, 1].

    WARNING: This assumption fails for immune activation contexts where
    activated/exhausted cells express MORE genes than quiescent progenitors.

    Returns
    -------
    scores : np.ndarray
        Per-cell stemness score in [0, 1].
    """
    if issparse(X):
        gene_counts = np.array((X > 0).sum(axis=1)).flatten().astype(float)
    else:
        gene_counts = (X > 0).sum(axis=1).astype(float)

    scores = (gene_counts - gene_counts.min()) / (gene_counts.max() - gene_counts.min() + 1e-10)
    return scores


def _select_root_cell(adata, root_gene=None, root_cell_type=None):
    """Select root cell for DPT.

    Priority:
      1. root_gene — cell with highest expression
      2. root_cell_type — cell in this cluster at DC1 extreme
      3. Auto — gene-count heuristic (highest gene count = most stem-like)

    Returns
    -------
    root_idx : int
    method : str
        Description of how root was selected.
    """
    if root_gene and root_gene in adata.var_names:
        expr = adata[:, root_gene].X
        if issparse(expr):
            expr = expr.toarray()
        root_idx = int(np.argmax(expr.flatten()))
        return root_idx, f"highest {root_gene}"

    if root_cell_type:
        ct_col = None
        for col in ["cell_type", "leiden"]:
            if col in adata.obs.columns:
                if root_cell_type in adata.obs[col].values:
                    ct_col = col
                    break
        if ct_col:
            mask = adata.obs[ct_col] == root_cell_type
            dc1 = adata.obsm["X_diffmap"][:, 0]
            candidates = np.where(mask)[0]
            root_idx = int(candidates[np.argmin(dc1[candidates])])
            return root_idx, f"{ct_col}={root_cell_type}, DC1 min"

    # Auto: gene-count heuristic (CytoTRACE-inspired)
    stemness = _compute_gene_count_score(adata.X)
    root_idx = int(np.argmax(stemness))
    logger.warning(
        "No root_gene provided — using gene-count heuristic (CytoTRACE-inspired). "
        "This assumes stem cells express more genes, which may fail for "
        "immune activation / T cell exhaustion. Provide root_gene for "
        "dynamic GRN analysis."
    )
    return root_idx, "gene-count max (auto, CytoTRACE heuristic)"


def _validate_and_fix_direction(adata, tau, root_gene=None):
    """Validate pseudotime direction and invert if needed.

    Strategy:
      1. If root_gene provided: check Spearman correlation with τ.
         Root gene (stem/progenitor marker) should DECREASE along τ.
         If ρ > +0.05 (increases), invert.
      2. If no root_gene: check gene-count correlation with τ.
         Gene count should DECREASE along τ (stem→differentiated).
         If ρ > +0.1, invert. Higher threshold because gene-count
         heuristic is less reliable.

    Returns
    -------
    tau : np.ndarray
        Corrected pseudotime (possibly inverted).
    inverted : bool
    """
    inverted = False

    if root_gene and root_gene in adata.var_names:
        expr = adata[:, root_gene].X
        if issparse(expr):
            expr = expr.toarray()
        expr = expr.flatten()

        expressing = expr > 0
        if expressing.sum() > 50:
            rho, _ = spearmanr(tau[expressing], expr[expressing])
            if rho > 0.05:
                tau = 1.0 - tau
                inverted = True
                logger.info(f"  Direction fix: {root_gene} had ρ={rho:+.3f} "
                            f"(should decrease along τ) → INVERTED")
            else:
                logger.info(f"  Direction OK: {root_gene} ρ={rho:+.3f} (decreases along τ)")
        else:
            logger.warning(f"  Direction check skipped: {root_gene} expressed in "
                           f"only {expressing.sum()} cells")
    else:
        # Fallback: gene-count heuristic
        stemness = _compute_gene_count_score(adata.X)
        rho, _ = spearmanr(tau, stemness)
        if rho > 0.1:
            tau = 1.0 - tau
            inverted = True
            logger.info(f"  Direction fix (gene-count heuristic): ρ={rho:+.3f} → INVERTED")
            logger.warning("  Gene-count inversion applied — verify this is correct "
                           "for your biology. Provide root_gene for reliable direction.")
        else:
            logger.info(f"  Direction check (gene-count heuristic): ρ={rho:+.3f} → OK")

    return tau, inverted


def compute_pseudotime(
    adata,
    root_gene: str = None,
    root_cell_type: str = None,
    n_dcs: int = 15,
):
    """Compute DPT pseudotime with automatic direction correction.

    Root cell selection priority:
      1. root_gene — cell with highest expression of this gene.
         Also used to validate and auto-correct pseudotime direction.
      2. root_cell_type — cell in this cluster at DC1 extreme.
      3. Auto — gene-count heuristic (CytoTRACE-inspired: highest gene
         count = most stem-like). WARNING: fails for immune activation.

    After DPT, direction is validated:
      - If root_gene provided: checks that root_gene DECREASES along τ.
      - If no root_gene: checks that gene count DECREASES along τ.
      - If validation fails, pseudotime is inverted (1-τ).

    For dynamic GRN analysis, always provide root_gene (a stem/progenitor
    marker) to ensure correct pseudotime direction. For static GRN
    benchmarking, direction doesn't affect results.

    Parameters
    ----------
    adata : AnnData
        Must have neighbors computed (.obsp['connectivities']).
    root_gene : str, optional
        Gene name for root cell selection AND direction validation.
        Should be a stem/progenitor marker (e.g., 'Tcf7' for T cells,
        'CD34' for hematopoiesis, 'SOX2' for neural).
    root_cell_type : str, optional
        Cell type or leiden cluster label for root selection.
    n_dcs : int
        Number of diffusion components.

    Returns
    -------
    AnnData
        With .obs['pseudotime'] in [0, 1], direction-validated.
    """
    sc.tl.diffmap(adata, n_comps=n_dcs)

    root_idx, method = _select_root_cell(adata, root_gene, root_cell_type)
    logger.info(f"Root cell: index {root_idx} ({method})")

    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata, n_branchings=0)

    # Normalize to [0, 1], handle infinities
    tau = adata.obs["dpt_pseudotime"].values.copy()
    tau[np.isinf(tau)] = np.nanmax(tau[~np.isinf(tau)])
    tau = (tau - np.nanmin(tau)) / (np.nanmax(tau) - np.nanmin(tau) + 1e-10)

    # Validate and fix direction
    tau, inverted = _validate_and_fix_direction(adata, tau, root_gene)

    adata.obs["pseudotime"] = tau

    logger.info(f"Pseudotime: range [{tau.min():.3f}, {tau.max():.3f}]"
                f"{' (inverted)' if inverted else ''}")

    return adata
