"""Ground truth network download from CollecTRI (OmniPath).

CollecTRI is a curated database of TF-target regulatory interactions
used as gold standard for GRN benchmarking.
"""

import os
import io
import logging
import urllib.request

import pandas as pd

logger = logging.getLogger(__name__)

COLLECTRI_URL = (
    "https://omnipathdb.org/interactions?"
    "datasets=collectri&genesymbols=yes&fields=sources,references"
    "&license=academic"
)

# OmniPath organism codes
ORGANISM_CODE = {
    "human": 9606,
    "mouse": 10090,
}


def download_collectri(organism: str = "human") -> pd.DataFrame:
    """Download full CollecTRI database from OmniPath.

    Parameters
    ----------
    organism : str
        'human' or 'mouse'.

    Returns
    -------
    pd.DataFrame
        Columns: Gene1 (TF), Gene2 (target).
    """
    org_code = ORGANISM_CODE.get(organism)
    url = COLLECTRI_URL
    if org_code:
        url += f"&organisms={org_code}"

    logger.info(f"Downloading CollecTRI ({organism})...")
    response = urllib.request.urlopen(url, timeout=120)
    data = response.read().decode("utf-8")
    collectri = pd.read_csv(io.StringIO(data), sep="\t")
    logger.info(f"  Raw CollecTRI: {len(collectri)} interactions")

    # Extract TF → target pairs
    gt = collectri[["source_genesymbol", "target_genesymbol"]].drop_duplicates()
    gt.columns = ["Gene1", "Gene2"]
    gt = gt[gt["Gene1"] != gt["Gene2"]]  # remove self-edges

    logger.info(f"  Unique edges: {len(gt)}, TFs: {gt['Gene1'].nunique()}")
    return gt


def build_ground_truth(
    gene_names: list,
    organism: str = "human",
    out_dir: str = None,
    gt_file: str = None,
) -> pd.DataFrame:
    """Build ground truth filtered to our gene set.

    Parameters
    ----------
    gene_names : list
        Gene names from data.csv columns.
    organism : str
        'human' or 'mouse'.
    out_dir : str, optional
        If provided, saves label.csv here.
    gt_file : str, optional
        Path to pre-existing GT file (skip download).

    Returns
    -------
    pd.DataFrame
        Ground truth with columns Gene1, Gene2.
    """
    if gt_file and os.path.exists(gt_file):
        logger.info(f"Loading GT from {gt_file}")
        gt = pd.read_csv(gt_file)
        if "Gene1" not in gt.columns:
            # Try common column name variants
            if "source_genesymbol" in gt.columns:
                gt = gt.rename(columns={"source_genesymbol": "Gene1",
                                        "target_genesymbol": "Gene2"})
            else:
                gt.columns = ["Gene1", "Gene2"] + list(gt.columns[2:])
    else:
        gt = download_collectri(organism)

    # Filter to genes in our dataset
    gene_set = set(gene_names)
    gt = gt[gt["Gene1"].isin(gene_set) & gt["Gene2"].isin(gene_set)].copy()
    gt = gt[["Gene1", "Gene2"]].drop_duplicates()
    gt = gt[gt["Gene1"] != gt["Gene2"]]

    n_tfs = gt["Gene1"].nunique()
    n_targets = gt["Gene2"].nunique()
    logger.info(f"Filtered GT: {len(gt)} edges, {n_tfs} TFs, {n_targets} targets")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "label.csv")
        gt.to_csv(path, index=False)
        logger.info(f"Saved {path}")

    return gt
