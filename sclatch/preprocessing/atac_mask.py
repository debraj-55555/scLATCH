"""ATAC mask construction via ChIP-Atlas TF binding data.

Builds a binary mask M[target, TF] = 1 where:
  - TF has experimental ChIP-seq binding near target gene (ChIP-Atlas)
  - Target gene's promoter region has an accessible ATAC peak

TF list comes from curated Aerts-lab TF lists ONLY, never from ground truth.
This prevents information leakage between evaluation GT and model prior.
The candidate-regulator universe is a curated Aerts-lab TF list shipped in priors/: the SCENIC+
allTFs lists (priors/allTFs_*.txt, the default) or the pySCENIC TF lists
(priors/hs_hgnc_tfs.txt, priors/mm_mgi_tfs.txt), chosen with tf_list=.

Defaults (unchanged from the paper, do not alter):
  - binding window: +/- 10 kb around the TSS (window=10000)
  - masking is over the curated TF list only, never over ground truth
  - the model then applies effective_mask = mask + mask_leak * (1 - mask),
    mask_leak = 0.4, with NO annealing (see the trainer / FINAL_DYNAMIC_CONFIG)

Reproducibility note:
  A TF missing from the local ChIP-Atlas cache is downloaded unless allow_download=False,
  and without a local TSS table the TSS positions come from Ensembl. Both sources change
  over time, so a freshly downloaded mask can drift from the one used in the paper. To
  reproduce the paper's masks, build from the cache in priors/chip_atlas/ with
  allow_download=False, or use the precomputed atac_mask.npz files in the figshare data
  record (https://doi.org/10.6084/m9.figshare.34335939). The caches and the TF lists are
  not shipped in the pip package; they live in the repository priors/.

Genome assemblies:
  human -> hg38
  mouse -> mm10
"""

import os
import json
import logging
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse

logger = logging.getLogger(__name__)

# Genome assembly per organism
GENOME_MAP = {"human": "hg38", "mouse": "mm10"}

# Curated TF lists of the Aerts laboratory, species-specific, independent of any GT.
# Read from the priors/ directory (the single location for all priors).
#   scenicplus : SCENIC+ allTFs lists (Bravo Gonzalez-Blas et al. 2023)  [default]
#   pyscenic   : pySCENIC TF lists     (Van de Sande et al. 2020)
TF_LISTS = {
    "scenicplus": {"human": "allTFs_hg38.txt", "mouse": "allTFs_mm.txt"},
    "pyscenic":   {"human": "hs_hgnc_tfs.txt", "mouse": "mm_mgi_tfs.txt"},
}
ALLTFS_NAME = TF_LISTS["scenicplus"]   # backward-compatible alias


def _disp(p):
    """Repo-relative display for a path (never changes the path used on disk)."""
    try:
        rp = os.path.relpath(str(p))
        return rp if not rp.startswith("..") else str(p)
    except Exception:
        return str(p)


def _resolve_repo_path(relpath: str) -> str:
    """Resolve a repo-root-relative prior path deterministically.

    Order: $SCLATCH_DATA_ROOT/relpath; else walk up from the working directory to the repo
    root (marked by pyproject.toml or .git) and use root/relpath; else relpath under the
    working directory. With the priors placed in the canonical repo dirs, the defaults work
    with no flags from anywhere inside the repo.
    """
    root = os.environ.get("SCLATCH_DATA_ROOT")
    if root:
        return os.path.join(root, relpath)
    d = os.path.abspath(os.getcwd())
    for _ in range(8):
        if os.path.exists(os.path.join(d, "pyproject.toml")) or os.path.isdir(os.path.join(d, ".git")):
            return os.path.join(d, relpath)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return relpath


# ── TF list (curated Aerts-lab lists, independent of any ground truth) ────

def get_tf_list(gene_names: list, organism: str = "human", tf_list: str = "scenicplus") -> list:
    """Get the candidate-regulator list from a curated species-specific TF list.

    No ground truth is read. The list is read from priors/ (TF_LISTS[tf_list][organism]),
    the single location for all priors.

    Parameters
    ----------
    gene_names : list
        All gene names in the dataset.
    organism : str
        'human' or 'mouse'.
    tf_list : str
        'scenicplus' (SCENIC+ allTFs, default) or 'pyscenic' (pySCENIC TF lists).

    Returns
    -------
    list
        Sorted list of TF names present in both the TF list and the gene set.
    """
    gene_set = set(gene_names)

    if tf_list not in TF_LISTS:
        raise ValueError(f"tf_list must be one of {sorted(TF_LISTS)}, got {tf_list!r}")
    fname = TF_LISTS[tf_list].get(organism, "")
    tf_file = _resolve_repo_path(os.path.join("priors", fname))
    if not (tf_file and os.path.exists(tf_file)):
        raise FileNotFoundError(
            f"{tf_list} TF list not found for {organism} at priors/{fname}. "
            f"See priors/README.md."
        )

    all_tfs = set(pd.read_csv(tf_file, header=None)[0].tolist())
    tfs = sorted(all_tfs & gene_set)

    _label = {"scenicplus": "SCENIC+", "pyscenic": "pySCENIC"}[tf_list]
    logger.info(f"TF list: {len(tfs)} TFs ({_label} {organism}, {fname}, "
                f"{len(all_tfs)} total, {len(gene_set)} genes in dataset)")

    return tfs


# ── TSS positions from Ensembl REST API ───────────────────────

# Default local TSS snapshots (relative to the working directory). Preferred over the
# live Ensembl REST call so the mask build is offline and reproduces the paper TSS
# snapshot. Download these from the data record (see README "Downloading priors and data").
# Local TSS snapshots, preferred over live Ensembl. priors/tss/ (canonical) is tried first,
# then the legacy data/ locations. Repo-root-relative; resolved by _resolve_repo_path.
DEFAULT_TSS_FILES = {
    "human": ["priors/tss/TSS_extend_1M.txt", "data/TSS_extend_1M.txt"],
    "mouse": ["priors/tss/mm10_gene_tss.csv", "data/chipseq/mm10_gene_tss.csv"],
}


def _load_tss_from_file(tss_file: str, gene_names: list) -> dict:
    """Parse a local TSS table into {gene: (chrom, tss)}.

    Handles both shipped formats:
      - data/chipseq/mm10_gene_tss.csv : gene_name,chr,tss,strand (chr like '11', strand +-1)
      - data/TSS_extend_1M.txt         : chr  1M-  1M+  symbol  TSS  strand (chr like 'chr1')
    Column names are matched case-insensitively; the chrom is normalized to a 'chr' prefix
    to match ATAC peak coordinates.
    """
    with open(tss_file) as fh:
        header = fh.readline()
    sep = "," if header.count(",") >= header.count("\t") else "\t"
    df = pd.read_csv(tss_file, sep=sep)
    cols = {c.lower(): c for c in df.columns}
    gene_col = next((cols[c] for c in ("gene_name", "symbol", "gene") if c in cols), None)
    chr_col = next((cols[c] for c in ("chr", "chrom", "chromosome") if c in cols), None)
    tss_col = next((cols[c] for c in ("tss",) if c in cols), None)
    if not (gene_col and chr_col and tss_col):
        raise ValueError(
            f"TSS file {tss_file} needs gene/chr/tss columns; found {list(df.columns)}")
    gene_set = set(gene_names)
    tss_dict = {}
    for g, ch, ts in zip(df[gene_col].astype(str), df[chr_col].astype(str), df[tss_col]):
        if g not in gene_set:
            continue
        ch = ch if ch.startswith("chr") else f"chr{ch}"
        try:
            tss_dict[g] = (ch, int(ts))
        except (ValueError, TypeError):
            pass
    return tss_dict


def get_tss_positions(gene_names: list, organism: str = "human",
                      tss_file: str = None) -> dict:
    """Get TSS positions, preferring a local file (offline) over Ensembl REST.

    Parameters
    ----------
    gene_names : list
        Gene names to query.
    organism : str
        'human' or 'mouse'.
    tss_file : str, optional
        Path to a local TSS table. If None, a default local snapshot for the organism
        (see DEFAULT_TSS_FILES) is used when present. Live Ensembl REST is used only when
        no local file is available. Using the local snapshot keeps the build offline and
        reproducible.

    Returns
    -------
    dict
        {gene_name: (chrom, tss_position)}
    """
    # 1) Local TSS file (offline, reproducible) — explicit path or organism default.
    candidate = tss_file
    if candidate is None:
        for p in DEFAULT_TSS_FILES.get(organism, []):
            rp = _resolve_repo_path(p)
            if os.path.exists(rp):
                candidate = rp
                break
    if candidate and os.path.exists(candidate):
        logger.info(f"Loading TSS from local file {_disp(candidate)} (offline)...")
        tss_dict = _load_tss_from_file(candidate, gene_names)
        logger.info(f"  TSS mapped from file: {len(tss_dict)}/{len(gene_names)}")
        if tss_dict:
            return tss_dict
        logger.warning("  Local TSS file matched no genes; falling back to Ensembl REST")

    # 2) Live Ensembl REST fallback (needs internet).
    species_map = {"human": "homo_sapiens", "mouse": "mus_musculus"}
    sp = species_map.get(organism, organism)
    tss_dict = {}
    batch_size = 200

    logger.info(f"Querying Ensembl for {len(gene_names)} gene TSS positions...")

    for i in range(0, len(gene_names), batch_size):
        batch = gene_names[i:i + batch_size]
        payload = json.dumps({"symbols": batch}).encode("utf-8")
        url = f"https://rest.ensembl.org/lookup/symbol/{sp}"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json",
                     "Accept": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read().decode("utf-8"))
            for gene, info in result.items():
                if isinstance(info, dict) and "seq_region_name" in info:
                    chrom = "chr" + info["seq_region_name"]
                    strand = info.get("strand", 1)
                    tss = info["start"] if strand == 1 else info["end"]
                    tss_dict[gene] = (chrom, tss)
        except Exception:
            pass

        if (i // batch_size) % 5 == 0 and i > 0:
            logger.info(f"  TSS query: {i}/{len(gene_names)}, mapped: {len(tss_dict)}")

    logger.info(f"  TSS mapped: {len(tss_dict)}/{len(gene_names)}")
    return tss_dict


# ── ATAC peak processing ─────────────────────────────────────

def load_atac_peaks(adata_atac, min_accessibility: float = 0.01) -> dict:
    """Extract accessible peak coordinates from ATAC AnnData.

    Parameters
    ----------
    adata_atac : AnnData
        ATAC peak matrix (cells x peaks).
    min_accessibility : float
        Minimum fraction of cells where peak is accessible.

    Returns
    -------
    dict
        {peak_name: (chrom, start, end)} for accessible peaks.
    """
    atac_mat = adata_atac.X
    if issparse(atac_mat):
        peak_acc = np.array((atac_mat > 0).mean(axis=0)).flatten()
    else:
        peak_acc = (atac_mat > 0).mean(axis=0)

    accessible_mask = peak_acc > min_accessibility
    logger.info(f"  Accessible peaks (>{min_accessibility*100:.0f}% cells): "
                f"{accessible_mask.sum()}/{len(accessible_mask)}")

    peak_coords = {}
    for pi, pname in enumerate(adata_atac.var_names):
        if not accessible_mask[pi]:
            continue
        parts = pname.replace("-", ":").split(":")
        if len(parts) >= 3:
            try:
                peak_coords[pname] = (parts[0], int(parts[1]), int(parts[2]))
            except ValueError:
                pass

    logger.info(f"  Parsed accessible peaks: {len(peak_coords)}")
    return peak_coords


def map_peaks_to_genes(
    gene_names: list, tss_dict: dict, peak_coords: dict, window: int = 10000,
) -> dict:
    """Find peaks within window of each gene's TSS.

    Returns
    -------
    dict
        {gene_name: [peak_ids]}
    """
    # Group peaks by chromosome and sort by midpoint so each gene is a binary-search
    # window lookup (same result as the pairwise scan, but O(genes log peaks)).
    from collections import defaultdict
    by_chrom = defaultdict(list)
    for pid, (pc, ps, pe) in peak_coords.items():
        by_chrom[pc].append(((ps + pe) // 2, pid))
    chrom_arr = {}
    for pc, lst in by_chrom.items():
        lst.sort(key=lambda x: x[0])
        chrom_arr[pc] = (np.array([m for m, _ in lst]), [p for _, p in lst])

    gene_peaks = {}
    for gene in gene_names:
        if gene not in tss_dict:
            continue
        chrom, tss = tss_dict[gene]
        if chrom not in chrom_arr:
            continue
        mids, pids = chrom_arr[chrom]
        lo = int(np.searchsorted(mids, tss - window, side="left"))
        hi = int(np.searchsorted(mids, tss + window, side="right"))
        if hi > lo:
            gene_peaks[gene] = pids[lo:hi]

    logger.info(f"  Genes with nearby peaks: {len(gene_peaks)}/{len(gene_names)}")
    return gene_peaks


# ── ChIP-Atlas query ──────────────────────────────────────────

def _parse_chipatlas_tsv(lines, tf, gene_to_idx, gene_peaks, binding_threshold):
    """Parse ChIP-Atlas TSV for a single TF."""
    if tf not in gene_to_idx:
        return []
    ji = gene_to_idx[tf]
    edges = []
    for line in lines[1:]:
        parts = line.split("\t")
        target = parts[0]
        if target not in gene_to_idx or target == tf:
            continue
        if target not in gene_peaks:
            continue
        scores = []
        for x in parts[1:]:
            if x and x != "NA" and not x.startswith("STRING"):
                try:
                    scores.append(float(x))
                except ValueError:
                    pass
        if scores and np.mean(scores) > binding_threshold:
            edges.append((gene_to_idx[target], ji))
    return edges


def _load_chipatlas_tf(args):
    """Load ChIP-Atlas data for a single TF — local cache first, then (optionally) HTTP."""
    tf, genome, gene_to_idx, gene_peaks, binding_threshold, cache_dir, allow_download = args
    if tf not in gene_to_idx:
        return tf, []

    # Local cache
    local_path = os.path.join(cache_dir, genome, f"{tf}.10.tsv")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        with open(local_path) as f:
            lines = f.read().strip().split("\n")
        return tf, _parse_chipatlas_tsv(lines, tf, gene_to_idx, gene_peaks, binding_threshold)

    # HTTP fallback with caching. Disabled with allow_download=False, which rebuilds strictly
    # from the local cache (e.g. to reproduce a deposited mask from its deposited cache).
    if not allow_download:
        return tf, []
    try:
        url = f"https://chip-atlas.dbcls.jp/data/{genome}/target/{tf}.10.tsv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = resp.read().decode("utf-8").strip()
        lines = data.split("\n")
        # Cache locally
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            f.write(data)
        return tf, _parse_chipatlas_tsv(lines, tf, gene_to_idx, gene_peaks, binding_threshold)
    except Exception:
        return tf, []


def _build_chipatlas_mask(
    tf_list, gene_names, gene_peaks, genome, cache_dir, binding_threshold=50,
    allow_download=True,
):
    """Build mask from ChIP-Atlas binding data with parallel queries."""
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)
    atac_mask = np.zeros((n_genes, n_genes), dtype=np.float32)

    local_dir = os.path.join(cache_dir, genome)
    n_cached = 0
    if os.path.isdir(local_dir):
        n_cached = len([f for f in os.listdir(local_dir) if f.endswith(".tsv")])

    query_args = [
        (tf, genome, gene_to_idx, gene_peaks, binding_threshold, cache_dir, allow_download)
        for tf in tf_list if tf in gene_to_idx
    ]

    logger.info(f"  ChIP-Atlas ({genome}): {len(query_args)} TFs, "
                f"{n_cached} cached locally")

    n_found = 0
    n_done = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_load_chipatlas_tf, args): args[0]
                   for args in query_args}
        for future in as_completed(futures):
            tf, edges = future.result()
            n_done += 1
            if edges:
                n_found += 1
                for gi, ji in edges:
                    atac_mask[gi, ji] = 1.0
            if n_done % 100 == 0:
                logger.info(f"    [{n_done}/{len(query_args)}] "
                            f"TFs with targets: {n_found}")

    logger.info(f"  ChIP-Atlas: {n_found}/{len(query_args)} TFs with binding, "
                f"{int(atac_mask.sum())} edges")
    return atac_mask


# ── Main entry point ──────────────────────────────────────────

def build_atac_mask(
    data_dir: str,
    organism: str = "human",
    out_dir: str = None,
    adata_atac=None,
    window: int = 10000,
    binding_threshold: float = 50,
    skip_atac_peaks: bool = False,
    cache_dir: str = None,
    tss_file: str = None,
    tf_list: str = "scenicplus",
    allow_download: bool = True,
):
    """Build TF-gene ATAC mask using ChIP-Atlas + ATAC accessibility.

    Parameters
    ----------
    data_dir : str
        Directory containing data.csv (for gene list).
    organism : str
        'human' or 'mouse'.
    out_dir : str, optional
        Where to save atac_mask.npz. Defaults to data_dir.
    adata_atac : AnnData, optional
        ATAC peak matrix. If None, tries to load from data_dir/adata_atac.h5ad.
    window : int
        Window (bp) around TSS to search for peaks.
    binding_threshold : float
        Minimum ChIP-Atlas binding score.
    skip_atac_peaks : bool
        Fallback only. If True, skip the ATAC accessibility intersection and use ChIP-Atlas
        binding alone (all genes treated as accessible). Less specific and not used for any
        paper benchmark; prints a loud warning. scLATCH is a multiome method, so prefer own
        scATAC or a reference atlas. Default False.
    cache_dir : str, optional
        ChIP-Atlas binding cache. Default: 'priors/chip_atlas' (canonical), else the legacy
        'data/chipseq/chip_atlas', with layout {cache_dir}/{genome}/{TF}.10.tsv (hg38 or mm10).
        Download this cache into priors/chip_atlas/ (see priors/README.md).
    tss_file : str, optional
        Local TSS table. Default: the organism snapshot in DEFAULT_TSS_FILES when present
        ('priors/tss/TSS_extend_1M.txt' human, 'priors/tss/mm10_gene_tss.csv' mouse), which
        makes the build fully offline. Live Ensembl REST is used only if no local file exists.

    tf_list : str
        Candidate-regulator universe: 'scenicplus' (SCENIC+ allTFs, default) or
        'pyscenic' (pySCENIC TF lists). Use the list a deposited mask was built with to
        reproduce it (see priors/README.md).
    allow_download : bool
        If a TF's ChIP-Atlas file is not in the cache, fetch it from chip-atlas.dbcls.jp
        (default True). Set False to build strictly from the local cache, which makes the
        result independent of network access and of later ChIP-Atlas updates.
    Returns
    -------
    np.ndarray
        (n_genes, n_genes) binary ATAC mask.
    """
    if out_dir is None:
        out_dir = data_dir
    if cache_dir is None:
        # Deterministic default: $SCLATCH_CHIPATLAS_DIR, else the repo's priors/chip_atlas
        # (canonical, if populated), else the legacy data/chipseq/chip_atlas. A directory that
        # exists but holds no {TF}.10.tsv counts as absent so the actionable error still fires.
        genome = GENOME_MAP[organism]
        import glob as _glob
        def _has_tsv(root):
            return bool(_glob.glob(os.path.join(root, genome, "*.tsv")))
        cache_dir = os.environ.get("SCLATCH_CHIPATLAS_DIR")
        if not cache_dir:
            priors_cache = _resolve_repo_path(os.path.join("priors", "chip_atlas"))
            legacy_cache = _resolve_repo_path(os.path.join("data", "chipseq", "chip_atlas"))
            if _has_tsv(priors_cache):
                cache_dir = priors_cache
            elif _has_tsv(legacy_cache):
                cache_dir = legacy_cache
            else:
                cache_dir = priors_cache  # canonical location for the actionable error

    genome = GENOME_MAP[organism]

    # Load gene list (prefer gene_names.csv, fallback to data.csv header)
    gene_names_csv = os.path.join(data_dir, "gene_names.csv")
    data_csv = os.path.join(data_dir, "data.csv")
    if os.path.exists(gene_names_csv):
        gene_names = pd.read_csv(gene_names_csv, header=None)[0].tolist()
    else:
        gene_names = list(pd.read_csv(data_csv, index_col=0, nrows=0).columns)
    n_genes = len(gene_names)
    logger.info(f"Building ATAC mask: {n_genes} genes, {organism} ({genome})")

    # Candidate regulators from the curated TF list (no ground truth)
    tf_names = get_tf_list(gene_names, organism, tf_list=tf_list)

    # Get TSS positions (local snapshot preferred -> offline & reproducible)
    logger.info("[Step 1] Getting TSS positions...")
    tss_dict = get_tss_positions(gene_names, organism, tss_file=tss_file)

    # ATAC peak processing. scLATCH is multiomic: the mask REQUIRES chromatin accessibility.
    if skip_atac_peaks:
        logger.warning(
            "[Step 2] skip_atac_peaks=True: treating ALL genes as accessible. This is a "
            "degraded, NON paper-faithful mask (ChIP-Atlas binding only, no ATAC "
            "intersection); it is less specific. The paper always intersects accessibility.")
        gene_peaks = {g: ["dummy"] for g in gene_names}
    else:
        logger.info("[Step 2] Loading ATAC accessibility (required)...")
        if adata_atac is None:
            atac_path = os.path.join(data_dir, "adata_atac.h5ad")
            if os.path.exists(atac_path):
                adata_atac = ad.read_h5ad(atac_path)
                logger.info(f"  Loaded {atac_path}: {adata_atac.n_obs} x {adata_atac.n_vars}")
            else:
                raise FileNotFoundError(
                    "No scATAC accessibility available for the mask. scLATCH is multiomic: "
                    "the regulatory mask is ChIP-Atlas binding within 10 kb of the TSS "
                    "INTERSECTED with accessible ATAC peaks. Supply accessibility as your own "
                    "scATAC (data/<name>/atac.h5ad, see data/README.md) or a matched-tissue "
                    "REFERENCE scATAC atlas (build_prior(reference_atac=...)). Only pass "
                    "skip_atac=True to accept the degraded, non paper-faithful ChIP-only mask.")

        peak_coords = load_atac_peaks(adata_atac)
        logger.info(f"[Step 3] Mapping peaks to genes (window={window}bp)...")
        gene_peaks = map_peaks_to_genes(gene_names, tss_dict, peak_coords, window)

    # Fail early with an actionable message if the ChIP-Atlas cache is absent or empty.
    import glob as _glob
    genome_cache = os.path.join(cache_dir, genome)
    if not _glob.glob(os.path.join(genome_cache, "*.tsv")):
        raise FileNotFoundError(
            f"ChIP-Atlas cache not found: {genome_cache}\n"
            f"Put the ChIP-Atlas binding cache at priors/chip_atlas/{genome}/ as {{TF}}.10.tsv "
            f"files (see priors/README.md), or set "
            f"SCLATCH_CHIPATLAS_DIR / pass --cache_dir. To skip the ATAC intersection use "
            f"--skip_atac_peaks. Precomputed masks are also provided; build_prior() loads one "
            f"if present.")

    # Build mask from ChIP-Atlas
    logger.info(f"[Step 4] Querying ChIP-Atlas...")
    atac_mask = _build_chipatlas_mask(
        tf_names, gene_names, gene_peaks, genome, cache_dir, binding_threshold,
        allow_download=allow_download,
    )

    # Report statistics
    n_edges = int(atac_mask.sum())
    n_tfs_in_mask = int((atac_mask.sum(axis=0) > 0).sum())
    n_targets_in_mask = int((atac_mask.sum(axis=1) > 0).sum())
    logger.info(f"\nATAC Mask Summary:")
    logger.info(f"  Edges:      {n_edges:,} ({n_edges / (n_genes**2) * 100:.3f}%)")
    logger.info(f"  TFs:        {n_tfs_in_mask}")
    logger.info(f"  Targets:    {n_targets_in_mask}")

    # Check GT coverage if label.csv exists (reporting only, not used for building)
    gt_file = os.path.join(data_dir, "label.csv")
    if os.path.exists(gt_file):
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        gt = pd.read_csv(gt_file)
        gt_in = sum(
            1 for _, r in gt.iterrows()
            if r["Gene1"] in gene_to_idx and r["Gene2"] in gene_to_idx
            and atac_mask[gene_to_idx[r["Gene2"]], gene_to_idx[r["Gene1"]]] > 0
        )
        logger.info(f"  GT coverage: {gt_in}/{len(gt)} ({gt_in/len(gt)*100:.1f}%)")

    # Save as sparse .npz (much smaller for sparse masks)
    os.makedirs(out_dir, exist_ok=True)
    from scipy.sparse import csr_matrix
    mask_sparse = csr_matrix(atac_mask)
    out_path = os.path.join(out_dir, "atac_mask.npz")
    from scipy.sparse import save_npz
    save_npz(out_path, mask_sparse)
    sparse_mb = os.path.getsize(out_path) / 1e6
    dense_mb = atac_mask.nbytes / 1e6
    logger.info(f"  Saved: {_disp(out_path)} ({sparse_mb:.1f} MB, was {dense_mb:.0f} MB dense)")

    return atac_mask
