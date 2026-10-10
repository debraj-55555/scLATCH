"""CLI entry points for scLATCH preprocessing, ground truth, and mask building."""

import argparse
import logging


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    """Main preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description="scLATCH: preprocess raw scRNA/multiome data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data_dir", required=True,
                        help="Path to raw data directory")
    parser.add_argument("--organism", required=True, choices=["human", "mouse"])
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: processed/<dataset_name>)")

    # QC
    parser.add_argument("--min_genes", type=int, default=200)
    parser.add_argument("--max_genes", type=int, default=6000)
    parser.add_argument("--max_pct_mt", type=float, default=20.0)
    parser.add_argument("--min_cells", type=int, default=10)

    # Embedding
    parser.add_argument("--n_pcs", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--resolution", type=float, default=0.8)
    parser.add_argument("--n_hvg_for_pca", type=int, default=3000,
                        help="HVGs for PCA only (all genes kept in output). 0=use all.")

    # Pseudotime
    parser.add_argument("--no_pseudotime", action="store_true",
                        help="Skip pseudotime (for non-trajectory data like cell line mixtures)")
    parser.add_argument("--root_gene", default=None)
    parser.add_argument("--root_cell_type", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    from sclatch.preprocessing.pipeline import preprocess
    preprocess(
        data_dir=args.data_dir, organism=args.organism, out_dir=args.out_dir,
        min_genes=args.min_genes, max_genes=args.max_genes,
        max_pct_mt=args.max_pct_mt, min_cells=args.min_cells,
        n_pcs=args.n_pcs, n_neighbors=args.n_neighbors,
        resolution=args.resolution, n_hvg_for_pca=args.n_hvg_for_pca,
        compute_pseudotime_flag=not args.no_pseudotime,
        root_gene=args.root_gene, root_cell_type=args.root_cell_type,
    )


def main_ground_truth():
    """Ground truth download CLI."""
    parser = argparse.ArgumentParser(
        description="scLATCH: download CollecTRI ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", required=True,
                        help="Processed data directory (must contain data.csv)")
    parser.add_argument("--organism", required=True, choices=["human", "mouse"])
    parser.add_argument("--gt_file", default=None,
                        help="Path to pre-existing GT file (skip download)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    import pandas as pd
    from sclatch.preprocessing.ground_truth import build_ground_truth

    data = pd.read_csv(f"{args.data_dir}/data.csv", index_col=0, nrows=0)
    gene_names = list(data.columns)

    build_ground_truth(
        gene_names=gene_names, organism=args.organism,
        out_dir=args.data_dir, gt_file=args.gt_file,
    )


def main_atac_mask():
    """ATAC mask building CLI."""
    parser = argparse.ArgumentParser(
        description="scLATCH: build ChIP-Atlas ATAC mask (SCENIC+ TF list, no GT leakage)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", required=True,
                        help="Processed data directory (must contain data.csv)")
    parser.add_argument("--organism", required=True, choices=["human", "mouse"])
    parser.add_argument("--out_dir", default=None,
                        help="Where to write atac_mask.npz (default: data_dir)")
    parser.add_argument("--window", type=int, default=10000,
                        help="Window (bp) around TSS for peak search")
    parser.add_argument("--binding_threshold", type=float, default=50,
                        help="ChIP-Atlas binding score threshold")
    parser.add_argument("--skip_atac_peaks", action="store_true",
                        help="Skip ATAC intersection (ChIP-Atlas binding for all genes)")
    parser.add_argument("--cache_dir", default=None,
                        help="ChIP-Atlas binding cache dir (default: priors/chip_atlas). "
                             "Layout {cache_dir}/{genome}/{TF}.10.tsv (genome hg38 or mm10).")
    parser.add_argument("--tss_file", default=None,
                        help="Local TSS table for offline builds (default: the organism "
                             "snapshot priors/tss/TSS_extend_1M.txt (human) or "
                             "priors/tss/mm10_gene_tss.csv (mouse) when present; otherwise "
                             "live Ensembl REST).")
    parser.add_argument("--tf_list", choices=["scenicplus", "pyscenic"], default="scenicplus",
                        help="Candidate-regulator universe: SCENIC+ allTFs (default) or pySCENIC TF "
                             "lists. Use the list a deposited mask was built with to reproduce it.")
    parser.add_argument("--no_download", action="store_true",
                        help="Build strictly from the local ChIP-Atlas cache; do not fetch missing "
                             "TFs from chip-atlas.dbcls.jp.")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    from sclatch.preprocessing.atac_mask import build_atac_mask

    build_atac_mask(
        data_dir=args.data_dir, organism=args.organism, out_dir=args.out_dir,
        window=args.window, binding_threshold=args.binding_threshold,
        skip_atac_peaks=args.skip_atac_peaks, cache_dir=args.cache_dir,
        tss_file=args.tss_file, tf_list=args.tf_list,
        allow_download=not args.no_download,
    )


if __name__ == "__main__":
    main()
