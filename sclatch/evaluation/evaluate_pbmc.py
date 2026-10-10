"""Standalone PBMC GRN evaluation.

Loads a trained model's output from its results folder and evaluates
against one or more ground truth networks. Stores results back to the
same folder.

Completely decoupled from training — no model weights loaded, just the
adjacency matrix. Can be run independently at any time.

Usage:
    # As module
    from sclatch.evaluation.evaluate_pbmc import evaluate_run
    evaluate_run('results/my_run/', 'processed/pbmc_10k/')

    # As script
    python -m sclatch.evaluation.evaluate_pbmc \
        --run_dir results/my_run/ \
        --data_dir processed/pbmc_10k/
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


def load_adjacency(run_dir):
    """Load adjacency matrix from a training run folder.

    Supports:
        adjacency.npz — sparse (new format, pre-normalized)
        adjacency.npy — dense (old format, may or may not be normalized)

    All formats are loaded as pre-normalized (compatible with evaluation).

    Returns
    -------
    A : np.ndarray (n_genes, n_genes)
    """
    from scipy.sparse import load_npz

    npz_path = os.path.join(run_dir, 'adjacency.npz')
    npy_path = os.path.join(run_dir, 'adjacency.npy')

    if os.path.exists(npz_path):
        A = load_npz(npz_path).toarray()
        logger.info(f"  Loaded A: {A.shape} from adjacency.npz (sparse, pre-normalized)")
        return A
    elif os.path.exists(npy_path):
        A = np.load(npy_path)
        # Check if this is old format (pre-normalized) or raw
        has_old_result = os.path.exists(os.path.join(run_dir, 'result.json'))
        has_new_config = os.path.exists(os.path.join(run_dir, 'training_config.json'))
        if has_new_config and not has_old_result:
            # New raw format — need to normalize
            n = A.shape[0]
            grn = 1.0 / (n - 1)
            A = A / grn
            logger.info(f"  Loaded A: {A.shape} from adjacency.npy (raw, normalized)")
        else:
            logger.info(f"  Loaded A: {A.shape} from adjacency.npy (pre-normalized)")
        return A
    else:
        raise FileNotFoundError(f"No adjacency.npz or adjacency.npy in {run_dir}")


def load_gene_names(data_dir):
    """Load gene names from data directory."""
    gn_path = os.path.join(data_dir, 'gene_names.csv')
    if os.path.exists(gn_path):
        return pd.read_csv(gn_path, header=None)[0].tolist()
    # Fallback: from data.csv header
    return pd.read_csv(os.path.join(data_dir, 'data.csv'),
                        nrows=0).columns[1:].tolist()


def load_ground_truths(data_dir):
    """Load all available ground truth files from data directory.

    Returns
    -------
    gts : dict {name: DataFrame with Gene1, Gene2 columns}
    """
    gts = {}
    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith('label') and fname.endswith('.csv'):
            gt = pd.read_csv(os.path.join(data_dir, fname))
            if 'Gene1' in gt.columns and 'Gene2' in gt.columns:
                name = fname.replace('.csv', '').replace('label_', '').replace('label', 'collectri')
                gts[name] = gt
                logger.info(f"  GT '{name}': {len(gt)} edges, "
                            f"{gt['Gene1'].nunique()} TFs")
    return gts


def evaluate_adjacency(A, gene_names, gt_df, normalize=False):
    """Evaluate adjacency matrix against a ground truth.

    Args:
        A: (n_genes, n_genes) adjacency matrix. A[i,j] = edge j→i.
           Expected to be pre-normalized (from load_adjacency or save_results).
        gene_names: list of gene names matching A rows/columns.
        gt_df: DataFrame with Gene1 (TF), Gene2 (target) columns.
        normalize: if True, divide A by gene_reg_norm (only for raw A).

    Returns:
        dict with AUPRC_ratio, EPR, AUROC, n_gt_edges, n_gt_tfs, n_eval_pairs.
    """
    n_genes = len(gene_names)
    g2i = {g: i for i, g in enumerate(gene_names)}

    if normalize:
        gene_reg_norm = 1.0 / (n_genes - 1)
        A = A / gene_reg_norm

    A_abs = np.abs(A.copy())
    np.fill_diagonal(A_abs, 0)

    # Filter GT to genes in our dataset
    gt_tfs = sorted(set(gt_df['Gene1']) & set(gene_names))
    gt_set = set(zip(gt_df['Gene1'], gt_df['Gene2']))

    if not gt_tfs:
        return {'AUPRC_ratio': 0, 'EPR': 0, 'AUROC': 0.5,
                'n_gt_edges': 0, 'n_gt_tfs': 0, 'n_eval_pairs': 0}

    # Vectorized evaluation
    tf_indices = np.array([g2i[tf] for tf in gt_tfs])
    n_tfs = len(tf_indices)
    tf_idx_to_pos = {int(j): k for k, j in enumerate(tf_indices)}

    A_tf = A_abs[:, tf_indices]  # (n_genes, n_tfs)

    label_mat = np.zeros((n_genes, n_tfs), dtype=np.int8)
    for tf, target in gt_set:
        if tf in g2i and target in g2i:
            tf_gene_idx = g2i[tf]
            if tf_gene_idx in tf_idx_to_pos:
                label_mat[g2i[target], tf_idx_to_pos[tf_gene_idx]] = 1

    # Mask self-edges
    mask = np.ones((n_genes, n_tfs), dtype=bool)
    for k, j in enumerate(tf_indices):
        mask[j, k] = False

    scores = A_tf[mask]
    labels = label_mat[mask]
    n_true = int(labels.sum())

    if n_true == 0 or n_true >= len(labels):
        return {'AUPRC_ratio': 0, 'EPR': 0, 'AUROC': 0.5,
                'n_gt_edges': n_true, 'n_gt_tfs': len(gt_tfs),
                'n_eval_pairs': len(scores)}

    pos_rate = labels.mean()
    auprc = average_precision_score(labels, scores)
    auprc_ratio = auprc / pos_rate

    top_k_idx = np.argpartition(-scores, n_true)[:n_true]
    epr = (labels[top_k_idx].sum() / n_true) / pos_rate

    auroc = roc_auc_score(labels, scores)

    return {
        'AUPRC_ratio': round(float(auprc_ratio), 4),
        'EPR': round(float(epr), 4),
        'AUROC': round(float(auroc), 4),
        'n_gt_edges': n_true,
        'n_gt_tfs': len(gt_tfs),
        'n_eval_pairs': len(scores),
    }


def evaluate_run(run_dir, data_dir, normalize=True):
    """Evaluate a training run against all available GTs.

    Loads adjacency.npy from run_dir, evaluates against all label*.csv
    in data_dir, saves results to run_dir/evaluation_results.json.

    Args:
        run_dir: path to training output folder (contains adjacency.npy)
        data_dir: path to processed data (contains gene_names.csv, label*.csv)
        normalize: whether to divide A by gene_reg_norm

    Returns:
        results: dict {gt_name: {metrics}}
    """
    logger.info(f"Evaluating run: {run_dir}")
    logger.info(f"Data dir: {data_dir}")

    A = load_adjacency(run_dir)  # always returns pre-normalized A
    normalize = False  # load_adjacency handles normalization
    gene_names = load_gene_names(data_dir)
    gts = load_ground_truths(data_dir)

    if not gts:
        logger.warning("No ground truth files found!")
        return {}

    results = {}
    print(f"\n{'GT Name':<30s} {'AUPRC':>8s} {'EPR':>8s} {'AUROC':>8s} "
          f"{'Edges':>8s} {'TFs':>5s}")
    print("-" * 73)

    for gt_name, gt_df in gts.items():
        m = evaluate_adjacency(A, gene_names, gt_df, normalize=normalize)
        results[gt_name] = m
        print(f"{gt_name:<30s} {m['AUPRC_ratio']:>8.3f} {m['EPR']:>8.3f} "
              f"{m['AUROC']:>8.4f} {m['n_gt_edges']:>8d} {m['n_gt_tfs']:>5d}")

    # Save results
    out_path = os.path.join(run_dir, 'evaluation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved to {out_path}")

    return results


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(
        description='Evaluate scLATCH run against ground truth(s)')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to training output folder (has adjacency.npy)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed data (has gene_names.csv, label*.csv)')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip dividing A by gene_reg_norm')
    args = parser.parse_args()

    print("=" * 60)
    print("scLATCH Evaluation")
    print("=" * 60)

    results = evaluate_run(args.run_dir, args.data_dir,
                            normalize=not args.no_normalize)


if __name__ == '__main__':
    main()
