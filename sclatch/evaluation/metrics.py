"""
Unified GRN evaluation metrics.
Used for ALL methods (scLATCH and baselines) to ensure fair comparison.

Metrics:
  - AUPRC Ratio (primary)
  - Early Precision Ratio (EPR)
  - AUROC (supplementary)
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_grn(predicted_edges_df, ground_truth_df, tf_list, gene_list):
    """
    Evaluate predicted GRN against ground truth.

    Args:
        predicted_edges_df: DataFrame with columns ['TF', 'target', 'weight']
        ground_truth_df: DataFrame with columns ['TF', 'target'] (binary edges)
        tf_list: list of TF names (regulators)
        gene_list: list of all gene names (targets)

    Returns:
        dict with AUPRC, AUPRC_ratio, EPR, AUROC
    """
    # Build ground truth set
    gt_set = set(zip(ground_truth_df['TF'], ground_truth_df['target']))

    # Build weight lookup from predictions
    weight_dict = {}
    for _, row in predicted_edges_df.iterrows():
        weight_dict[(row['TF'], row['target'])] = abs(row['weight'])

    # All possible TF->gene pairs (excluding self-regulation)
    all_pairs = [(tf, g) for tf in tf_list for g in gene_list if tf != g]

    y_true = np.array([1 if (tf, g) in gt_set else 0 for (tf, g) in all_pairs])
    y_score = np.array([weight_dict.get((tf, g), 0.0) for (tf, g) in all_pairs])

    n_positive = y_true.sum()
    if n_positive == 0:
        raise ValueError("No ground truth edges found in the provided gene/TF lists")

    edge_density = n_positive / len(y_true)

    # AUPRC
    auprc = average_precision_score(y_true, y_score)
    auprc_ratio = auprc / edge_density

    # EPR: top-k where k = number of GT edges
    k = n_positive
    top_k_idx = np.argsort(y_score)[::-1][:k]
    ep = y_true[top_k_idx].sum() / k
    epr = ep / edge_density

    # AUROC
    auroc = roc_auc_score(y_true, y_score)

    return {
        'AUPRC': auprc,
        'AUPRC_ratio': auprc_ratio,
        'EPR': epr,
        'AUROC': auroc,
        'n_gt_edges': int(n_positive),
        'n_total_pairs': len(all_pairs),
        'edge_density': edge_density,
    }


def evaluate_adjacency_matrix(A, gene_names, ground_truth_df, tf_list=None,
                               evaluate_mask=None):
    """
    Evaluate an adjacency matrix A (n_genes x n_genes) against ground truth.
    Convenience wrapper around evaluate_grn.

    Args:
        A: numpy array (n_genes x n_genes), A[i,j] = weight of edge gene_j -> gene_i
           (DeepSEM convention: rows=targets, columns=regulators)
        gene_names: list of gene names matching A's rows/columns
        ground_truth_df: DataFrame with columns ['TF', 'target']
        tf_list: list of TF names; if None, inferred from ground truth
        evaluate_mask: optional binary mask (same shape as A) restricting evaluation

    Returns:
        dict with metrics
    """
    import pandas as pd

    gene_names = list(gene_names)
    if tf_list is None:
        tf_list = sorted(set(ground_truth_df['TF']) & set(gene_names))

    A_eval = np.abs(A.copy())
    np.fill_diagonal(A_eval, 0)

    if evaluate_mask is not None:
        A_eval = A_eval * evaluate_mask

    # Extract edges: A[i,j] = edge from gene_j (TF) -> gene_i (target)
    rows = []
    for j, tf in enumerate(gene_names):
        if tf not in tf_list:
            continue
        for i, target in enumerate(gene_names):
            if i == j:
                continue
            w = A_eval[i, j]
            if w > 0 or True:  # include all pairs for proper AUPRC
                rows.append({'TF': tf, 'target': target, 'weight': w})

    predicted_df = pd.DataFrame(rows)
    return evaluate_grn(predicted_df, ground_truth_df, tf_list, gene_names)


def compute_epr_deepsem_style(A, truth_edges, evaluate_mask):
    """
    EPR computation matching DeepSEM's exact implementation for comparison.

    Args:
        A: adjacency matrix (n_genes x n_genes)
        truth_edges: set of (sender_idx, receiver_idx) tuples
        evaluate_mask: binary mask array

    Returns:
        (Ep, Epr) - edge precision count and early precision ratio
    """
    num_nodes = A.shape[0]
    num_truth_edges = len(truth_edges)
    A = np.abs(A)
    if evaluate_mask is None:
        evaluate_mask = np.ones_like(A) - np.eye(len(A))
    A = A * evaluate_mask
    A_val = list(np.sort(np.abs(A.reshape(-1, 1)), 0)[:, 0])
    A_val.reverse()
    cutoff_all = A_val[num_truth_edges]
    A_indicator_all = np.zeros([num_nodes, num_nodes])
    A_indicator_all[np.abs(A) > cutoff_all] = 1
    idx_rec, idx_send = np.where(A_indicator_all)
    A_edges = set(zip(idx_send, idx_rec))
    overlap_A = A_edges.intersection(truth_edges)
    Ep = len(overlap_A)
    Epr = 1.0 * Ep / ((num_truth_edges ** 2) / np.sum(evaluate_mask))
    return Ep, Epr
