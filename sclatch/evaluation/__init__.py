"""scLATCH evaluation metrics.

Functions:
    evaluate_grn — evaluate predicted edges against ground truth.
    evaluate_adjacency_matrix — evaluate A matrix directly.
    compute_epr_deepsem_style — DeepSEM-compatible EPR.
    evaluate_run — evaluate a training run folder against all GTs.
"""

from sclatch.evaluation.metrics import (
    evaluate_grn,
    evaluate_adjacency_matrix,
    compute_epr_deepsem_style,
)
from sclatch.evaluation.evaluate_pbmc import evaluate_run

__all__ = [
    "evaluate_grn",
    "evaluate_adjacency_matrix",
    "compute_epr_deepsem_style",
    "evaluate_run",
]
