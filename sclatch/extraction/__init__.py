"""GRN extraction methods beyond raw adjacency matrix.

Modules:
    jacobian  — Jacobian-based GRN (captures NN + A contributions)
    perturbation — In-silico TF perturbation scoring
"""

from sclatch.extraction.jacobian import extract_jacobian_grn
from sclatch.extraction.perturbation import extract_perturbation_grn

__all__ = [
    "extract_jacobian_grn",
    "extract_perturbation_grn",
]
