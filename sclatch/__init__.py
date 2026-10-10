# Quiet third-party noise and expose output control before anything heavy imports.
from sclatch._verbose import (quiet_external_warnings as _qw, set_verbosity,
                              get_verbosity)
_qw()

"""scLATCH: diffusion-based dynamic gene-regulatory-network inference.

scLATCH trains a conditional denoising diffusion model whose denoiser embeds a
learnable gene x gene adjacency A; the converged A is read out directly as the
GRN. A low-rank, pseudotime- and condition-gated term yields a dynamic network
A(tau, k) carrying an architectural bistability prior.

High-level API (see data/README.md and priors/README.md for the folder conventions):
    sclatch.load_dataset(name)                 -> Dataset      (reads data/<name>/)
    sclatch.preprocess(dataset, root_gene=...) -> Dataset      (normalize, embed, pseudotime)
    sclatch.build_prior(dataset, organism=...) -> mask         (ChIP-Atlas ∩ ATAC ∩ TF list)
    sclatch.train(dataset, mask)               -> TrainedModel (exact paper params)
    sclatch.load_model(path, dataset=...)      -> TrainedModel (released checkpoint)

Package structure:
    sclatch.api            — high-level Dataset / TrainedModel API
    sclatch.preprocessing  — preprocessing pipeline internals (incl. ChIP/ATAC mask)
    sclatch.model          — scLATCH diffusion model
    sclatch.training       — single-stage dynamic training (canonical)
    sclatch.evaluation     — AUPRC-ratio, EPR, AUROC metrics
"""

__version__ = "0.1.0"

from sclatch.model import scLATCH, DiffusionSchedule
from sclatch.evaluation import evaluate_grn, evaluate_adjacency_matrix
from sclatch.training import train_dynamic, FINAL_DYNAMIC_CONFIG, TWO_STAGE
from sclatch.api import (
    Dataset,
    TrainedModel,
    load_dataset,
    preprocess,
    build_prior,
    train,
    load_model,
)

__all__ = [
    "set_verbosity", "get_verbosity",
    "scLATCH", "DiffusionSchedule",
    "evaluate_grn", "evaluate_adjacency_matrix",
    "train_dynamic", "FINAL_DYNAMIC_CONFIG", "TWO_STAGE",
    "Dataset", "TrainedModel",
    "load_dataset", "preprocess", "build_prior", "train", "load_model",
]
