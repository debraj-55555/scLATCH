"""In-silico perturbation-based GRN extraction.

For each TF, perturb its expression to the population mean and measure the
effect on all genes' noise predictions. This captures causal direction
(TF → target) through the full model.

For this architecture (per-gene MLP + (I-A) mixing):
    effect[g,h] ≈ (I-A)[g,h] × NN_h'(x_h) × Δx_h

Similar to Jacobian but uses finite differences — more robust to numerical issues.
"""

import numpy as np
import torch


def extract_perturbation_grn(model, x_batch, t_values, continuous=None,
                              categorical=None, tf_indices=None, device='cpu'):
    """Extract GRN via in-silico TF perturbation (mean-value knockout).

    For each TF, set its expression to the population mean and measure
    the change in noise prediction for all genes.

    Args:
        model: trained scLATCH model (in eval mode)
        x_batch: (n_cells, n_genes) expression tensor (clean x_0)
        t_values: list of diffusion timesteps to average over
        continuous: dict {key: (n_cells,) tensor} or None
        categorical: dict {key: (n_cells,) tensor} or None
        tf_indices: list of TF column indices to perturb
        device: torch device

    Returns:
        P: (n_genes, n_tfs) numpy array
           P[g,j] = mean absolute effect of perturbing TF j on gene g
    """
    model.eval()
    model = model.to(device)
    n_genes = x_batch.shape[1]

    if tf_indices is None:
        tf_indices = list(range(n_genes))
    n_tfs = len(tf_indices)

    from sclatch.model.diffusion import DiffusionSchedule
    schedule = DiffusionSchedule(T=5000, device=device)

    # Compute per-gene means for perturbation value
    gene_means = x_batch.mean(dim=0).to(device)

    P_sum = np.zeros((n_genes, n_tfs), dtype=np.float32)

    for t_val in t_values:
        t = torch.full((x_batch.shape[0],), t_val, device=device, dtype=torch.long)
        x_0 = x_batch.to(device)
        x_t, noise = schedule.q_sample(x_0, t)

        cont_dev = {k: v.to(device) for k, v in continuous.items()} if continuous else None
        cat_dev = {k: v.to(device) for k, v in categorical.items()} if categorical else None

        # Baseline prediction
        with torch.no_grad():
            pred_baseline = model(x_t, t, continuous=cont_dev, categorical=cat_dev)

        # Perturbation for each TF
        for j_idx, tf_col in enumerate(tf_indices):
            x_t_pert = x_t.clone()
            x_t_pert[:, tf_col] = gene_means[tf_col]  # set to population mean

            with torch.no_grad():
                pred_pert = model(x_t_pert, t, continuous=cont_dev, categorical=cat_dev)

            # Effect: absolute change in prediction per gene, averaged over batch
            effect = (pred_baseline - pred_pert).abs().mean(dim=0).cpu().numpy()
            P_sum[:, j_idx] += effect

        print(f"  Perturbation at t={t_val}: done ({n_tfs} TFs)")

    P_avg = P_sum / len(t_values)
    return P_avg
