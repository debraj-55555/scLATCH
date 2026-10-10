"""Jacobian-based GRN extraction.

For this architecture, the MLP operates per-gene: h[g] depends only on x_t[g].
The only cross-gene mixing is through (I-A) at the output. Therefore:

    J[g,j] = d(noise_pred_g) / d(x_t_j)
           = final'(·) × (I-A)[g,j] × NN_j'(x_t[j])

The Jacobian is essentially A scaled by the NN's per-gene derivative.
This scaling can up-weight TFs that are "active" at observed expression levels.
"""

import numpy as np
import torch


def extract_jacobian_grn(model, x_batch, t_values, continuous=None,
                          categorical=None, tf_indices=None, device='cpu'):
    """Extract GRN via Jacobian d(noise_pred)/d(x_t), averaged over samples and timesteps.

    Args:
        model: trained scLATCH model (in eval mode)
        x_batch: (n_cells, n_genes) expression tensor (clean x_0)
        t_values: list of diffusion timesteps to average over
        continuous: dict {key: (n_cells,) tensor} or None
        categorical: dict {key: (n_cells,) tensor} or None
        tf_indices: list of TF column indices to keep (None = all)
        device: torch device

    Returns:
        J: (n_genes, n_tfs) or (n_genes, n_genes) numpy array
           J[g,j] = mean influence of gene j on gene g's noise prediction
    """
    model.eval()
    model = model.to(device)
    n_genes = x_batch.shape[1]

    # Diffusion schedule for adding noise
    from sclatch.model.diffusion import DiffusionSchedule
    schedule = DiffusionSchedule(T=5000, device=device)

    all_J = []

    for t_val in t_values:
        # Create noisy input
        t = torch.full((x_batch.shape[0],), t_val, device=device, dtype=torch.long)
        x_0 = x_batch.to(device)
        x_t, noise = schedule.q_sample(x_0, t)

        x_t = x_t.detach().requires_grad_(True)

        # Prepare conditions
        cont_dev = {k: v.to(device) for k, v in continuous.items()} if continuous else None
        cat_dev = {k: v.to(device) for k, v in categorical.items()} if categorical else None

        # Forward pass
        with torch.enable_grad():
            noise_pred = model(x_t, t, continuous=cont_dev, categorical=cat_dev)

        # Extract Jacobian row-by-row: for each output gene g, compute gradient
        J_t = torch.zeros(n_genes, n_genes, dtype=torch.float32)

        for g in range(n_genes):
            if x_t.grad is not None:
                x_t.grad.zero_()
            # Backprop from gene g's noise prediction (summed over batch)
            noise_pred[:, g].sum().backward(retain_graph=(g < n_genes - 1))
            # Gradient: d(sum_b noise_pred[b,g]) / d(x_t[b,j]) → summed over batch
            # Average over batch to get mean sensitivity
            J_t[g, :] = x_t.grad.mean(dim=0).cpu()

        all_J.append(J_t.numpy())

        # Print progress
        print(f"  Jacobian at t={t_val}: ||J||_F = {np.linalg.norm(J_t.numpy()):.4f}")

    # Average over timesteps
    J_avg = np.mean(all_J, axis=0)

    # Keep only TF columns if specified
    if tf_indices is not None:
        J_avg = J_avg[:, tf_indices]

    return np.abs(J_avg)  # absolute value for GRN scoring


def extract_jacobian_grn_efficient(model, x_batch, t_values, continuous=None,
                                    categorical=None, tf_indices=None,
                                    device='cpu', batch_genes=500):
    """Memory-efficient Jacobian extraction — process genes in batches.

    For 21960 genes, computing the full Jacobian row-by-row requires retaining
    the computation graph for n_genes backward passes. This version processes
    output genes in batches, rebuilding the graph for each batch.

    Args:
        ... (same as extract_jacobian_grn)
        batch_genes: number of output genes per graph rebuild

    Returns:
        J: (n_genes, n_tfs) numpy array
    """
    model.eval()
    model = model.to(device)
    n_genes = x_batch.shape[1]
    n_tfs = len(tf_indices) if tf_indices is not None else n_genes

    from sclatch.model.diffusion import DiffusionSchedule
    schedule = DiffusionSchedule(T=5000, device=device)

    J_sum = np.zeros((n_genes, n_tfs), dtype=np.float32)
    n_t = len(t_values)

    for t_val in t_values:
        t = torch.full((x_batch.shape[0],), t_val, device=device, dtype=torch.long)
        x_0 = x_batch.to(device)
        x_t_base, _ = schedule.q_sample(x_0, t)

        # Process output genes in batches
        for g_start in range(0, n_genes, batch_genes):
            g_end = min(g_start + batch_genes, n_genes)

            x_t = x_t_base.clone().detach().requires_grad_(True)
            cont_dev = {k: v.to(device) for k, v in continuous.items()} if continuous else None
            cat_dev = {k: v.to(device) for k, v in categorical.items()} if categorical else None

            with torch.enable_grad():
                noise_pred = model(x_t, t, continuous=cont_dev, categorical=cat_dev)

            for g in range(g_start, g_end):
                if x_t.grad is not None:
                    x_t.grad.zero_()
                retain = (g < g_end - 1)
                noise_pred[:, g].sum().backward(retain_graph=retain)
                grad = x_t.grad.mean(dim=0).cpu()
                if tf_indices is not None:
                    J_sum[g, :] += grad[tf_indices].numpy()
                else:
                    J_sum[g, :] += grad.numpy()

            # Free graph
            del noise_pred, x_t
            if device != 'cpu':
                torch.cuda.empty_cache()

        print(f"  Jacobian at t={t_val}: done")

    J_avg = J_sum / n_t
    return np.abs(J_avg)
