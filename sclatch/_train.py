#!/usr/bin/env python
"""
Unified scLATCH training pipeline.

Data-agnostic: works with any dataset that follows the directory convention:

  data_dir/
  ├── data.csv              # (cells × genes) expression matrix, index_col=0
  ├── label.csv             # Ground truth: columns Gene1, Gene2
  ├── pseudotime.npy        # Optional: (n_cells,) float array
  ├── cell_metadata.csv     # Optional: index_col=0, columns used as categorical
  └── atac_mask.npy         # Optional: (n_genes, n_genes) binary mask

Examples:
  # RNA-only (BEELINE-style)
  sclatch-train --data_dir data/processed/mHSC-E_top2500

  # RNA + pseudotime
  sclatch-train --data_dir data/processed/huang_tcell --continuous pseudotime

  # Full model: RNA + pseudotime + condition + ATAC
  sclatch-train --data_dir data/processed/huang_tcell \\
      --continuous pseudotime --categorical condition --atac_mask

  # Dynamic A(τ) with temporal contrastive loss
  sclatch-train --data_dir data/processed/huang_tcell \\
      --continuous pseudotime --categorical condition --atac_mask \\
      --dynamic_rank 32 --lambda_temporal 0.05

  # Multi-dataset sweep
  sclatch-train --data_dir data/processed/mHSC-E_top2500 data/processed/mHSC-L_top2500

  # 5-seed ensemble
  sclatch-train --data_dir data/processed/huang_tcell --seeds 42 123 456 789 1024
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
import torch
from sclatch._verbose import log as _log, milestone as _ms, progress as _progress
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score

from sclatch.model import scLATCH


def _disp(p):
    """Repo-relative display for a path (never changes the path used on disk)."""
    try:
        rp = os.path.relpath(str(p))
        return rp if not rp.startswith("..") else str(p)
    except Exception:
        return str(p)


# ---------------------------------------------------------------------------
# AMP helpers (version-gated: modern torch.amp on new torch, torch.cuda.amp on old)
# ---------------------------------------------------------------------------
# The numerics are identical either way; this only picks the non-deprecated API so
# there is no FutureWarning on torch >= 2.4 while still running on older torch.

def _amp_autocast(enabled):
    """Return a CUDA autocast context. Uses torch.amp on new torch, torch.cuda.amp on old.

    When disabled (CPU path), returns a no-op context so behavior is byte-identical
    to the previous `torch.cuda.amp.autocast(enabled=False)`.
    """
    if not enabled:
        import contextlib
        return contextlib.nullcontext()
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "autocast"):
        try:
            return amp.autocast("cuda")            # torch >= 1.10 signature, no deprecation
        except TypeError:
            pass
    return torch.cuda.amp.autocast(enabled=True)   # legacy fallback


def _amp_grad_scaler():
    """Return a CUDA GradScaler using the non-deprecated API when available."""
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        try:
            return amp.GradScaler("cuda")          # torch >= 2.3
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler()             # legacy fallback


# ---------------------------------------------------------------------------
# Data loading (auto-detects available signals)
# ---------------------------------------------------------------------------

def normalize_expression(X):
    """Per-cell min-max → per-gene z-score (RegDiffusion convention)."""
    cell_min = X.min(axis=1, keepdims=True)
    cell_max = X.max(axis=1, keepdims=True)
    cell_range = cell_max - cell_min
    cell_range[cell_range == 0] = 1
    X = (X - cell_min) / cell_range
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X


def _auto_compute_pseudotime(data_dir, root_gene=None, root_cell_type=None):
    """Auto-compute DPT pseudotime from adata_rna.h5ad if pseudotime.npy missing.

    Uses sclatch.preprocessing.pseudotime with automatic direction correction:
    - If root_gene provided: uses highest-expressing cell + validates direction
    - If not: uses CytoTRACE gene-count heuristic + validates direction
    Direction validation inverts pseudotime if root_gene increases along τ.

    Returns path to saved pseudotime.npy, or None on failure.
    """
    import scanpy as sc

    adata_path = os.path.join(data_dir, 'adata_rna.h5ad')
    if not os.path.exists(adata_path):
        _log(f"  Cannot auto-compute pseudotime: {adata_path} not found")
        return None

    _log(f"  Auto-computing pseudotime from {adata_path} ...")
    adata = sc.read_h5ad(adata_path)

    # Compute neighbors if not already computed
    if 'neighbors' not in adata.uns:
        _log("  Computing neighbors (PCA → kNN)...")
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_pcs=50, n_neighbors=30)

    try:
        from sclatch.preprocessing.pseudotime import compute_pseudotime
        adata = compute_pseudotime(adata, root_gene=root_gene,
                                   root_cell_type=root_cell_type)
        tau = adata.obs['pseudotime'].values.astype(np.float32)
        tau_path = os.path.join(data_dir, 'pseudotime.npy')
        np.save(tau_path, tau)
        _log(f"  Pseudotime computed and saved to {tau_path}")
        _log(f"  Range: [{tau.min():.3f}, {tau.max():.3f}], "
              f"median={np.median(tau):.3f}")
        return tau_path
    except Exception as e:
        _log(f"  Auto-pseudotime failed: {e}")
        return None


def load_dataset(data_dir, continuous_keys=None, categorical_keys=None,
                 use_atac_mask=False, mask_name=None, prior_matrix=None,
                 init_prior=None, root_gene=None,
                 root_cell_type=None):
    """
    Load any dataset following the directory convention.

    Returns dict with:
      X: (n_cells, n_genes) normalized expression
      gene_names: list of gene names
      gt: DataFrame with Gene1, Gene2 columns
      signals: dict of {name: numpy array} for continuous/categorical
      atac_mask: (n_genes, n_genes) or None
      meta: dict with categorical column mappings
    """
    continuous_keys = continuous_keys or []
    categorical_keys = categorical_keys or []

    # Expression — prefer data.npy (fast) over data.csv (slow)
    npy_path = os.path.join(data_dir, 'data.npy')
    csv_path = os.path.join(data_dir, 'data.csv')
    gene_names_path = os.path.join(data_dir, 'gene_names.csv')
    if os.path.exists(npy_path) and os.path.exists(gene_names_path):
        X = np.load(npy_path).astype(np.float32)
        gene_names = pd.read_csv(gene_names_path, header=None)[0].tolist()
    else:
        data = pd.read_csv(csv_path, index_col=0)
        gene_names = list(data.columns)
        X = data.values.astype(np.float32)
    X = normalize_expression(X)
    n_cells, n_genes = X.shape

    _ms(f"  Expression: {n_cells} cells × {n_genes} genes")

    # Ground truth — NOT used during training, only for post-training evaluation.
    # Loaded here only for dataset metadata (TF count). Evaluation is separate.
    gt = pd.DataFrame(columns=['Gene1', 'Gene2'])
    for f in sorted(os.listdir(data_dir)):
        if f.startswith('label') and f.endswith('.csv'):
            gt = pd.read_csv(os.path.join(data_dir, f))
            _log(f"  GT (for eval only): {f} — {len(gt)} edges, "
                  f"{gt['Gene1'].nunique()} TFs")
            break
    TFs = sorted(set(gt['Gene1']) & set(gene_names)) if len(gt) > 0 else []

    # Continuous signals
    signals = {}
    if 'pseudotime' in continuous_keys:
        tau_path = os.path.join(data_dir, 'pseudotime.npy')
        if not os.path.exists(tau_path):
            # Auto-compute from adata_rna.h5ad if available
            _log(f"  pseudotime.npy not found — attempting auto-computation...")
            tau_path = _auto_compute_pseudotime(
                data_dir, root_gene=root_gene, root_cell_type=root_cell_type)
        if tau_path and os.path.exists(tau_path):
            signals['pseudotime'] = np.load(tau_path).astype(np.float32)
            _log(f"  Pseudotime: [{signals['pseudotime'].min():.3f}, "
                  f"{signals['pseudotime'].max():.3f}]")
        else:
            _log(f"  WARNING: pseudotime unavailable — skipping --continuous pseudotime")
            continuous_keys = [k for k in continuous_keys if k != 'pseudotime']

    # Generic continuous: look for {name}.npy
    for key in continuous_keys:
        if key == 'pseudotime':
            continue
        path = os.path.join(data_dir, f'{key}.npy')
        if os.path.exists(path):
            signals[key] = np.load(path).astype(np.float32)
            _log(f"  Continuous '{key}': loaded from {key}.npy")
        else:
            _log(f"  WARNING: --continuous {key} but {path} not found, skipping")

    # Categorical signals (from cell_metadata.csv columns)
    cat_mappings = {}  # {col_name: {value: int}}
    cat_arrays = {}    # {col_name: np.array of ints}
    meta_df = None
    if categorical_keys:
        meta_path = os.path.join(data_dir, 'cell_metadata.csv')
        if os.path.exists(meta_path):
            meta_df = pd.read_csv(meta_path, index_col=0)
            for key in categorical_keys:
                if key in meta_df.columns:
                    uniq = sorted(meta_df[key].unique())
                    mapping = {v: i for i, v in enumerate(uniq)}
                    cat_mappings[key] = mapping
                    cat_arrays[key] = meta_df[key].map(mapping).values.astype(np.int64)
                    _log(f"  Categorical '{key}': {len(uniq)} categories {uniq}")
                else:
                    _log(f"  WARNING: --categorical {key} not in cell_metadata.csv, skipping")
        else:
            _log(f"  WARNING: --categorical specified but cell_metadata.csv not found")
            categorical_keys = []

    # ATAC mask
    atac_mask = None
    if use_atac_mask:
        # Try mask_name variant first, then default
        candidates = []
        if mask_name:
            candidates.append(os.path.join(data_dir, f'atac_mask_{mask_name}.npy'))
            candidates.append(os.path.join(data_dir, f'atac_mask_{mask_name}.npz'))
            candidates.append(os.path.join(data_dir, f'{mask_name}.npz'))
            candidates.append(os.path.join(data_dir, f'{mask_name}_mask.npz'))
        candidates.append(os.path.join(data_dir, 'atac_mask.npy'))
        candidates.append(os.path.join(data_dir, 'atac_mask.npz'))

        for path in candidates:
            if os.path.exists(path):
                if path.endswith('.npz'):
                    try:
                        from scipy.sparse import load_npz as _load_sparse
                        atac_mask = _load_sparse(path).toarray()
                    except Exception:
                        atac_mask = np.load(path)['arr_0']
                else:
                    atac_mask = np.load(path)
                _log(f"  ATAC mask: {os.path.basename(path)} — "
                      f"{atac_mask.sum():.0f} non-zero edges "
                      f"({atac_mask.sum()/(n_genes*n_genes)*100:.2f}%)")
                break
        if atac_mask is None:
            _log(f"  WARNING: --atac_mask but no mask file found")

    # Override mask with continuous prior matrix if specified
    if prior_matrix:
        prior_path = os.path.join(data_dir, prior_matrix)
        if os.path.exists(prior_path):
            atac_mask = np.load(prior_path)
            _log(f"  Prior matrix (as mask): {prior_matrix} — "
                  f"{(atac_mask > 0).sum():.0f} nonzero, "
                  f"range=[{atac_mask.min():.3f}, {atac_mask.max():.3f}]")

    # Load initialization prior
    init_prior_data = None
    if init_prior:
        ip_path = os.path.join(data_dir, init_prior)
        if os.path.exists(ip_path):
            init_prior_data = np.load(ip_path)
            _log(f"  Init prior: {init_prior} — "
                  f"{(init_prior_data != 0).sum():.0f} nonzero")


    # Build truth_pairs for fast eval
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    return {
        'X': X,
        'gene_names': gene_names,
        'n_genes': n_genes,
        'n_cells': n_cells,
        'TFs': TFs,
        'gene_to_idx': gene_to_idx,
        'signals': signals,
        'cat_mappings': cat_mappings,
        'cat_arrays': cat_arrays,
        'atac_mask': atac_mask,
        'init_prior': init_prior_data,
        'continuous_keys': [k for k in continuous_keys if k in signals],
        'categorical_keys': {k: len(v) for k, v in cat_mappings.items()},
        'data_dir': data_dir,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_diffusion(T, beta_start, beta_end, device):
    """Build diffusion schedule tensors."""
    scale = 1000 / T
    betas = torch.linspace(scale * beta_start, scale * beta_end, T)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0).to(device)
    return torch.sqrt(alpha_bars), torch.sqrt(1.0 - alpha_bars)


def _run_training_loop(model, loader, optimizer, sqrt_ab, sqrt_om, T,
                       device, n_steps, sparse_coef, eval_every, ds,
                       gene_reg_norm, cont_names, cat_names, label="",
                       no_mask_anneal=False, lambda_mask=0.0,
                       lambda_pergene=0.0,
                       early_stop_threshold=0.001,
                       no_early_stop=False):
    """Core training loop. Returns final model state + metrics.

    Model selection: uses the FINAL checkpoint (like RegDiffusion).
    GT metrics are logged for monitoring but do NOT influence training or
    checkpoint selection. This ensures zero GT leakage into the model.

    Convergence: tracked via adj_change (mean absolute change in A per step).
    When adj_change → 0, the model has converged. User can inspect logs.
    """
    best_state = None
    prev_A = None  # for adj_change tracking

    # Loss-convergence early stopping:
    # Stop when loss relative change drops below 10% (plateau detected).
    # Requires loss to have dropped at least 50% from initial (learning happened).
    # This fires at ~step 250 for PBMC (21960 genes) and ~step 150 for smaller datasets.
    prev_eval_loss = None
    initial_loss = None
    min_steps_before_stop = 150  # never stop before this

    # AMP for large gene sets — halves all intermediate tensor sizes
    use_amp = (ds['n_genes'] > 10000 and device != 'cpu')
    # GradScaler not needed with bfloat16 params (bf16 has same range as f32)
    # Only use scaler if all params are float32
    has_bf16 = any(p.dtype == torch.bfloat16 for p in model.parameters())
    scaler = _amp_grad_scaler() if (use_amp and not has_bf16) else None
    if use_amp:
        _log(f"  Using AMP (mixed precision) for {ds['n_genes']} genes"
              f"{', no GradScaler (bf16 params)' if has_bf16 else ''}")

    # Mask annealing: start with mask_leak for fast learning,
    # anneal to 1.0 so uncovered GT edges can rank properly (AUROC)
    mask_anneal = (not no_mask_anneal and
                   hasattr(model, 'set_mask_leak') and
                   model._atac_mask_binary is not None and
                   model.mask_leak < 1.0)
    if mask_anneal:
        leak_start = model.mask_leak
        anneal_start = int(n_steps * 0.25)   # start annealing at 25%
        anneal_end = int(n_steps * 0.75)     # reach leak=1.0 at 75%
        _log(f"  Mask annealing: leak {leak_start:.2f} -> 1.0 "
              f"(steps {anneal_start}-{anneal_end})")

    _desc = f"  training ({label})" if label else "  training"
    _bar = _progress(range(n_steps), total=n_steps, desc=_desc)
    for step in _bar:
        # Anneal mask leak
        if mask_anneal:
            if step < anneal_start:
                leak = leak_start
            elif step < anneal_end:
                frac = (step - anneal_start) / (anneal_end - anneal_start)
                leak = leak_start + frac * (1.0 - leak_start)
            else:
                leak = 1.0
            if step % 25 == 0:  # update every 25 steps (not every step)
                model.set_mask_leak(leak)

        model.train()
        batch = next(iter(loader))
        idx = 0
        x_0 = batch[idx].to(device); idx += 1
        continuous = {}
        for key in cont_names:
            continuous[key] = batch[idx].to(device); idx += 1
        categorical = {}
        for key in cat_names:
            categorical[key] = batch[idx].to(device); idx += 1

        optimizer.zero_grad()
        t = torch.randint(0, T, (x_0.shape[0],), device=device)
        noise = torch.randn_like(x_0)
        x_t = sqrt_ab[t].unsqueeze(-1) * x_0 + sqrt_om[t].unsqueeze(-1) * noise

        with _amp_autocast(use_amp):
            need_h = (lambda_pergene > 0 and step > 10)
            if need_h:
                noise_pred, h_features = model(x_t, t,
                                   continuous=continuous if continuous else None,
                                   categorical=categorical if categorical else None,
                                   return_h=True)
            else:
                noise_pred = model(x_t, t,
                                   continuous=continuous if continuous else None,
                                   categorical=categorical if categorical else None)
            loss = F.mse_loss(noise, noise_pred)

            if step > 10:
                loss += model.sparsity_loss(
                    lambda_static=sparse_coef,
                    lambda_dynamic=sparse_coef * 0.2)

                # Per-gene feature reconstruction loss — gives A focused gradient
                # Manual backward inside, returns detached loss (no double graph)
                if lambda_pergene > 0:
                    pg_loss = model.per_gene_feature_loss(
                        h_features, n_sample=500)
                    # pg_loss is detached — backward already done inside the method
                    # Gradient accumulated to adj_A.grad manually

                # Gene-masking auxiliary loss (BERT-style, legacy)
                if lambda_mask > 0:
                    loss += lambda_mask * model.gene_masking_loss(x_0)

                # Temporal contrastive loss (for dynamic A)
                lambda_temporal = getattr(model, '_lambda_temporal', 0)
                if lambda_temporal > 0 and 'pseudotime' in (continuous or {}):
                    tau = continuous['pseudotime']
                    perm = torch.randperm(tau.shape[0], device=tau.device)
                    cond_orig = model.cond_encoder(
                        t, continuous=continuous, categorical=categorical)
                    cont_shuf = {k: v[perm] for k, v in continuous.items()}
                    cat_shuf = ({k: v[perm] for k, v in categorical.items()}
                                if categorical else None)
                    cond_shuf = model.cond_encoder(
                        t[perm], continuous=cont_shuf, categorical=cat_shuf)
                    ramp = min(1.0, step / (n_steps * 0.3))
                    loss += (lambda_temporal * ramp *
                             model.temporal_contrastive_loss(
                                 cond_orig, cond_shuf, tau, tau[perm]))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if step % eval_every == eval_every - 1 or step == n_steps - 1:
            model.eval()

            # Track A convergence (like RegDiffusion's adj_change)
            with torch.no_grad():
                # Sample random rows for adj_change (not just top-left corner)
                idx = torch.randperm(model.n_genes, device=model.adj_A.device)[:2000]
                A_sample = model.adj_A.data[idx][:, idx].float()
                if prev_A is not None:
                    adj_change = float((A_sample - prev_A).abs().mean())
                else:
                    adj_change = float('inf')
                prev_A = A_sample.clone()

            # Print ONLY loss + adj_change. NO GT metrics during training.
            if hasattr(_bar, "set_postfix_str"):
                _bar.set_postfix_str(f"loss={loss.item():.4f} adj_Δ={adj_change:.6f}")
            _log(f"  {label} Step {step+1:4d} | loss={loss.item():.4f} "
                  f"adj_Δ={adj_change:.6f}")

            # Loss-convergence early stopping:
            # Stop when loss relative change < 10% (plateau) AND loss has dropped ≥50% from start
            current_loss = loss.item()
            if initial_loss is None:
                initial_loss = current_loss
            if (not no_early_stop) and prev_eval_loss is not None and step >= min_steps_before_stop:
                rel_change = abs(current_loss - prev_eval_loss) / (prev_eval_loss + 1e-8)
                total_drop = (initial_loss - current_loss) / (initial_loss + 1e-8)
                if rel_change < 0.10 and total_drop > 0.50:
                    _ms(f"  Converged at step {step+1}: "
                          f"|Δloss|={rel_change:.3f} < 10%, "
                          f"total drop={total_drop:.1%} from {initial_loss:.4f}")
                    if hasattr(_bar, "close"): _bar.close()
                    break
            prev_eval_loss = current_loss

    # Save FINAL converged model state
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, 0.0


def train_single(ds, args, seed=42):
    """
    Train scLATCH. Single-stage by default (the canonical protocol): A_static + NN and the
    rank-32 dynamic delta(τ) train jointly.

    Optional two-stage (--two_stage, ablation only):
      Stage 1: Learn static A via denoising + sparsity (benchmark-quality GRN)
      Stage 2: Learn dynamic delta(τ) on top of frozen A_static (if dynamic_rank>0)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_genes = ds['n_genes']
    gene_reg_norm = 1.0 / (n_genes - 1)
    device = args.device

    continuous_keys = ds['continuous_keys'] or None
    categorical_keys = ds['categorical_keys'] or None

    # Expression correlation matrix — used for:
    # 1. Correlation-informed A initialization (gives all GT edges a head start)
    # 2. Sign consistency loss (A signs match expression correlation)
    _log(f"  Computing expression correlation matrix ({n_genes} genes)...")

    # A initialization: uniform (like RegDiffusion) or correlation-informed
    use_corr_init = getattr(args, 'corr_init', False)
    if use_corr_init:
        corr_init = np.abs(np.nan_to_num(np.corrcoef(ds['X'].T).astype(np.float32), 0))
        np.fill_diagonal(corr_init, 0)
        _log(f"  A init: correlation-informed (mean |corr|={corr_init.mean():.4f})")
    else:
        corr_init = None
        _log(f"  A init: uniform (gene_reg_norm * 5)")

    model = scLATCH(
        n_genes=n_genes,
        hidden_dims=args.hidden_dims,
        cond_dim=args.cond_dim,
        continuous_cond_keys=continuous_keys,
        categorical_cond_keys=categorical_keys,
        adj_dropout=args.adj_dropout,
        sparse_threshold_coef=args.sparse_threshold_coef,
        atac_mask=ds['atac_mask'],
        mask_leak=getattr(args, 'mask_leak', 0.0),
        init_prior=corr_init,
        dynamic_rank=args.dynamic_rank,
        adj_attention=getattr(args, 'adj_attention', False),
        attn_dim=getattr(args, 'attn_dim', 32),
    ).to(device)

    has_dynamic = args.dynamic_rank > 0

    # Store loss hyperparams on model for the training loop
    model._lambda_temporal = args.lambda_temporal if has_dynamic else 0

    n_params_nn = sum(p.numel() for p in model.get_nn_params())
    n_params_A = n_genes * n_genes
    n_params_dyn = sum(p.numel() for p in model.get_dynamic_params())
    _log(f"  Model: {n_params_nn:,} nn + {n_params_A:,} A_static"
          + (f" + {n_params_dyn:,} dynamic" if has_dynamic else ""))
    _log(f"  Conditioning: continuous={continuous_keys}, "
          f"categorical={categorical_keys}")

    # Build tensors
    tensors = [torch.tensor(ds['X'], dtype=torch.float32)]
    cont_names = ds['continuous_keys']
    for key in cont_names:
        tensors.append(torch.tensor(ds['signals'][key], dtype=torch.float32))
    cat_names = list(ds['categorical_keys'].keys())
    for key in cat_names:
        tensors.append(torch.tensor(ds['cat_arrays'][key], dtype=torch.long))

    dataset = TensorDataset(*tensors)
    sampler = torch.utils.data.RandomSampler(
        dataset, replacement=True, num_samples=args.batch_size)
    loader = DataLoader(dataset, sampler=sampler,
                        batch_size=args.batch_size, drop_last=True)

    # Diffusion schedule
    sqrt_ab, sqrt_om = _build_diffusion(
        args.T, args.beta_start, args.beta_end, device)

    lr_adj = gene_reg_norm / args.lr_adj_divisor
    t0 = time.time()

    if has_dynamic and getattr(args, 'two_stage', False):
        # ══ TWO-STAGE: early-stop on static, freeze it, then train dynamic ══
        # STAGE 1 — A_static + NN, dynamic FROZEN, EARLY-STOP at static convergence
        model.freeze_dynamic()
        opt1 = torch.optim.Adam([
            {'params': model.get_nn_params(),     'lr': args.lr_nn, 'weight_decay': args.wd_nn},
            {'params': model.get_static_params(), 'lr': lr_adj,     'weight_decay': args.wd_adj},
        ], betas=(0.9, 0.99))
        _log(f"\n  ── STAGE 1: A_static + NN, dynamic frozen, EARLY-STOP (max {args.n_steps}) ──")
        _run_training_loop(
            model, loader, opt1, sqrt_ab, sqrt_om, args.T, device, args.n_steps,
            args.sparse_coef, args.eval_every, ds, gene_reg_norm, cont_names, cat_names,
            label="stage1", no_mask_anneal=args.no_mask_anneal, lambda_mask=args.lambda_mask,
            lambda_pergene=args.lambda_pergene, early_stop_threshold=args.early_stop_threshold,
            no_early_stop=getattr(args, 'no_early_stop', False))
        # STAGE 2 — FREEZE A_static + NN, train ONLY dynamic δA(τ) on the fixed backbone
        model.unfreeze_dynamic()
        for p in model.get_static_params(): p.requires_grad = False
        for p in model.get_nn_params():     p.requires_grad = False
        s2 = getattr(args, 'stage2_steps', None) or 1000
        opt2 = torch.optim.Adam([
            {'params': model.get_dynamic_params(), 'lr': lr_adj * 2, 'weight_decay': args.wd_adj * 0.5},
        ], betas=(0.9, 0.99))
        _log(f"\n  ── STAGE 2: dynamic δA(τ) on FROZEN A_static ({s2} steps) ──")
        best_state, best_score = _run_training_loop(
            model, loader, opt2, sqrt_ab, sqrt_om, args.T, device, s2,
            args.sparse_coef, args.eval_every, ds, gene_reg_norm, cont_names, cat_names,
            label="stage2", no_mask_anneal=True, lambda_mask=args.lambda_mask,
            lambda_pergene=args.lambda_pergene, early_stop_threshold=args.early_stop_threshold,
            no_early_stop=True)
    else:
        # ── Single-stage: static A + dynamic A(τ) together (documented protocol) ──
        if has_dynamic:
            model.unfreeze_dynamic()
            _log(f"\n  ── Training: A_static + dynamic A(τ) rank={args.dynamic_rank} "
                  f"({args.n_steps} steps) ──")
        else:
            _log(f"\n  ── Training: Static GRN ({args.n_steps} steps) ──")
        param_groups = [
            {'params': model.get_nn_params(), 'lr': args.lr_nn, 'weight_decay': args.wd_nn},
            {'params': model.get_static_params(), 'lr': lr_adj, 'weight_decay': args.wd_adj},
        ]
        if has_dynamic:
            param_groups.append(
                {'params': model.get_dynamic_params(), 'lr': lr_adj * 2, 'weight_decay': args.wd_adj * 0.5})
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.99))
        best_state, best_score = _run_training_loop(
            model, loader, optimizer, sqrt_ab, sqrt_om, args.T,
            device, args.n_steps, args.sparse_coef, args.eval_every,
            ds, gene_reg_norm, cont_names, cat_names, label="",
            no_mask_anneal=args.no_mask_anneal,
            lambda_mask=args.lambda_mask,
            lambda_pergene=args.lambda_pergene,
            early_stop_threshold=args.early_stop_threshold,
            no_early_stop=getattr(args, 'no_early_stop', False))

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    model.eval()
    elapsed = time.time() - t0
    _log(f"  Training complete ({elapsed:.0f}s)")

    return model, elapsed


def save_results(model, ds, elapsed, save_dir, args, gene_reg_norm,
                 max_edges=1_000_000):
    """Save training outputs to a single folder (memory-efficient).

    Saves:
        model.pt               — checkpoint (only NN params, NOT full adj_A)
        adjacency.npz          — sparse A matrix (only non-trivial edges)
        training_config.json   — hyperparameters + metadata

    Memory optimizations:
        - A extracted as bf16 on CPU, converted to sparse immediately
        - Edge list saved as top max_edges only (not all 482M)
        - Model checkpoint excludes adj_A (saved in adjacency.npz instead)
        - Total disk: ~50 MB instead of ~20 GB
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract A on CPU, normalize, and convert to sparse immediately
    with torch.no_grad():
        A = model.get_adj().cpu().float().numpy()  # clean A (after thresh + mask)
    A = A / gene_reg_norm  # normalize (same as old format for compatibility)
    A_abs = np.abs(A)
    np.fill_diagonal(A_abs, 0)

    # Save adjacency as sparse .npz (only non-zero entries)
    from scipy.sparse import csr_matrix, save_npz
    A_sparse = csr_matrix(A_abs)
    save_npz(os.path.join(save_dir, 'adjacency.npz'), A_sparse)
    n_nonzero = A_sparse.nnz
    _log(f"  Adjacency: {n_nonzero:,} non-zero edges "
          f"({A_sparse.data.nbytes / 1e6:.1f} MB sparse)")

    # Save top edges as compact TSV (not all 482M)
    gene_names = np.array(ds['gene_names'])
    if n_nonzero > max_edges:
        # Exactly the max_edges largest weights. A threshold test (>=) would keep every tie at
        # the cut-off, and a short, unconverged run has near-uniform A (all edges tied).
        top_idx = np.argpartition(A_sparse.data, -max_edges)[-max_edges:]
        top_data = A_sparse.data[top_idx]
        # Get row/col indices for top edges
        rows, cols = A_sparse.nonzero()
        top_rows = rows[top_idx]
        top_cols = cols[top_idx]
    else:
        rows, cols = A_sparse.nonzero()
        top_rows, top_cols, top_data = rows, cols, A_sparse.data

    # Sort by weight descending
    order = np.argsort(-top_data)
    edges = pd.DataFrame({
        'TF': gene_names[top_cols[order]],
        'Target': gene_names[top_rows[order]],
        'EdgeWeight': top_data[order].astype(np.float32),
    })
    edges.to_csv(os.path.join(save_dir, 'GRN_inference_result.tsv'),
                 sep='\t', index=False)
    _log(f"  Edge list: {len(edges):,} edges saved")

    # Model checkpoint (full state dict — includes adj_A for reproducibility)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

    # Training config
    config = {
        'elapsed_s': round(elapsed, 1),
        'n_genes': ds['n_genes'],
        'n_cells': ds['n_cells'],
        'n_edges': n_nonzero,
        'gene_reg_norm': gene_reg_norm,
        'data_dir': ds.get('data_dir'),
        'args': {k: v for k, v in vars(args).items()
                 if not k.startswith('_')},
    }
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)

    _log(f"  Saved to {_disp(save_dir)}/")
    _log(f"    adjacency.npz  — {n_nonzero:,} edges ({A_sparse.data.nbytes/1e6:.1f} MB)")
    _log(f"    model.pt       — full state dict")
    _log(f"    GRN_inference_result.tsv — top {len(edges):,} edges")
    _log(f"    training_config.json — hyperparameters")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='scLATCH: Unified training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data
    g = parser.add_argument_group('Data')
    g.add_argument('--data_dir', type=str, nargs='+', required=True,
                   help='Path(s) to dataset directory(ies)')
    g.add_argument('--continuous', type=str, nargs='*', default=[],
                   help='Continuous signal names (e.g. pseudotime velocity)')
    g.add_argument('--categorical', type=str, nargs='*', default=[],
                   help='Categorical column names from cell_metadata.csv '
                        '(e.g. condition batch)')
    g.add_argument('--atac_mask', action='store_true',
                   help='Load ATAC mask from data_dir/atac_mask.npy')
    g.add_argument('--mask_name', type=str, default=None,
                   help='ATAC mask variant name (e.g. chipatlas, motif, combined). '
                        'Loads data_dir/atac_mask_{name}.npy instead of atac_mask.npy')
    g.add_argument('--mask_leak', type=float, default=0.4,
                   help='Leak rate for non-mask edges [0=hard mask, 0.4=recommended]. '
                        'Non-mask edges get this weight instead of 0.')
    g.add_argument('--no_mask_anneal', action='store_true',
                   help='Disable mask annealing (keep mask_leak fixed throughout training)')
    g.add_argument('--root_gene', type=str, default=None,
                   help='Root gene for DPT pseudotime (e.g. TCF7, CD34). Used when '
                        'pseudotime.npy is missing and must be auto-computed. '
                        'Also used for direction validation (root gene should '
                        'decrease along τ). If not given, uses gene-count heuristic.')
    g.add_argument('--root_cell_type', type=str, default=None,
                   help='Root cell type for DPT (e.g. naive, HSC). Used when '
                        'pseudotime.npy is missing and root_gene is not given.')
    g.add_argument('--corr_init', action='store_true',
                   help='Initialize A from expression correlation (default: uniform init)')
    g.add_argument('--init_prior', type=str, default=None,
                   help='Prior matrix .npy file for A initialization (in data_dir). '
                        'E.g. prior_mask_corr.npy, expr_correlation.npy')
    g.add_argument('--prior_matrix', type=str, default=None,
                   help='Continuous prior .npy file to use AS the mask (replaces binary). '
                        'E.g. prior_mask_corr.npy, prior_hybrid.npy')
    g.add_argument('--lambda_mask', type=float, default=0.0,
                   help='Gene-masking auxiliary loss weight (BERT-style). '
                        'Gives A direct O(N)-per-gene gradient. 0.5 recommended.')
    g.add_argument('--lambda_pergene', type=float, default=0.0,
                   help='Per-gene feature reconstruction loss weight. '
                        'Predicts each gene MLP features from other genes through A. '
                        'N× stronger gradient than denoising. 1.0 recommended.')

    # Model
    g = parser.add_argument_group('Model')
    g.add_argument('--hidden_dims', type=int, nargs='+', default=[16, 16, 16],
                   help='MLP block hidden dimensions')
    g.add_argument('--cond_dim', type=int, default=64,
                   help='Condition embedding dimension')
    g.add_argument('--adj_dropout', type=float, default=0.3,
                   help='Adjacency dropout rate during training')
    g.add_argument('--sparse_threshold_coef', type=float, default=0.2,
                   help='Soft threshold = gene_reg_norm * coef (edges below threshold pushed to 0)')
    g.add_argument('--dynamic_rank', type=int, default=32,
                   help='Rank of dynamic A(τ) perturbation (0=static, 32=recommended with pseudotime)')
    g.add_argument('--adj_attention', action='store_true',
                   help='Use attention-based A (per-TF softmax competition)')
    g.add_argument('--attn_dim', type=int, default=32,
                   help='Query/key dimension for attention-based A')

    # Regularization
    g = parser.add_argument_group('Regularization')
    g.add_argument('--lambda_temporal', type=float, default=0.05,
                   help='Temporal contrastive loss weight (A varies with τ). '
                        'Default 0.05; set 0 for static benchmark runs.')
    g.add_argument('--two_stage', action='store_true',
                   help='Two-stage: Stage1 early-stop on static (dynamic frozen); '
                        'Stage2 freeze A_static, train only δA(τ).')
    g.add_argument('--stage2_steps', type=int, default=1000,
                   help='Stage-2 (dynamic) steps for --two_stage')

    # Diffusion
    g = parser.add_argument_group('Diffusion')
    g.add_argument('--T', type=int, default=5000,
                   help='Number of diffusion timesteps')
    g.add_argument('--beta_start', type=float, default=0.0001)
    g.add_argument('--beta_end', type=float, default=0.02)

    # Training
    g = parser.add_argument_group('Training')
    g.add_argument('--n_steps', type=int, default=2000,
                   help='Number of training iterations')
    g.add_argument('--batch_size', type=int, default=256,
                   help='Batch size (sampled with replacement)')
    g.add_argument('--lr_nn', type=float, default=1e-3,
                   help='Learning rate for NN parameters')
    g.add_argument('--lr_adj_divisor', type=float, default=50,
                   help='A learning rate = gene_reg_norm / divisor')
    g.add_argument('--wd_nn', type=float, default=0.1,
                   help='Weight decay for NN parameters')
    g.add_argument('--wd_adj', type=float, default=0.01,
                   help='Weight decay for A')
    g.add_argument('--sparse_coef', type=float, default=0.025,
                   help='Sparsity loss coefficient on A (L1 regularization for edge selection)')
    g.add_argument('--eval_every', type=int, default=50,
                   help='Evaluate every N steps')
    g.add_argument('--early_stop_threshold', type=float, default=0.001,
                   help='adj_change threshold for early stopping')
    g.add_argument('--early_stop_patience', type=int, default=50,
                   help='Number of consecutive checks below threshold')
    g.add_argument('--no_early_stop', action='store_true',
                   help='Disable loss-convergence early stopping (run all n_steps). '
                        'Default behavior unchanged when omitted; production runs '
                        'continue to use the 10%%/50%% loss-plateau criterion.')

    # Seeds / output
    g = parser.add_argument_group('Output')
    g.add_argument('--seeds', type=int, nargs='+', default=[42],
                   help='Random seed(s) — multiple for ensemble')
    g.add_argument('--save_dir', type=str, default='results',
                   help='Base output directory')
    g.add_argument('--run_name', type=str, default=None,
                   help='Run name (default: auto from data_dir + signals)')
    g.add_argument('--device', type=str, default='cuda',
                   help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Auto-select device
    if args.device == 'cuda' and not torch.cuda.is_available():
        _log("  CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Process each dataset
    for data_dir in args.data_dir:
        dataset_name = os.path.basename(data_dir.rstrip('/'))

        _log(f"\n{'='*60}")
        _log(f"scLATCH: {dataset_name}")
        _log(f"{'='*60}")

        ds = load_dataset(
            data_dir,
            continuous_keys=list(args.continuous),
            categorical_keys=list(args.categorical),
            use_atac_mask=args.atac_mask,
            mask_name=args.mask_name,
            prior_matrix=getattr(args, 'prior_matrix', None),
            init_prior=getattr(args, 'init_prior', None),
            root_gene=getattr(args, 'root_gene', None),
            root_cell_type=getattr(args, 'root_cell_type', None),
        )

        # Auto-generate run name
        if args.run_name:
            run_name = args.run_name
        else:
            parts = ['sclatch']
            if ds['continuous_keys']:
                parts.extend(ds['continuous_keys'])
            if ds['categorical_keys']:
                parts.extend(ds['categorical_keys'].keys())
            if ds['atac_mask'] is not None:
                parts.append('atac')
            run_name = '_'.join(parts)

        # Multi-seed training
        all_metrics = []
        for seed in args.seeds:
            seed_suffix = f'seed{seed}' if len(args.seeds) > 1 else ''
            save_dir = os.path.join(args.save_dir, dataset_name, run_name)
            if seed_suffix:
                save_dir = os.path.join(save_dir, seed_suffix)

            _ms(f"\n--- Seed {seed} ---")
            model, elapsed = train_single(ds, args, seed)
            grn = 1.0 / (ds['n_genes'] - 1)

            # Save converged model (single checkpoint — no GT-based selection)
            save_results(model, ds, elapsed, save_dir, args, grn)

            # Evaluate against ground truth ONLY when label*.csv files exist (optional
            # benchmark step). Analysis-only datasets have no GT, so skip it silently.
            data_dir = ds['data_dir']
            has_gt = (os.path.isdir(data_dir) and
                      any(f.startswith('label') and f.endswith('.csv')
                          for f in os.listdir(data_dir)))
            if has_gt:
                from sclatch.evaluation.evaluate_pbmc import evaluate_run
                results = evaluate_run(save_dir, data_dir)
                all_metrics.append(results)

            _log(f"  Saved to {_disp(save_dir)}/ ({elapsed:.0f}s)")

        # Ensemble summary
        if len(args.seeds) > 1 and all_metrics:
            # Collect from first GT (usually 'collectri')
            first_gt = list(all_metrics[0].keys())[0] if all_metrics[0] else None
            if first_gt:
                auprc_vals = [m[first_gt]['AUPRC_ratio'] for m in all_metrics
                              if first_gt in m]
                auroc_vals = [m[first_gt]['AUROC'] for m in all_metrics
                              if first_gt in m]
                _log(f"\n  Ensemble ({len(args.seeds)} seeds, GT={first_gt}):")
                _log(f"    AUPRC_ratio: {np.mean(auprc_vals):.3f} "
                      f"± {np.std(auprc_vals):.3f}")
                _log(f"    AUROC:       {np.mean(auroc_vals):.3f} "
                      f"± {np.std(auroc_vals):.3f}")


if __name__ == '__main__':
    main()
