"""FINAL training stage for scLATCH dynamic A(τ) — SINGLE-STAGE (canonical).

This module pins the *final* training protocol used for **every** dynamic model in the
paper (PBMC, BMMC, mouse-skin, T-cell, LCMV, ovarian). It delegates to the packaged trainer
(`sclatch._train`, exposed as the `sclatch-train` console script) so there is ONE source of
truth and the package can never diverge from the trained checkpoints.

SINGLE-STAGE means A_static + NN and the rank-32 dynamic δA(τ) are trained **jointly** with
an internal warm-start (`dynamic_active` flips False→True so the random U,V do not corrupt
early denoising), under temporal-contrastive + sign-consistency regularizers. This is the
production path.

It is NOT the deprecated two-stage schedule (`sclatch-train --two_stage`: Stage-1 static
early-stop, then Stage-2 δA on a frozen backbone), which is retained in the trainer only for
ablation and must NOT be used for production dynamic models (it can yield τ-invariant gates).
"""
import os, sys, subprocess

# Exact production config (the paper's single-stage dynamic protocol). Single seed for biology.
FINAL_DYNAMIC_CONFIG = dict(
    continuous=['pseudotime'], atac_mask=True, mask_leak=0.4, no_mask_anneal=True,
    hidden_dims=[16, 16, 16], cond_dim=64, adj_dropout=0.3,
    sparse_threshold_coef=0.2, dynamic_rank=32, lambda_temporal=0.05,
    T=5000, n_steps=2000, batch_size=256, lr_nn=1e-3, lr_adj_divisor=50,
    wd_nn=0.1, wd_adj=0.01, sparse_coef=0.025,
)
TWO_STAGE = False  # final protocol is single-stage; never set True for production


def build_command(data_dir, save_dir, run_name='dynamic', seed=42, python=None,
                  repo_root=None, **overrides):
    """Build the canonical trainer command for FINAL single-stage dynamic training.

    The command runs the packaged trainer as a module (``python -m sclatch._train``)
    so it works from an installed wheel regardless of the working directory. The
    ``sclatch-train`` console script and ``python -m sclatch._train`` execute the exact
    same code, so there is no divergence from the trained checkpoints.
    """
    cfg = {**FINAL_DYNAMIC_CONFIG, **overrides}
    cmd = [python or sys.executable, '-u', '-m', 'sclatch._train',
           '--data_dir', data_dir, '--save_dir', save_dir, '--run_name', run_name,
           '--seeds', str(seed)]
    for k, v in cfg.items():
        flag = '--' + k
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        elif isinstance(v, (list, tuple)):
            cmd.append(flag); cmd += [str(x) for x in v]
        else:
            cmd += [flag, str(v)]
    # deliberately NO --two_stage
    return cmd


def train_dynamic(data_dir, save_dir, run_name='dynamic', seed=42, python=None,
                  repo_root=None, dry_run=False, cwd=None, **overrides):
    """Run the FINAL single-stage dynamic training via the packaged trainer.

    Args:
        data_dir: processed dataset dir (needs data.npy, gene_names.csv, pseudotime.npy,
                  atac_mask.npz, cell_metadata.csv).
        save_dir: output dir; the model lands in `{save_dir}/{dataset}/{run_name}/model.pt`.
        seed: training seed (single seed for the published biology).
        cwd: working directory for the subprocess (defaults to the caller's cwd, so
             relative data_dir/save_dir paths resolve as the user expects).
        repo_root: kept for backward compatibility; if given, used as the cwd.
        overrides: override any FINAL_DYNAMIC_CONFIG entry (e.g. n_steps=3000).
        dry_run: if True, only return the command list without executing.
    """
    cmd = build_command(data_dir, save_dir, run_name, seed, python, repo_root, **overrides)
    print('[sclatch.train_dynamic] FINAL single-stage dynamic:\n  ' + ' '.join(cmd))
    if dry_run:
        return cmd
    return subprocess.run(cmd, cwd=cwd or repo_root, check=True)
