"""
scLATCH: Diffusion-based dynamic Gene Regulatory Network Inference.

Production model combining:
  - RegDiffusion architecture: (I-A) at output, per-gene embeddings
  - Generic multi-signal conditioning (pseudotime, cell type, batch, etc.)
  - Dynamic A(τ): low-rank pseudotime-dependent adjacency perturbation
  - Temporal contrastive loss: A varies across pseudotime
  - Two-phase training: warm-start static → unfreeze dynamic

Architecture:
  A(τ) = A_static + Σ_k gate_k(τ) · (u_k ⊗ v_k)

  where gate_k(τ) maps pseudotime to K scalar weights via sinusoidal MLP.
  When dynamic_rank=0, reduces to static A.

Training losses:
  L = L_denoise + λ_sparse_static * |A_static| + λ_sparse_dynamic * |ΔA|
      + λ_temporal * L_temporal_contrast
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ─── Building blocks (unchanged from v3) ───────────────────────────

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class GeneEmbedding(nn.Module):
    def __init__(self, n_genes, gene_dim):
        super().__init__()
        self.gene_emb = nn.Parameter(torch.randn(n_genes, gene_dim - 1))

    def forward(self, x):
        batch_emb = self.gene_emb.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([x.unsqueeze(-1), batch_emb], dim=-1)


class ConditionedBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim, dropout=0.1):
        super().__init__()
        self.l1 = nn.Linear(in_dim, out_dim)
        self.l2 = nn.Linear(out_dim, out_dim)
        self.cond_mlp = nn.Linear(cond_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Tanh()

    def forward(self, x, cond):
        h = self.dropout(self.act(self.l1(x)))
        h = h + self.act(self.cond_mlp(cond)).unsqueeze(1)
        return self.act(self.l2(h))


class ConditionEncoder(nn.Module):
    """Encodes diffusion time, continuous covariates, and categorical labels.

    Fix v2: Three changes vs original:
    (1) Continuous covariates (pseudotime ∈ [0,1]) now use a learned MLP
        instead of SinusoidalPositionEmbedding (designed for integers 0-5000).
        SinePE wastes 75% of dimensions for [0,1] inputs and produces only
        L2=0.003 variation across τ∈[0,1] — 3264× weaker than condition embedding.
    (2) Each component is L2-normalised before summation so no single signal
        can dominate the combined cond vector by scale alone.
    (3) Diffusion time still uses SinePE (correct range) but is also normalised.
    """

    def __init__(self, d_cond, continuous_keys=None, categorical_keys=None):
        super().__init__()
        self.d_cond = d_cond
        # Diffusion time: SinePE is correct here (t is integer in [0, T])
        self.time_enc = nn.Sequential(
            SinusoidalPositionEmbedding(d_cond),
            nn.Linear(d_cond, d_cond), nn.Tanh())
        self.continuous_keys = continuous_keys or []
        self.continuous_encs = nn.ModuleDict()
        for key in self.continuous_keys:
            # Fix (1): learned MLP encoder for scalars in [0,1].
            # Linear(1→d) gives proper gradient flow and L2~1.8 variation
            # across the full [0,1] range (vs 0.003 for SinePE).
            self.continuous_encs[key] = nn.Sequential(
                nn.Linear(1, d_cond), nn.Tanh(),
                nn.Linear(d_cond, d_cond), nn.Tanh())
        self.categorical_keys = categorical_keys or {}
        self.categorical_encs = nn.ModuleDict()
        for key, n_cats in self.categorical_keys.items():
            self.categorical_encs[key] = nn.Embedding(n_cats, d_cond)

    def forward(self, t, continuous=None, categorical=None):
        # Fix (2): L2-normalise each component so no signal dominates by scale.
        # The gate network can then learn appropriate weights for each via its
        # own parameters rather than being forced to ignore the weakest signal.
        parts = [F.normalize(self.time_enc(t.float()), dim=-1)]
        if continuous:
            for key, val in continuous.items():
                if key in self.continuous_encs:
                    enc = self.continuous_encs[key](val.float().unsqueeze(-1))
                    parts.append(F.normalize(enc, dim=-1))
        if categorical:
            for key, val in categorical.items():
                if key in self.categorical_encs:
                    enc = self.categorical_encs[key](val)
                    parts.append(F.normalize(enc, dim=-1))
        return sum(parts)


# ─── Dynamic adjacency ─────────────────────────────────────────────

class DynamicAdjacency(nn.Module):
    """
    Low-rank condition-aware adjacency perturbation.

    A(τ, k) = A_static + U @ diag(gate(c)) @ V

    gate(c) maps the full condition vector c (containing pseudotime τ,
    condition label k, and diffusion time t) to K scalar weights.
    Each mode (u_k, v_k) represents a regulatory program that can
    activate or deactivate along pseudotime AND differ across conditions.
    """

    def __init__(self, n_genes, rank=16, cond_dim=64):
        super().__init__()
        self.rank = rank
        # Fix v3: 10× larger init so delta_A is visible relative to A_static.
        # Old scale=0.01/√rank gave max delta_A≈0.000075 vs A_static mean≈0.0006
        # (ratio 0.13×). New scale=0.1/√rank gives delta_A in same order as A_static.
        scale = 0.1 / math.sqrt(rank)
        self.U = nn.Parameter(torch.randn(n_genes, rank) * scale)
        self.V = nn.Parameter(torch.randn(rank, n_genes) * scale)
        # Gate network: takes full cond vector (cond_dim,) instead of scalar τ.
        # Pseudotime and condition label are already encoded in cond by ConditionEncoder.
        self.gate_net = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.Tanh(),
            nn.Linear(cond_dim, cond_dim), nn.Tanh(),
            nn.Linear(cond_dim, rank), nn.Tanh(),
        )

    def forward(self, cond):
        """cond: (batch, cond_dim) → gate: (batch, rank)"""
        return self.gate_net(cond)

    def compute_dynamic_h(self, h, gate):
        """Memory-efficient h @ delta_A via rank decomposition."""
        temp = torch.einsum('bgd,gr->brd', h, self.U)
        temp = temp * gate.unsqueeze(-1)
        return torch.einsum('brd,rg->bgd', temp, self.V)

    def get_delta_A(self, cond):
        """Get delta_A matrix for a condition vector (for analysis).

        Args:
            cond: (batch, cond_dim) tensor from build_cond_vector()
        """
        gate = self.gate_net(cond.to(self.U.device))
        return torch.einsum('br,gr,rh->bgh', gate, self.U, self.V)


# ─── Main model ────────────────────────────────────────────────────

class scLATCH(nn.Module):
    """
    scLATCH: Conditional diffusion model for dynamic GRN inference.

    When dynamic_rank=0: static model (equivalent to v3).
    When dynamic_rank>0: pseudotime-dependent A(τ).
    When adj_attention=True: attention-based A with per-TF competition.

    Args:
        n_genes: number of genes
        hidden_dims: MLP block dimensions [default: [16,16,16]]
        cond_dim: condition embedding dimension [default: 64]
        continuous_cond_keys: list of continuous signal names
        categorical_cond_keys: dict {name: n_categories}
        adj_dropout: dropout rate on A during training [default: 0.3]
        sparse_threshold_coef: soft threshold = gene_reg_norm * coef [default: 0.2, paper]
        atac_mask: optional (n_genes, n_genes) binary numpy array
        mask_leak: float in [0,1], weight for non-mask edges [default: 0.4, paper]
            With leak>0, non-mask edges get reduced weight instead of being zeroed:
            effective_mask = mask + leak * (1 - mask). leak=0 gives a hard mask.
        dynamic_rank: rank of dynamic perturbation [default: 32, paper; 0 = static]
        adj_attention: if True, learn A via attention (query/key per gene)
        attn_dim: dimension of query/key embeddings for attention A
    """

    def __init__(
        self,
        n_genes,
        hidden_dims=None,
        cond_dim=64,
        continuous_cond_keys=None,
        categorical_cond_keys=None,
        adj_dropout=0.3,
        sparse_threshold_coef=0.2,
        atac_mask=None,
        mask_leak=0.4,
        init_prior=None,
        dynamic_rank=32,
        adj_attention=False,
        attn_dim=32,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 16]

        self.n_genes = n_genes
        self.gene_dim = hidden_dims[0]
        self.adj_dropout = adj_dropout
        self.gene_reg_norm = 1.0 / (n_genes - 1)
        self.sparse_threshold = self.gene_reg_norm * sparse_threshold_coef
        self.dynamic_rank = dynamic_rank
        self.adj_attention = adj_attention

        # Use bfloat16 for A when n_genes is large (saves ~4 GB for 21960 genes)
        # bf16 has same range as f32, just less precision (7-bit mantissa)
        # Safe for GRN: we care about relative ranking, not exact values
        self._use_bf16_adj = (n_genes > 10000)
        adj_dtype = torch.bfloat16 if self._use_bf16_adj else torch.float32

        # Static adjacency — scalar or attention-based
        init_val = self.gene_reg_norm * 5
        if adj_attention:
            self.attn_query = nn.Parameter(torch.randn(n_genes, attn_dim) * 0.02)
            self.attn_key = nn.Parameter(torch.randn(n_genes, attn_dim) * 0.02)
            self.attn_scale = attn_dim ** 0.5
            self.attn_temperature = nn.Parameter(torch.ones(n_genes) * 1.0)

        # A initialization: from prior (correlation) if provided, else uniform
        if init_prior is not None:
            prior_t = torch.from_numpy(init_prior).to(adj_dtype)
            # Scale so mean A value ≈ init_val (gene_reg_norm * 5)
            prior_mean = float(prior_t[prior_t > 0].mean()) if (prior_t > 0).any() else 1.0
            scale = init_val / prior_mean
            self.adj_A = nn.Parameter(prior_t * scale)
        else:
            self.adj_A = nn.Parameter(
                torch.ones(n_genes, n_genes, dtype=adj_dtype) * init_val)

        self.mask_leak = mask_leak
        self._current_leak = mask_leak
        self.mask_bias = None

        # Multiplicative ATAC mask (soft: mask=1.0, non-mask=leak)
        if atac_mask is not None:
            self.register_buffer('_atac_mask_binary',
                                 torch.from_numpy(atac_mask).bool())
            mask_t = torch.from_numpy(atac_mask).half()
            if mask_leak > 0:
                mask_t = mask_t + mask_leak * (1.0 - mask_t)
            self.register_buffer('atac_mask', mask_t)
        else:
            self._atac_mask_binary = None
            self.atac_mask = None

        # Dynamic adjacency (optional)
        if dynamic_rank > 0:
            self.dynamic_adj = DynamicAdjacency(n_genes, rank=dynamic_rank,
                                                 cond_dim=cond_dim)
        else:
            self.dynamic_adj = None
        # dynamic_active: False during Stage 1 (warm-start) so the random
        # large-scale delta_A (scale=0.1/√rank) does not corrupt denoising.
        # Set True only when unfreeze_dynamic() is called at Stage 2 start.
        self.dynamic_active = False

        # Condition encoder
        self.cond_encoder = ConditionEncoder(
            d_cond=cond_dim,
            continuous_keys=continuous_cond_keys,
            categorical_keys=categorical_cond_keys,
        )

        # Gene embedding + MLP blocks + output (same as v3)
        self.gene_emb = nn.Sequential(
            GeneEmbedding(n_genes, self.gene_dim), nn.Tanh())
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            in_d = hidden_dims[i]
            out_d = hidden_dims[i + 1] - 1
            self.blocks.append(ConditionedBlock(in_d, out_d, cond_dim))
        self.final = nn.Linear(hidden_dims[-1] - 1, 1)

    def _update_effective_mask(self):
        """Recompute effective mask from binary mask + current leak."""
        if self._atac_mask_binary is not None:
            m = self._atac_mask_binary.half()  # bool -> float16
            if self._current_leak > 0:
                m = m + self._current_leak * (1.0 - m)
            # Replace in-place to avoid memory leak from accumulating buffers
            if hasattr(self, 'atac_mask') and self.atac_mask is not None:
                self.atac_mask.data = m
            else:
                self.atac_mask = m
        else:
            self.atac_mask = None

    def set_mask_leak(self, leak: float):
        """Update mask leak (for annealing during training)."""
        self._current_leak = leak
        self.mask_leak = leak
        self._update_effective_mask()

    # ─── Adjacency operations ──────────────────────────────────

    def soft_thresholding(self, x, tau):
        return torch.sign(x) * F.relu(torch.abs(x) - tau)

    def _get_attention_weights(self):
        """Compute attention-based per-TF target reweighting.

        Instead of softmax (which collapses to uniform), use sigmoid
        attention: each target gets an independent relevance score in [0,1].
        This creates per-TF competition without the collapse problem.

          relevance[i, j] = σ(q_j · k_i / sqrt(d))

        Combined with the scalar A magnitude, this acts as a learned
        per-TF gate: high-relevance targets get amplified, low ones
        get suppressed. Unlike softmax, scores don't sum to 1, so
        the gradient doesn't push all targets to be equal.

        Returns: (n_genes, n_genes) relevance weight matrix in [0,1]
        """
        # q_j: TF j as regulator, k_i: gene i as target
        scores = torch.mm(self.attn_key, self.attn_query.T)  # (G, G)
        scores = scores / self.attn_scale

        # Sigmoid: independent per-edge relevance in [0, 1]
        # Temperature controls sharpness
        relevance = torch.sigmoid(scores * self.attn_temperature.unsqueeze(0).clamp(min=0.1))

        # Zero out non-mask and diagonal
        if self.atac_mask is not None:
            relevance = relevance * self.atac_mask
        relevance.fill_diagonal_(0)

        return relevance

    def _get_static_clean_A(self, training_mask=None):
        """Get clean static A with thresholding + mask.

        Uses gradient checkpointing for large gene sets to save GPU memory
        (recomputes intermediates during backward instead of storing them).
        """
        if self.n_genes > 10000:
            return torch.utils.checkpoint.checkpoint(
                self._get_static_clean_A_impl, training_mask,
                use_reentrant=False)
        return self._get_static_clean_A_impl(training_mask)

    def _get_static_clean_A_impl(self, training_mask=None):
        """Implementation of clean A computation."""
        A = self.soft_thresholding(self.adj_A, self.sparse_threshold)
        A = A.clone()
        A.diagonal().zero_()
        if training_mask is not None:
            A = A * training_mask.to(A.dtype)
        if self.atac_mask is not None:
            A = A * self.atac_mask.to(A.dtype)
        return A

    def get_I_minus_A(self):
        """Get (I - A_static) for forward pass, with dropout during training."""
        if self.training and self.adj_dropout > 0:
            if self.n_genes > 10000:
                col_mask = (torch.rand(self.n_genes, device=self.adj_A.device)
                            > self.adj_dropout).to(self.adj_A.dtype)
                col_mask /= (1 - self.adj_dropout)
                clean_A = self._get_static_clean_A()
                clean_A = clean_A * col_mask.unsqueeze(0)
            else:
                dropout_mask = (torch.rand_like(self.adj_A) > self.adj_dropout).float()
                dropout_mask /= (1 - self.adj_dropout)
                dropout_mask.fill_diagonal_(0)
                clean_A = self._get_static_clean_A(dropout_mask)
        else:
            clean_A = self._get_static_clean_A()
        # I - A: subtract A from identity without materializing dense eye
        # clone to avoid in-place on computation graph
        I_minus_A = clean_A.neg()
        idx = torch.arange(self.n_genes, device=clean_A.device)
        I_minus_A[idx, idx] += 1.0
        return I_minus_A

    def get_adj(self):
        """Get the clean static adjacency (no dropout)."""
        return self._get_static_clean_A()

    def build_cond_vector(self, tau, categorical=None):
        """Build condition vector for analysis (uses t=0 dummy diffusion time).

        Args:
            tau: scalar, list, or (batch,) tensor of pseudotime values
            categorical: optional dict {key: (batch,) int tensor}
        Returns:
            cond: (batch, cond_dim) tensor
        """
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor([tau] if not hasattr(tau, '__len__') else tau,
                               dtype=torch.float32, device=self.adj_A.device)
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)
        batch = tau.shape[0]
        t_dummy = torch.zeros(batch, device=tau.device, dtype=torch.float32)
        continuous = {'pseudotime': tau}
        return self.cond_encoder(t_dummy, continuous=continuous,
                                 categorical=categorical)

    def get_adj_dynamic(self, cond):
        """Get A(τ,k) = A_static + delta_A(cond), with ATAC mask.

        Args:
            cond: (batch, cond_dim) condition vector from build_cond_vector()
                  or from cond_encoder() during training.
        """
        A_static = self.get_adj()
        if self.dynamic_adj is None or cond is None:
            return A_static
        delta = self.dynamic_adj.get_delta_A(cond)  # (batch, G, G)
        A_dyn = A_static.unsqueeze(0) + delta
        # Zero diagonal without materializing eye matrix
        eye = torch.eye(self.n_genes, device=A_dyn.device)
        A_dyn = A_dyn * (1.0 - eye).unsqueeze(0)
        if self.atac_mask is not None:
            A_dyn = A_dyn * self.atac_mask.unsqueeze(0)
        return A_dyn.squeeze(0) if A_dyn.shape[0] == 1 else A_dyn

    # ─── Regularization losses ─────────────────────────────────

    def temporal_contrastive_loss(self, cond_a, cond_b, tau_a, tau_b,
                                   cat_a=None, cat_b=None):
        """
        Encourage A(τ,k) to vary with pseudotime by operating on the adjacency
        matrix directly — not just gate vectors.

        Fix v3:
        (1) Loss operates on A-level differences, not gate-level.
            Old: L = -|Δτ| · ||gate_a - gate_b||²
              → dL/dU = 0, dL/dV = 0: U,V get NO gradient from this loss.
            New: L = -|Δτ| · ||delta_A(c_a) - delta_A(c_b)||²_F
              → gradient flows through U @ diag(gate) @ V to all parameters.

        (2) Use ALL pairs weighted by |Δτ| (revert same-condition masking).
            T cell analysis showed cross-condition pairs (naive τ≈0.06 vs
            exhausted τ≈0.87, |Δτ|=0.80) ARE biologically meaningful temporal
            comparisons — naive IS early pseudotime. Same-condition masking
            removed the only pairs with meaningful |Δτ| signal.
            Both T cell and PBMC benefit from the full pair set.
        """
        if self.dynamic_adj is None:
            return torch.tensor(0.0, device=self.adj_A.device)

        # Compute delta_A for each cond vector: (batch, G, G) but memory-heavy.
        # Use Frobenius norm via U,V decomposition: ||ΔdeltaA||_F²
        # delta_A(c) = U @ diag(gate(c)) @ V
        # delta_A_a - delta_A_b = U @ diag(gate_a - gate_b) @ V
        # ||delta_A_a - delta_A_b||_F² = ||U @ diag(Δg) @ V||_F²
        gate_a = self.dynamic_adj(cond_a)            # (batch, rank)
        gate_b = self.dynamic_adj(cond_b)            # (batch, rank)
        delta_g = gate_a - gate_b                    # (batch, rank)

        # ||U @ diag(Δg) @ V||_F² = sum_r [ Δg_r² * ||u_r||² * ||v_r||² ]
        # where u_r = U[:,r], v_r = V[r,:]
        U = self.dynamic_adj.U                       # (G, rank)
        V = self.dynamic_adj.V                       # (rank, G)
        u_norms_sq = U.pow(2).sum(dim=0)             # (rank,)
        v_norms_sq = V.pow(2).sum(dim=1)             # (rank,)
        uv_sq = u_norms_sq * v_norms_sq              # (rank,) — per-mode scale
        # delta_A diff squared, Frobenius: sum over rank of Δg_r² * ||u_r||²||v_r||²
        a_diff_sq = (delta_g.pow(2) * uv_sq.unsqueeze(0)).sum(dim=-1)  # (batch,)

        tau_diff = (tau_a - tau_b).abs()             # (batch,)
        return -(tau_diff * a_diff_sq).mean()

    def sparsity_loss(self, lambda_static=0.25, lambda_dynamic=0.05):
        """Separate sparsity budgets for static and dynamic components."""
        A_static = self.get_adj()
        loss = lambda_static * A_static.abs().mean()
        if self.dynamic_adj is not None:
            U_norm = self.dynamic_adj.U.abs().mean()
            V_norm = self.dynamic_adj.V.abs().mean()
            loss += lambda_dynamic * (U_norm + V_norm)
        return loss

    def gene_masking_loss(self, x_0, mask_fraction=0.15):
        """BERT-style gene masking loss.

        Randomly mask genes and predict their expression through A.
        Gives A direct O(N)-per-gene gradient instead of O(1/N²) from denoising.

        L_mask = MSE(x_0[masked], (x_0 @ A)[masked])

        Combined with ATAC mask (constrains which edges exist) and sparsity
        (prunes weak edges), this learns direct regulatory relationships.
        """
        A = self._get_static_clean_A().to(x_0.dtype)
        # Per-sample random mask (different genes masked per cell)
        mask = torch.rand_like(x_0) < mask_fraction  # (batch, n_genes)
        # Predict: each gene's expression from all other genes through A
        # x_pred[b, g] = sum_j x_0[b, j] * A[g, j]
        x_pred = x_0 @ A.T  # (batch, n_genes) — A[g,j] means j→g
        # Loss only on masked positions
        if mask.any():
            return torch.nn.functional.mse_loss(x_pred[mask], x_0[mask])
        return torch.tensor(0.0, device=x_0.device)

    def per_gene_feature_loss(self, h, n_sample=500):
        """Per-gene feature reconstruction through A (memory-efficient).

        For sampled target genes, predict their MLP features from all other genes'
        features through A: h_pred[g] = Σ_j A[g,j] * h[j]

        Memory optimization (inspired by RegDiffusion-ME):
        - Extracts sampled rows of adj_A as a NEW tensor (.clone()) so backward
          produces a small (n_sample, G) gradient, not a sparse grad for full adj_A.
        - Manually accumulates gradient to adj_A.data via torch.no_grad() update.
        - h is detached: no MLP gradient from this loss.
        - Total extra memory: ~50 MB for n_sample=500 (vs 8+ GB without this trick).

        Args:
            h: (B, G, D) MLP features BEFORE A mixing
            n_sample: number of target genes to sample per step
        """
        n_sample = min(n_sample, self.n_genes)
        indices = torch.randperm(self.n_genes, device=h.device)[:n_sample]

        # Clone sampled rows — creates a NEW leaf tensor, gradient is (n_sample, G)
        # not a sparse index into the full (G, G) adj_A
        A_rows = self.adj_A.data[indices, :].clone().requires_grad_(True)

        # Apply soft thresholding on sampled rows only
        A_clean = self.soft_thresholding(A_rows, self.sparse_threshold)
        # Zero self-edges
        self_mask = torch.ones(n_sample, self.n_genes, device=h.device)
        for k, idx in enumerate(indices):
            self_mask[k, idx] = 0.0
        A_clean = A_clean * self_mask
        # Apply ATAC mask if present
        if self.atac_mask is not None:
            A_clean = A_clean * self.atac_mask[indices, :].to(A_clean.dtype)

        # h is detached: gradient flows to A_rows only
        h_det = h.detach()
        h_pred = torch.einsum('sg,bgd->bsd', A_clean.to(h_det.dtype), h_det)
        h_true = h_det[:, indices, :]

        loss = F.mse_loss(h_pred, h_true)

        # Manual backward: compute grad for A_rows, then apply to adj_A
        if A_rows.grad is not None:
            A_rows.grad.zero_()
        loss.backward(retain_graph=False)

        if A_rows.grad is not None:
            # Accumulate gradient to the actual parameter's .grad
            if self.adj_A.grad is None:
                self.adj_A.grad = torch.zeros_like(self.adj_A)
            self.adj_A.grad[indices, :] += A_rows.grad.to(self.adj_A.grad.dtype)

        # Return loss value (detached — backward already done manually)
        return loss.detach()

    # ─── Forward pass ──────────────────────────────────────────

    def forward(self, x_t, t, continuous=None, categorical=None,
                return_h=False):
        cond = self.cond_encoder(t, continuous, categorical)
        original = x_t.unsqueeze(-1)
        h = self.gene_emb(x_t)

        for i, block in enumerate(self.blocks):
            if i != 0:
                h = torch.cat([h, original], dim=-1)
            h = block(h, cond)

        # Save h before A mixing (for per-gene feature loss)
        h_pre_A = h

        # Static (I - A) at output — cast to h's dtype for einsum
        I_minus_A = self.get_I_minus_A().to(h.dtype)
        h_out = torch.einsum('bgd,gh->bhd', h, I_minus_A)

        # Dynamic perturbation — gate network sees full cond (τ + condition).
        # Only active in Stage 2 (dynamic_active=True); skipped during Stage 1
        # warm-start so the uninitialised large-scale U,V don't corrupt denoising.
        if self.dynamic_adj is not None and self.dynamic_active:
            gate = self.dynamic_adj(cond)
            h_dynamic = self.dynamic_adj.compute_dynamic_h(h, gate)
            h_out = h_out - h_dynamic  # subtract: (I - A) means A contributes negatively

        noise_pred = self.final(h_out).squeeze(-1)

        if return_h:
            return noise_pred, h_pre_A
        return noise_pred

    # ─── Utility ───────────────────────────────────────────────

    def freeze_dynamic(self):
        """Freeze dynamic parameters for Phase 1 (warm-start) training."""
        self.dynamic_active = False
        if self.dynamic_adj is not None:
            for p in self.dynamic_adj.parameters():
                p.requires_grad = False

    def unfreeze_dynamic(self):
        """Unfreeze dynamic parameters for Phase 2 training."""
        self.dynamic_active = True
        if self.dynamic_adj is not None:
            for p in self.dynamic_adj.parameters():
                p.requires_grad = True

    def get_dynamic_params(self):
        """Return dynamic parameters (for separate optimizer group)."""
        if self.dynamic_adj is not None:
            return list(self.dynamic_adj.parameters())
        return []

    def get_static_params(self):
        """Return static A parameters (scalar + attention if enabled)."""
        params = [self.adj_A]
        if self.adj_attention:
            params.extend([self.attn_query, self.attn_key, self.attn_temperature])
        return params

    def get_nn_params(self):
        """Return all non-adjacency parameters."""
        adj_ids = {id(self.adj_A)}
        if self.adj_attention:
            adj_ids.update([id(self.attn_query), id(self.attn_key),
                            id(self.attn_temperature)])
        if self.dynamic_adj is not None:
            adj_ids.update(id(p) for p in self.dynamic_adj.parameters())
        return [p for p in self.parameters() if id(p) not in adj_ids]
