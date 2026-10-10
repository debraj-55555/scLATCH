"""Generic visualization and readout helpers for a trained scLATCH model.

These functions work on ANY trained model and any user dataset. They distill the
core, reusable analyses used in the paper into data-first helpers: each analysis
function returns plain numpy / pandas objects, and each ``plot_*`` function draws
them onto a matplotlib Axes. matplotlib is imported lazily, so importing this
module never requires it (install with ``pip install 'sclatch[viz]'`` to plot).

All the dynamic readouts exploit the low-rank structure of the dynamic term
``A(tau, k) = A_static + U @ diag(gate(cond)) @ V`` (rank r), so they stay cheap
even for tens of thousands of genes: the whole dynamic state lives in the r-dim
``gate(cond)`` vector.

Analyses
--------
reconfiguration_curve      ||A(tau) - A(tau0)||_F versus tau (network rewiring).
localize_commitment_boundary  tau where rewiring rate peaks / inflects.
tf_strength_matrix         per-TF regulon strength S_TF(tau) = ||A(tau)[:, TF]||_1.
gate_entropy_curve         entropy of the rank-mode gate over tau (Delta-S).
gate_trajectory            raw gate(tau) matrix (n_tau, rank).
network_state_coordinate   1-D per-cell network-state coordinate (for landscape).
quasi_potential_landscape  empirical quasi-potential U(s) = -log rho(s).
predict_knockout           in-silico TF knockout signature from the adjacency.
"""

import numpy as np
import torch

# Muted palette (paper Fig-1 spirit): brick-orange, steel-blue, purple, muted-green.
PALETTE = ["#c1663f", "#4a7ba6", "#8a6ba6", "#6a9a5b", "#b0902f", "#7a7a7a"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default_grn_norm(model):
    return 1.0 / (model.n_genes - 1)


def _as_tensor(x, device):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    return x.to(device)


@torch.no_grad()
def _gate_at(model, tau, categorical=None):
    """gate(cond) in R^rank for one or many tau values. Returns (n_tau, rank)."""
    if model.dynamic_adj is None:
        raise ValueError("model has no dynamic term (dynamic_rank=0); "
                         "dynamic readouts require a model trained with dynamic_rank>0.")
    device = model.adj_A.device
    tau_t = _as_tensor(np.atleast_1d(np.asarray(tau, dtype=np.float32)), device)
    cond = model.build_cond_vector(tau_t, categorical=categorical)
    return model.dynamic_adj.gate_net(cond).float().cpu().numpy()


@torch.no_grad()
def adjacency_at_tau(model, tau, categorical=None, normalize=True):
    """Dense dynamic adjacency A(tau, k) as a numpy array (n_genes, n_genes).

    A[g, j] is the weight of regulator j on target g. Materializes the full
    matrix, so use only for moderate gene counts; the curve/strength/entropy
    helpers below avoid this and scale to large gene sets.
    """
    device = model.adj_A.device
    cond = model.build_cond_vector(float(tau), categorical=categorical)
    A = model.get_adj_dynamic(cond).float().cpu().numpy()
    if A.ndim == 3:
        A = A[0]
    if normalize:
        A = A / _default_grn_norm(model)
    return A


# ---------------------------------------------------------------------------
# 1. A(tau) network-reconfiguration curve
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconfiguration_curve(model, tau_grid=None, categorical=None, ref_tau=None):
    """||A(tau) - A(ref_tau)||_F across a tau grid (network rewiring curve).

    Uses the exact low-rank identity
        ||U diag(d) V||_F^2 = d^T ( (U^T U) . (V V^T) ) d,   d = gate(tau) - gate(ref),
    so it never materializes a genes x genes matrix. Matches the Frobenius
    trajectory used for the paper's dynamic-network figures.

    Returns
    -------
    tau_grid : np.ndarray (n_tau,)
    frob     : np.ndarray (n_tau,)  Frobenius norm of the dynamic rewiring vs ref.
    """
    if tau_grid is None:
        tau_grid = np.linspace(0.0, 1.0, 51)
    tau_grid = np.asarray(tau_grid, dtype=np.float32)
    ref_tau = float(tau_grid[0]) if ref_tau is None else float(ref_tau)

    gates = _gate_at(model, tau_grid, categorical=categorical)          # (n_tau, r)
    gate_ref = _gate_at(model, ref_tau, categorical=categorical)[0]     # (r,)

    U = model.dynamic_adj.U.detach().float().cpu().numpy()             # (G, r)
    V = model.dynamic_adj.V.detach().float().cpu().numpy()             # (r, G)
    M = (U.T @ U) * (V @ V.T)                                          # (r, r)

    frob = np.empty(len(tau_grid), dtype=np.float64)
    for i, g in enumerate(gates):
        d = (g - gate_ref)
        frob[i] = np.sqrt(max(float(d @ M @ d), 0.0))
    return tau_grid, frob


def localize_commitment_boundary(tau_grid, frob):
    """Locate where the network reconfigures fastest along tau.

    Returns dict with:
      peak_rate_tau : tau of maximum |dA/dtau| (steepest rewiring).
      inflection_tau: first tau where d2A/dtau2 changes sign (onset of commitment).
    """
    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    frob = np.asarray(frob, dtype=np.float64)
    d1 = np.gradient(frob, tau_grid)
    d2 = np.gradient(d1, tau_grid)
    peak_rate_tau = float(tau_grid[int(np.argmax(np.abs(d1)))])

    inflection_tau = None
    for i in range(len(d2) - 1):
        if d2[i] * d2[i + 1] < 0:
            # linear interpolation of the zero crossing
            t0, t1 = tau_grid[i], tau_grid[i + 1]
            f0, f1 = d2[i], d2[i + 1]
            inflection_tau = float(t0 - f0 * (t1 - t0) / (f1 - f0))
            break
    return {"peak_rate_tau": peak_rate_tau, "inflection_tau": inflection_tau}


def plot_reconfiguration_curve(tau_grid, frob, ax=None, boundary=None,
                               color=PALETTE[0], label=None):
    """Plot the ||A(tau) - A(tau0)||_F rewiring curve; mark a boundary if given."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(tau_grid, frob, color=color, lw=2.0, label=label)
    if boundary is not None:
        ax.axvline(boundary, color="#555", ls="--", lw=1.0)
    ax.set_xlabel("pseudotime tau")
    ax.set_ylabel(r"$\|A(\tau) - A(\tau_0)\|_F$")
    if label:
        ax.legend(frameon=False, fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# 3. Per-TF regulon strength S_TF(tau) and gene x tau heatmap
# ---------------------------------------------------------------------------

@torch.no_grad()
def tf_strength_matrix(model, tf_names, gene_names, tau_grid=None,
                       categorical=None, normalize=True):
    """Per-TF regulon strength S_TF(tau) = ||A(tau)[:, TF]||_1 over a tau grid.

    Computes each TF column of A(tau) from the static column plus the low-rank
    dynamic contribution (U @ (gate * V[:, TF])), applying the same mask and
    self-edge removal as ``get_adj_dynamic`` — without materializing the full
    matrix. Scales to large gene sets.

    Returns
    -------
    pandas.DataFrame  index = TF names present in the dataset, columns = tau grid,
                      values = column-L1 regulon strength.
    """
    import pandas as pd
    if tau_grid is None:
        tau_grid = np.linspace(0.0, 1.0, 26)
    tau_grid = np.asarray(tau_grid, dtype=np.float32)

    g2i = {g: i for i, g in enumerate(gene_names)}
    tfs = [t for t in tf_names if t in g2i]
    tf_idx = np.array([g2i[t] for t in tfs], dtype=np.int64)

    device = model.adj_A.device
    grn = _default_grn_norm(model) if normalize else 1.0

    # static clean adjacency, columns for the TFs only
    A_static = model.get_adj().float()                         # (G, G) once
    static_cols = A_static[:, torch.as_tensor(tf_idx, device=A_static.device)].cpu().numpy()  # (G, n_tf)

    U = model.dynamic_adj.U.detach().float().cpu().numpy() if model.dynamic_adj is not None else None
    V = model.dynamic_adj.V.detach().float().cpu().numpy() if model.dynamic_adj is not None else None
    mask = None
    if getattr(model, "atac_mask", None) is not None:
        mask = model.atac_mask.float().cpu().numpy()

    gates = _gate_at(model, tau_grid, categorical=categorical) if U is not None else None

    out = np.zeros((len(tfs), len(tau_grid)), dtype=np.float64)
    for ti in range(len(tau_grid)):
        for k, j in enumerate(tf_idx):
            col = static_cols[:, k].copy()
            if U is not None:
                col = col + U @ (gates[ti] * V[:, j])
            if mask is not None:
                col = col * mask[:, j]
            col[j] = 0.0
            out[k, ti] = np.abs(col).sum() / grn
    return pd.DataFrame(out, index=tfs, columns=[round(float(t), 4) for t in tau_grid])


def plot_tf_strength_heatmap(strength_df, ax=None, boundary=None,
                             row_zscore=True, max_tfs=40):
    """gene x tau heatmap of regulon strength (per-row z-scored by default)."""
    import matplotlib.pyplot as plt
    df = strength_df
    if len(df) > max_tfs:
        # keep the most dynamic TFs (largest range across tau)
        rng = df.max(axis=1) - df.min(axis=1)
        df = df.loc[rng.sort_values(ascending=False).index[:max_tfs]]
    Z = df.values.astype(float)
    if row_zscore:
        Z = (Z - Z.mean(axis=1, keepdims=True)) / (Z.std(axis=1, keepdims=True) + 1e-9)
        vmin, vmax, cmap, cbar_label = -2, 2, "RdBu_r", "regulon strength (z)"
    else:
        vmin, vmax, cmap, cbar_label = None, None, "magma", "regulon strength"
    tau = np.asarray(df.columns, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, max(2.5, 0.18 * len(df))))
    im = ax.imshow(Z, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[tau[0], tau[-1], len(df), 0])
    if boundary is not None:
        ax.axvline(boundary, color="k", ls="--", lw=1.1)
    ax.set_yticks(np.arange(len(df)) + 0.5)
    ax.set_yticklabels(df.index, fontsize=6)
    ax.set_xlabel("pseudotime tau")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# 4. Gate entropy (Delta-S) over tau
# ---------------------------------------------------------------------------

@torch.no_grad()
def gate_trajectory(model, tau_grid=None, categorical=None):
    """Raw gate(tau) trajectory, shape (n_tau, rank)."""
    if tau_grid is None:
        tau_grid = np.linspace(0.0, 1.0, 51)
    tau_grid = np.asarray(tau_grid, dtype=np.float32)
    return tau_grid, _gate_at(model, tau_grid, categorical=categorical)


def gate_entropy_curve(model, tau_grid=None, categorical=None):
    """Entropy of the (normalized |gate|) rank-mode distribution across tau.

    Low entropy = one regulatory program dominates (committed state); high
    entropy = programs mixed. The entropy minimum often coincides with the
    commitment boundary.

    Returns (tau_grid, entropy).
    """
    tau_grid, gates = gate_trajectory(model, tau_grid, categorical)
    p = np.abs(gates)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-10)
    ent = -(p * np.log(p + 1e-10)).sum(axis=1)
    return tau_grid, ent


def plot_gate_entropy(tau_grid, entropy, ax=None, color=PALETTE[2], boundary=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(tau_grid, entropy, color=color, lw=2.0)
    if boundary is not None:
        ax.axvline(boundary, color="#555", ls="--", lw=1.0)
    ax.set_xlabel("pseudotime tau")
    ax.set_ylabel("gate entropy (nats)")
    return ax


# ---------------------------------------------------------------------------
# 6. Quasi-potential landscape (bistability)
# ---------------------------------------------------------------------------

@torch.no_grad()
def network_state_coordinate(model, pseudotime, categorical=None, sign_align=True):
    """1-D per-cell network-state coordinate from the dynamic gate.

    The dynamic reconfiguration of A lives entirely in the rank-r gate vector,
    scaled by each mode's norm sqrt(||u_r|| ||v_r||). This projects the per-cell
    scaled gate onto its top principal component, giving a cheap, faithful proxy
    for the network-state coordinate used in the paper's landscape (which used
    MDS-1 of the full A(tau)). Feed the result to quasi_potential_landscape.

    Parameters
    ----------
    pseudotime : (n_cells,) array of per-cell tau values.

    Returns (n_cells,) network-state coordinate.
    """
    pseudotime = np.asarray(pseudotime, dtype=np.float32)
    gates = _gate_at(model, pseudotime, categorical=categorical)        # (n, r)
    U = model.dynamic_adj.U.detach().float().cpu().numpy()
    V = model.dynamic_adj.V.detach().float().cpu().numpy()
    scale = np.sqrt((U ** 2).sum(0) * (V ** 2).sum(1) + 1e-12)          # (r,)
    Xg = gates * scale[None, :]
    Xg = Xg - Xg.mean(0, keepdims=True)
    # PC1 via SVD
    _, _, Vt = np.linalg.svd(Xg, full_matrices=False)
    s = Xg @ Vt[0]
    if sign_align:
        # orient so the coordinate increases with pseudotime
        if np.corrcoef(s, pseudotime)[0, 1] < 0:
            s = -s
    return s


def quasi_potential_landscape(state_coord, n_grid=200, bw_method="scott"):
    """Empirical quasi-potential U(s) = -log rho(s) from a 1-D state coordinate.

    Two wells separated by a barrier => bistable (a switch); one broad basin =>
    a continuum. Returns (grid, U) with U shifted so its minimum is 0.
    """
    from scipy.stats import gaussian_kde
    s = np.asarray(state_coord, dtype=np.float64)
    grid = np.linspace(s.min(), s.max(), n_grid)
    rho = gaussian_kde(s, bw_method=bw_method)(grid)
    U = -np.log(rho + 1e-4)
    U -= U.min()
    return grid, U


def plot_quasi_potential(grid, U, ax=None, color=PALETTE[1]):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(grid, U, color=color, lw=2.0)
    ax.set_xlabel("network-state coordinate s")
    ax.set_ylabel("quasi-potential U(s) = -log rho")
    return ax


# ---------------------------------------------------------------------------
# 5. In-silico knockout from the adjacency
# ---------------------------------------------------------------------------

def predict_knockout(A, gene_names, tf_name, method="direct"):
    """Predict the expression signature of knocking out one TF from adjacency A.

    A[g, j] is the effect of regulator j on target g (the scLATCH convention).
      direct  : immediate targets, signature = -A[:, TF].
      leontief: propagated steady-state effect, signature = -(I - A)^{-1}[:, TF].

    Returns a pandas Series (index = gene_names) of predicted delta-expression;
    negative = down on knockout of an activator, positive = up (de-repression).
    """
    import pandas as pd
    gene_names = list(gene_names)
    j = gene_names.index(tf_name)
    A = np.asarray(A, dtype=np.float64)
    if method == "direct":
        pred = -A[:, j]
    elif method == "leontief":
        n = A.shape[0]
        pred = -np.linalg.inv(np.eye(n) - A)[:, j]
    else:
        raise ValueError("method must be 'direct' or 'leontief'")
    return pd.Series(pred, index=gene_names, name=f"KO_{tf_name}")


def plot_knockout_barplot(signature, top_n=15, ax=None):
    """Bar plot of the strongest predicted responders to a TF knockout."""
    import matplotlib.pyplot as plt
    s = signature.drop(labels=[signature.name.replace("KO_", "")], errors="ignore")
    top = s.reindex(s.abs().sort_values(ascending=False).index[:top_n])[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 0.28 * len(top) + 0.6))
    colors = [PALETTE[3] if v > 0 else PALETTE[0] for v in top.values]
    ax.barh(np.arange(len(top)), top.values, color=colors)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top.index, fontsize=7)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("predicted delta-expression")
    ax.set_title(signature.name, fontsize=9)
    return ax


# ---------------------------------------------------------------------------
# Network (node-link) visualizations
# ---------------------------------------------------------------------------

def _spring_layout(G, seed=0):
    try:
        import networkx as nx
        return nx.spring_layout(G, seed=seed, k=None)
    except Exception:
        import numpy as _np
        rng = _np.random.default_rng(seed)
        return {n: rng.normal(size=2) for n in G.nodes()}


def plot_network(edges_df, title=None, ax=None, weight_col="weight", label_hubs=8):
    """Node-link GRN diagram from an edge list.

    edges_df has columns regulator_TF, target_gene, weight, sign. TFs (regulators) are hub
    nodes, targets are leaf nodes; node size scales with out-degree, edge width with the
    absolute weight, edge color with sign (activation warm, repression cool). Needs networkx
    and matplotlib (the `viz` extra); both are imported lazily.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 5.2))
    if edges_df is None or len(edges_df) == 0:
        ax.text(0.5, 0.5, "no edges", ha="center", va="center"); ax.axis("off"); return ax

    G = nx.DiGraph()
    tfs = set(edges_df["regulator_TF"])
    w = edges_df[weight_col].astype(float).values
    wmax = float(np.abs(w).max()) or 1.0
    for _, r in edges_df.iterrows():
        G.add_edge(r["regulator_TF"], r["target_gene"], weight=float(r[weight_col]),
                   sign=int(r["sign"]) if "sign" in edges_df.columns else int(np.sign(r[weight_col])))
    pos = _spring_layout(G, seed=0)
    outdeg = dict(G.out_degree())
    tf_nodes = [n for n in G.nodes() if n in tfs]
    tg_nodes = [n for n in G.nodes() if n not in tfs]
    nx.draw_networkx_nodes(G, pos, nodelist=tg_nodes, node_size=30, node_color="#c9c9c9",
                           ax=ax, linewidths=0)
    nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes, ax=ax, linewidths=0,
                           node_size=[120 + 60 * outdeg.get(n, 0) for n in tf_nodes],
                           node_color=PALETTE[1])
    for u, v, d in G.edges(data=True):
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-", alpha=0.5,
                                    lw=0.4 + 2.6 * abs(d["weight"]) / wmax,
                                    color=(PALETTE[0] if d["sign"] >= 0 else PALETTE[1])))
    hubs = sorted(tf_nodes, key=lambda n: -outdeg.get(n, 0))[:label_hubs]
    nx.draw_networkx_labels(G, pos, labels={n: n for n in hubs}, font_size=8, ax=ax)
    ax.set_title(title or "GRN", fontsize=10); ax.axis("off")
    ax.plot([], [], color=PALETTE[0], lw=2, label="activation (+)")
    ax.plot([], [], color=PALETTE[1], lw=2, label="repression (-)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    return ax


def plot_differential_network(change_df, title=None, ax=None, label_hubs=8):
    """Node-link diagram of the edges that change most between two states.

    change_df has columns regulator_TF, target_gene, weight_a, weight_b, change. Edge color
    marks gained (change > 0, stronger in state a) vs lost (change < 0); width scales with the
    absolute change. Needs networkx and matplotlib (the `viz` extra), imported lazily.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 5.2))
    if change_df is None or len(change_df) == 0:
        ax.text(0.5, 0.5, "no changing edges", ha="center", va="center"); ax.axis("off"); return ax
    G = nx.DiGraph()
    tfs = set(change_df["regulator_TF"])
    ch = change_df["change"].astype(float).values
    cmax = float(np.abs(ch).max()) or 1.0
    for _, r in change_df.iterrows():
        G.add_edge(r["regulator_TF"], r["target_gene"], change=float(r["change"]))
    pos = _spring_layout(G, seed=0)
    outdeg = dict(G.out_degree())
    tf_nodes = [n for n in G.nodes() if n in tfs]
    tg_nodes = [n for n in G.nodes() if n not in tfs]
    nx.draw_networkx_nodes(G, pos, nodelist=tg_nodes, node_size=30, node_color="#c9c9c9",
                           ax=ax, linewidths=0)
    nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes, ax=ax, linewidths=0,
                           node_size=[120 + 60 * outdeg.get(n, 0) for n in tf_nodes],
                           node_color="#555")
    for u, v, d in G.edges(data=True):
        gained = d["change"] > 0
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-", alpha=0.6,
                                    lw=0.4 + 2.6 * abs(d["change"]) / cmax,
                                    color=(PALETTE[3] if gained else PALETTE[0])))
    hubs = sorted(tf_nodes, key=lambda n: -outdeg.get(n, 0))[:label_hubs]
    nx.draw_networkx_labels(G, pos, labels={n: n for n in hubs}, font_size=8, ax=ax)
    ax.set_title(title or "differential GRN", fontsize=10); ax.axis("off")
    ax.plot([], [], color=PALETTE[3], lw=2, label="gained in a")
    ax.plot([], [], color=PALETTE[0], lw=2, label="lost in a")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    return ax
