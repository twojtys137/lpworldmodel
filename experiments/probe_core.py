"""Checkpoint-independent numerical kernels for frozen-model experiments."""
from __future__ import annotations

import numpy as np
import torch


def scene_variance_diagnostics(observations):
    """Separate scene, temporal and spatial variation in (N,T,P,D) latents.

    Scenes must be sampled from distinct trajectories. Variance across N is
    computed at each fixed relative time, patch and channel BEFORE averaging.
    Pooling patches would let an image-independent positional template look
    noncollapsed. These statistics alone cannot establish representation quality:
    an unchanging dataset or a changing but task-irrelevant code are confounders.
    """
    if isinstance(observations, torch.Tensor):
        observations = observations.detach().cpu().numpy()
    z = np.asarray(observations, dtype=np.float64)
    if z.ndim != 4 or any(size < 1 for size in z.shape) or not np.isfinite(z).all():
        raise ValueError("Expected finite nonempty (scenes, time, patches, channels)")
    n, t, p, d = z.shape
    pooled = float(z.reshape(-1, d).var(axis=0).mean())
    scene = float(z.var(axis=0).mean()) if n >= 2 else None
    temporal = float(z.var(axis=1).mean()) if t >= 2 else None
    spatial = float(z.mean(axis=(0, 1)).var(axis=0).mean())
    return {"n_trajectories": n, "n_frames": t, "n_patches": p, "n_channels": d,
            "across_trajectory_variance_fixed_time_patch_channel": scene,
            "within_trajectory_temporal_variance_fixed_patch_channel": temporal,
            "spatial_template_variance": spatial,
            "pooled_channel_variance": pooled,
            "scene_to_pooled_variance_ratio": (scene / pooled
                if scene is not None and pooled > 1e-12 else None),
            "spatial_template_fraction_of_pooled_variance": (spatial / pooled
                if pooled > 1e-12 else None),
            "interpretation": "Low scene variance with high spatial-template variance is "
                "consistent with a positional shortcut; it is not proof of collapse. "
                "Check diversity of observed scenes and prediction/action diagnostics. "
                "Relative times are sampled segment positions, not synchronized physics."}


def dynamics_diagnostics(observations, prediction, zero_prediction, history_length):
    """Within-checkpoint comparison with persistence and a zero-action rollout.

    Inputs are (T,P,D), (P,D), (P,D). 'Observed change' is change in this
    encoder's dataset embeddings, not a ground-truth physical-state metric.
    Undefined ratios on a zero observed change are null, not zero success/error.
    """
    tensors = [torch.as_tensor(value).detach().to(dtype=torch.float64)
               for value in (observations, prediction, zero_prediction)]
    z, predicted, zero = tensors
    if z.ndim != 3 or min(z.shape) < 1 or not 1 <= history_length < z.shape[0]:
        raise ValueError("Expected (time, patches, channels) with observed future frames")
    if predicted.shape != z.shape[1:] or zero.shape != predicted.shape:
        raise ValueError("Predictions must have shape (patches, channels)")
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("Non-finite dynamics probe features")
    persistence = float((z[-1] - z[history_length-1]).square().mean())
    model_error = float((predicted-z[-1]).square().mean())
    action_response = float((predicted-zero).square().mean().sqrt())
    change = persistence ** .5
    defined = persistence > 1e-12
    return {"observed_change_rms": change,
            "within_trajectory_temporal_variance_fixed_patch_channel": float(
                z.var(dim=0, unbiased=False).mean()),
            "persistence_comparison_defined": defined,
            "model_to_persistence_mse_ratio": model_error / persistence if defined else None,
            "model_beats_persistence": model_error < persistence if defined else None,
            "action_response_to_observed_change_ratio": action_response / change if defined else None}


@torch.no_grad()
def latent_rollout(wm, history, actions):
    """Linked latent history; actions include history actions, exactly as wm.rollout."""
    if wm.action_conditioning != "adaln":
        raise ValueError("The frozen latent probe currently supports AdaLN LpWM/LeWM only")
    if actions.shape[1] < history.shape[1]:
        raise ValueError("At least one future action is required")
    prediction, _ = wm.rollout_from_latent({"visual": history}, actions)
    return prediction["visual"]


def fit_bases(observed_states, goal_residuals, seed=0):
    """Channel bases: centered state PCA, uncentered residual SVD, random Haar.

    Residual centering would discard a real part of the squared goal cost.
    Each input row is one patch. No test goal may enter either fit matrix.
    """
    x = np.asarray(observed_states, dtype=np.float64)
    r = np.asarray(goal_residuals, dtype=np.float64)
    if x.ndim != 2 or r.ndim != 2 or x.shape[1] != r.shape[1]:
        raise ValueError("Expected two row matrices with the same channel dimension")
    if not np.isfinite(x).all() or not np.isfinite(r).all():
        raise ValueError("Non-finite calibration features")
    x = x - x.mean(axis=0)
    _, pca = np.linalg.eigh(x.T @ x / max(len(x), 1))
    _, goal = np.linalg.eigh(r.T @ r / max(len(r), 1))
    random, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(x.shape[1], x.shape[1])))
    return {"state_pca": pca[:, ::-1].copy(), "goal_residual": goal[:, ::-1].copy(),
            "random": random}


def compression_rows(residuals, bases, ranks):
    """Finite candidate certificate; not an out-of-sample/simulator bound.

    R has (goals, candidates, patches, channels). Scores retain denominator D,
    hence 0 <= full - projected and finite-set regret <= max discarded energy.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim != 4 or not np.isfinite(residuals).all():
        raise ValueError("Expected finite (goals, candidates, patches, channels)")
    d = residuals.shape[-1]
    full = np.mean(residuals**2, axis=(-2, -1))
    rows = []
    for name, basis in bases.items():
        if not np.allclose(basis.T @ basis, np.eye(d), atol=2e-6):
            raise ValueError("Basis must be orthonormal and complete")
        for rank in sorted(set(ranks) | {d}):
            if not 1 <= rank <= d:
                continue
            score = np.sum((residuals @ basis[:, :rank])**2, axis=-1).mean(axis=-1) / d
            for goal in range(len(full)):
                chosen = int(np.argmin(score[goal]))
                regret = float(full[goal, chosen] - full[goal].min())
                tail = float(np.max(full[goal] - score[goal]))
                rows.append({"basis": name, "rank": rank, "dimension": d, "goal": goal,
                             "relative_cost_mae": float(np.mean(abs(full[goal]-score[goal])) /
                                                        max(float(full[goal].mean()), 1e-12)),
                             "chosen_candidate": chosen, "full_best_candidate": int(full[goal].argmin()),
                             "finite_set_regret": regret, "finite_set_tail_bound": max(tail, 0),
                             "certificate_pass": bool(regret <= tail + 1e-9)})
    return rows


@torch.no_grad()
def locality_probe(predictor, history, action_embeddings, epsilons=(1e-3, 1e-2), seed=0):
    """Actual central differences, not the router's straight-through gradient.

    One direction per source patch at the final history frame. Report support
    switches separately: a graph recorded at one point is not a uniform graph.
    """
    if not hasattr(predictor, "generator_graph"):
        return {"supported": False, "reason": "Predictor has no generator graph"}
    if history.shape[0] != 1:
        raise ValueError("Use one sample; generator_graph records sample zero")
    old = predictor.record_graph
    predictor.record_graph = True
    try:
        predictor(history, action_embeddings)
        base = {k: v.clone() for k, v in predictor.generator_graph().items()}
        adjacency = (base["edge_support"] & base["law_support"].unsqueeze(-1)).any(0)
        adjacency.fill_diagonal_(True)  # identity, self routing and temporal terms
        p, d = history.shape[-2:]
        rng = torch.Generator(device=history.device).manual_seed(seed)
        directions = torch.randn(p, d, device=history.device, generator=rng, dtype=history.dtype)
        directions /= directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = max(float(history.square().mean().sqrt()), 1e-2)
        rows = []
        for relative_epsilon in epsilons:
            epsilon = relative_epsilon * scale
            off_energy = total_energy = 0.0
            switches = 0
            stable_off = stable_total = 0.0
            for source in range(p):
                bump = torch.zeros_like(history)
                bump[0, -1, source] = epsilon * directions[source]
                plus = predictor(history + bump, action_embeddings)[0, -1]
                graph_plus = {k: v.clone() for k, v in predictor.generator_graph().items()}
                minus = predictor(history - bump, action_embeddings)[0, -1]
                graph_minus = predictor.generator_graph()
                switched = any(not torch.equal(base[k], graph_plus[k]) or
                               not torch.equal(base[k], graph_minus[k]) for k in base)
                switches += int(switched)
                energy = ((plus - minus) / (2*epsilon)).square().sum(-1)
                off = float(energy[~adjacency[:, source]].sum())
                total = float(energy.sum())
                off_energy += off
                total_energy += total
                if not switched:
                    stable_off += off
                    stable_total += total
            rows.append({"relative_epsilon": relative_epsilon, "epsilon": epsilon,
                         "sources": p, "sources_with_support_switch": switches,
                         "off_graph_energy_fraction": off_energy / max(total_energy, 1e-30),
                         "stable_off_graph_energy_fraction": (stable_off / max(stable_total, 1e-30)
                                                               if switches < p else None)})
        return {"supported": True, "graph_density_with_self": float(adjacency.float().mean()),
                "scope": "one-step, sampled directions, not a global Lipschitz certificate", "rows": rows}
    finally:
        predictor.record_graph = old


def paired_bootstrap(a, b, seed=0, draws=10000):
    """Goal-paired descriptive CI; training-seed uncertainty is a separate level."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("Paired, nonempty one-dimensional observations required")
    diff = b-a
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(len(diff), size=(draws, len(diff)))].mean(1)
    return {"n_goals": len(diff), "delta_b_minus_a": float(diff.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist()}
