import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(alpha, base, mode="last", basis_path=None, basis_key="goal_residual", rank=None):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")
    basis = None
    if basis_path is not None:
        with np.load(basis_path, allow_pickle=False) as archive:
            matrix = archive[basis_key].copy()
        if rank is None or not 1 <= rank <= matrix.shape[1]:
            raise ValueError("A valid projection rank is required with basis_path")
        matrix = matrix[:, :rank]
        if not np.isfinite(matrix).all() or not np.allclose(matrix.T @ matrix, np.eye(rank), atol=2e-5):
            raise ValueError("Objective projection must have orthonormal columns")
        basis = torch.from_numpy(matrix)

    def visual_error(pred, target):
        difference = pred - target
        if basis is None:
            return difference.square().mean(dim=-1)
        if difference.shape[-1] != basis.shape[0]:
            raise ValueError("Projection channel dimension does not match this checkpoint")
        projected = difference @ basis.to(device=difference.device, dtype=difference.dtype)
        # Preserve full-space MSE scale (D, not rank). This matters with proprio.
        return projected.square().sum(dim=-1) / difference.shape[-1]

    def objective_fn_last(z_obs_pred, z_obs_tgt):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        visual = visual_error(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"])
        loss_visual = visual.mean(dim=tuple(range(1, visual.ndim)))
        loss = loss_visual
        if "proprio" in z_obs_pred and "proprio" in z_obs_tgt:
            loss_proprio = metric(
                z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]
            ).mean(dim=tuple(range(1, z_obs_pred["proprio"].ndim)))
            loss = loss + alpha * loss_proprio
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        visual = visual_error(z_obs_pred["visual"], z_obs_tgt["visual"])
        loss_visual = visual.mean(dim=tuple(range(2, visual.ndim)))
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss = loss_visual
        if "proprio" in z_obs_pred and "proprio" in z_obs_tgt:
            loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
                dim=tuple(range(2, z_obs_pred["proprio"].ndim))
            )
            loss_proprio = (loss_proprio * coeffs).mean(dim=1)
            loss = loss + alpha * loss_proprio
        return loss

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError
