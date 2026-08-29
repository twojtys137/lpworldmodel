"""Shared building blocks for InfoJEPA.

Ported from the LeWM codebase (lucas-maes/le-wm, `module.py`). LeWM is the
baseline that the InfoJEPA / RDMReg method extends, so these blocks live under
the project name rather than the baseline's:
  - SIGReg            : anti-collapse regularizer (isotropic-Gaussian goodness-of-fit)
  - FeedForward / Attention / Block / ConditionalBlock / Transformer : transformer stack
  - ARPredictor      : autoregressive predictor with AdaLN-zero action conditioning
  - Embedder         : action encoder (Conv1d smoothing + MLP)
  - MLP              : projector / pred_proj head
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def modulate(x, shift, scale):
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU).

    Pushes the marginal distribution of the embeddings toward a standard
    isotropic Gaussian via an Epps-Pulley characteristic-function goodness-of-fit
    test on random 1D projections (Cramer-Wold). Parameterless (buffers only).
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        returns: scalar statistic averaged over projections and time
        """
        # sample random unit projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


class FeedForward(nn.Module):
    """FeedForward network used in Transformers."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with optional causal masking."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, causal=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.causal = causal
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, attn_mask=None):
        """
        x : (B, L, D)
        attn_mask : optional (L, L) bool/float mask (True/0 = attend). When given,
            it overrides `is_causal` (e.g. the block-causal mask used for patches).
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=drop
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=drop, is_causal=self.causal
            )
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, causal=True):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout, causal=causal
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, causal=False):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout, causal=causal
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
        causal=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout, causal=causal)
            )

    def forward(self, x, c=None, attn_mask=None):
        x = self.input_proj(x)

        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            if isinstance(block, Block):
                x = block(x, attn_mask=attn_mask)
            else:
                x = block(x, c, attn_mask=attn_mask)
        x = self.norm(x)

        x = self.output_proj(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (N, D)
        """
        return self.net(x)


class Embedder(nn.Module):
    """Action/proprio encoder: Conv1d (k=1) smoothing + MLP.

    Follows the InfoJEPA encoder-construction convention (`in_chans`, `emb_dim`),
    so it is a drop-in for the Hydra `action_encoder` / `proprio_encoder` groups.
    Exposes `.emb_dim` (consumed by train.py).
    """

    def __init__(self, in_chans=10, emb_dim=192, smoothed_dim=None, mlp_scale=4):
        super().__init__()
        smoothed_dim = smoothed_dim or in_chans
        self.emb_dim = emb_dim
        self.patch_embed = nn.Conv1d(in_chans, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        returns: (B, T, emb_dim)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction.

    Causal transformer with AdaLN-zero action conditioning. The LeWM `pred_proj`
    head is folded in as `output_proj` so it is owned (and optimized) by the
    predictor module.

    Works for any number of patch tokens per frame:
      - num_patches == 1 (CLS): one token per frame; plain temporal causality.
      - num_patches  > 1 (patch features): `num_patches` tokens per frame with a
        factorized (temporal + spatial) pos-embed and a block-causal mask
        (full attention within a frame and across past frames, causal over time).

    forward(x, c):
        x: (B, T, P, input_dim)   embeddings (P == num_patches)
        c: (B, T, act_dim==input_dim)   per-frame action embeddings (conditioning)
        returns: (B, T, P, output_dim)
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        num_patches=1,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        pred_proj_hidden=2048,
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        self.num_patches = num_patches
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames, 1, input_dim) * 0.02)
        self.spatial_pos = (
            nn.Parameter(torch.randn(1, 1, num_patches, input_dim) * 0.02)
            if num_patches > 1
            else None
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
            causal=True,
        )
        self.output_proj = MLP(
            input_dim=output_dim,
            hidden_dim=pred_proj_hidden,
            output_dim=output_dim,
            norm_fn=nn.LayerNorm,
        )

    @staticmethod
    def _block_causal_mask(T, P, device):
        """(T*P, T*P) bool mask, True = attend. Token (t,p) attends to (t',p')
        iff t' <= t (full attention within current + past frames). Frame-major
        flattening: token_index // P == frame_index."""
        frame_idx = torch.arange(T * P, device=device) // P
        return frame_idx[None, :] <= frame_idx[:, None]

    def forward(self, x, c):
        """
        x: (B, T, P, input_dim)
        c: (B, T, act_dim)
        """
        B, T, P, _ = x.shape
        x = x + self.temporal_pos[:, :T]
        if self.spatial_pos is not None:
            x = x + self.spatial_pos
        c = c.unsqueeze(2).expand(B, T, P, c.size(-1))  # broadcast action over patches

        x = rearrange(x, "b t p d -> b (t p) d")  # frame-major
        c = rearrange(c, "b t p a -> b (t p) a")
        x = self.dropout(x)

        attn_mask = self._block_causal_mask(T, P, x.device) if P > 1 else None
        x = self.transformer(x, c, attn_mask=attn_mask)

        x = self.output_proj(rearrange(x, "b l d -> (b l) d"))
        x = rearrange(x, "(b t p) d -> b t p d", b=B, t=T, p=P)
        return x


class LinearDynamicsPredictor(nn.Module):
    """(Action-conditioned) LINEAR dynamics predictor -- the linear-core rungs
    (LTI(1)/LTI(k)/MLP∘LTI/MLP∘LTV) of the predictor-complexity ladder. Used via the
    VWorldModel "adaln" path -- a misnomer that gates the from-scratch JEPA path (action
    passed separately, shared link on enc+pred outputs), NOT AdaLN conditioning: this
    predictor consumes the action linearly (P(a)z or Wz+Ba), not via AdaLN. Same forward(x, c)
    contract as ARPredictor (x=(B,T,P,D) codes, c=(B,T,D) action embeddings, returns
    (B,T,P,D)), and the map is LINEAR in the code. `action_linear`/`additive` use only the
    current frame z_t (no temporal attention); `var` is history-aware (last `num_frames`
    frames, a causal linear map) — the fair linear analog of the transformer's num_hist
    window, needed where one frame is not a Markov-sufficient state (e.g. pusht: velocity).
    output[:, t] predicts frame t+1. For adaln the model returns the PRE-link u; VWorldModel
    then applies the (fixed, non-learned) link, so the code evolves in the linked space.

    mode="action_linear" (excluded rung, config linear_pa): P(a) = diag(d(a)) + U(a) V(a)^T -- a
        per-action diagonal rescale plus a rank-r mixing, with d/U/V from small MLPs of the action.
    mode="additive"    (LTI(1), config linear_wb): z_{t+1} = W z_t + B a_t -- fully linear,
        single-frame, action additive.
    mode="var"         (LTI(k), config linear_var): z_{t+1} = sum_{k=0}^{H-1} A_k z_{t-k} + B a_t,
        H=num_frames (=num_hist). A first-order linear system on the STACKED history [z_t..z_{t-H+1}]
        (state-augmentation) -- can capture 2nd-order dynamics (velocity via z_t,z_{t-1}).
        CAUSAL: output[:,t] uses only frames <= t; missing lags (t<k, incl. rollout
        cold-start where T<H) are dropped (= zero, exact since A_k*0=0), mirroring the
        transformer's min(num_hist, L) growing context. H=1 reduces exactly to `additive`.
    mode="mlp_var"  (MLP∘LTI(k), config mlp_var): z_{t+1} = W ReLU(sum_{k} A_k z_{t-k} + B a_t) -- a
        1-hidden-layer MLP on the LTI(k) features. The nonlinear CORE is shared by dense & sparse
        (only the output link differs), so the dense/sparse comparison is complexity-matched
        (unlike `var`, where sparse gets a "free" RepReLU nonlinearity dense lacks). Sits between
        LTI(k) (linear) and Shallow-AdaLN (attention) on the complexity ladder.
    mode="ltv" (MLP∘LTV(k), config ltv): z_{t+1} = W ReLU(sum_k A_k(z_t) z_{t-k} + B(z_t) a_t)
        with A_k(z_t)/B(z_t) = fixed base + STATE-GATED low-rank correction (the input
        never emits a matrix, only a small gate g(z_t)=sigmoid(gate·z_t)∈R^r selects among fixed
        low-rank directions). rank r = # state-selected modes; U-factors 0-inited so it starts == mlp_var.
        Shared nonlinear core across dense/sparse (only the link differs). Between MLP∘LTI and Shallow-AdaLN.
    mode="sparse_ltv" is the same operator bank, but exactly `topk` of its `rank` modes are
        active per token/lag.  Sparsity therefore lives in the dynamics operator rather than
        in the representation.  Use it with patch features and the identity link.

    Initialised near the identity map (d(a)=1, low-rank=0 / W=I / lag0=I & older-lags=0, B=0)
    so the untrained predictor is z_{t+1}=z_t (mirrors AdaLN-zero); the invariance loss shapes it.
    EXCEPTION: `mlp_var` uses NON-identity muP fan_in-scaled init (the inner ReLU precludes a clean
    identity map on a signed code; init variance ~ 1/fan_in per the muP recipe).
    """

    def __init__(
        self,
        *,
        input_dim,
        output_dim=None,
        num_frames=1,
        num_patches=1,
        hidden_dim=None,
        mode="action_linear",
        rank=16,
        topk=2,
        gate_temperature=1.0,
        gate_balance_weight=0.0,
        act_hidden=256,
        **kwargs,
    ):
        super().__init__()
        D = input_dim
        out = output_dim or input_dim
        assert out == D, f"linear predictor assumes output_dim==input_dim (code space), got {out} vs {D}"
        assert mode in {"action_linear", "additive", "var", "mlp_var", "ltv", "sparse_ltv"}, (
            f"mode {mode} not supported"
        )
        self.mode = mode
        self.rank = rank
        self.topk = topk
        self.gate_temperature = gate_temperature
        self.gate_balance_weight = gate_balance_weight
        self._aux_losses = {}
        self._diagnostics = {}
        if mode == "additive":  # LTI(1): z' = W z + B a
            self.W = nn.Linear(D, D, bias=True)
            self.B = nn.Linear(D, D, bias=False)   # action embedding has dim D (adaln)
            nn.init.eye_(self.W.weight)
            nn.init.zeros_(self.W.bias)
            nn.init.zeros_(self.B.weight)           # AdaLN-zero style: no action effect at init
        elif mode == "var":
            self.n_lags = num_frames
            self.lags = nn.ModuleList([nn.Linear(D, D, bias=(k == 0)) for k in range(num_frames)])
            self.B = nn.Linear(D, D, bias=False)    # action embedding has dim D (adaln)
            nn.init.eye_(self.lags[0].weight); nn.init.zeros_(self.lags[0].bias)  # lag0 (z_t) = identity
            for k in range(1, num_frames):
                nn.init.zeros_(self.lags[k].weight)  # older frames ignored at init -> z' = z_t
            nn.init.zeros_(self.B.weight)            # no action effect at init
        elif mode == "mlp_var":
            self.n_lags = num_frames
            self.lags = nn.ModuleList([nn.Linear(D, D, bias=(k == 0)) for k in range(num_frames)])
            self.B = nn.Linear(D, D, bias=False)     # action embedding has dim D (adaln)
            self.W = nn.Linear(D, D, bias=True)      # nonlinear readout: z' = W ReLU(var_core)
            for lin in (*self.lags, self.B, self.W):
                nn.init.normal_(lin.weight, mean=0.0, std=D ** -0.5)  # muP hidden init N(0, 1/fan_in)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
        elif mode in {"ltv", "sparse_ltv"}:  # DATA-DEPENDENT VAR + nonlinear readout:
            r = rank
            assert 1 <= topk <= r, f"topk must be in [1, rank], got {topk} for rank={r}"
            assert gate_temperature > 0, "gate_temperature must be positive"
            self.n_lags = num_frames
            self.rank = r
            self.lags = nn.ModuleList([nn.Linear(D, D, bias=(k == 0)) for k in range(num_frames)])
            self.B = nn.Linear(D, D, bias=False)
            self.W = nn.Linear(D, D, bias=True)                                   # nonlinear readout
            self.Vlag = nn.ModuleList([nn.Linear(D, r, bias=False) for _ in range(num_frames)])  # down-proj
            self.Ulag = nn.ModuleList([nn.Linear(r, D, bias=False) for _ in range(num_frames)])  # up-proj (0-init)
            self.VB = nn.Linear(D, r, bias=False)
            self.UB = nn.Linear(r, D, bias=False)                                 # up-proj (0-init)
            self.gate = nn.Linear(D, (num_frames + 1) * r)                        # per-lag + B gates, from z_t
            for m in (*self.lags, self.B, self.W, *self.Vlag, self.VB, self.gate):
                nn.init.normal_(m.weight, mean=0.0, std=D ** -0.5)                # muP fan_in init
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            for m in (*self.Ulag, self.UB):
                nn.init.zeros_(m.weight)                                          # LoRA-style: LTV off at init
        else:
            self.to_d = nn.Sequential(
                nn.Linear(D, act_hidden), nn.SiLU(), nn.Linear(act_hidden, D)
            )
            self.to_uv = nn.Sequential(
                nn.Linear(D, act_hidden), nn.SiLU(), nn.Linear(act_hidden, 2 * D * rank)
            )
            nn.init.zeros_(self.to_d[-1].weight); nn.init.ones_(self.to_d[-1].bias)
            nn.init.zeros_(self.to_uv[-1].weight); nn.init.zeros_(self.to_uv[-1].bias)

    def auxiliary_losses(self):
        """Weighted routing losses produced by the most recent forward pass."""
        return self._aux_losses

    def diagnostics(self):
        """Detached routing statistics produced by the most recent forward pass."""
        return self._diagnostics

    def _ltv_gates(self, logits):
        soft = torch.sigmoid(logits / self.gate_temperature)
        if self.mode != "sparse_ltv":
            self._aux_losses = {}
            self._diagnostics = {}
            return soft

        indices = logits.topk(self.topk, dim=-1).indices
        support = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
        # Binary top-k in the forward pass, sigmoid gradients in the backward pass.
        gates = support.detach() - soft.detach() + soft

        usage = support.mean(dim=(0, 1, 2, 3))
        differentiable_usage = gates.mean(dim=(0, 1, 2, 3))
        usage_prob = usage / usage.sum().clamp_min(1e-8)
        if self.rank > 1:
            usage_entropy = -(usage_prob * usage_prob.clamp_min(1e-8).log()).sum() / math.log(self.rank)
        else:
            usage_entropy = usage.new_tensor(1.0)
        balance = (differentiable_usage - float(self.topk) / self.rank).square().mean()
        switch = (
            (support[:, 1:] - support[:, :-1]).abs().mean()
            if support.shape[1] > 1
            else support.new_zeros(())
        )
        self._aux_losses = {}
        if self.gate_balance_weight > 0:
            self._aux_losses["generator_balance_loss"] = self.gate_balance_weight * balance
        self._diagnostics = {
            "generator_active_fraction": support.mean().detach(),
            "generator_usage_entropy": usage_entropy.detach(),
            "generator_switch_rate": switch.detach(),
        }
        return gates

    def forward(self, x, c):
        """x: (B,T,P,D) codes; c: (B,T,D) action embeddings -> (B,T,P,D)."""
        B, T, P, D = x.shape
        if self.mode == "var":
            out = self.lags[0](x)                       # A z_t                       (B,T,P,D)
            for k in range(1, self.n_lags):
                if k >= T:
                    break                               # no z_{t-k} available (T<num_hist, e.g. cold-start)
                xk = self.lags[k](x[:, : T - k])        # lag-k on frames [0 .. T-1-k]
                out = out + torch.cat([x.new_zeros(B, k, P, D), xk], dim=1)  # place at output positions [k:]
            return out + self.B(c).unsqueeze(2)         # + B a_t   (B,T,1,D) broadcast over patches
        if self.mode == "mlp_var":
            u = self.lags[0](x)                          # A z_t                       (B,T,P,D)
            for k in range(1, self.n_lags):
                if k >= T:
                    break                                # z_{t-k} unavailable (cold-start T<num_hist)
                xk = self.lags[k](x[:, : T - k])
                u = u + torch.cat([x.new_zeros(B, k, P, D), xk], dim=1)
            u = u + self.B(c).unsqueeze(2)               # + B a_t
            return self.W(torch.relu(u))
        if self.mode in {"ltv", "sparse_ltv"}:  # state-gated low-rank VAR core -> ReLU -> readout W
            r = self.rank
            gate_logits = self.gate(x).view(B, T, P, self.n_lags + 1, r)
            g = self._ltv_gates(gate_logits)                                    # dense or exact top-k gates
            core = self.lags[0](x) + self.Ulag[0](g[..., 0, :] * self.Vlag[0](x))   # k=0 (no shift)
            for k in range(1, self.n_lags):
                if k >= T:
                    break                                # z_{t-k} unavailable (cold-start T<num_hist)
                xk = x[:, : T - k]                       # z_{t-k} feeding output positions [k:]
                base_k = self.lags[k](xk)                # A_k z_{t-k}
                corr_k = self.Ulag[k](g[:, k:, :, k, :] * self.Vlag[k](xk))   # U_k( g_k(z_t) ⊙ V_k z_{t-k} )
                core = core + torch.cat([x.new_zeros(B, k, P, D), base_k + corr_k], dim=1)
            Ba = self.B(c).unsqueeze(2)                  # base B a_t                 (B,T,1,D)
            corr_B = self.UB(g[..., self.n_lags, :] * self.VB(c).unsqueeze(2))   # U_B( g_B(z_t) ⊙ V_B a_t )
            core = core + Ba + corr_B
            return self.W(torch.relu(core))              # PRE-link; VWorldModel applies identity/reprelu
        if self.mode == "additive":
            Wz = self.W(x)                          # (B,T,P,D)
            Ba = self.B(c).unsqueeze(2)             # (B,T,1,D) broadcast over patches
            return Wz + Ba
        d = self.to_d(c)                            # (B,T,D)
        uv = self.to_uv(c).view(B, T, D, 2 * self.rank)
        U, V = uv[..., : self.rank], uv[..., self.rank :]   # (B,T,D,r) each
        diag_part = d.unsqueeze(2) * x              # (B,T,P,D)  diag(d) z
        Vtz = torch.einsum("btdr,btpd->btpr", V, x)         # V^T z  (B,T,P,r)
        lowrank = torch.einsum("btdr,btpr->btpd", U, Vtz)   # U (V^T z)  (B,T,P,D)
        return diag_part + lowrank


class SparseGeneratorPredictor(nn.Module):
    """Dense patch state with sparse, shared relational dynamics generators.

    The state ``x`` remains a dense signed tensor ``(B,T,P,D)``.  A bank of
    ``num_laws`` shared low-rank operators routes messages between patches; exactly
    ``law_topk`` laws and ``edge_topk`` source patches are active for each target
    patch.  With the default fixed identity base, every learned state change --
    relational, temporal, or action-induced -- passes through that sparse bank.
    No slot or object labels are introduced.  Because every routing and
    message function is shared over patch indices, the predictor is equivariant to
    a simultaneous permutation of the patch axis.

    A chunked pairwise router bounds activation memory at O(B*T*M*C*P), where C is
    ``query_chunk_size``, while keeping the exact same function as the unchunked
    implementation.  This matters for Colab-scale experiments with 64+ patches.
    """

    def __init__(
        self,
        *,
        input_dim,
        output_dim=None,
        num_frames=1,
        num_patches=1,
        hidden_dim=None,
        num_laws=8,
        law_rank=32,
        law_topk=2,
        edge_topk=8,
        query_chunk_size=16,
        gate_temperature=1.0,
        gate_balance_weight=0.0,
        base_mode="identity",
        exclude_self_edges=True,
        record_graph=False,
        **kwargs,
    ):
        super().__init__()
        D = input_dim
        out = output_dim or D
        assert out == D, f"sparse generator assumes output_dim==input_dim, got {out} vs {D}"
        assert num_laws >= 1 and 1 <= law_topk <= num_laws
        assert law_rank >= 1 and edge_topk >= 1 and query_chunk_size >= 1
        assert gate_temperature > 0
        assert base_mode in {"identity", "learned"}

        self.num_frames = num_frames
        self.num_patches = num_patches
        self.num_laws = num_laws
        self.law_rank = law_rank
        self.law_topk = law_topk
        self.edge_topk = edge_topk
        self.query_chunk_size = query_chunk_size
        self.gate_temperature = gate_temperature
        self.gate_balance_weight = gate_balance_weight
        self.base_mode = base_mode
        self.exclude_self_edges = exclude_self_edges
        self.record_graph = record_graph

        # The default base is a fixed identity residual, so no dense dynamics path
        # can bypass the sparse law bank. A learned dense VAR is retained only as an
        # explicit ablation/control.
        if base_mode == "learned":
            self.lags = nn.ModuleList([nn.Linear(D, D, bias=(k == 0)) for k in range(num_frames)])
            self.B = nn.Linear(D, D, bias=False)
            nn.init.eye_(self.lags[0].weight)
            nn.init.zeros_(self.lags[0].bias)
            for lag in self.lags[1:]:
                nn.init.zeros_(lag.weight)
            nn.init.zeros_(self.B.weight)
        else:
            self.lags = None
            self.B = None

        # Shared law router and low-rank operator bank.
        self.state_norm = nn.LayerNorm(D)
        self.action_norm = nn.LayerNorm(D)
        self.q_proj = nn.Linear(D, num_laws * law_rank, bias=False)
        self.k_proj = nn.Linear(D, num_laws * law_rank, bias=False)
        self.v_proj = nn.Linear(D, num_laws * law_rank, bias=False)
        self.history_value = nn.ModuleList(
            [nn.Linear(D, num_laws * law_rank, bias=False) for _ in range(num_frames - 1)]
        )
        self.action_value = nn.Linear(D, num_laws * law_rank, bias=False)
        self.state_gate = nn.Linear(D, num_laws)
        self.action_gate = nn.Linear(D, num_laws, bias=False)
        self.law_up = nn.Parameter(torch.empty(num_laws, law_rank, D))

        for module in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            *self.history_value,
            self.action_value,
            self.state_gate,
            self.action_gate,
        ):
            nn.init.normal_(module.weight, mean=0.0, std=D ** -0.5)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        # Near-identity, but not exactly zero: the router receives gradient on step one.
        nn.init.normal_(self.law_up, mean=0.0, std=1e-3 * law_rank ** -0.5)

        self._aux_losses = {}
        self._diagnostics = {}
        self._last_graph = None

    def auxiliary_losses(self):
        return self._aux_losses

    def diagnostics(self):
        return self._diagnostics

    def generator_graph(self):
        """Routing support for sample 0 at the final time, or ``None`` if disabled."""
        return self._last_graph

    @staticmethod
    def _straight_through_sparse_distribution(logits, topk, temperature):
        dense = torch.softmax(logits / temperature, dim=-1)
        indices = logits.topk(topk, dim=-1).indices
        support = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
        sparse = dense * support
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # Sparse normalized values forward; dense-softmax gradients backward.
        routed = sparse.detach() - dense.detach() + dense
        return routed, support

    def _dense_temporal_core(self, x, c):
        if self.base_mode == "identity":
            return x
        B, T, P, D = x.shape
        core = self.lags[0](x)
        for k in range(1, self.num_frames):
            if k >= T:
                break
            lagged = self.lags[k](x[:, : T - k])
            core = core + torch.cat([x.new_zeros(B, k, P, D), lagged], dim=1)
        return core + self.B(c).unsqueeze(2)

    def forward(self, x, c):
        """x: (B,T,P,D), c: (B,T,D) -> dense signed (B,T,P,D)."""
        B, T, P, D = x.shape
        M, R = self.num_laws, self.law_rank
        if c.shape[:2] != (B, T) or c.shape[-1] != D:
            raise ValueError(f"expected action shape {(B, T, D)}, got {tuple(c.shape)}")

        state = self.state_norm(x)
        action = self.action_norm(c)
        q = self.q_proj(state).view(B, T, P, M, R).permute(0, 1, 3, 2, 4)
        k = self.k_proj(state).view(B, T, P, M, R).permute(0, 1, 3, 2, 4)
        v = self.v_proj(state).view(B, T, P, M, R).permute(0, 1, 3, 2, 4)
        temporal_value = x.new_zeros(B, T, M, P, R)
        for lag, projection in enumerate(self.history_value, start=1):
            if lag >= T:
                break
            delta = x[:, lag:] - x[:, : T - lag]
            delta_value = projection(delta).view(B, T - lag, P, M, R).permute(0, 1, 3, 2, 4)
            padding = x.new_zeros(B, lag, M, P, R)
            temporal_value = temporal_value + torch.cat([padding, delta_value], dim=1)
        action_value = self.action_value(action).view(B, T, M, R).unsqueeze(3)

        law_logits = self.state_gate(state) + self.action_gate(action).unsqueeze(2)
        law_distribution, law_support = self._straight_through_sparse_distribution(
            law_logits, self.law_topk, self.gate_temperature
        )  # (B,T,P,M)
        law_gate = law_distribution.permute(0, 1, 3, 2)  # (B,T,M,P)

        max_edges = P - 1 if self.exclude_self_edges and P > 1 else P
        edge_topk = min(self.edge_topk, max_edges)
        chunks = []
        graph_chunks = []
        source_index = torch.arange(P, device=x.device)
        for start in range(0, P, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, P)
            q_chunk = q[:, :, :, start:stop]
            scores = torch.einsum("btmcr,btmpr->btmcp", q_chunk, k) / math.sqrt(R)
            if self.exclude_self_edges and P > 1:
                query_index = torch.arange(start, stop, device=x.device)
                self_mask = query_index[:, None] == source_index[None, :]
                scores = scores.masked_fill(self_mask.view(1, 1, 1, stop - start, P), -torch.inf)

            edge_weights, edge_support = self._straight_through_sparse_distribution(
                scores, edge_topk, self.gate_temperature
            )
            message = torch.einsum("btmcp,btmpr->btmcr", edge_weights, v)
            message = message + temporal_value[:, :, :, start:stop] + action_value
            message = torch.einsum("btmcr,mrd->btmcd", message, self.law_up)
            message = message * law_gate[:, :, :, start:stop].unsqueeze(-1)
            chunks.append(message.sum(dim=2))
            if self.record_graph:
                graph_chunks.append(edge_support[0, -1].detach().bool())

        relation_update = torch.cat(chunks, dim=2)
        output = self._dense_temporal_core(x, c) + relation_update

        usage = law_support.mean(dim=(0, 1, 2))
        usage_prob = usage / usage.sum().clamp_min(1e-8)
        if M > 1:
            usage_entropy = -(usage_prob * usage_prob.clamp_min(1e-8).log()).sum() / math.log(M)
        else:
            usage_entropy = usage.new_tensor(1.0)
        differentiable_usage = law_distribution.mean(dim=(0, 1, 2))
        balance = (differentiable_usage - 1.0 / M).square().mean()
        switch = (
            (law_support[:, 1:] - law_support[:, :-1]).abs().mean()
            if T > 1
            else law_support.new_zeros(())
        )
        self._aux_losses = {}
        if self.gate_balance_weight > 0:
            self._aux_losses["generator_balance_loss"] = self.gate_balance_weight * balance
        self._diagnostics = {
            "generator_active_fraction": law_support.mean().detach(),
            "generator_edge_fraction": x.new_tensor(float(edge_topk) / P),
            "generator_usage_entropy": usage_entropy.detach(),
            # Diagnostic only: never optimized, because fixed patch-index persistence is
            # not transport-covariant when objects move through the image.
            "generator_switch_rate": switch.detach(),
        }
        if self.record_graph:
            self._last_graph = {
                "law_support": law_support[0, -1].detach().bool().transpose(0, 1),
                "edge_support": torch.cat(graph_chunks, dim=1),
            }
        else:
            self._last_graph = None
        return output





def reprelu(x):
    """Straight-through "reparameterized ReLU": ReLU on the forward pass, GELU gradient on
    the backward pass. Forward value == ReLU(x) exactly (the detached ReLU/GELU terms cancel
    in value), so hard zeros are preserved; the backward gradient is GELU'(x), which is
    nonzero for x < 0 -- so a thresholded/zeroed coordinate (soft) or group (group) keeps
    receiving gradient instead of a dead zero. Drop-in for a max(., 0) whose zero region
    would otherwise stop learning."""
    return torch.relu(x).detach() + F.gelu(x) - F.gelu(x).detach()


def swd(z, target, num_projections=8192):
    """Sliced-Wasserstein-2 between z and target, shape (N, *batch, D).
    Dim 0 (N) is the SAMPLE axis (sorted to form the empirical 1-D CDF); the last dim
    (D) is the feature/support axis (random projections live in R^D, shared); any
    MIDDLE dims are extra BATCH axes (batched-matmul style) -> an independent SWD per
    slice, averaged. So features across the batch axes are NOT pooled as i.i.d. samples."""
    D = z.shape[-1]
    proj = torch.randn(num_projections, D, device=z.device)
    proj = proj / proj.norm(dim=1, keepdim=True)
    pz = torch.sort(z @ proj.T, dim=0).values  # (N, *batch, num_proj), sorted over N
    pt = torch.sort(target @ proj.T, dim=0).values
    return ((pz - pt) ** 2).mean()


def gng_unit_sigma(p):
    """Auto sigma so the zero-mean Generalized Gaussian GN_p has UNIT variance, for the
    Gamma-based sampler below: sigma = sqrt(Gamma(1/p)/Gamma(3/p)) / p^(1/p)
    (p=1 -> Laplace scale 1/sqrt2 -> var 1; p=2 -> sigma 1 -> standard normal)."""
    return math.sqrt(math.gamma(1.0 / p) / math.gamma(3.0 / p)) / (p ** (1.0 / p))


def sample_generalized_gaussian(shape, p=2.0, device="cpu"):
    """Unified base sampler: zero-mean unit-variance Generalized Gaussian GN_p.
    p=2 -> Gaussian, p=1 -> Laplace, general p via the Gamma trick. The link function
    h(.) is applied to this base (outside) to form the target -> identity/relu
    targets all come from one interface (see RDMReg)."""
    assert p > 0.0, "p must be > 0"
    sigma = gng_unit_sigma(p)
    if p == 1.0:
        return torch.distributions.Laplace(
            torch.tensor(0.0, device=device), torch.tensor(sigma, device=device)
        ).sample(shape)
    if p == 2.0:
        return sigma * torch.randn(shape, device=device)
    sign = torch.empty(shape, device=device).bernoulli_(0.5) * 2 - 1
    g = torch.distributions.Gamma(1.0 / p, 1.0).sample(shape).to(device)
    gn = sign * (p * g).pow(1.0 / p)
    return sigma * gn


class Link(nn.Module):
    """Architectural link function h(.) defining the representation space the predictor
    and planner operate in. Owned by VWorldModel and applied to encoder & predictor
    outputs (one shared instance -> tied threshold).

      identity       -> u                              (dense; pairs with gaussian target)
      relu           -> relu(u)                         (rectified; pairs with rectified-GG)
      reprelu        -> relu(u) fwd, gelu-grad bwd         (rectified, straight-through;
                        zeroed coords keep receiving gradient -- pairs with rectified-GG)
    """

    def __init__(self, kind="identity"):
        super().__init__()
        assert kind in ("identity", "relu", "reprelu"), f"link {kind} not supported"
        self.kind = kind

    def forward(self, u):
        if self.kind == "identity":
            return u
        if self.kind == "relu":
            return F.relu(u)
        return reprelu(u)


AGG_PATTERNS = {
    "btp": "b t p d -> (b t p) d",
    "b": "b t p d -> b (t p) d",
    "bp": "b t p d -> (b p) t d",
    "bt": "b t p d -> (b t) p d",
}


class RDMReg(nn.Module):
    """Sliced-Wasserstein distribution-matching regularizer (param-free).

    Unified target interface: always sample a zero-mean, unit-variance Generalized
    Gaussian base GN_p (p=2 Gaussian, p=1 Laplace, auto sigma) and apply the SAME link
    h(.) that produced the features. So the target is self-consistent with the
    representation space:
      identity link       -> target = GN_p                 (dense)
      relu / reprelu link -> target = relu(GN_p)           (rectified GG)
    The target is detached (a reference).

    `agg` chooses how (b,t,p,d) features map onto SWD's (sample, *batch, feature) axes
    (see AGG_PATTERNS). NB: the SWD scale is O(1), so `reg_weight` should be large
    (~100), unlike SIGReg's N-scaled statistic.
    """

    def __init__(self, target_p=2.0, num_projections=8192, agg="btp", mu=0.0):
        super().__init__()
        assert agg in AGG_PATTERNS, f"agg {agg} not in {list(AGG_PATTERNS)}"
        self.target_p = target_p          # GN_p shape: 2 -> Gaussian, 1 -> Laplace
        self.num_projections = num_projections
        self.agg = agg
        self.mu = mu                      # target-mean shift: with the relu link, mu<0 pushes

    def reg_loss(self, z, link=None):
        """z: linked encoder features (b, t, p, d). Reshaped per `agg` into
        (sample, *batch, feature). `link` is the model's link h(.); target is h(GN_p)."""
        z = rearrange(z, AGG_PATTERNS[self.agg])
        base = sample_generalized_gaussian(z.shape, self.target_p, z.device)
        if self.mu != 0.0:
            base = base + self.mu
        target = (link(base) if link is not None else base).detach()
        return swd(z, target, self.num_projections)
