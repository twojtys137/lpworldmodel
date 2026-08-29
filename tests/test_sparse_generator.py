import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.infojepa_modules import Embedder, LinearDynamicsPredictor, Link, SparseGeneratorPredictor
from models.visual_world_model import VWorldModel


class TinyPatchEncoder(nn.Module):
    name = "tiny_patch"
    emb_dim = 16
    latent_ndim = 2
    num_patches = 9
    patch_size = 1

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, self.emb_dim, kernel_size=1)

    def forward(self, image):
        tokens = F.adaptive_avg_pool2d(self.proj(image), (3, 3))
        return tokens.flatten(2).transpose(1, 2)


class SparseGeneratorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(2, 3, 9, 16)
        self.action = torch.randn(2, 3, 16)

    def test_sparse_ltv_has_exact_mode_budget(self):
        model = LinearDynamicsPredictor(
            input_dim=16,
            num_frames=3,
            mode="sparse_ltv",
            rank=8,
            topk=2,
            gate_balance_weight=0.1,
        )
        output = model(self.x, self.action)
        self.assertEqual(output.shape, self.x.shape)
        self.assertAlmostEqual(float(model.diagnostics()["generator_active_fraction"]), 0.25, places=6)

        loss = output.square().mean() + sum(model.auxiliary_losses().values())
        loss.backward()
        self.assertIsNotNone(model.gate.weight.grad)
        self.assertTrue(torch.isfinite(model.gate.weight.grad).all())

    def test_relational_generator_has_exact_law_and_edge_budgets(self):
        model = SparseGeneratorPredictor(
            input_dim=16,
            num_frames=3,
            num_patches=9,
            num_laws=4,
            law_rank=6,
            law_topk=2,
            edge_topk=3,
            query_chunk_size=4,
            record_graph=True,
        )
        output = model(self.x, self.action)
        graph = model.generator_graph()

        self.assertEqual(output.shape, self.x.shape)
        self.assertIsNone(model.lags)  # no learnable dense dynamics bypass by default
        self.assertEqual(graph["law_support"].shape, (4, 9))
        self.assertEqual(graph["edge_support"].shape, (4, 9, 9))
        self.assertTrue(torch.equal(graph["law_support"].sum(dim=0), torch.full((9,), 2)))
        self.assertTrue(torch.equal(graph["edge_support"].sum(dim=-1), torch.full((4, 9), 3)))
        self.assertAlmostEqual(float(model.diagnostics()["generator_active_fraction"]), 0.5, places=6)
        self.assertAlmostEqual(float(model.diagnostics()["generator_edge_fraction"]), 1.0 / 3.0, places=6)

    def test_predictor_is_patch_permutation_equivariant(self):
        model = SparseGeneratorPredictor(
            input_dim=16,
            num_frames=3,
            num_patches=9,
            num_laws=4,
            law_rank=6,
            law_topk=2,
            edge_topk=3,
            query_chunk_size=4,
        ).eval()
        permutation = torch.randperm(self.x.shape[2])

        output = model(self.x, self.action)
        permuted_output = model(self.x[:, :, permutation], self.action)

        torch.testing.assert_close(permuted_output, output[:, :, permutation], rtol=1e-5, atol=1e-6)

    def test_dense_generator_control_activates_full_support(self):
        model = SparseGeneratorPredictor(
            input_dim=16,
            num_frames=3,
            num_patches=9,
            num_laws=4,
            law_rank=6,
            law_topk=4,
            edge_topk=9,
            record_graph=True,
        )
        model(self.x, self.action)
        graph = model.generator_graph()
        self.assertTrue(graph["law_support"].all())
        # Self edges are excluded; all eight remaining sources are active.
        self.assertTrue(torch.equal(graph["edge_support"].sum(dim=-1), torch.full((4, 9), 8)))

    def test_chunking_does_not_change_the_operator(self):
        chunked = SparseGeneratorPredictor(
            input_dim=16,
            num_frames=3,
            num_patches=9,
            num_laws=4,
            law_rank=6,
            law_topk=2,
            edge_topk=3,
            query_chunk_size=3,
        ).eval()
        unchunked = SparseGeneratorPredictor(
            input_dim=16,
            num_frames=3,
            num_patches=9,
            num_laws=4,
            law_rank=6,
            law_topk=2,
            edge_topk=3,
            query_chunk_size=9,
        ).eval()
        unchunked.load_state_dict(chunked.state_dict())

        torch.testing.assert_close(
            chunked(self.x, self.action),
            unchunked(self.x, self.action),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_state_is_dense_and_signed(self):
        model = SparseGeneratorPredictor(
            input_dim=16,
            num_frames=1,
            num_patches=9,
            num_laws=4,
            law_rank=6,
            law_topk=1,
            edge_topk=2,
        )
        negative_state = -torch.ones(1, 1, 9, 16)
        output = model(negative_state, torch.zeros(1, 1, 16))
        self.assertTrue((output < 0).all())
        self.assertFalse(hasattr(model, "slots"))

    def test_world_model_adds_generator_loss_and_diagnostics(self):
        predictor = SparseGeneratorPredictor(
            input_dim=16,
            num_frames=3,
            num_patches=9,
            num_laws=4,
            law_rank=6,
            law_topk=2,
            edge_topk=3,
            gate_balance_weight=0.1,
        )
        model = VWorldModel(
            image_size=12,
            num_hist=3,
            num_pred=1,
            encoder=TinyPatchEncoder(),
            proprio_encoder=nn.Identity(),
            action_encoder=Embedder(in_chans=2, emb_dim=16),
            decoder=None,
            predictor=predictor,
            action_conditioning="adaln",
            train_encoder=True,
            train_predictor=True,
            train_decoder=False,
            regularizer=None,
            link=Link("identity"),
        )
        obs = {"visual": torch.randn(2, 4, 3, 12, 12), "proprio": torch.zeros(2, 4, 1)}
        action = torch.randn(2, 4, 2)

        _, _, _, loss, components = model(obs, action)
        self.assertIn("generator_balance_loss", components)
        self.assertIn("generator_active_fraction", components)
        self.assertAlmostEqual(float(components["l0_frac"]), 1.0, places=6)
        loss.backward()
        self.assertIsNotNone(predictor.state_gate.weight.grad)


if __name__ == "__main__":
    unittest.main()
