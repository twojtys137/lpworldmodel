"""Legacy metadata regression tests; runnable with standard-library unittest."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "lewm_checkpoint", Path(__file__).parents[1] / "scripts/lewm_checkpoint.py")
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


class LegacyMetadataTests(unittest.TestCase):
    def setUp(self):
        self.state = {"state_dict": {}, "optimizer_states": [{}], "lr_schedulers": [{}],
                      "loops": {"fit_loop": {"epoch_loop.batch_progress": {"is_last_batch": True}}},
                      "epoch": 5, "global_step": 83598,
                      "hyper_parameters": {"system.working_dir": "/content/lewm"},
                      "wandb": {"id": "run"}}
        self.recipe = {"output_model_name": "run", "seed": 3072, "trainer": {"max_epochs": 10}}

    def validate(self, recipe=None):
        metadata.validate_resume(self.state, "run", 10, 3072, legacy_recipe=recipe)

    def test_legacy_with_original_recipe(self):
        before = copy.deepcopy(self.state)
        self.validate(self.recipe)
        self.assertEqual(before, self.state)

    def test_missing_recipe_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "recipe mismatch"):
            self.validate()

    def test_wrong_recipe_fields_are_rejected(self):
        for recipe in ({**self.recipe, "seed": 0},
                       {**self.recipe, "output_model_name": "other"},
                       {**self.recipe, "trainer": {"max_epochs": 4}}):
            with self.subTest(recipe=recipe), self.assertRaisesRegex(ValueError, "recipe mismatch"):
                self.validate(recipe)

    def test_missing_or_wrong_wandb_is_rejected(self):
        for wandb in ({}, {"id": "other"}):
            self.state["wandb"] = wandb
            with self.assertRaisesRegex(ValueError, "W&B run ID"):
                self.validate(self.recipe)

    def test_conflicting_embedded_metadata_is_not_overwritten(self):
        self.state["hyper_parameters"]["seed"] = 0
        with self.assertRaisesRegex(ValueError, "recipe mismatch"):
            self.validate(self.recipe)

    def test_mid_epoch_and_model_only_are_still_rejected(self):
        self.state["loops"]["fit_loop"]["epoch_loop.batch_progress"]["is_last_batch"] = False
        with self.assertRaisesRegex(ValueError, "completed epoch"):
            self.validate(self.recipe)
        del self.state["optimizer_states"]
        with self.assertRaisesRegex(ValueError, "missing optimizer_states"):
            self.validate(self.recipe)

    def test_modern_recipe_needs_no_sidecar(self):
        self.state["hyper_parameters"] = {"output_model_name": "run", "seed": 3072,
                                           "trainer.max_epochs": 10}
        del self.state["wandb"]
        self.validate()

    def test_sidecar_is_bound_to_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.ckpt"
            path.write_bytes(b"unchanged checkpoint")
            self.assertIsNone(metadata.load_legacy_recipe(path))
            path.with_suffix(".recipe.json").write_text(json.dumps({
                "checkpoint_sha256": metadata.checkpoint_digest(path), "config": self.recipe}))
            self.assertEqual(metadata.load_legacy_recipe(path), self.recipe)
            path.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "does not match checkpoint"):
                metadata.load_legacy_recipe(path)
