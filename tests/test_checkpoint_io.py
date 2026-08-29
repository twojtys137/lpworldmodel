import unittest
from unittest import mock

from utils import load_trusted_checkpoint


class TrustedCheckpointTest(unittest.TestCase):
    def test_explicitly_disables_weights_only(self):
        sentinel = object()
        with mock.patch("utils.torch.load", return_value=sentinel) as torch_load:
            result = load_trusted_checkpoint("model_latest.pth", map_location="cpu")

        self.assertIs(result, sentinel)
        torch_load.assert_called_once_with(
            "model_latest.pth", map_location="cpu", weights_only=False
        )


if __name__ == "__main__":
    unittest.main()
