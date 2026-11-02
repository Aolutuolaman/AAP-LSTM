"""Optional integration checks using checkpoints supplied outside the repository."""
import json
import os
from pathlib import Path
import unittest
import numpy as np
from aap_lstm.cli import sha256_file
from aap_lstm.tokens import encode

CHECKPOINT_DIR = os.environ.get("AAP_CHECKPOINT_DIR")


@unittest.skipUnless(CHECKPOINT_DIR, "Set AAP_CHECKPOINT_DIR to test private reference checkpoints.")
class CheckpointTests(unittest.TestCase):
    def test_all_reference_weights_load_and_predict(self):
        from aap_lstm.model import load_predictor
        import tensorflow as tf
        registry_path = Path(__file__).resolve().parents[1] / "aap_lstm" / "checkpoints.json"
        registry = json.loads(registry_path.read_text())["checkpoints"]
        inputs = encode(["UAR", "UND", "UCQ"])
        for alias, entry in registry.items():
            with self.subTest(model=alias):
                tf.keras.backend.clear_session()
                checkpoint = Path(CHECKPOINT_DIR) / entry["filename"]
                self.assertEqual(sha256_file(checkpoint), entry["sha256"])
                model = load_predictor(checkpoint)
                probabilities = model(inputs, training=False).numpy()
                self.assertEqual(probabilities.shape, (3, 3, 23))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue((probabilities >= 0).all())
                np.testing.assert_allclose(probabilities.sum(axis=-1), 1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
