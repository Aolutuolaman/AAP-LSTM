import unittest
import numpy as np
from aap_lstm.sampling import SamplingConfig, sample_candidates, temperature_probabilities
from aap_lstm.tokens import VOCABULARY, START, END, PAD, encode, decode_candidate


class SamplingTests(unittest.TestCase):
    def test_checkpoint_token_contract(self):
        self.assertEqual(VOCABULARY, "ARNDCEQGHILKMFPSTWYVUBZ")
        self.assertEqual((START, PAD, END), (20, 21, 22))
        self.assertEqual(encode(["UAR"])[0].argmax(axis=1).tolist(), [20, 0, 1])
        self.assertEqual(decode_candidate([START, 0, END, PAD, 1]), "A")

    def test_bad_input_fails_without_tokenizer_loop(self):
        for sequences in ([], [""], ["AX"], ["A", "AA"]):
            with self.subTest(sequences=sequences), self.assertRaises(ValueError):
                encode(sequences)

    def test_temperature_preserves_zeros_and_normalizes(self):
        probabilities = np.zeros((1, 23))
        probabilities[0, :2] = [0.8, 0.2]
        scaled = temperature_probabilities(probabilities, 2)
        np.testing.assert_allclose(scaled[0, :2], [2/3, 1/3])
        self.assertEqual(scaled[0, 2:].sum(), 0)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                temperature_probabilities(probabilities, value)
        for invalid in (np.zeros((1, 23)), np.full((1, 23), np.nan), np.ones((1, 22)), -np.ones((1, 23))):
            with self.assertRaises(ValueError):
                temperature_probabilities(invalid, 1)

    def test_tiny_temperature_is_finite(self):
        probabilities = np.ones((2, 23))
        probabilities[:, 0] = 10
        result = temperature_probabilities(probabilities, 1e-320)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[:, 0], [1, 1])

    def test_seeded_generation_unique_and_canonical(self):
        def predict(inputs):
            probabilities = np.zeros((inputs.shape[0], 23))
            probabilities[:, [0, 1, END]] = [0.4, 0.4, 0.2]
            return probabilities
        config = SamplingConfig(num=10, min_length=1, max_length=4,
            max_steps=4, batch_size=30, max_trials=3)
        first, counts = sample_candidates(predict, config)
        second, _ = sample_candidates(predict, config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(1 <= len(item) <= 4 and set(item) <= set("AR") for item in first))
        self.assertEqual(counts["returned"], 10)

    def test_invalid_tokens_exhaust_budget_without_leaking_to_output(self):
        def predict(inputs):
            probabilities = np.zeros((inputs.shape[0], 23))
            probabilities[:, PAD] = 1
            return probabilities
        config = SamplingConfig(num=1, min_length=1, max_length=2,
            max_steps=2, batch_size=3, max_trials=2)
        peptides, counts = sample_candidates(predict, config)
        self.assertEqual(peptides, [])
        self.assertEqual(counts["invalid_tokens"], 6)
        self.assertEqual(counts["trials"], 2)

    def test_invalid_config(self):
        for settings in ({"num": 0}, {"seed": -1}, {"max_length": 51}, {"min_length": 20, "max_length": 10}, {"temperature": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                SamplingConfig(**settings).validate()


if __name__ == "__main__":
    unittest.main()
