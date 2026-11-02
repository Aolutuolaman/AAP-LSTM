"""Temperature sampling and candidate filtering for the checkpoint interface."""
from dataclasses import dataclass
import math
import numpy as np
from .tokens import AMINO_ACIDS, VOCABULARY, START, decode_candidate


@dataclass(frozen=True)
class SamplingConfig:
    num: int = 20
    min_length: int = 10
    max_length: int = 50
    temperature: float = 1.1
    batch_size: int = 200
    max_trials: int = 20
    seed: int = 71
    max_steps: int = 50

    def validate(self):
        for name in ("num", "min_length", "max_length", "batch_size", "max_trials", "seed", "max_steps"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer.")
        if self.num < 1 or self.batch_size < 1 or self.max_trials < 1:
            raise ValueError("num, batch_size and max_trials must be positive.")
        if not 1 <= self.min_length <= self.max_length <= self.max_steps <= 50:
            raise ValueError("Require 1 <= min_length <= max_length <= max_steps <= 50.")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be in [0, 4294967295].")


def temperature_probabilities(probabilities, temperature):
    """Apply temperature scaling, retaining exact zero-probability events."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(VOCABULARY):
        raise ValueError("Model predictions must have shape (batch, 23).")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Model predictions must be finite and non-negative.")
    if (probabilities.sum(axis=1) <= 0).any():
        raise ValueError("Each prediction must have positive probability mass.")
    with np.errstate(divide="ignore"):
        logits = np.log(probabilities)
    # Subtract before dividing to avoid overflow at very small temperatures.
    with np.errstate(over="ignore", under="ignore"):
        scaled = np.exp((logits - logits.max(axis=1, keepdims=True)) / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def sample_candidates(predict, config):
    """Return unique canonical peptides and counters within a fixed trial budget.

    predict receives a float32 (batch, time, 23) array and returns last-step
    probabilities of shape (batch, 23). All 23 tokens remain available during
    sampling. Rows continue for max_steps; the first Z terminates decoding.
    """
    config.validate()
    rng = np.random.RandomState(config.seed)
    eye = np.eye(len(VOCABULARY), dtype=np.float32)
    accepted = {}
    counts = {"trials": 0, "sampled_rows": 0, "invalid_tokens": 0,
              "outside_length": 0, "duplicates": 0}
    for trial in range(config.max_trials):
        rows = np.full((config.batch_size, config.max_steps + 1), START, dtype=np.int64)
        for step in range(config.max_steps):
            probabilities = temperature_probabilities(predict(eye[rows[:, :step + 1]]), config.temperature)
            if probabilities.shape[0] != config.batch_size:
                raise ValueError("Model returned an unexpected batch size.")
            rows[:, step + 1] = [rng.choice(len(VOCABULARY), p=row) for row in probabilities]
        counts["trials"] = trial + 1
        counts["sampled_rows"] += config.batch_size
        for row in rows:
            peptide = decode_candidate(row)
            if any(token not in AMINO_ACIDS for token in peptide):
                counts["invalid_tokens"] += 1
            elif not config.min_length <= len(peptide) <= config.max_length:
                counts["outside_length"] += 1
            elif peptide in accepted:
                counts["duplicates"] += 1
            else:
                accepted[peptide] = None
        if len(accepted) >= config.num:
            break
    peptides = list(accepted)[:config.num]
    counts["unique_eligible"] = len(accepted)
    counts["returned"] = len(peptides)
    return peptides, counts
