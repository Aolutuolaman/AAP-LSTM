"""Checkpoint vocabulary. Token indices are part of the model interface."""
import numpy as np

AMINO_ACIDS = "ARNDCEQGHILKMFPSTWYV"
VOCABULARY = AMINO_ACIDS + "UBZ"
START, PAD, END = (VOCABULARY.index(token) for token in "UBZ")
_INDEX = {token: index for index, token in enumerate(VOCABULARY)}
_EYE = np.eye(len(VOCABULARY), dtype=np.float32)


def encode(sequences):
    """Encode equally sized strings; reject unknown symbols and empty input."""
    if not sequences or not sequences[0]:
        raise ValueError("At least one non-empty sequence is required.")
    width = len(sequences[0])
    if any(len(sequence) != width for sequence in sequences):
        raise ValueError("Sequences must have equal lengths.")
    try:
        indices = [[_INDEX[token] for token in sequence] for sequence in sequences]
    except KeyError as exc:
        raise ValueError(f"Unknown token: {exc.args[0]!r}") from exc
    return _EYE[np.asarray(indices, dtype=np.int64)]


def decode_candidate(indices):
    """Remove the leading start token and stop at the first end token."""
    if not len(indices) or indices[0] != START:
        raise ValueError("A sampled row must start with U.")
    tokens = []
    for index in indices[1:]:
        if index == END:
            break
        tokens.append(VOCABULARY[index])
    return "".join(tokens)
