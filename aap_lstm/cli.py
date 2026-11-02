"""Command-line entry point for checkpoint-based sequence generation."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata, resources
import json
import os
from pathlib import Path
import platform
import sys
from . import __version__
from .sampling import SamplingConfig, sample_candidates
from .tokens import VOCABULARY


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser():
    result = argparse.ArgumentParser(description="Generate peptide candidates from an externally supplied AAP-LSTM checkpoint.")
    selection = result.add_mutually_exclusive_group()
    selection.add_argument("--model", choices=("v1", "v2", "ang2", "bfgf", "mmps", "vegfr2"), help="Checkpoint alias (default: v1).")
    selection.add_argument("--weights", type=Path, help="Explicit legacy .hdf5 or .h5 weights path.")
    result.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory containing checkpoints (default: ./models).")
    result.add_argument("--num", type=int, default=20)
    result.add_argument("--min-length", type=int, default=10)
    result.add_argument("--max-length", type=int, default=50)
    result.add_argument("--temperature", type=float, default=1.1)
    result.add_argument("--batch-size", type=int, default=200)
    result.add_argument("--max-trials", type=int, default=20)
    result.add_argument("--seed", type=int, default=71)
    result.add_argument("--output", type=Path, default=Path("outputs/peptides.fasta"))
    return result


def write_results(output, peptides, record):
    """Create both output files exclusively; roll back our files on write failure."""
    output = Path(output)
    sidecar = Path(str(output) + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Output or metadata already exists; choose a new --output path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    fasta = "".join(f">AAP_{index:06d}\n{peptide}\n" for index, peptide in enumerate(peptides, 1))
    record = dict(record, fasta_sha256=hashlib.sha256(fasta.encode("utf-8")).hexdigest())
    created = []
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            created.append(output)
            handle.write(fasta)
        with sidecar.open("x", encoding="utf-8", newline="\n") as handle:
            created.append(sidecar)
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except BaseException:
        for item in created:
            item.unlink(missing_ok=True)
        raise


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    config = SamplingConfig(num=args.num, min_length=args.min_length,
        max_length=args.max_length, temperature=args.temperature,
        batch_size=args.batch_size, max_trials=args.max_trials, seed=args.seed)
    try:
        config.validate()
    except ValueError as exc:
        argument_parser.error(str(exc))
    registry = json.loads(resources.files("aap_lstm").joinpath("checkpoints.json").read_text(encoding="utf-8"))["checkpoints"]
    alias = args.model or "v1"
    checkpoint = (args.weights if args.weights is not None else args.model_dir / registry[alias]["filename"]).resolve()
    if not checkpoint.is_file():
        argument_parser.error(f"Checkpoint not found: {checkpoint}. Supply --weights or --model-dir; weights are not bundled.")
    if checkpoint.suffix.lower() not in (".hdf5", ".h5"):
        argument_parser.error("Supply legacy Keras HDF5 weights (.hdf5 or .h5).")
    output = args.output.resolve()
    if output.exists() or Path(str(output) + ".json").exists():
        argument_parser.error("Output or metadata already exists; choose a new --output path.")
    try:
        checksum = sha256_file(checkpoint)
    except OSError as exc:
        argument_parser.error(f"Cannot read checkpoint: {exc}")
    matches = [name for name, info in registry.items() if info["sha256"] == checksum]
    if args.weights is None and checksum != registry[alias]["sha256"]:
        argument_parser.error("Checkpoint hash does not match the selected reference model. Use --weights explicitly for a custom compatible checkpoint.")
    # Use CPU for the documented validation path; set before importing TensorFlow.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError as exc:
        print(f"TensorFlow is unavailable: {exc}. Install requirements.txt in this environment.", file=sys.stderr)
        return 1
    try:
        from .model import load_predictor
        tf.keras.utils.set_random_seed(config.seed)
        model = load_predictor(checkpoint)
        @tf.function(input_signature=[tf.TensorSpec((None, None, 23), tf.float32)])
        def predict_tensor(inputs):
            return model(inputs, training=False)[:, -1, :]
        peptides, counts = sample_candidates(lambda inputs: predict_tensor(inputs).numpy(), config)
        complete = len(peptides) == config.num
        record = {"software": "AAP-LSTM inference", "version": __version__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_filename": checkpoint.name, "checkpoint_sha256": checksum,
            "matching_reference_aliases": matches, "sampling": asdict(config),
            "vocabulary": VOCABULARY, "complete": complete, "counts": counts,
            "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "tensorflow": tf.__version__, "keras": metadata.version("keras"),
                        "numpy": metadata.version("numpy"), "device": "CPU"}}
        write_results(output, peptides, record)
    except (ImportError, OSError, ValueError, RuntimeError, tf.errors.OpError) as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(peptides)}/{config.num} candidates to {output}")
    print(f"Run metadata: {output}.json")
    if not complete:
        print("Sampling budget exhausted; output is partial. Increase --max-trials if needed.", file=sys.stderr)
        return 3
    return 0
