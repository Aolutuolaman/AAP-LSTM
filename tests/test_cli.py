import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from aap_lstm.cli import main, write_results


class CliTests(unittest.TestCase):
    def test_missing_checkpoint_is_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
                main(["--weights", str(Path(directory) / "missing.hdf5")])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("weights are not bundled", errors.getvalue())

    def test_output_and_metadata_match(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "peptides.fasta"
            write_results(output, ["AR", "ND"], {"complete": True})
            self.assertEqual(output.read_text(), ">AAP_000001\nAR\n>AAP_000002\nND\n")
            metadata = json.loads(Path(str(output) + ".json").read_text())
            self.assertEqual(metadata["fasta_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                write_results(output, ["CC"], {})
            self.assertIn("AR", output.read_text())

    def test_existing_sidecar_prevents_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "peptides.fasta"
            sidecar = Path(str(output) + ".json")
            sidecar.write_text("existing")
            with self.assertRaises(FileExistsError):
                write_results(output, ["AR"], {})
            self.assertFalse(output.exists())
            self.assertEqual(sidecar.read_text(), "existing")

    def test_bad_metadata_rolls_back_only_new_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "peptides.fasta"
            with self.assertRaises(ValueError):
                write_results(output, ["AR"], {"invalid": float("nan")})
            self.assertFalse(output.exists())
            self.assertFalse(Path(str(output) + ".json").exists())


if __name__ == "__main__":
    unittest.main()
