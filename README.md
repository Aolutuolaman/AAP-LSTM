# AAP-LSTM inference

Generate peptide sequence candidates from compatible AAP-LSTM checkpoints.
This repository provides the inference component of the AAP-LSTM workflow:
a two-layer LSTM, temperature sampling, length filtering and FASTA export.
It does not contain the training pipeline, comparison predictors or the full
experimental-analysis workflow from the manuscript. Generated sequences are
candidates; model output does not establish biological activity.

**Model weights are supplied separately.** The repository is source-only.
It can be installed and its unit tests run without a checkpoint; peptide
generation requires an authorized compatible checkpoint. There is currently
no verified public download endpoint for the six fine-tuned reference files.

## Install

Use 64-bit Python 3.10–3.12. Validated on Windows CPU with Python 3.12.14,
TensorFlow 2.16.2, Keras 3.3.3 and NumPy 1.26.4. Run from the repository directory:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python generate_cli.py --help
```

On Linux, use .venv/bin/python in place of .venv\Scripts\python.
The CLI uses CPU inference. GPU configuration is outside the validated scope.
An installed command is also available with python -m pip install . followed
by aap-generate --help.

## Generate candidates

Pass an external checkpoint directly; the file does not need to be copied
into this repository:

```powershell
.venv\Scripts\python generate_cli.py --weights "C:/checkpoints/finetuned_gen1.hdf5" --num 20 --seed 71 --output outputs/peptides.fasta
```

Alternatively, select a reference alias and the directory containing it:

```powershell
.venv\Scripts\python generate_cli.py --model vegfr2 --model-dir "C:/checkpoints" --num 20 --output outputs/vegfr2.fasta
```

| Alias  | Expected file         |
| ------ | --------------------- |
| v1     | finetuned_gen1.hdf5   |
| v2     | finetuned_gen2.hdf5   |
| ang2   | finetuned_Ang2.hdf5   |
| bfgf   | finetuned_BFGF.hdf5   |
| mmps   | finetuned_MMPs.hdf5   |
| vegfr2 | finetuned_VEGFR2.hdf5 |

Reference aliases verify the SHA-256 fingerprint in
aap_lstm/checkpoints.json. For a different compatible checkpoint, use
--weights explicitly. File names alone do not establish model identity.
External initial pretrained weights are not interchangeable with the six
fine-tuned reference variants merely because the architecture matches.

The command creates a FASTA file and a .fasta.json sidecar containing the
checkpoint hash, seed, sampling settings, runtime versions, filtering counts
and output hash. Existing files are never overwritten. A new output path is
required for another run.

Exit codes: 0 = requested count generated; 1 = runtime failure;
2 = invalid arguments, missing checkpoint or existing output;
3 = sampling budget exhausted and a partial output was written.
Inspect the sidecar's complete field before using results downstream.

## Model and sampling

- Two 256-unit LSTM layers with dropout settings 0.3 and 0.5, both returning
  sequences, followed by a 23-class softmax. Dropout is inactive at inference.
- Fixed vocabulary order: ARNDCEQGHILKMFPSTWYVUBZ. U is the start token,
  B is padding, and Z is the end token.
- Defaults: temperature 1.1; 50 sampling steps; retained length 10–50;
  batch size 200; at most 20 trials; seed 71; 20 requested candidates.
- All 23 output tokens remain available during sampling. Each row is sampled
  for 50 steps, then decoded up to its first Z. Rows without Z may produce
  a 50-residue candidate. Internal U/B tokens cause the candidate to be
  rejected; they are not converted to amino acids.
- Candidates are deduplicated in first-seen order. Each run records rejected
  sequences and any shortfall.

This release preserves the checkpoint interface and original default
sampling budget. It replaces legacy configuration/tokenizer helpers and adds
numerically stable temperature scaling, explicit random seeding, canonical
amino-acid filtering and ordered deduplication. These changes, and a newer
TensorFlow runtime, can change sampled sequences relative to historical
runs. It is not a bit-for-bit reproduction of the manuscript's candidate
list or reported performance. A fixed seed is intended for repeatability in
a fixed environment, not identical results on every platform.

## Tests

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

Unit tests run without weights. The optional integration check loads all six
reference checkpoints when AAP_CHECKPOINT_DIR is set:

```powershell
$env:AAP_CHECKPOINT_DIR = "C:/checkpoints"
.venv\Scripts\python -m unittest discover -s tests -v
```

## Provenance and license

The original workflow used the externally pretrained LSTM described by
Zhang et al. (2023), *Deep Learning-Based Bioactive Therapeutic Peptide
Generation and Screening*, DOI: [10.1021/acs.jcim.2c01485](https://doi.org/10.1021/acs.jcim.2c01485),
from [New_peptide_iteration/LSTM_Pep](https://github.com/haiping1010/New_peptide_iteration/tree/master/LSTM_Pep).


The source and documentation in this repository are released under the
[MIT license](LICENSE). This license does not cover external weights or data.
