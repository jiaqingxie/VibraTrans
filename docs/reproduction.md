# Reproducing the current experiments

## Included artifacts

- `configs/paper_tasks.json`: all 120 completed configurations, covering five
  protocols × three architectures × two directions × four seeds.
- `configs/splits/`: the five original split files, unchanged.
- `results/benchmark.csv`: every selected model's test scores.
- `results/checkpoints.json`: selected epoch, raw/EMA choice, checksum and
  validation score for each checkpoint.
- `figures/`: the current five main figures, plus the available Figure 2–5
  numerical source tables. Figure 1 is the architecture overview with illustrative
  examples. PDF checksums are in `figures/manifest.json`.

The main figures are supplied as finished artifacts. The table bundle supports
inspection and replotting; it is not advertised as a self-contained recreation
of every molecular inset and page layout. Full data arrays are obtained from their original providers. Two inference-only exports of the original
QM9S seed-0 checkpoints are provided as separate release assets. Training
commands create new checkpoints.

## Model and training recipe

VibraTrans uses width 384, eight blocks, six attention heads and fixed sinusoidal
positions. The source enters through ordered patches and a global source
encoder. Patch width is 20. The output is
a residual added to the source at time zero, with one network evaluation.
Velocity values are clipped to [-2, 2]; the final prediction is clipped to [0, 1].
The endpoint training loss uses the value before the final clipping.

The common endpoint objective is 0.6 L1 + 0.4 L2, plus first/second-difference
shape loss (weight 0.02) and target-variance-normalized squared error (0.005).
VibraTrans additionally uses weighted velocity supervision on interpolated paired
spectra (0.1). The PE Transformer and CNN share the endpoint objective.

All models start from independent initialization. AdamW uses learning rate
2e-4, weight decay 1e-4, gradient clipping at norm one, five-epoch warmup and
cosine decay. EMA decay is 0.9995. Training runs for 100 epochs. Batches are 192 for QM9S/QMe14S and 512 for
ViBench-Full. Direction-specific dropout is 0 for IR→Raman
and 0.05 for Raman→IR.

The original campaign used PyTorch `2.7.0a0+ecf3bae40a.nv25.02` on NVIDIA H200,
BF16 training and FP32 evaluation, with TF32 disabled. Runtime settings retain
the campaign's GEMM implementation of kernel-three 1D convolutions. The CPU
path supports implementation checks; new runs on a different numerical stack
need not reproduce every floating-point value or selected epoch exactly.

Validation evaluates initial weights, epoch one, every fifth epoch and the final
epoch. Raw and EMA candidates minimize the larger of MAE/reference-MAE and
(1−R²)/(1−reference-R²). The recorded reference constants are included directly
in each task; no earlier reference checkpoint is needed or loaded to initialize
training. Model/recipe development used previously inspected partitions;
checkpoint selection within each run uses validation only.

Each run writes `run.json`, `history.json`, `best.pt`, `last.pt` and
`complete.json`. An existing run directory is not overwritten. Evaluation
writes ordered predictions, row indices, per-spectrum metrics and a summary.
Undefined correlation/R² values are counted separately. Seeds are summarized
using sample standard deviation.

## Frozen features and chemical analysis

`vibratrans predict --features-output features.npy` captures mean final-block
tokens during the same source-only evaluation. PE features are read before the
separate final normalization. Neither target spectra nor molecular structures
enter feature extraction.

Install `pip install -e '.[analysis]'` to use `vibratrans.analysis`:

- `descriptor(mol)` returns the ten graph-derived targets in `TARGETS` order.
- `ridge_grid(features, labels, split)` fits training-only feature/label scalers,
  selects one alpha from {0.1, 1, 10, 100, 1000} using mean validation R² across
  targets, and returns test predictions, the selected alpha and validation scores.
- `matches(reference_positions, predicted_positions, tolerance=20)` computes
  maximum-cardinality one-to-one peak matches, minimizing distance among such
  matchings.

The paper's feature probe selects 20,000/5,000/10,000 train/validation/test
identities by ascending SHA256 of `"SpectraFlow-probe-2\0" + canonical_smiles`
within each fixed partition. All compared feature matrices must use the same
rows. Morgan fingerprints and descriptor labels require molecular structures.
Figure 5 retains the archived VAE control on matched identities; it is not one
of the 120 newly trained primary models.

Peak-region R² uses reference bins with normalized intensity ≥0.05, centers
within each molecule's region and weights by its reference variance. Full-spectrum
R² is the arithmetic mean of per-spectrum R². These aggregations are different.
Chemical analyses show preserved bands and shifts; they do not establish
uniform superiority on every chemical metric.

## Verification of the reorganization

The automated tests exercise training, checkpoint reload, prediction, feature
capture and evaluation for all three architectures on small synthetic arrays.
They also cover output clipping versus endpoint training, deterministic batches,
split integrity, undefined metrics and validation-only probe selection.

The [checkpoint comparison report](refactor_verification.json) compares the
reorganized code directly with the original campaign code on real QM9S checkpoints, both directions and all three architectures. It records
initialization, state-dictionary compatibility, predictions, features, losses
and gradients. This is an equivalence check on a CPU runtime, not another
120-run GPU campaign.

## Current naming and supplementary ablation

The public model/package name is VibraTrans. Recorded architecture fields retain
`vibradit` so that original state-dictionary checkpoints load without conversion.
The 120-task matrix is the primary final recipe. The separate three-term
loss-removal study in the SI sets the variance-normalized weight to zero in every
arm. Its per-seed source tables retain 71 completed ablations and one divergence,
24 reused controls and separately marked completion of nine original fits.
Those outcomes must not be counted as additional primary models.
