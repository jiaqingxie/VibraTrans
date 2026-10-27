# Data sources, preparation and exact row order

The small runnable examples require no full dataset. Full benchmark replication
uses original third-party releases; spectra are not redistributed in bulk here.
The recorded URLs, source versions, checksums and LMDB field names are in
[`raw_sources.json`](raw_sources.json). Prepared-array hashes and exact splits
are in [`datasets.json`](datasets.json). Source licensing remains with each provider.

## Prepare QM9S and QMe14S

Install conversion dependencies: `pip install -e '.[data]'`.

- QM9S: [Figshare version 3](https://doi.org/10.6084/m9.figshare.24235333.v3).
  Download `qm9s.pt`, `ir_boraden.csv` and `raman_boraden.csv` into `raw/qm9s/`.
  Preserve the original filenames (including `boraden`). The broadened matrices
  follow the author's PT list order; the separate CSV archive is not that row map.
- QMe14S: [the authors' shared release](https://figshare.com/s/889262a4e999b5c9a5b3).
  Download `IR_broaden.zip` and `Raman_broaden.zip` into `raw/qme14s/`.
  Conversion pairs numeric filename IDs and checks molecular identity in both modalities.

```bash
python tools/prepare_original.py --dataset qm9s --raw raw/qm9s --output data/qm9s
python tools/prepare_original.py --dataset qme14s --raw raw/qme14s --output data/qme14s
```

These conversion scripts preserve the experiment's array generation logic.
They retain native spectral bins and store raw float32 values. The QM9S PT and
ViBench LMDB files use their authors' serialized object formats; use the exact
provider files and verify their recorded checksums before conversion.

## Prepare ViBench-Full

The paper reconstructs a 91,949-row collection from eight released components
in the order below. It does not claim byte identity with the authors' HDF5 file.
Download each source LMDB listed in `raw_sources.json`, preserving its filename.
For each row, run the converter with the indicated keys and source version:

```bash
python tools/prepare_lmdb.py --input raw/vibench/geom_test.lmdb --output prepared/geom --ir-key q_ir --raman-key q_raman --source-version 'Figshare 28579832 version 2'
```

| Output domain | Original filename | IR key | Raman key | Source version |
|---|---|---|---|---|
| geom | [geom_test.lmdb](https://ndownloader.figshare.com/files/54272948) | `q_ir` | `q_raman` | Figshare 28579832 version 2 |
| mols | [mols_test.lmdb](https://ndownloader.figshare.com/files/68403160) | `ir` | `raman` | figshare-28579832-v3 |
| nist_ir | [nist_ir_test.lmdb](https://ndownloader.figshare.com/files/54272933) | `ir` | `raman` | Figshare 28579832 version 2 |
| pahs | [pahs_test.lmdb](https://ndownloader.figshare.com/files/54272927) | `ir` | `raman` | Figshare 28579832 version 2 |
| peptide | [peptide_test.lmdb](https://ndownloader.figshare.com/files/54272942) | `ir` | `raman` | Figshare 28579832 version 2 |
| peptide_mod | [peptide_mod_test.lmdb](https://ndownloader.figshare.com/files/54272930) | `ir` | `raman` | Figshare 28579832 version 2 |
| qm9 | [qm9_test.lmdb](https://ndownloader.figshare.com/files/54272957) | `ir` | `raman` | Figshare 28579832 version 2 |
| zinc15 | [zinc15_test.lmdb](https://ndownloader.figshare.com/files/54272951) | `ir` | `raman` | Figshare 28579832 version 2 |

```bash
python tools/merge_vibench.py --prepared-root prepared --output data/vibench_full
```

This uses the same source order and float32 concatenation as the original
campaign. No interpolation, extra normalization or axis reinterpretation occurs.

## Verify and use the recorded splits

```text
data/qm9s/          ir.npy, raman.npy
data/qme14s/        ir.npy, raman.npy
data/vibench_full/  ir.npy, raman.npy
```

```bash
vibratrans verify-data --task qm9s_vibratrans_ir_seed0 --data-root data --full-hash
vibratrans verify-data --task qme14s_vibratrans_ir_seed0 --data-root data --full-hash
vibratrans verify-data --task vibench_full_vibratrans_ir_seed0 --data-root data --full-hash
```

Full hashes must match before describing a run as reproducing the recorded
protocol. Exact split indices must never be applied to reordered arrays.
Scaffold tasks share the parent collection's arrays. Calibration rows in the
scaffold files are excluded from training/validation/test and checkpoint selection.

QM9S has 3,501 bins on the verified 500–4,000 cm⁻¹ inclusive axis; QMe14S has
3,500 native ordered bins with an unverified physical endpoint convention;
ViBench-Full has 1,024 native bins with unresolved complete physical coordinates.
The model independently normalizes source and target by
`(x-min(x))/(max(x)-min(x)+1e-8)` when reading each spectrum.

The three computed collections overlap chemically. QMe14S retains 472 test
rows sharing connectivity with training or validation (419 with training),
out of 27,693 test rows. The primary scores retain this original partition;
the two scaffold protocols are separate evaluations.

## Custom data

Copy one primary task into your own JSON configuration. Supply its paired
array directory, native length, disjoint integer train/val/test split, split
SHA256 and validation-reference constants. Relative split paths resolve from
the parent of the configuration directory. Custom runs are not paper replications.
