# Software review

The `VibraTrans_minimal_code.zip` release asset contains the runnable core,
fixed primary task configurations and splits, data preparation scripts, tests,
documentation and example inputs/outputs. It excludes manuscript PDFs, full
datasets, training logs and large weights. Separate source-data files support
inspection of plotted and tabulated results.

## Requirements and installation

- Python 3.10+; NumPy >=1.24; PyTorch >=2.4.
- Tested: Linux x86-64, Python 3.12, NumPy 1.26.4, PyTorch 2.4.0+cu121 on CPU.
- `pip install -e .` installs the core after a suitable PyTorch installation.
- `pip install -e '.[analysis,dev]'` adds descriptor/peak analysis and tests.
- `pip install -e '.[data]'` adds original-data conversion dependencies.
- CPU examples need no GPU. Full training used H200 GPUs and the numerical
  stack documented in `reproduction.md`.
- Installation time depends on the PyTorch download. The synthetic run takes a
  few seconds after installation. The real-model example needs two ~91 MB
  checkpoints. Output directories must be new.

## Demonstrations and expected outputs

1. `vibratrans demo --output-dir outputs/demo` exercises training, reload,
   prediction, features and evaluation entirely offline. Expected dimensions
   and scope are in the README and `demo_report.json`.
2. `python examples/run_pretrained.py --download --output-dir outputs/paper-demo`
   runs original paper checkpoints on three real held-out spectra, verifies
   checksums and compares predictions/features with supplied expected arrays.
   Offline review uses the two separately downloaded release assets.
3. Full benchmark replication uses `data.md`, array-hash verification and the
   fixed primary task configurations.

## Scope and limitations

The core includes all three primary architectures and the final training
objective, plus descriptor, ridge-probe and peak-matching helpers. The minimal
package does not orchestrate every supplementary figure layout or historical
experiment. Source tables retain reported outcomes and failure statuses.
Retraining on a different numerical stack may change scores and selected epochs.

Third-party data and dependencies retain their original terms. QM9S example
attribution accompanies the data. This revision does not introduce a blanket
software license; the authors must approve one before describing the repository
as open-source licensed. Public access and permission to reuse are distinct.
