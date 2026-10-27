# Three real QM9S test examples

These are the first three rows, in their recorded order, from the published
QM9S test split. No selection by prediction quality was used. They demonstrate
the actual paper checkpoints; three examples cannot estimate test accuracy.

Original data: **QM9S dataset, version 3**, DOI
[10.6084/m9.figshare.24235333.v3](https://doi.org/10.6084/m9.figshare.24235333.v3).
The Figshare release is licensed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Please retain this attribution when redistributing the supplied example spectra.
The original dataset creators are credited by the linked dataset record.

`ir.npy` and `raman.npy` contain unmodified native 3,501-bin rows from the prepared
arrays. Coordinates are 500–4,000 cm⁻¹ inclusive. The model applies independent
sample-wise min–max normalization. `manifest.json` records row indices,
identities, checkpoint checksums and release URLs.

`expected_from_ir.npy` and `expected_from_raman.npy` are predictions from the
selected original seed-0 checkpoints; `expected_features_*.npy` contain mean
final-block features. Outputs were generated with PyTorch 2.4.0+cu121 and NumPy
1.26.4 on CPU. The example allows rtol=1e-4, atol=1e-5 for runtime differences.
Weights are exported without optimizer state; every model parameter is unchanged.
