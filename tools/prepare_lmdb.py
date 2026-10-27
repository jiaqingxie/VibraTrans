"""Materialize paired ViBench spectra, retaining row identifiers and SMILES.

No interpolation is performed and no array is interpreted as a wavenumber axis
without metadata. In particular, GEOM k_ir/k_raman are alternative spectra,
not coordinate arrays. Specify the paired spectral keys explicitly.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle

import lmdb
import numpy as np
from rdkit import Chem


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else ''


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ir-key', default='ir')
    p.add_argument('--raman-key', default='raman')
    p.add_argument('--source-version', required=True)
    a = p.parse_args()
    if (a.output / 'manifest.json').exists():
        raise FileExistsError(f'Already prepared: {a.output}')
    a.output.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(a.input), subdir=False, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        # Store this ordering explicitly; never rely on implicit LMDB iteration.
        keys.sort(key=lambda k: (0, int(k)) if k.isdigit() else (1, k))
        first = pickle.loads(txn.get(keys[0]))
        length = np.asarray(first[a.ir_key]).size
        arrays = {mode: np.lib.format.open_memmap(a.output / f'{mode}.npy.part',
                  mode='w+', dtype='float32', shape=(len(keys), length)) for mode in ('ir', 'raman')}
        invalid_smiles = 0
        constants = dict(ir=0, raman=0)
        unique = set()
        with (a.output / 'molecules.csv.part').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['row', 'record_id', 'filename', 'smiles', 'canonical_smiles'])
            writer.writeheader()
            for i, key in enumerate(keys):
                record = pickle.loads(txn.get(key))
                for mode, field in [('ir', a.ir_key), ('raman', a.raman_key)]:
                    spectrum = np.asarray(record[field], dtype=np.float32)
                    if spectrum.ndim != 1 or spectrum.size != length or not np.isfinite(spectrum).all():
                        raise ValueError(f'{key!r}: invalid {field} shape or non-finite intensity')
                    constants[mode] += int(spectrum.max() == spectrum.min())
                    arrays[mode][i] = spectrum
                smiles = record.get('smiles', '')
                can = canonical(smiles)
                invalid_smiles += int(not can)
                if can:
                    unique.add(can)
                writer.writerow(dict(row=i, record_id=key.decode(), filename=record.get('filename', ''),
                                     smiles=smiles, canonical_smiles=can))
                if (i + 1) % 10000 == 0:
                    print(f'{a.input.name}: {i + 1}/{len(keys)}', flush=True)
    env.close()
    for mode in arrays:
        arrays[mode].flush()
        os.replace(a.output / f'{mode}.npy.part', a.output / f'{mode}.npy')
    os.replace(a.output / 'molecules.csv.part', a.output / 'molecules.csv')
    meta = {'source_file': str(a.input.resolve()), 'source_version': a.source_version,
            'ir_key': a.ir_key, 'raman_key': a.raman_key, 'rows': len(keys), 'length': length,
            'unique_canonical_smiles': len(unique), 'invalid_smiles': invalid_smiles,
            'constant_spectra': constants, 'axis': 'ordered index; physical range requires source metadata',
            'normalization': 'raw values stored; min-max normalization at model input'}
    provenance = a.input.with_name(a.input.name + '.provenance.json')
    if provenance.exists():
        meta['source_provenance'] = json.loads(provenance.read_text())
    meta['molecules_sha256'] = hashlib.sha256((a.output / 'molecules.csv').read_bytes()).hexdigest()
    (a.output / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
