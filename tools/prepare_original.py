"""Prepare original QM9S/QMe14S archives without resampling spectral bins.

QM9S broadened matrices follow the author's qm9s.pt list order. The separately
released per-molecule CSV archive has a different order and is not a row map.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile

import numpy as np
from rdkit import Chem


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES in source metadata: {smiles}')
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def members(archive, prefix=None):
    result = {}
    for info in archive.infolist():
        if info.is_dir() or not info.filename.endswith('.csv'):
            continue
        stem = Path(info.filename).stem
        pattern = rf'{prefix}_(\d+)' if prefix else r'(\d+)'
        match = re.fullmatch(pattern, stem)
        if not match:
            raise ValueError(f'Unexpected archive member: {info.filename}')
        key = int(match[1])
        if key in result:
            raise ValueError(f'Duplicate identifier {key}')
        result[key] = info.filename
    if not result:
        raise ValueError('No molecular records')
    return result


def create_arrays(output, count, length):
    return {m: np.lib.format.open_memmap(output / f'{m}.npy.part', mode='w+',
            dtype='float32', shape=(count, length)) for m in ('ir', 'raman')}


def spectrum(text, length, label):
    value = np.fromstring(text, sep=',', dtype=np.float32)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f'Invalid spectrum: {label}, shape={value.shape}')
    return value


def qme14(raw, output):
    with zipfile.ZipFile(raw/'IR_broaden.zip') as zi, zipfile.ZipFile(raw/'Raman_broaden.zip') as zr:
        ir, raman = members(zi, 'IR'), members(zr, 'Raman')
        if set(ir) != set(raman):
            raise ValueError('IR and Raman molecule identifiers differ')
        keys = sorted(ir)
        arrays = create_arrays(output, len(keys), 3500)
        rows, constants = [], dict(ir=0, raman=0)
        for row, key in enumerate(keys):
            pair = [z.read(names[key]).decode().splitlines() for z,names in [(zi,ir),(zr,raman)]]
            identities = [canonical(lines[0].strip()) for lines in pair]
            if identities[0] != identities[1]:
                raise ValueError(f'Molecular identity mismatch at {key}')
            for mode, lines in zip(('ir','raman'), pair):
                v = spectrum(next(line for line in reversed(lines) if line.strip()), 3500, f'{mode}:{key}')
                arrays[mode][row] = v
                constants[mode] += int(v.max() == v.min())
            rows.append(dict(row=row, record_id=key, filename=ir[key], smiles=pair[0][0].strip(),
                             canonical_smiles=identities[0], original_qm9_id=''))
            if (row+1)%10000 == 0:
                print(f'QMe14S: {row+1}/{len(keys)} paired and identity checked',flush=True)
    return arrays, rows, dict(source_version='Figshare shared item 28260197, files 51890792/51890552',
        source_url='https://figshare.com/s/889262a4e999b5c9a5b3',
        axis='native ordered bins; source states 500–4000 cm^-1 but endpoint convention is not supplied',
        axis_units='sample index', source_range_cm_inverse=[500,4000], length=3500,
        pairing='matching numeric filename IDs and matching canonical SMILES in both modalities',
        constant_spectra=constants)


def qm9(raw, output):
    import torch
    rows = []
    records=torch.load(raw/'qm9s.pt',map_location='cpu',weights_only=False,mmap=True)
    for row,record in enumerate(records):
        smiles=record.smile
        rows.append(dict(row=row,record_id=row,filename=f'qm9s.pt[{row}]',smiles=smiles,
            canonical_smiles=canonical(smiles),original_qm9_id=int(record.number)))
        if (row+1)%20000==0:
            print(f'QM9S: {row+1}/{len(records)} author PT molecular identities read',flush=True)
    arrays = create_arrays(output,len(rows),3501)
    constants = dict(ir=0,raman=0)
    grid = np.arange(500.,4001.)
    for mode in ('ir','raman'):
        with (raw/f'{mode}_boraden.csv').open() as f:
            header = f.readline().strip().split(',')
            np.testing.assert_array_equal(np.asarray(header[1:],dtype=float), grid)
            count = 0
            for row,line in enumerate(f):
                row_id,values = line.split(',',1)
                if int(row_id) != row or row >= len(rows):
                    raise ValueError(f'{mode}: unexpected zero-based row ID {row_id}')
                v = spectrum(values,3501,f'{mode}:{row_id}')
                arrays[mode][row] = v
                constants[mode] += int(v.max() == v.min())
                count += 1
                if count%20000 == 0:
                    print(f'QM9S {mode}: {count}/{len(rows)} spectra',flush=True)
            if count != len(rows):
                raise ValueError(f'{mode}: row count does not match molecular metadata')
    np.save(output/'wavenumbers.npy',grid)
    return arrays,rows,dict(source_version='Figshare 24235333 version 3',
        source_url='https://figshare.com/articles/dataset/QM9S_dataset/24235333',
        axis='500–4000 cm^-1 inclusive, verified from both source matrix headers',
        axis_units='cm^-1',length=3501,constant_spectra=constants,
        pairing='same zero-based IR/Raman matrix row; author qm9s.pt list order supplies molecular identity')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',choices=['qm9s','qme14s'],required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if (a.output/'manifest.json').exists():
        raise FileExistsError(f'Already prepared: {a.output}')
    a.output.mkdir(parents=True,exist_ok=True)
    arrays,rows,meta=(qm9 if a.dataset=='qm9s' else qme14)(a.raw,a.output)
    for mode,array in arrays.items():
        array.flush()
        os.replace(a.output/f'{mode}.npy.part',a.output/f'{mode}.npy')
    with (a.output/'molecules.csv.part').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    os.replace(a.output/'molecules.csv.part',a.output/'molecules.csv')
    meta.update(rows=len(rows),unique_canonical_smiles=len({r['canonical_smiles']for r in rows}),
        invalid_smiles=0,normalization='raw values stored; per-spectrum min-max at model input',
        interpolation='none: all native bins retained',
        source_provenance=[json.loads(f.read_text())for f in sorted(a.raw.glob('*.provenance.json'))],
        molecules_sha256=sha256(a.output/'molecules.csv'),
        arrays_sha256={mode:sha256(a.output/f'{mode}.npy')for mode in arrays})
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':
    main()
