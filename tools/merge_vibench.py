"""Recreate the paper's eight-domain native-bin concatenation without resampling."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    domains = ('geom', 'mols', 'nist_ir', 'pahs', 'peptide', 'peptide_mod', 'qm9', 'zinc15')
    args.output.mkdir(parents=True, exist_ok=False)
    counts = []
    for name in domains:
        shapes = [np.load(args.prepared_root / name / f'{m}.npy', mmap_mode='r').shape
                  for m in ('ir', 'raman')]
        if shapes[0] != shapes[1] or len(shapes[0]) != 2 or shapes[0][1] != 1024:
            raise ValueError(f'Unexpected paired shapes for {name}: {shapes}')
        counts.append(shapes[0][0])
    if sum(counts) != 91949:
        raise ValueError('The recorded source versions contain exactly 91,949 rows.')
    for modality in ('ir', 'raman'):
        target = np.lib.format.open_memmap(args.output / f'{modality}.npy', mode='w+',
                                         dtype='float32', shape=(sum(counts), 1024))
        start = 0
        for domain, count in zip(domains, counts):
            x = np.load(args.prepared_root / domain / f'{modality}.npy', mmap_mode='r')
            if not np.isfinite(x).all():
                raise ValueError(f'Nonfinite source values: {domain}/{modality}')
            target[start:start + count] = x
            start += count
        target.flush()
    (args.output / 'concatenation.json').write_text(json.dumps(dict(zip(domains, counts)), indent=2))


if __name__ == '__main__':
    main()
