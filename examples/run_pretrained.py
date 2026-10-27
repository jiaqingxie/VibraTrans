"""Run the released paper checkpoints on three fixed held-out QM9S spectra."""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from vibratrans.inference import predict_file
from vibratrans.runtime import digest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='Download two paper checkpoints (~183 MB total)')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output directory; existing predictions are not overwritten.')
    examples = Path(__file__).resolve().parent / 'qm9s'
    manifest = json.loads((examples / 'manifest.json').read_text())
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for record in manifest['checkpoints']:
        checkpoint = args.checkpoint_dir / record['filename']
        if not checkpoint.exists() and args.download:
            temporary = checkpoint.with_suffix('.download')
            urllib.request.urlretrieve(record['url'], temporary)
            if digest(temporary) != record['sha256']:
                temporary.unlink()
                raise ValueError('Downloaded checkpoint checksum mismatch.')
            temporary.rename(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f'{checkpoint}: use --download or supply the release asset.')
        if digest(checkpoint) != record['sha256']:
            raise ValueError(f'Checkpoint checksum mismatch: {checkpoint.name}')
        source = record['source']
        prediction = args.output_dir / f'predicted_from_{source}.npy'
        features = args.output_dir / f'features_from_{source}.npy'
        predict_file(checkpoint, examples / f'{source}.npy', prediction,
                     features_output=features, device='cpu')
        actual = np.load(prediction)
        expected = np.load(examples / record['expected_prediction'])
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.load(features), np.load(examples / record['expected_features']),
                                   rtol=1e-4, atol=1e-5)
        reports.append(dict(source=source, rows=len(actual), passed=True,
                            max_prediction_difference=float(np.max(np.abs(actual - expected)))))
    (args.output_dir / 'verification.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
