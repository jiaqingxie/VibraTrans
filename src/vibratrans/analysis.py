"""Chemical descriptors, one-to-one peak matching and frozen-feature ridge probes.

Install the analysis extra to use this module. Functions retain the current
campaign's definitions; input row ordering must be shared across representations.
"""

import numpy as np
from rdkit.Chem import Crippen, rdMolDescriptors
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

TARGETS = [
    "LogP",
    "TPSA",
    "H-bond donors",
    "H-bond acceptors",
    "Rotatable bonds",
    "Ring count",
    "Fraction sp3 carbon",
    "Aromatic rings",
    "Molar refractivity",
    "Labute ASA",
]


def descriptor(mol):
    return [
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        Crippen.MolMR(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
    ]


def ridge_grid(x, y, split):
    train, val, test = [split == s for s in ["train", "val", "test"]]
    scaler = StandardScaler().fit(x[train])
    z = scaler.transform(x).astype("float64")
    ys = StandardScaler().fit(y[train])
    target = ys.transform(y[train])
    # Centered training Gram matrix is reused for all regularization values.
    dual = train.sum() < z.shape[1]
    covariance = z[train] @ z[train].T if dual else z[train].T @ z[train]
    cross = target if dual else z[train].T @ target
    eigenvalues, vectors = eigh(covariance, check_finite=False)
    eigenvalues = np.maximum(eigenvalues, 0)
    projected = vectors.T @ cross
    best = None
    grid = []
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        coefficients = vectors @ (projected / (eigenvalues[:, None] + alpha))
        if dual:
            coefficients = z[train].T @ coefficients
        val_prediction = ys.inverse_transform(z[val] @ coefficients)
        score = float(r2_score(y[val], val_prediction, multioutput="uniform_average"))
        grid.append(dict(alpha=alpha, validation_mean_r2=score))
        if best is None or score > best[0]:
            best = (score, alpha, coefficients)
    prediction = ys.inverse_transform(z[test] @ best[2])
    return prediction, best[1], grid


def matches(true, predicted, tolerance=20):
    result = np.zeros(len(true), bool)
    if not len(true) or not len(predicted):
        return result
    distance = abs(true[:, None] - predicted[None, :])
    penalty = (len(true) + 1) * (tolerance + 1)
    cost = np.concatenate(
        [
            np.where(distance <= tolerance, distance, penalty * 10),
            np.full((len(true), len(true)), penalty),
        ],
        axis=1,
    )
    a, b = linear_sum_assignment(cost)
    keep = b < len(predicted)
    result[a[keep]] = distance[a[keep], b[keep]] <= tolerance
    return result
