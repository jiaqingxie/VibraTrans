"""Per-spectrum metrics; constant-target scores stay undefined."""

import numpy as np


def per_sample_metrics(prediction, target):
    p, y = prediction.astype("float64"), target.astype("float64")
    error = p - y
    pc, yc = p - p.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    ss = (yc**2).sum(1)
    denominator = np.sqrt((pc**2).sum(1) * ss)
    r2 = np.full(len(p), np.nan)
    pearson = np.full(len(p), np.nan)
    np.divide((error**2).sum(1), ss, out=r2, where=ss > 0)
    r2 = 1 - r2
    np.divide((pc * yc).sum(1), denominator, out=pearson, where=denominator > 0)
    return dict(mae=np.abs(error).mean(1), rmse=np.sqrt((error**2).mean(1)), r2=r2, pearson=pearson)
