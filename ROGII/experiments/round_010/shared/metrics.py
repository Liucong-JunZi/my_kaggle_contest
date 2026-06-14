"""Standard metrics. Always offset-space (target = TVT - last_known_tvt)."""
import numpy as np
import pandas as pd


def perwell_rmse(target: np.ndarray, pred: np.ndarray, wells: np.ndarray) -> float:
    df = pd.DataFrame({"target": target, "pred": pred, "well": wells})
    rmses = df.groupby("well").apply(
        lambda g: np.sqrt(np.mean((g["target"] - g["pred"]) ** 2))
    )
    return float(rmses.mean())


def perwell_rmse_arr(target: np.ndarray, pred: np.ndarray, wells: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({"target": target, "pred": pred, "well": wells})
    rmses = df.groupby("well").apply(
        lambda g: np.sqrt(np.mean((g["target"] - g["pred"]) ** 2))
    )
    return rmses.values


def flat_rmse(target: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - pred) ** 2)))
