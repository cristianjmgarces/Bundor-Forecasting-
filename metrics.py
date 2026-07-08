from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_shape(y_true, y_pred)

    nonzero_mask = y_true != 0
    if not nonzero_mask.any():
        logger.warning("MAPE: no hay observaciones con y_true != 0, devolviendo NaN")
        return float("nan")

    return float(
        np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]) / np.abs(y_true[nonzero_mask]))
        * 100
    )


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_shape(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_shape(y_true, y_pred)

    numerator = 2.0 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)

    valid_mask = denominator != 0
    if not valid_mask.any():
        logger.warning("SMAPE: todos los valores son cero, devolviendo 0.0")
        return 0.0

    return float(np.mean(numerator[valid_mask] / denominator[valid_mask]) * 100)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return {
        "MAPE": mape(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }


def _check_shape(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true y y_pred deben tener la misma forma: "
            f"{y_true.shape} vs {y_pred.shape}"
        )
    if len(y_true) == 0:
        raise ValueError("Los arrays no pueden estar vacíos.")