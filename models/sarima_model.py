from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np

from config import (
    SARIMA_MAX_P,
    SARIMA_MAX_D,
    SARIMA_MAX_Q,
    SARIMA_SEASONAL_PERIODS,
)
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class SARIMAModel(BaseModel):
    def __init__(
        self,
        max_p: int = SARIMA_MAX_P,
        max_d: int = SARIMA_MAX_D,
        max_q: int = SARIMA_MAX_Q,
        seasonal_periods: list[int] | None = None,
    ) -> None:
        super().__init__(name="SARIMA")
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self._seasonal_periods = seasonal_periods or SARIMA_SEASONAL_PERIODS

        self._result = None
        self._best_order: tuple = ()
        self._best_seasonal_order: tuple = ()
        self._best_aic: float = np.inf
        self._fitted_vals: np.ndarray = np.array([])
        self._n: int = 0

    def fit(self, series: np.ndarray) -> "SARIMAModel":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        series = np.asarray(series, dtype=float)
        self._n = len(series)

        if self._n < 10:
            raise ValueError(
                f"SARIMA requiere al menos 10 observaciones, recibió {self._n}."
            )

        best_aic = np.inf
        best_result = None
        best_order = (0, 1, 0)
        best_seasonal = (0, 0, 0, 0)

        for s in self._seasonal_periods:
            if self._n < 2 * s:
                continue
            candidates = [
                ((1, 1, 1), (1, 0, 0, s)),
                ((0, 1, 1), (0, 1, 1, s)),
                ((1, 1, 0), (1, 0, 0, s)),
                ((0, 1, 1), (0, 0, 1, s)),
                ((1, 1, 1), (0, 1, 1, s)),
            ]
            for order, seasonal_order in candidates:
                try:
                    mdl = SARIMAX(
                        series,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = mdl.fit(disp=False, maxiter=50, method="lbfgs")
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_result = res
                        best_order = order
                        best_seasonal = seasonal_order
                except Exception as exc:
                    logger.debug("SARIMA%s x %s falló: %s", order, seasonal_order, exc)

        if best_result is None:
            logger.warning("SARIMA: usando fallback (1,1,1)(1,0,0,s)")
            for s in self._seasonal_periods:
                try:
                    mdl = SARIMAX(
                        series, order=(1, 1, 1), seasonal_order=(1, 0, 0, s),
                        enforce_stationarity=False, enforce_invertibility=False,
                    )
                    best_result = mdl.fit(disp=False, maxiter=30)
                    best_order = (1, 1, 1)
                    best_seasonal = (1, 0, 0, s)
                    best_aic = best_result.aic
                    break
                except Exception:
                    continue
            if best_result is None:
                raise RuntimeError("SARIMA no pudo ajustarse a la serie.")

        self._result = best_result
        self._best_order = best_order
        self._best_seasonal_order = best_seasonal
        self._best_aic = best_aic
        fv = best_result.fittedvalues
        self._fitted_vals = np.maximum(
            fv.to_numpy() if hasattr(fv, "to_numpy") else np.asarray(fv), 0.0
        )
        self._is_fitted = True

        logger.info(
            "SARIMA entrenado: order=%s, seasonal=%s, AIC=%.2f",
            best_order,
            best_seasonal,
            best_aic,
        )
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        fc = self._result.forecast(steps=horizon)
        return np.maximum(
            fc.to_numpy() if hasattr(fc, "to_numpy") else np.asarray(fc), 0.0
        )

    def fitted_values(self) -> np.ndarray:
        self._require_fitted()
        return self._fitted_vals.copy()

    def get_params(self) -> dict[str, Any]:
        self._require_fitted()
        p, d, q = self._best_order
        P, D, Q, s = self._best_seasonal_order
        return {
            "p": p, "d": d, "q": q,
            "P": P, "D": D, "Q": Q, "s": s,
            "aic": self._best_aic,
            "n_training_obs": self._n,
        }