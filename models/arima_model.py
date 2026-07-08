from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np

from config import ARIMA_MAX_D, ARIMA_MAX_P, ARIMA_MAX_Q
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class ARIMAModel(BaseModel):
    def __init__(
        self,
        max_p: int = ARIMA_MAX_P,
        max_d: int = ARIMA_MAX_D,
        max_q: int = ARIMA_MAX_Q,
    ) -> None:
        super().__init__(name="ARIMA")
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q

        self._result = None
        self._best_order: tuple = (0, 1, 0)
        self._best_aic: float = np.inf
        self._fitted_vals: np.ndarray = np.array([])
        self._n: int = 0

    def fit(self, series: np.ndarray) -> "ARIMAModel":
        from statsmodels.tsa.arima.model import ARIMA

        series = np.asarray(series, dtype=float)
        self._n = len(series)

        if self._n < 5:
            raise ValueError(
                f"ARIMA requiere al menos 5 observaciones, recibió {self._n}."
            )

        best_aic = np.inf
        best_result = None
        best_order = (0, 1, 0)

        for order in itertools.product(
            range(self.max_p + 1),
            range(self.max_d + 1),
            range(self.max_q + 1),
        ):
            p, d, q = order
            if p == 0 and q == 0:
                continue
            try:
                mdl = ARIMA(series, order=order)
                res = mdl.fit(method_kwargs={"maxiter": 50})
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_result = res
                    best_order = order
            except Exception as exc:
                logger.debug("ARIMA%s falló: %s", order, exc)

        if best_result is None:
            logger.warning("ARIMA: usando fallback (0,1,0)")
            try:
                mdl = ARIMA(series, order=(0, 1, 0))
                best_result = mdl.fit()
                best_order = (0, 1, 0)
                best_aic = best_result.aic
            except Exception as exc:
                raise RuntimeError(f"ARIMA no pudo ajustarse: {exc}") from exc

        self._result = best_result
        self._best_order = best_order
        self._best_aic = best_aic
        fv = best_result.fittedvalues
        self._fitted_vals = np.maximum(
            fv.to_numpy() if hasattr(fv, "to_numpy") else np.asarray(fv), 0.0
        )
        self._is_fitted = True

        logger.info(
            "ARIMA entrenado: order=%s, AIC=%.2f", best_order, best_aic
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
        return {
            "p": p, "d": d, "q": q,
            "aic": self._best_aic,
            "n_training_obs": self._n,
        }