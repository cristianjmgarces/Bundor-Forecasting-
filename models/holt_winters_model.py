from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config import SARIMA_SEASONAL_PERIODS
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class HoltWintersModel(BaseModel):
    def __init__(self, seasonal_periods: list[int] | None = None) -> None:
        super().__init__(name="HoltWinters")
        self._seasonal_periods = seasonal_periods or SARIMA_SEASONAL_PERIODS
        self._model = None
        self._result = None
        self._best_config: dict = {}
        self._fitted_vals: np.ndarray = np.array([])
        self._n: int = 0

    def fit(self, series: np.ndarray) -> "HoltWintersModel":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        series = np.asarray(series, dtype=float)
        self._n = len(series)

        if self._n < 4:
            raise ValueError(
                f"HoltWinters requiere al menos 4 observaciones, recibió {self._n}."
            )

        all_positive = bool(np.all(series > 0))
        best_aic = np.inf
        best_result = None
        best_cfg: dict = {}

        trend_options = ["add", "mul", None]
        seasonal_options = ["add", "mul", None] if all_positive else ["add", None]

        for period in self._seasonal_periods:
            if period >= self._n:
                continue
            for trend in trend_options:
                for seasonal in seasonal_options:
                    if seasonal is None and trend is None:
                        continue
                    try:
                        mdl = ExponentialSmoothing(
                            series,
                            trend=trend,
                            seasonal=seasonal,
                            seasonal_periods=period if seasonal else None,
                            initialization_method="estimated",
                        )
                        res = mdl.fit(optimized=True)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_result = res
                            best_cfg = {
                                "trend": trend,
                                "seasonal": seasonal,
                                "seasonal_period": period if seasonal else None,
                                "aic": res.aic,
                            }
                    except Exception as exc:
                        logger.debug(
                            "HW config (%s,%s,p=%d) falló: %s",
                            trend,
                            seasonal,
                            period,
                            exc,
                        )

        if best_result is None:
            try:
                mdl = ExponentialSmoothing(series, initialization_method="estimated")
                best_result = mdl.fit(optimized=True)
                best_cfg = {"trend": None, "seasonal": None, "aic": best_result.aic}
            except Exception as exc:
                raise RuntimeError(
                    f"HoltWinters no pudo ajustarse a la serie: {exc}"
                ) from exc

        self._result = best_result
        self._best_config = best_cfg
        fv = best_result.fittedvalues
        self._fitted_vals = np.maximum(
            fv.to_numpy() if hasattr(fv, "to_numpy") else np.asarray(fv), 0.0
        )
        self._is_fitted = True

        logger.info(
            "HoltWinters entrenado: config=%s, AIC=%.2f",
            best_cfg,
            best_cfg.get("aic", np.nan),
        )
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        fc = self._result.forecast(horizon)
        return np.maximum(
            fc.to_numpy() if hasattr(fc, "to_numpy") else np.asarray(fc), 0.0
        )

    def fitted_values(self) -> np.ndarray:
        self._require_fitted()
        return self._fitted_vals.copy()

    def get_params(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            **self._best_config,
            "n_training_obs": self._n,
            "alpha": getattr(self._result, "params", {}).get("smoothing_level", None),
            "beta": getattr(self._result, "params", {}).get("smoothing_trend", None),
            "gamma": getattr(self._result, "params", {}).get("smoothing_seasonal", None),
        }