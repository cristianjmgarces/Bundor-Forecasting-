from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config import SPECTRAL_N_HARMONICS, SPECTRAL_PERIOD
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class SpectralModel(BaseModel):
    def __init__(
        self,
        n_harmonics: int = SPECTRAL_N_HARMONICS,
        period: int = SPECTRAL_PERIOD,
    ) -> None:
        super().__init__(name="SpectralAnalysis")
        self.n_harmonics = n_harmonics
        self.period = period
        self._omega = (2.0 * np.pi) / period

        self._n: int = 0
        self._coefficients: np.ndarray = np.array([])
        self._fitted: np.ndarray = np.array([])
        self._r_squared_adj: float = float("nan")
        self._p_values: np.ndarray = np.array([])
        self._vif: list[float] = []

    def fit(self, series: np.ndarray) -> "SpectralModel":
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        series = np.asarray(series, dtype=float)
        self._n = len(series)

        n_params = 2 * self.n_harmonics + 2
        min_required = n_params + 8
        if self._n < min_required:
            raise ValueError(
                f"SpectralModel requiere al menos {min_required} observaciones "
                f"para {self.n_harmonics} armónicos (T={self.period}), "
                f"recibió {self._n}."
            )

        t = np.arange(1, self._n + 1, dtype=float)
        X = self._build_design_matrix(t)
        X_const = sm.add_constant(X, prepend=True)

        result = sm.OLS(series, X_const).fit()

        self._coefficients = np.asarray(result.params, dtype=float)
        self._fitted = np.maximum(np.asarray(result.fittedvalues, dtype=float), 0.0)
        self._r_squared_adj = float(result.rsquared_adj)
        self._p_values = np.asarray(result.pvalues, dtype=float)

        try:
            self._vif = [
                float(variance_inflation_factor(X_const, i))
                for i in range(1, X_const.shape[1])
            ]
        except Exception as exc:
            logger.warning("No se pudo calcular VIF: %s", exc)
            self._vif = []

        self._is_fitted = True

        logger.info(
            "SpectralModel (regresión armónica) entrenado: %d obs, %d armónicos, "
            "T=%d, R²_adj=%.4f",
            self._n,
            self.n_harmonics,
            self.period,
            self._r_squared_adj,
        )
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        t_future = np.arange(self._n + 1, self._n + horizon + 1, dtype=float)
        X_future = self._build_design_matrix(t_future)
        X_future_const = np.column_stack([np.ones(horizon), X_future])
        forecast = X_future_const @ self._coefficients
        return np.clip(forecast, 0.0, None)

    def fitted_values(self) -> np.ndarray:
        self._require_fitted()
        return self._fitted.copy()

    def get_params(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "n_harmonics": self.n_harmonics,
            "period_T": self.period,
            "omega_W": round(self._omega, 6),
            "n_training_obs": self._n,
            "intercept": round(float(self._coefficients[0]), 4)
            if len(self._coefficients) else None,
            "trend_coef": round(float(self._coefficients[1]), 4)
            if len(self._coefficients) > 1 else None,
            "r_squared_adj": round(self._r_squared_adj, 4),
            "max_p_value": round(float(np.max(self._p_values)), 4)
            if len(self._p_values) else None,
            "max_vif": round(max(self._vif), 4) if self._vif else None,
        }

    def _build_design_matrix(self, t: np.ndarray) -> np.ndarray:
        columns = [t]
        for k in range(1, self.n_harmonics + 1):
            columns.append(np.sin(t * self._omega * k))
            columns.append(np.cos(t * self._omega * k))
        return np.column_stack(columns)