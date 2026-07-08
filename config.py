REQUIRED_COLUMNS = ["año", "semana", "variedad", "formato", "cliente", "suma_de_litros"]

COLUMN_TYPES = {
    "año": "int",
    "semana": "int",
    "variedad": "str",
    "formato": "str",
    "cliente": "str",
    "suma_de_litros": "float",
}

SEMANA_MIN = 1
SEMANA_MAX = 53
ANIO_MIN = 1900
ANIO_MAX = 2100

MODEL_NAMES = ["SpectralAnalysis", "HoltWinters", "SARIMA", "ARIMA"]

PRIMARY_METRIC = "SMAPE"
SECONDARY_METRIC = "MSE"
TERTIARY_METRIC = "MAPE"

ARIMA_MAX_P = 2
ARIMA_MAX_D = 1
ARIMA_MAX_Q = 2

SARIMA_SEASONAL_PERIODS = [52]
SARIMA_MAX_P = 1
SARIMA_MAX_D = 1
SARIMA_MAX_Q = 1

MODEL_TIMEOUT_SECONDS = 45

SPECTRAL_N_HARMONICS = 5
SPECTRAL_PERIOD = 52

MIN_SEASONAL_CYCLES = 2
DEFAULT_SEASONAL_PERIOD = 52

PLOT_FIGURE_SIZE = (14, 6)
PLOT_HISTORICAL_COLOR = "royalblue"
PLOT_FORECAST_COLOR = "darkorange"
PLOT_HISTORICAL_LABEL = "Serie Histórica"
PLOT_FORECAST_LABEL = "Pronóstico"
PLOT_TITLE_TEMPLATE = "Pronóstico de Litros — {model_name}"

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"