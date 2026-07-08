from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from config import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

APP_PATH = Path(__file__).parent / "ui" / "streamlit_app.py"


def launch_streamlit() -> None:
    logger.info("Iniciando Bundor Forecasting App…")
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH), "--server.headless", "false"]
    subprocess.run(cmd, check=True)


def main() -> None:
    launch_streamlit()


if __name__ == "__main__":
    main()