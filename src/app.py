"""Entry point for the installed application: run it from the folder that holds the modules."""

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

from main import main  # noqa: E402  (needs APP_DIR on sys.path first)

raise SystemExit(main())
