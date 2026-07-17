"""
CRISTAL - A Python anomaly detection framework based on the Christoffel Function.
Copyright (C) 2026 Florian Grivet

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging

# pylint: disable=unused-variable
from .__version__ import __author__, __date__, __license__, __version__
from .backend import *
from .commons import *
from .config import *
from .core import *
from .evaluation import *
from .preprocessing import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("CRISTAL v%s (released: %s)\nCopyright (C) 2026 Florian Grivet", __version__, __date__)
