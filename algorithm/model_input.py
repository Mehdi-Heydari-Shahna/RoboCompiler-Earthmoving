"""Read physical graph records from an explicitly supplied JSON file."""

import json
from pathlib import Path

from .compiler import ModelError, validate


def load_graph(path):
    """Load and validate external physical records without supplying defaults."""
    with Path(path).open("r", encoding="utf-8") as stream:
        physical = json.load(stream)
    if not isinstance(physical, dict):
        raise ModelError("The external model must be a JSON object")
    validate(physical)
    return physical
