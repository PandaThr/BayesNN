"""
Public interface for the BayesNN package.
"""

from .BayesianLinear import BayesianLinear
from .loss import (
    _Loss,
    BNN_VBLoss,
    BNN_AlphaDivergenceLoss,
)

__all__ = [
    "BayesianLinear",
    "_Loss",
    "BNN_VBLoss",
    "BNN_AlphaDivergenceLoss",
]

__version__ = "0.1.0"
