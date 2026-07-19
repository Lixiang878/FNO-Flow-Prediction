"""fno-flow-prediction: FNO vs UNet for parametric PDE surrogate modelling.

The package is intentionally split into a zero-dependency core (numpy) that can
generate data and run the model *architectures* offline, plus an optional torch
path (lazy-imported) used only for real training. This mirrors the "offline-first,
real model optional" pattern of the sibling portfolio projects.
"""

from .data import burgers_solver, generate_dataset
from .models import FNO1D, UNet1D
from .baseline import lowres_solver_error, relative_l2

__all__ = [
    "burgers_solver",
    "generate_dataset",
    "FNO1D",
    "UNet1D",
    "lowres_solver_error",
    "relative_l2",
]
