"""Application-specific dataset loaders for SURGE."""

from __future__ import annotations

from ..dataset import SurrogateDataset

from .m3dc1 import M3DC1Dataset
from .xgc import XGCDataset

__all__ = [
    "SurrogateDataset",
    "M3DC1Dataset",
    "XGCDataset",
]
