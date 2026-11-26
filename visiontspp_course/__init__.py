"""Course improvement modules built on top of VisionTS++."""

from .model import VisionTSppCourse
from .grouping import balanced_group_sizes, build_static_groups, compute_window_permutation

__all__ = [
    "VisionTSppCourse",
    "balanced_group_sizes",
    "build_static_groups",
    "compute_window_permutation",
]
