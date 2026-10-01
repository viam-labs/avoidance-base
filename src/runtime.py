"""Module parent robot, used to read the frame system."""

from __future__ import annotations

from typing import Optional

_MODULE: Optional[object] = None


def set_module(module: object) -> None:
    global _MODULE
    _MODULE = module


def get_parent_robot() -> Optional[object]:
    """Viam ``RobotClient`` connected to the module parent, if available."""
    if _MODULE is None:
        return None
    return getattr(_MODULE, "parent", None)
