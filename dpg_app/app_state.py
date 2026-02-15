"""
Centralized application state management.

Replaces tkinter's StringVar system with a cleaner dataclass-based approach
using DearPyGui's value registry for reactive updates.
"""

import dearpygui.dearpygui as dpg
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
import copy

# Global DPI scale factor (set by main.py on startup)
_dpi_scale = 1.0


def set_dpi_scale(scale: float):
    """Set the global DPI scale factor."""
    global _dpi_scale
    _dpi_scale = scale


def get_dpi_scale() -> float:
    """Get the global DPI scale factor."""
    return _dpi_scale


def scaled(value: int) -> int:
    """Scale a pixel value by the DPI scale factor."""
    return int(value * _dpi_scale)

# Import defaults from equations module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from equations import DEFAULTS, PARAM_ORDER, PARAM_LABELS


# Parameter tooltips for user guidance
PARAM_TOOLTIPS = {
    "m": "Module - tooth size scaling factor (mm)",
    "z_f": "Number of flexspline teeth (typically 100)",
    "z_c": "Number of circular spline teeth (z_f + 2)",
    "w0": "Maximum radial deformation of wave generator (mm)",
    "r1": "Radius of convex arc AB (mm)",
    "c1": "X-offset of convex arc center O1 (mm)",
    "e1": "Y-offset of convex arc center O1 (mm)",
    "r2": "Radius of concave arc CD (mm)",
    "c2": "X-offset of concave arc center O2 (mm)",
    "e2": "Y-offset of concave arc center O2 (mm)",
    "ha": "Addendum height - tooth tip above pitch circle (mm)",
    "hf": "Dedendum height - tooth root below pitch circle (mm)",
    "mu_s": "Ring wall coefficient: s = mu_s * m * z_f",
    "mu_t": "Cup wall coefficient: t = mu_t * s",
}

# Parameter groupings for collapsible UI sections
PARAM_GROUPS = {
    "Basic Geometry": ["m", "z_f", "z_c", "w0"],
    "Convex Arc (AB)": ["r1", "c1", "e1"],
    "Concave Arc (CD)": ["r2", "c2", "e2"],
    "Tooth Heights": ["ha", "hf"],
    "Wall Coefficients": ["mu_s", "mu_t"],
}


class UndoManager:
    """Manages undo/redo history for parameter states."""

    def __init__(self, max_history: int = 50):
        self._history: List[Dict[str, float]] = []
        self._position: int = -1
        self._max_history = max_history

    def push(self, state: Dict[str, float]):
        """Push a new state onto the history stack."""
        # Truncate forward history if we're not at the end
        self._history = self._history[:self._position + 1]
        self._history.append(copy.deepcopy(state))

        # Limit history size
        if len(self._history) > self._max_history:
            self._history.pop(0)
        else:
            self._position += 1

    def undo(self) -> Optional[Dict[str, float]]:
        """Return the previous state, or None if at beginning."""
        if self._position > 0:
            self._position -= 1
            return copy.deepcopy(self._history[self._position])
        return None

    def redo(self) -> Optional[Dict[str, float]]:
        """Return the next state, or None if at end."""
        if self._position < len(self._history) - 1:
            self._position += 1
            return copy.deepcopy(self._history[self._position])
        return None

    def can_undo(self) -> bool:
        return self._position > 0

    def can_redo(self) -> bool:
        return self._position < len(self._history) - 1

    def clear(self):
        """Clear all history."""
        self._history.clear()
        self._position = -1


class AppState:
    """
    Centralized application state manager.

    Manages all parameter values, fillet radii, smoothing factor,
    and provides change notification callbacks.
    """

    # Class-level state (singleton pattern)
    _params: Dict[str, float] = {}
    _smooth: float = 0.001
    _fillet_add: float = 0.15
    _fillet_ded: float = 0.1
    _auto_update: bool = False
    _undo_manager: UndoManager = UndoManager()
    _change_callbacks: List[Callable] = []
    _initialized: bool = False

    @classmethod
    def initialize(cls):
        """Initialize state with default values."""
        if cls._initialized:
            return

        cls._params = {key: float(DEFAULTS[key]) for key in PARAM_ORDER}
        cls._smooth = 0.001
        cls._fillet_add = 0.15
        cls._fillet_ded = 0.1
        cls._auto_update = False
        cls._undo_manager = UndoManager()
        cls._change_callbacks = []

        # Push initial state to undo history
        cls._undo_manager.push(cls._params.copy())
        cls._initialized = True

    @classmethod
    def get_params(cls) -> Dict[str, float]:
        """Get a copy of all parameter values."""
        return cls._params.copy()

    @classmethod
    def get_param(cls, key: str) -> float:
        """Get a single parameter value."""
        return cls._params.get(key, 0.0)

    # Tab prefixes for widget synchronization
    _TAB_PREFIXES = ["tab21", "tab22", "tab_fs", "tab_cs", "tab_ov"]

    @classmethod
    def set_param(cls, key: str, value: float, record_undo: bool = True):
        """Set a single parameter value."""
        if key in cls._params:
            old_value = cls._params[key]
            cls._params[key] = value

            # Update DearPyGui widgets in all tabs
            for prefix in cls._TAB_PREFIXES:
                widget_tag = f"{prefix}_param_{key}"
                if dpg.does_item_exist(widget_tag):
                    dpg.set_value(widget_tag, value)

            # Record for undo if value changed
            if record_undo and old_value != value:
                cls._undo_manager.push(cls._params.copy())

            # Notify callbacks
            cls._notify_change(key, value)

    @classmethod
    def update_params(cls, params: Dict[str, float], record_undo: bool = True):
        """Update multiple parameters at once."""
        changed = False
        for key, value in params.items():
            if key in cls._params and cls._params[key] != value:
                cls._params[key] = value
                changed = True

                # Update DearPyGui widgets in all tabs
                for prefix in cls._TAB_PREFIXES:
                    widget_tag = f"{prefix}_param_{key}"
                    if dpg.does_item_exist(widget_tag):
                        dpg.set_value(widget_tag, value)

        if changed and record_undo:
            cls._undo_manager.push(cls._params.copy())
            cls._notify_change("*", None)

    @classmethod
    def reset_to_defaults(cls):
        """Reset all parameters to default values."""
        cls._params = {key: float(DEFAULTS[key]) for key in PARAM_ORDER}
        cls._smooth = 0.001
        cls._fillet_add = 0.15
        cls._fillet_ded = 0.1

        # Update all DearPyGui widgets in all tabs
        for key, value in cls._params.items():
            for prefix in cls._TAB_PREFIXES:
                widget_tag = f"{prefix}_param_{key}"
                if dpg.does_item_exist(widget_tag):
                    dpg.set_value(widget_tag, value)

        # Update smooth and fillet widgets
        for prefix in cls._TAB_PREFIXES:
            smooth_tag = f"{prefix}_param_smooth"
            if dpg.does_item_exist(smooth_tag):
                dpg.set_value(smooth_tag, cls._smooth)

            fillet_add_tag = f"{prefix}_param_fillet_add"
            if dpg.does_item_exist(fillet_add_tag):
                dpg.set_value(fillet_add_tag, cls._fillet_add)

            fillet_ded_tag = f"{prefix}_param_fillet_ded"
            if dpg.does_item_exist(fillet_ded_tag):
                dpg.set_value(fillet_ded_tag, cls._fillet_ded)

        cls._undo_manager.push(cls._params.copy())
        cls._notify_change("*", None)

    @classmethod
    def get_smooth(cls) -> float:
        return cls._smooth

    @classmethod
    def set_smooth(cls, value: float):
        cls._smooth = value
        for prefix in cls._TAB_PREFIXES:
            widget_tag = f"{prefix}_param_smooth"
            if dpg.does_item_exist(widget_tag):
                dpg.set_value(widget_tag, value)

    @classmethod
    def get_fillet_add(cls) -> float:
        return cls._fillet_add

    @classmethod
    def set_fillet_add(cls, value: float):
        cls._fillet_add = value
        for prefix in cls._TAB_PREFIXES:
            widget_tag = f"{prefix}_param_fillet_add"
            if dpg.does_item_exist(widget_tag):
                dpg.set_value(widget_tag, value)

    @classmethod
    def get_fillet_ded(cls) -> float:
        return cls._fillet_ded

    @classmethod
    def set_fillet_ded(cls, value: float):
        cls._fillet_ded = value
        for prefix in cls._TAB_PREFIXES:
            widget_tag = f"{prefix}_param_fillet_ded"
            if dpg.does_item_exist(widget_tag):
                dpg.set_value(widget_tag, value)

    @classmethod
    def get_auto_update(cls) -> bool:
        return cls._auto_update

    @classmethod
    def set_auto_update(cls, value: bool):
        cls._auto_update = value

    @classmethod
    def undo(cls):
        """Undo the last parameter change."""
        state = cls._undo_manager.undo()
        if state:
            cls.update_params(state, record_undo=False)

    @classmethod
    def redo(cls):
        """Redo the last undone change."""
        state = cls._undo_manager.redo()
        if state:
            cls.update_params(state, record_undo=False)

    @classmethod
    def add_change_callback(cls, callback: Callable):
        """Register a callback for parameter changes."""
        cls._change_callbacks.append(callback)

    @classmethod
    def remove_change_callback(cls, callback: Callable):
        """Remove a change callback."""
        if callback in cls._change_callbacks:
            cls._change_callbacks.remove(callback)

    @classmethod
    def _notify_change(cls, key: str, value: Any):
        """Notify all registered callbacks of a change."""
        for callback in cls._change_callbacks:
            try:
                callback(key, value)
            except Exception as e:
                print(f"Error in change callback: {e}")

    @classmethod
    def read_from_widgets(cls, tag_prefix: str = None) -> Dict[str, float]:
        """Read current values from DearPyGui widgets into params dict.

        Args:
            tag_prefix: Tab-specific prefix (e.g., "tab21", "tab_fs"). If None,
                        tries to find any matching widget.
        """
        params = {}

        # List of possible prefixes if none specified
        prefixes = [tag_prefix] if tag_prefix else ["tab21", "tab22", "tab_fs", "tab_cs", "tab_ov"]

        for key in PARAM_ORDER:
            value = None
            for prefix in prefixes:
                widget_tag = f"{prefix}_param_{key}"
                if dpg.does_item_exist(widget_tag):
                    value = dpg.get_value(widget_tag)
                    break

            if value is not None:
                params[key] = value
            else:
                params[key] = cls._params.get(key, DEFAULTS.get(key, 0.0))

        # Also read smooth and fillet values
        for prefix in prefixes:
            smooth_tag = f"{prefix}_param_smooth"
            if dpg.does_item_exist(smooth_tag):
                cls._smooth = dpg.get_value(smooth_tag)
                break

        for prefix in prefixes:
            fillet_add_tag = f"{prefix}_param_fillet_add"
            if dpg.does_item_exist(fillet_add_tag):
                cls._fillet_add = dpg.get_value(fillet_add_tag)
                break

        for prefix in prefixes:
            fillet_ded_tag = f"{prefix}_param_fillet_ded"
            if dpg.does_item_exist(fillet_ded_tag):
                cls._fillet_ded = dpg.get_value(fillet_ded_tag)
                break

        cls._params = params
        return params


def get_param_label(key: str) -> str:
    """Get the display label for a parameter."""
    return PARAM_LABELS.get(key, key)


def get_param_tooltip(key: str) -> str:
    """Get the tooltip text for a parameter."""
    return PARAM_TOOLTIPS.get(key, "")


def get_param_groups() -> Dict[str, List[str]]:
    """Get parameter groupings for UI organization."""
    return PARAM_GROUPS.copy()
