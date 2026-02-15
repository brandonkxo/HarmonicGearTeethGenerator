"""
Reusable parameter input panel widget.

Provides collapsible parameter groups with validation and tooltips.
"""

import dearpygui.dearpygui as dpg
from typing import Callable, Optional

from dpg_app.app_state import (
    AppState, PARAM_GROUPS, get_param_label, get_param_tooltip, scaled
)
from equations import DEFAULTS, PARAM_ORDER


def create_parameter_panel(
    tag_prefix: str,
    on_change: Optional[Callable] = None,
    include_smooth: bool = False,
    include_fillets: bool = False
):
    """
    Create a parameter input panel with collapsible groups.

    Args:
        tag_prefix: Prefix for widget tags (e.g., "tab21")
        on_change: Callback when any parameter changes
        include_smooth: Whether to include smoothing factor input
        include_fillets: Whether to include fillet radius inputs
    """
    dpg.add_text("Parameters", color=(180, 180, 255))
    dpg.add_separator()
    dpg.add_spacer(height=5)

    # Create collapsible groups
    for group_name, param_keys in PARAM_GROUPS.items():
        # Default some groups to closed
        default_open = group_name not in ["Wall Coefficients"]

        with dpg.collapsing_header(label=group_name, default_open=default_open):
            for key in param_keys:
                _create_param_input(
                    key=key,
                    tag_prefix=tag_prefix,
                    on_change=on_change
                )
            dpg.add_spacer(height=3)

    # Smoothing factor (optional)
    if include_smooth:
        dpg.add_spacer(height=5)
        dpg.add_separator()
        dpg.add_spacer(height=5)
        _create_special_input(
            key="smooth",
            label="Smoothing (s)",
            default=AppState.get_smooth(),
            tooltip="B-spline smoothing factor for conjugate profile",
            tag_prefix=tag_prefix,
            on_change=on_change
        )

    # Fillet radii (optional)
    if include_fillets:
        dpg.add_spacer(height=5)
        dpg.add_separator()
        dpg.add_spacer(height=5)
        dpg.add_text("Fillet Radii", color=(150, 150, 150))

        _create_special_input(
            key="fillet_add",
            label="Addendum fillet r",
            default=AppState.get_fillet_add(),
            tooltip="Fillet radius at tooth tip (addendum)",
            tag_prefix=tag_prefix,
            on_change=on_change
        )

        _create_special_input(
            key="fillet_ded",
            label="Dedendum fillet r",
            default=AppState.get_fillet_ded(),
            tooltip="Fillet radius at tooth root (dedendum)",
            tag_prefix=tag_prefix,
            on_change=on_change
        )



def _create_param_input(
    key: str,
    tag_prefix: str,
    on_change: Optional[Callable] = None
):
    """Create a single parameter input with label and tooltip."""
    label = get_param_label(key)
    tooltip = get_param_tooltip(key)
    default = float(DEFAULTS.get(key, 0.0))

    with dpg.group(horizontal=True):
        dpg.add_text(label, indent=10)
        dpg.add_spacer()

        # Determine format based on parameter type
        if key in ("z_f", "z_c"):
            # Integer parameters
            input_tag = f"{tag_prefix}_param_{key}"
            dpg.add_input_int(
                tag=input_tag,
                default_value=int(default),
                width=scaled(130),
                min_value=1,
                min_clamped=True,
                callback=_make_param_callback(key, on_change)
            )
        else:
            # Float parameters
            input_tag = f"{tag_prefix}_param_{key}"
            dpg.add_input_float(
                tag=input_tag,
                default_value=default,
                width=scaled(130),
                format="%.4f",
                callback=_make_param_callback(key, on_change)
            )

        # Add tooltip
        if tooltip:
            with dpg.tooltip(parent=input_tag):
                dpg.add_text(tooltip, wrap=250)


def _create_special_input(
    key: str,
    label: str,
    default: float,
    tooltip: str,
    tag_prefix: str,
    on_change: Optional[Callable] = None
):
    """Create a special parameter input (smooth, fillets)."""
    with dpg.group(horizontal=True):
        dpg.add_text(label, indent=10)
        dpg.add_spacer()

        input_tag = f"{tag_prefix}_param_{key}"
        dpg.add_input_float(
            tag=input_tag,
            default_value=default,
            width=scaled(130),
            format="%.4f",
            callback=_make_special_callback(key, on_change)
        )

        if tooltip:
            with dpg.tooltip(parent=input_tag):
                dpg.add_text(tooltip, wrap=250)


def _make_param_callback(key: str, on_change: Optional[Callable]):
    """Create a callback for parameter changes."""
    def callback(sender, app_data, user_data):
        # Update state
        AppState.set_param(key, float(app_data), record_undo=True)

        # Auto-update plot
        if on_change:
            on_change()

    return callback


def _make_special_callback(key: str, on_change: Optional[Callable]):
    """Create a callback for special parameter changes (smooth, fillets)."""
    def callback(sender, app_data, user_data):
        # Update appropriate state
        if key == "smooth":
            AppState.set_smooth(float(app_data))
        elif key == "fillet_add":
            AppState.set_fillet_add(float(app_data))
        elif key == "fillet_ded":
            AppState.set_fillet_ded(float(app_data))

        # Auto-update plot
        if on_change:
            on_change()

    return callback


def create_button_row(
    tag_prefix: str,
    on_update: Callable,
    on_reset: Optional[Callable] = None,
    include_export: bool = False,
    on_export: Optional[Callable] = None,
    export_formats: list = None
):
    """
    Create a row of action buttons.

    Args:
        tag_prefix: Prefix for widget tags
        on_update: Callback for Update button
        on_reset: Callback for Reset button (defaults to AppState.reset_to_defaults)
        include_export: Whether to include Export button
        on_export: Callback for Export button
        export_formats: List of export format options
    """
    dpg.add_spacer(height=10)

    with dpg.group(horizontal=True):
        # Update button
        update_btn = dpg.add_button(
            label="Update",
            tag=f"btn_update_{tag_prefix}",
            callback=on_update,
            width=scaled(95)
        )
        dpg.bind_item_theme(update_btn, "theme_button_update")

        # Reset button
        reset_callback = on_reset if on_reset else lambda: AppState.reset_to_defaults()
        reset_btn = dpg.add_button(
            label="Reset",
            callback=reset_callback,
            width=scaled(95)
        )
        dpg.bind_item_theme(reset_btn, "theme_button_reset")

        # Export button (optional)
        if include_export and on_export:
            export_btn = dpg.add_button(
                label="Export",
                callback=on_export,
                width=scaled(95)
            )
            dpg.bind_item_theme(export_btn, "theme_button_export")

    # Export format selector (optional)
    if include_export and export_formats:
        dpg.add_spacer(height=5)
        with dpg.group(horizontal=True):
            dpg.add_text("Format:", indent=10)
            dpg.add_combo(
                items=export_formats,
                tag=f"{tag_prefix}_export_format",
                default_value=export_formats[0],
                width=scaled(120)
            )
