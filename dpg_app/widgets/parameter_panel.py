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

    with dpg.table(header_row=False, borders_innerH=False, borders_innerV=False,
                   borders_outerH=False, borders_outerV=False, policy=dpg.mvTable_SizingStretchProp):
        dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
        dpg.add_table_column(width_fixed=True, init_width_or_weight=scaled(130))

        with dpg.table_row():
            # Label column
            dpg.add_text(label, indent=10)

            # Input column
            if key in ("z_f", "z_c"):
                # Integer parameters
                input_tag = f"{tag_prefix}_param_{key}"
                dpg.add_input_int(
                    tag=input_tag,
                    default_value=int(default),
                    width=scaled(120),
                    min_value=1,
                    min_clamped=True
                )
            else:
                # Float parameters
                input_tag = f"{tag_prefix}_param_{key}"
                dpg.add_input_float(
                    tag=input_tag,
                    default_value=default,
                    width=scaled(120),
                    format="%.4f"
                )

    # Add handler for deactivation (click away or Enter)
    handler_tag = f"{input_tag}_handler"
    if dpg.does_item_exist(handler_tag):
        dpg.delete_item(handler_tag)
    with dpg.item_handler_registry(tag=handler_tag):
        dpg.add_item_deactivated_after_edit_handler(
            callback=_make_param_callback(key, on_change, input_tag)
        )
    dpg.bind_item_handler_registry(input_tag, handler_tag)

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
    with dpg.table(header_row=False, borders_innerH=False, borders_innerV=False,
                   borders_outerH=False, borders_outerV=False, policy=dpg.mvTable_SizingStretchProp):
        dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
        dpg.add_table_column(width_fixed=True, init_width_or_weight=scaled(130))

        with dpg.table_row():
            # Label column
            dpg.add_text(label, indent=10)

            # Input column
            input_tag = f"{tag_prefix}_param_{key}"
            dpg.add_input_float(
                tag=input_tag,
                default_value=default,
                width=scaled(120),
                format="%.4f"
            )

    # Add handler for deactivation (click away or Enter)
    handler_tag = f"{input_tag}_handler"
    if dpg.does_item_exist(handler_tag):
        dpg.delete_item(handler_tag)
    with dpg.item_handler_registry(tag=handler_tag):
        dpg.add_item_deactivated_after_edit_handler(
            callback=_make_special_callback(key, on_change, input_tag)
        )
    dpg.bind_item_handler_registry(input_tag, handler_tag)

    if tooltip:
        with dpg.tooltip(parent=input_tag):
            dpg.add_text(tooltip, wrap=250)


def _make_param_callback(key: str, on_change: Optional[Callable], widget_tag: str):
    """Create a callback for parameter changes."""
    def callback(sender, app_data, user_data=None):
        # Get value from the widget using stored tag
        value = dpg.get_value(widget_tag)

        if value is None:
            return

        # Update state
        AppState.set_param(key, float(value), record_undo=True)

        # Enforce harmonic drive constraint: z_c = z_f + 2
        if key == "z_f":
            new_z_c = int(value) + 2
            AppState.set_param("z_c", float(new_z_c), record_undo=False)
            # Update all z_c widgets across tabs
            _update_linked_widgets("z_c", new_z_c)
        elif key == "z_c":
            new_z_f = int(value) - 2
            AppState.set_param("z_f", float(new_z_f), record_undo=False)
            # Update all z_f widgets across tabs
            _update_linked_widgets("z_f", new_z_f)

        # Auto-update plot
        if on_change:
            on_change()

    return callback


def _update_linked_widgets(key: str, value):
    """Update all widgets for a parameter across all tabs."""
    # List of known tab prefixes
    tab_prefixes = ["tab21", "tab22", "tab_fs", "tab_cs", "tab_ov"]

    for prefix in tab_prefixes:
        widget_tag = f"{prefix}_param_{key}"
        if dpg.does_item_exist(widget_tag):
            dpg.set_value(widget_tag, int(value) if key in ("z_f", "z_c") else value)


def _make_special_callback(key: str, on_change: Optional[Callable], widget_tag: str):
    """Create a callback for special parameter changes (smooth, fillets)."""
    def callback(sender, app_data, user_data=None):
        # Get value from the widget using stored tag
        value = dpg.get_value(widget_tag)

        if value is None:
            return

        # Update appropriate state
        if key == "smooth":
            AppState.set_smooth(float(value))
        elif key == "fillet_add":
            AppState.set_fillet_add(float(value))
        elif key == "fillet_ded":
            AppState.set_fillet_ded(float(value))

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
