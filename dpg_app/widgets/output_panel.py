"""
Reusable output display panel widget.

Provides category dropdown and formatted output values.
"""

import dearpygui.dearpygui as dpg
from typing import Dict, List, Callable, Optional, Any


# Output categories for each tab type
OUTPUT_CATEGORIES = {
    "flexspline_tooth": {
        "Wall Thickness": ["s", "t", "ds"],
        "Deformation Angles": ["alpha", "delta"],
        "Arc Lengths": ["l1", "l2", "l3", "h1"],
        "Pitch Geometry": ["rp"],
    },
    "conjugate_tooth": {
        "Wall Thickness": ["s", "t"],
        "Pitch Geometry": ["rp_c"],
        "Segment Counts": ["count_AB", "count_BC", "count_CD"],
    },
    "flexspline_full": {
        "Wall Thickness": ["s", "t", "ds"],
        "Circle Radii": ["rp", "rm", "r_add", "r_ded"],
        "Tooth Heights": ["ha", "hf"],
        "Mesh Info": ["z_f", "chain_points"],
    },
    "circular_spline": {
        "Wall Thickness": ["s", "t"],
        "Circle Radii": ["rp_c", "r_add", "r_ded"],
        "Tooth Heights": ["ha", "hf"],
        "Mesh Info": ["z_c", "chain_points"],
    },
    "radial_modification": {
        "Flexspline": ["z_f", "rp"],
        "Circular Spline": ["z_c", "rp_c"],
        "Modification": ["d_max", "interference_count"],
    },
}

# Display labels for output values
OUTPUT_LABELS = {
    "s": "Ring wall s",
    "t": "Cup wall t",
    "ds": "Neutral offset ds",
    "alpha": "Angle \u03b1",
    "delta": "Angle \u03b4",
    "l1": "Arc length l\u2081",
    "l2": "Line length l\u2082",
    "l3": "Arc length l\u2083",
    "h1": "Height h\u2081",
    "rp": "Pitch radius rp",
    "rp_c": "Pitch radius rp_c",
    "rm": "Neutral radius rm",
    "r_add": "Addendum radius",
    "r_ded": "Dedendum radius",
    "ha": "Addendum h\u2090",
    "hf": "Dedendum h\u1DA0",
    "z_f": "Flexspline teeth",
    "z_c": "Circular spline teeth",
    "chain_points": "Chain points",
    "count_AB": "Segment AB points",
    "count_BC": "Segment BC points",
    "count_CD": "Segment CD points",
    "d_max": "Max penetration d_max",
    "interference_count": "Interference points",
}

# Units for output values
OUTPUT_UNITS = {
    "s": "mm",
    "t": "mm",
    "ds": "mm",
    "alpha": "\u00b0",
    "delta": "\u00b0",
    "l1": "mm",
    "l2": "mm",
    "l3": "mm",
    "h1": "mm",
    "rp": "mm",
    "rp_c": "mm",
    "rm": "mm",
    "r_add": "mm",
    "r_ded": "mm",
    "ha": "mm",
    "hf": "mm",
    "d_max": "mm",
}


def create_output_panel(
    tag_prefix: str,
    tab_type: str,
    on_category_change: Optional[Callable] = None
):
    """
    Create an output display panel with category dropdown.

    Args:
        tag_prefix: Prefix for widget tags
        tab_type: Type of tab (key in OUTPUT_CATEGORIES)
        on_category_change: Callback when category changes
    """
    categories = OUTPUT_CATEGORIES.get(tab_type, {})
    category_names = list(categories.keys())

    if not category_names:
        dpg.add_text("No output categories defined", color=(150, 150, 150))
        return

    dpg.add_spacer(height=10)
    dpg.add_separator()
    dpg.add_spacer(height=5)

    dpg.add_text("Output", color=(180, 180, 255))
    dpg.add_spacer(height=5)

    # Category dropdown
    dpg.add_combo(
        items=category_names,
        tag=f"{tag_prefix}_output_category",
        default_value=category_names[0],
        width=-1,
        callback=lambda s, a: _on_category_selected(tag_prefix, tab_type, a, on_category_change)
    )

    dpg.add_spacer(height=5)

    # Output display area
    with dpg.child_window(
        tag=f"{tag_prefix}_output_display",
        height=120,
        border=False
    ):
        # Create text widgets for each possible output value (initially hidden)
        for category, keys in categories.items():
            for key in keys:
                dpg.add_text(
                    "",
                    tag=f"{tag_prefix}_out_{key}",
                    show=False,
                    color=(200, 200, 200)
                )

    # Show initial category
    _show_category_outputs(tag_prefix, tab_type, category_names[0])


def _on_category_selected(
    tag_prefix: str,
    tab_type: str,
    category: str,
    callback: Optional[Callable]
):
    """Handle category selection change."""
    _show_category_outputs(tag_prefix, tab_type, category)
    if callback:
        callback(category)


def _show_category_outputs(tag_prefix: str, tab_type: str, category: str):
    """Show outputs for the selected category, hide others."""
    categories = OUTPUT_CATEGORIES.get(tab_type, {})

    # Hide all outputs first
    for cat_keys in categories.values():
        for key in cat_keys:
            widget_tag = f"{tag_prefix}_out_{key}"
            if dpg.does_item_exist(widget_tag):
                dpg.configure_item(widget_tag, show=False)

    # Show outputs for selected category
    selected_keys = categories.get(category, [])
    for key in selected_keys:
        widget_tag = f"{tag_prefix}_out_{key}"
        if dpg.does_item_exist(widget_tag):
            dpg.configure_item(widget_tag, show=True)


def update_output_values(tag_prefix: str, values: Dict[str, Any]):
    """
    Update output display with new values.

    Args:
        tag_prefix: Widget tag prefix
        values: Dictionary of key -> value pairs
    """
    for key, value in values.items():
        widget_tag = f"{tag_prefix}_out_{key}"
        if dpg.does_item_exist(widget_tag):
            label = OUTPUT_LABELS.get(key, key)
            unit = OUTPUT_UNITS.get(key, "")

            if isinstance(value, float):
                if abs(value) < 0.0001 and value != 0:
                    text = f"{label}: {value:.6f} {unit}"
                else:
                    text = f"{label}: {value:.4f} {unit}"
            elif isinstance(value, int):
                text = f"{label}: {value}"
            else:
                text = f"{label}: {value}"

            dpg.set_value(widget_tag, text)


def create_info_text(tag_prefix: str, initial_text: str = "Click Update to compute."):
    """Create a simple info text display."""
    dpg.add_spacer(height=10)
    dpg.add_separator()
    dpg.add_spacer(height=5)

    dpg.add_text("Status", color=(180, 180, 255))
    dpg.add_text(
        initial_text,
        tag=f"{tag_prefix}_info",
        wrap=310,
        color=(150, 150, 150)
    )


def update_info_text(tag_prefix: str, text: str, color: tuple = None):
    """Update the info text display."""
    widget_tag = f"{tag_prefix}_info"
    if dpg.does_item_exist(widget_tag):
        dpg.set_value(widget_tag, text)
        if color:
            dpg.configure_item(widget_tag, color=color)


def create_legend(tag_prefix: str, items: List[tuple]):
    """
    Create a color legend.

    Args:
        tag_prefix: Widget tag prefix
        items: List of (label, color_tuple) pairs
    """
    dpg.add_spacer(height=10)
    dpg.add_separator()
    dpg.add_spacer(height=5)

    dpg.add_text("Legend", color=(180, 180, 255))
    dpg.add_spacer(height=3)

    for label, color in items:
        with dpg.group(horizontal=True):
            # Color swatch
            with dpg.drawlist(width=16, height=16):
                dpg.draw_rectangle(
                    (0, 0), (16, 16),
                    fill=color,
                    color=color
                )
            dpg.add_spacer(width=5)
            dpg.add_text(label, color=(180, 180, 180))
