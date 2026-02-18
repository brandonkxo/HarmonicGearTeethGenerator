"""
Reusable output display panel widget.

Provides category dropdown and formatted output values.
"""

import dearpygui.dearpygui as dpg
from typing import Dict, List, Callable, Optional, Any
import os

from dpg_app.app_state import scaled

# Path to reference images
REF_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Ref Images")

# Categories that have reference images
CATEGORY_REF_IMAGES = {
    "Circle Radii": "Flexspline Ref Radii.png",
}

# Track loaded textures
_loaded_textures = {}


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
        "Circle Radii": ["rp", "rm", "rb", "r_add", "r_ded"],
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
    },
}

# Display labels for output values
OUTPUT_LABELS = {
    "s": "Ring wall s",
    "t": "Cup wall t",
    "ds": "Neutral offset ds",
    "alpha": "Angle α",
    "delta": "Angle δ",
    "l1": "Arc length l₁",
    "l2": "Line length l₂",
    "l3": "Arc length l₃",
    "h1": "Height h₁",
    "rp": "Pitch radius rp",
    "rp_c": "Pitch radius rpc",
    "rm": "Neutral radius rm",
    "rb": "Inner radius rb",
    "r_add": "Addendum radius",
    "r_ded": "Dedendum radius",
    "ha": "Addendum",
    "hf": "Dedendum",
    "z_f": "Flexspline teeth",
    "z_c": "Circular spline teeth",
    "chain_points": "Chain points",
    "count_AB": "Segment AB points",
    "count_BC": "Segment BC points",
    "count_CD": "Segment CD points",
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
    "rb": "mm",
    "r_add": "mm",
    "r_ded": "mm",
    "ha": "mm",
    "hf": "mm",
}


def _load_texture(image_path: str) -> Optional[int]:
    """Load an image as a texture, returning the texture tag."""
    if image_path in _loaded_textures:
        return _loaded_textures[image_path]

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    try:
        width, height, channels, data = dpg.load_image(image_path)

        # Create texture registry if it doesn't exist
        if not dpg.does_item_exist("ref_texture_registry"):
            dpg.add_texture_registry(tag="ref_texture_registry")

        texture_tag = f"texture_{os.path.basename(image_path)}"
        if not dpg.does_item_exist(texture_tag):
            dpg.add_static_texture(
                width=width,
                height=height,
                default_value=data,
                tag=texture_tag,
                parent="ref_texture_registry"
            )

        _loaded_textures[image_path] = (texture_tag, width, height)
        return _loaded_textures[image_path]
    except Exception as e:
        print(f"Error loading texture: {e}")
        return None


def show_reference_image(sender, app_data, user_data):
    """Show reference image for a category in a popup window."""
    category = user_data
    print(f"show_reference_image called with category: {category}")

    if category not in CATEGORY_REF_IMAGES:
        print(f"Category '{category}' not in CATEGORY_REF_IMAGES")
        return

    image_file = CATEGORY_REF_IMAGES[category]
    image_path = os.path.join(REF_IMAGES_DIR, image_file)
    print(f"Image path: {image_path}")
    print(f"Image exists: {os.path.exists(image_path)}")

    # Clean up existing window
    if dpg.does_item_exist("ref_image_window"):
        dpg.delete_item("ref_image_window")

    # Load texture
    texture_info = _load_texture(image_path)
    print(f"Texture info: {texture_info}")
    if texture_info is None:
        print("Failed to load texture")
        return

    texture_tag, img_width, img_height = texture_info

    # Scale image to fit reasonably on screen
    max_width = scaled(800)
    max_height = scaled(600)

    scale = min(max_width / img_width, max_height / img_height, 1.0)
    display_width = int(img_width * scale)
    display_height = int(img_height * scale)

    # Create popup window
    window_width = display_width + scaled(20)
    window_height = display_height + scaled(50)

    print(f"Creating window with size {window_width}x{window_height}")
    with dpg.window(
        label=f"{category} Reference",
        tag="ref_image_window",
        modal=True,
        width=window_width,
        height=window_height,
        pos=(dpg.get_viewport_width() // 2 - window_width // 2,
             dpg.get_viewport_height() // 2 - window_height // 2),
        no_resize=False,
        no_collapse=True,
        on_close=lambda: dpg.delete_item("ref_image_window")
    ):
        dpg.add_image(texture_tag, width=display_width, height=display_height)
        dpg.add_spacer(height=5)
        dpg.add_button(
            label="Close",
            callback=lambda: dpg.delete_item("ref_image_window"),
            width=-1
        )
    print("Window created successfully")


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
        height=scaled(120),
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

        # Add help buttons for categories with reference images (initially hidden)
        for category in categories.keys():
            if category in CATEGORY_REF_IMAGES:
                dpg.add_spacer(height=5, tag=f"{tag_prefix}_help_spacer_{category}", show=False)
                dpg.add_button(
                    label="Explain these radii?",
                    tag=f"{tag_prefix}_help_btn_{category}",
                    callback=show_reference_image,
                    user_data=category,
                    show=False,
                    small=True
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

    # Hide all outputs and help buttons first
    for cat, cat_keys in categories.items():
        for key in cat_keys:
            widget_tag = f"{tag_prefix}_out_{key}"
            if dpg.does_item_exist(widget_tag):
                dpg.configure_item(widget_tag, show=False)

        # Hide help buttons
        help_btn_tag = f"{tag_prefix}_help_btn_{cat}"
        help_spacer_tag = f"{tag_prefix}_help_spacer_{cat}"
        if dpg.does_item_exist(help_btn_tag):
            dpg.configure_item(help_btn_tag, show=False)
        if dpg.does_item_exist(help_spacer_tag):
            dpg.configure_item(help_spacer_tag, show=False)

    # Show outputs for selected category
    selected_keys = categories.get(category, [])
    for key in selected_keys:
        widget_tag = f"{tag_prefix}_out_{key}"
        if dpg.does_item_exist(widget_tag):
            dpg.configure_item(widget_tag, show=True)

    # Show help button if this category has a reference image
    if category in CATEGORY_REF_IMAGES:
        help_btn_tag = f"{tag_prefix}_help_btn_{category}"
        help_spacer_tag = f"{tag_prefix}_help_spacer_{category}"
        if dpg.does_item_exist(help_btn_tag):
            dpg.configure_item(help_btn_tag, show=True)
        if dpg.does_item_exist(help_spacer_tag):
            dpg.configure_item(help_spacer_tag, show=True)


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
        wrap=scaled(310),
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
