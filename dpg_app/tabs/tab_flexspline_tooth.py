"""
Tab 2.1 - Flexspline Tooth Profile

Visualizes the three-segment tooth profile (AB convex, BC tangent, CD concave).
"""

import dearpygui.dearpygui as dpg
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from equations import compute_profile

from dpg_app.app_state import AppState
from dpg_app.widgets.parameter_panel import create_parameter_panel, create_button_row
from dpg_app.widgets.output_panel import (
    create_output_panel, update_output_values, create_info_text, update_info_text
)
from dpg_app.themes import COLORS

# Module-level cache for last computation result
_last_result = None


def create_tab_flexspline_tooth():
    """Create the Tab 2.1 content."""
    with dpg.group(horizontal=True):
        # Left panel - Parameters
        with dpg.child_window(width=340, border=True):
            create_parameter_panel(
                tag_prefix="tab21",
                on_change=_on_param_change,
                include_smooth=False,
                include_fillets=False
            )

            create_button_row(
                tag_prefix="tab21",
                on_update=_update_plot,
                on_reset=_reset_and_update
            )

            create_output_panel(
                tag_prefix="tab21",
                tab_type="flexspline_tooth"
            )

            create_info_text("tab21", "Click Update to compute tooth profile.")

        # Right panel - Plot
        with dpg.child_window(width=-1, border=True):
            _create_plot()


def _create_plot():
    """Create the visualization plot."""
    with dpg.plot(
        label="Flexspline Tooth Profile",
        tag="tab21_plot",
        width=-1,
        height=-1,
        equal_aspects=True,
        anti_aliased=True
    ):
        dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)

        # X axis
        dpg.add_plot_axis(
            dpg.mvXAxis,
            label="X_R (mm)",
            tag="tab21_x"
        )

        # Y axis
        with dpg.plot_axis(
            dpg.mvYAxis,
            label="Y_R (mm)",
            tag="tab21_y"
        ):
            # Placeholder series - will be populated on update
            pass


def _update_plot():
    """Update the plot with current parameters."""
    global _last_result

    # Read parameters from this tab's widgets
    params = AppState.read_from_widgets("tab21")

    # Compute profile
    result = compute_profile(params)

    if "error" in result:
        update_info_text("tab21", f"Error: {result['error']}", color=(255, 100, 100))
        return

    _last_result = result

    # Clear existing series
    _clear_plot_series()

    # Get Y axis for adding series
    y_axis = "tab21_y"

    # Draw reference zones (addendum, pitch, dedendum)
    _draw_reference_zones(result, params, y_axis)

    # Draw the three segments
    pts_AB = result["pts_AB"]
    pts_BC = result["pts_BC"]
    pts_CD = result["pts_CD"]

    # AB segment (convex) - Red
    if pts_AB:
        dpg.add_line_series(
            [p[0] for p in pts_AB],
            [p[1] for p in pts_AB],
            label="AB (convex)",
            tag="series_AB",
            parent=y_axis
        )
        dpg.bind_item_theme("series_AB", "theme_line_AB")

    # BC segment (tangent) - Blue
    if pts_BC:
        dpg.add_line_series(
            [p[0] for p in pts_BC],
            [p[1] for p in pts_BC],
            label="BC (tangent)",
            tag="series_BC",
            parent=y_axis
        )
        dpg.bind_item_theme("series_BC", "theme_line_BC")

    # CD segment (concave) - Green
    if pts_CD:
        dpg.add_line_series(
            [p[0] for p in pts_CD],
            [p[1] for p in pts_CD],
            label="CD (concave)",
            tag="series_CD",
            parent=y_axis
        )
        dpg.bind_item_theme("series_CD", "theme_line_CD")

    # Draw mirrored profile
    _draw_mirrored_profile(pts_AB, pts_BC, pts_CD, y_axis)

    # Draw circle centers
    _draw_circle_centers(result, y_axis)

    # Fit axes to data
    dpg.fit_axis_data("tab21_x")
    dpg.fit_axis_data("tab21_y")

    # Update output values
    _update_outputs(result)

    # Update status
    total_points = len(pts_AB) + len(pts_BC) + len(pts_CD)
    update_info_text("tab21", f"Profile computed: {total_points} points", color=(100, 255, 100))

    # Update main status bar
    from dpg_app.main import update_point_count
    update_point_count(total_points)


def _clear_plot_series():
    """Clear all existing plot series."""
    series_tags = [
        "series_AB", "series_BC", "series_CD",
        "series_mirror_AB", "series_mirror_BC", "series_mirror_CD",
        "series_addendum_zone", "series_pitch_zone", "series_dedendum_zone",
        "series_center_O1", "series_center_O2"
    ]
    for tag in series_tags:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)


def _draw_reference_zones(result, params, y_axis):
    """Draw reference zone lines (addendum, pitch, dedendum)."""
    ds = result["ds"]
    hf = params["hf"]
    ha = params["ha"]
    m = params["m"]

    # Y positions for reference lines
    y_add = ds + hf + ha  # Addendum
    y_pitch = ds + hf     # Pitch circle level
    y_ded = ds            # Dedendum

    # X range based on half tooth width
    x_min = -m * math.pi / 4
    x_max = m * math.pi / 2 + m * math.pi / 4

    # Addendum zone line
    dpg.add_line_series(
        [x_min, x_max],
        [y_add, y_add],
        label="Addendum",
        tag="series_addendum_zone",
        parent=y_axis
    )
    dpg.bind_item_theme("series_addendum_zone", "theme_line_addendum")

    # Pitch zone line
    dpg.add_line_series(
        [x_min, x_max],
        [y_pitch, y_pitch],
        label="Pitch",
        tag="series_pitch_zone",
        parent=y_axis
    )
    dpg.bind_item_theme("series_pitch_zone", "theme_line_pitch")

    # Dedendum zone line
    dpg.add_line_series(
        [x_min, x_max],
        [y_ded, y_ded],
        label="Dedendum",
        tag="series_dedendum_zone",
        parent=y_axis
    )
    dpg.bind_item_theme("series_dedendum_zone", "theme_line_dedendum")


def _draw_mirrored_profile(pts_AB, pts_BC, pts_CD, y_axis):
    """Draw the mirrored (symmetric) profile."""
    # Mirror about x=0
    if pts_AB:
        mirrored = [(-p[0], p[1]) for p in pts_AB]
        dpg.add_line_series(
            [p[0] for p in mirrored],
            [p[1] for p in mirrored],
            tag="series_mirror_AB",
            parent=y_axis
        )
        dpg.bind_item_theme("series_mirror_AB", "theme_line_mirror")

    if pts_BC:
        mirrored = [(-p[0], p[1]) for p in pts_BC]
        dpg.add_line_series(
            [p[0] for p in mirrored],
            [p[1] for p in mirrored],
            tag="series_mirror_BC",
            parent=y_axis
        )
        dpg.bind_item_theme("series_mirror_BC", "theme_line_mirror")

    if pts_CD:
        mirrored = [(-p[0], p[1]) for p in pts_CD]
        dpg.add_line_series(
            [p[0] for p in mirrored],
            [p[1] for p in mirrored],
            tag="series_mirror_CD",
            parent=y_axis
        )
        dpg.bind_item_theme("series_mirror_CD", "theme_line_mirror")


def _draw_circle_centers(result, y_axis):
    """Draw markers for circle centers O1 and O2."""
    x1 = result["x1_R"]
    y1 = result["y1_R"]
    x2 = result["x2_R"]
    y2 = result["y2_R"]

    # O1 center (small scatter around point to make it visible)
    dpg.add_scatter_series(
        [x1], [y1],
        label="O\u2081",
        tag="series_center_O1",
        parent=y_axis
    )

    # O2 center
    dpg.add_scatter_series(
        [x2], [y2],
        label="O\u2082",
        tag="series_center_O2",
        parent=y_axis
    )


def _update_outputs(result):
    """Update output panel with computed values."""
    # Convert angles to degrees for display
    alpha_deg = math.degrees(result["alpha"])
    delta_deg = math.degrees(result["delta"])

    values = {
        "s": result["s"],
        "t": result["t"],
        "ds": result["ds"],
        "alpha": alpha_deg,
        "delta": delta_deg,
        "l1": result["l1"],
        "l2": result["l2"],
        "l3": result["l3"],
        "h1": result["h1"],
        "rp": result["rp"],
    }

    update_output_values("tab21", values)


def _on_param_change():
    """Called when a parameter changes."""
    _update_plot()


def _reset_and_update():
    """Reset parameters and update plot."""
    AppState.reset_to_defaults()
    _update_plot()
