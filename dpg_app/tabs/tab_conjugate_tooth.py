"""
Tab 2.2 - Conjugate Circular Spline Tooth Profile

Visualizes the conjugate profile with envelope points and B-spline curve.
"""

import dearpygui.dearpygui as dpg
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from equations import compute_profile, compute_conjugate_profile, smooth_conjugate_profile

from dpg_app.app_state import AppState
from dpg_app.widgets.parameter_panel import create_parameter_panel, create_button_row
from dpg_app.widgets.output_panel import (
    create_output_panel, update_output_values, create_info_text, update_info_text
)
from dpg_app.export_manager import show_export_dialog

# Module-level cache
_last_result = None
_last_smoothed = None


def create_tab_conjugate_tooth():
    """Create the Tab 2.2 content."""
    with dpg.group(horizontal=True):
        # Left panel - Parameters
        with dpg.child_window(width=340, border=True):
            create_parameter_panel(
                tag_prefix="tab22",
                on_change=_on_param_change,
                include_smooth=True,
                include_fillets=False
            )

            create_button_row(
                tag_prefix="tab22",
                on_update=_update_plot,
                on_reset=_reset_and_update,
                include_export=True,
                on_export=_export_txt
            )

            # Segment visibility controls
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=5)
            dpg.add_text("Show Segments", color=(180, 180, 255))

            dpg.add_checkbox(label="AB (convex)", tag="tab22_show_AB", default_value=True,
                           callback=lambda: _update_plot())
            dpg.add_checkbox(label="BC (tangent)", tag="tab22_show_BC", default_value=True,
                           callback=lambda: _update_plot())
            dpg.add_checkbox(label="CD (concave)", tag="tab22_show_CD", default_value=True,
                           callback=lambda: _update_plot())

            create_output_panel(
                tag_prefix="tab22",
                tab_type="conjugate_tooth"
            )

            create_info_text("tab22", "Click Update to compute conjugate profile.")

        # Right panel - Plot
        with dpg.child_window(width=-1, border=True):
            _create_plot()


def _create_plot():
    """Create the visualization plot."""
    with dpg.plot(
        label="Conjugate Circular Spline Tooth",
        tag="tab22_plot",
        width=-1,
        height=-1,
        equal_aspects=True,
        anti_aliased=True
    ):
        dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)

        dpg.add_plot_axis(dpg.mvXAxis, label="X (mm)", tag="tab22_x")
        with dpg.plot_axis(dpg.mvYAxis, label="Y (mm)", tag="tab22_y"):
            pass


def _update_plot():
    """Update the plot with current parameters."""
    global _last_result, _last_smoothed

    # Read parameters from this tab's widgets
    params = AppState.read_from_widgets("tab22")
    smooth_val = AppState.get_smooth()

    # Compute conjugate profile
    update_info_text("tab22", "Computing conjugate profile...", color=(255, 200, 100))

    result = compute_conjugate_profile(params)

    if "error" in result:
        update_info_text("tab22", f"Error: {result['error']}", color=(255, 100, 100))
        return

    # Smooth the profile
    smoothed = smooth_conjugate_profile(result, s=smooth_val)
    if "error" in smoothed:
        update_info_text("tab22", f"Smoothing error: {smoothed['error']}", color=(255, 100, 100))
        return

    _last_result = result
    _last_smoothed = smoothed

    # Clear existing series
    _clear_plot_series()

    y_axis = "tab22_y"

    # Get segment visibility
    show_AB = dpg.get_value("tab22_show_AB")
    show_BC = dpg.get_value("tab22_show_BC")
    show_CD = dpg.get_value("tab22_show_CD")

    seg_branches = result.get("seg_branches", {})

    # Draw raw envelope points as scatter
    if show_AB and seg_branches.get("AB"):
        pts = seg_branches["AB"]
        dpg.add_scatter_series(
            [p[0] for p in pts], [p[1] for p in pts],
            label="AB points",
            tag="series_conj_AB",
            parent=y_axis
        )
        dpg.bind_item_theme("series_conj_AB", "theme_line_AB")

    if show_BC and seg_branches.get("BC"):
        pts = seg_branches["BC"]
        dpg.add_scatter_series(
            [p[0] for p in pts], [p[1] for p in pts],
            label="BC points",
            tag="series_conj_BC",
            parent=y_axis
        )
        dpg.bind_item_theme("series_conj_BC", "theme_line_BC")

    if show_CD and seg_branches.get("CD"):
        pts = seg_branches["CD"]
        dpg.add_scatter_series(
            [p[0] for p in pts], [p[1] for p in pts],
            label="CD points",
            tag="series_conj_CD",
            parent=y_axis
        )
        dpg.bind_item_theme("series_conj_CD", "theme_line_CD")

    # Draw smoothed flank curve
    smoothed_flank = smoothed.get("smoothed_flank", [])
    if smoothed_flank:
        dpg.add_line_series(
            [p[0] for p in smoothed_flank],
            [p[1] for p in smoothed_flank],
            label="Flank (B-spline)",
            tag="series_flank",
            parent=y_axis
        )
        dpg.bind_item_theme("series_flank", "theme_line_flexspline")

    # Draw mirrored flank
    if smoothed_flank:
        mirrored = [(-p[0], p[1]) for p in smoothed_flank]
        dpg.add_line_series(
            [p[0] for p in mirrored],
            [p[1] for p in mirrored],
            label="Mirrored",
            tag="series_flank_mirror",
            parent=y_axis
        )
        dpg.bind_item_theme("series_flank_mirror", "theme_line_mirror")

    # Fit axes
    dpg.fit_axis_data("tab22_x")
    dpg.fit_axis_data("tab22_y")

    # Update outputs
    _update_outputs(result, smoothed)

    # Update status
    n_pts = result.get("n_pts", 0)
    update_info_text("tab22", f"Computed {n_pts} envelope points", color=(100, 255, 100))


def _clear_plot_series():
    """Clear all existing plot series."""
    tags = [
        "series_conj_AB", "series_conj_BC", "series_conj_CD",
        "series_flank", "series_flank_mirror"
    ]
    for tag in tags:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)


def _update_outputs(result, smoothed):
    """Update output panel."""
    seg_branches = result.get("seg_branches", {})

    values = {
        "s": result.get("s", 0),
        "t": result.get("t", 0),
        "rp_c": result.get("rp_c", 0),
        "count_AB": len(seg_branches.get("AB", [])),
        "count_BC": len(seg_branches.get("BC", [])),
        "count_CD": len(seg_branches.get("CD", [])),
    }
    update_output_values("tab22", values)


def _export_txt():
    """Export conjugate roots to text file."""
    if _last_result is None:
        update_info_text("tab22", "No data to export. Click Update first.", color=(255, 200, 100))
        return

    # Build text content
    lines = ["# Conjugate Profile Roots\n"]
    lines.append("# phi_rad, phi_deg, l, x_g, y_local\n\n")

    raw_roots = _last_result.get("raw_roots", [])
    for phi, l, xg, y_local, seg in raw_roots:
        phi_deg = math.degrees(phi)
        lines.append(f"{phi:.6f}, {phi_deg:.4f}, {l:.6f}, {xg:.6f}, {y_local:.6f}\n")

    content = "".join(lines)

    # Save using file dialog
    from dpg_app.export_manager import write_txt
    import os

    export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
    os.makedirs(export_dir, exist_ok=True)
    filepath = os.path.join(export_dir, "conjugate_roots.txt")

    if write_txt(filepath, content):
        update_info_text("tab22", f"Exported to conjugate_roots.txt", color=(100, 255, 100))
    else:
        update_info_text("tab22", "Export failed!", color=(255, 100, 100))


def _on_param_change():
    """Called when a parameter changes."""
    _update_plot()


def _reset_and_update():
    """Reset parameters and update plot."""
    AppState.reset_to_defaults()
    _update_plot()
