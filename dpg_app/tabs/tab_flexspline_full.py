"""
Tab 2.3 - Full Flexspline Gear

Visualizes the complete flexspline gear with all teeth arranged on the pitch circle.
"""

import dearpygui.dearpygui as dpg
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from equations import build_full_flexspline, build_deformed_flexspline

from dpg_app.app_state import AppState, scaled
from dpg_app.widgets.parameter_panel import create_parameter_panel, create_button_row
from dpg_app.widgets.output_panel import (
    create_output_panel, update_output_values, create_info_text, update_info_text
)
from dpg_app.export_manager import show_export_dialog

# Module-level state
_last_result = None
_deformed = False
_first_update = True


def create_tab_flexspline_full():
    """Create the Tab 2.3 content."""
    with dpg.group(horizontal=True):
        # Left panel - Parameters
        with dpg.child_window(width=scaled(340), border=True):
            create_parameter_panel(
                tag_prefix="tab_fs",
                on_change=_on_param_change,
                include_smooth=True,
                include_fillets=True
            )

            create_button_row(
                tag_prefix="fs",
                on_update=_update_plot,
                on_reset=_reset_and_update,
                include_export=True,
                on_export=_export_curve,
                export_formats=[".sldcrv", ".dxf"]
            )

            # Toggle button
            dpg.add_spacer(height=10)
            dpg.add_button(
                label="Show Deformed",
                tag="btn_deform_fs",
                callback=_toggle_deformed,
                width=scaled(150)
            )

            create_output_panel(
                tag_prefix="tab_fs",
                tab_type="flexspline_full"
            )

            create_info_text("tab_fs", "Click Update to compute flexspline gear.")

        # Right panel - Plot
        with dpg.child_window(width=-1, border=True):
            _create_plot()


def _create_plot():
    """Create the visualization plot."""
    with dpg.plot(
        label="Flexspline Gear",
        tag="tab_fs_plot",
        width=-1,
        height=-1,
        equal_aspects=True,
        anti_aliased=True
    ):
        dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)

        dpg.add_plot_axis(dpg.mvXAxis, label="X (mm)", tag="tab_fs_x")
        with dpg.plot_axis(dpg.mvYAxis, label="Y (mm)", tag="tab_fs_y"):
            pass


def _update_plot():
    """Update the plot with current parameters."""
    global _last_result

    params = AppState.read_from_widgets("tab_fs")
    fillet_add = AppState.get_fillet_add()
    fillet_ded = AppState.get_fillet_ded()
    smooth_val = AppState.get_smooth()

    update_info_text("tab_fs", "Computing flexspline...", color=(255, 200, 100))

    # Choose deformed or undeformed
    if _deformed:
        result = build_deformed_flexspline(params, r_fillet_add=fillet_add, r_fillet_ded=fillet_ded, smooth=smooth_val)
    else:
        result = build_full_flexspline(params, r_fillet_add=fillet_add, r_fillet_ded=fillet_ded, smooth=smooth_val)

    if "error" in result:
        update_info_text("tab_fs", f"Error: {result['error']}", color=(255, 100, 100))
        return

    _last_result = result

    # Clear and redraw
    _clear_plot_series()

    y_axis = "tab_fs_y"
    chain = result.get("chain_xy", [])

    if not chain:
        update_info_text("tab_fs", "No chain data generated", color=(255, 100, 100))
        return

    # Draw gear outline
    x_data = [p[0] for p in chain]
    y_data = [p[1] for p in chain]

    # Close the loop
    x_data.append(chain[0][0])
    y_data.append(chain[0][1])

    dpg.add_line_series(
        x_data, y_data,
        label="Flexspline",
        tag="series_fs_chain",
        parent=y_axis
    )
    dpg.bind_item_theme("series_fs_chain", "theme_line_flexspline")

    # Draw reference circles
    _draw_reference_circles(result, params, y_axis)

    # Fit axis on first update
    global _first_update
    if _first_update:
        dpg.fit_axis_data("tab_fs_x")
        dpg.fit_axis_data("tab_fs_y")
        _first_update = False

    # Update outputs
    _update_outputs(result, params)

    mode = "Deformed" if _deformed else "Undeformed"
    update_info_text("tab_fs", f"Flexspline computed ({mode}): {len(chain)} points", color=(100, 255, 100))

    from dpg_app.main import update_point_count, update_mode
    update_point_count(len(chain))
    update_mode(mode)


def _clear_plot_series():
    """Clear existing series."""
    tags = ["series_fs_chain", "series_fs_add", "series_fs_pitch", "series_fs_ded"]
    for tag in tags:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)


def _draw_reference_circles(result, params, y_axis):
    """Draw reference circles (addendum, pitch, dedendum)."""
    rm = result.get("rm", 0)
    ds = result.get("ds", 0)
    ha = params["ha"]
    hf = params["hf"]
    w0 = params["w0"]

    # Calculate reference radii (from original gear_viewer.py)
    r_ded = rm + ds                 # dedendum circle radius
    rp = rm + ds + hf               # pitch circle radius
    r_add = rm + ds + hf + ha       # addendum circle radius

    # Store for output display
    result["_r_add"] = r_add
    result["_r_ded"] = r_ded

    # Generate circle points
    n_pts = 180
    for r, tag, theme, label in [
        (r_add, "series_fs_add", "theme_line_addendum", "Addendum"),
        (rp, "series_fs_pitch", "theme_line_pitch", "Pitch"),
        (r_ded, "series_fs_ded", "theme_line_dedendum", "Dedendum"),
    ]:
        if r > 0:
            angles = [2 * math.pi * i / n_pts for i in range(n_pts + 1)]

            if _deformed:
                # Deformed elliptical shape - offset from neutral layer
                # Each reference circle deforms based on its distance from rm
                offset = r - rm  # How far this circle is from neutral layer
                x_pts = [(rm + w0 * math.cos(2 * a) + offset) * math.cos(a) for a in angles]
                y_pts = [(rm + w0 * math.cos(2 * a) + offset) * math.sin(a) for a in angles]
            else:
                # Perfect circle
                x_pts = [r * math.cos(a) for a in angles]
                y_pts = [r * math.sin(a) for a in angles]

            dpg.add_line_series(
                x_pts, y_pts,
                label=label,
                tag=tag,
                parent=y_axis
            )
            dpg.bind_item_theme(tag, theme)


def _update_outputs(result, params):
    """Update output panel."""
    chain = result.get("chain_xy", [])
    rm = result.get("rm", 0)
    ds = result.get("ds", 0)
    s = result.get("s", 0)
    ha = params["ha"]
    hf = params["hf"]

    # Calculate reference radii
    r_ded = rm + ds
    r_add = rm + ds + hf + ha
    rb = rm - (s - ds)  # inner radius of flexspline

    values = {
        "s": s,
        "t": result.get("t", 0),
        "ds": ds,
        "rp": result.get("rp", 0),
        "rm": rm,
        "rb": rb,
        "r_add": r_add,
        "r_ded": r_ded,
        "ha": ha,
        "hf": hf,
        "z_f": int(params["z_f"]),
        "chain_points": len(chain),
    }
    update_output_values("tab_fs", values)


def _toggle_deformed():
    """Toggle deformed/undeformed state."""
    global _deformed
    _deformed = not _deformed

    btn_text = "Show Undeformed" if _deformed else "Show Deformed"
    dpg.configure_item("btn_deform_fs", label=btn_text)

    _update_plot()


def _export_curve():
    """Export flexspline curve."""
    if _last_result is None:
        update_info_text("tab_fs", "No data to export. Click Update first.", color=(255, 200, 100))
        return

    chain = _last_result.get("chain_xy", [])
    if not chain:
        return

    show_export_dialog("flexspline", chain, [".sldcrv", ".dxf"], closed=True)


def _on_param_change():
    """Called when parameter changes."""
    _update_plot()


def _reset_and_update():
    """Reset and update."""
    AppState.reset_to_defaults()
    _update_plot()
