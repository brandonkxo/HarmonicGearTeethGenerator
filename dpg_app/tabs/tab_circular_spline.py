"""
Tab 2.4 - Full Circular Spline Gear

Visualizes the complete circular spline gear with conjugate tooth profile.
"""

import dearpygui.dearpygui as dpg
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from equations import (
    compute_conjugate_profile, smooth_conjugate_profile, build_full_circular_spline
)

from dpg_app.app_state import AppState, scaled
from dpg_app.widgets.parameter_panel import create_parameter_panel, create_button_row
from dpg_app.widgets.output_panel import (
    create_output_panel, update_output_values, create_info_text, update_info_text
)
from dpg_app.export_manager import show_export_dialog

# Module-level state
_last_result = None
_zoomed = False
_first_update = True


def create_tab_circular_spline():
    """Create the Tab 2.4 content."""
    with dpg.group(horizontal=True):
        # Left panel - Parameters
        with dpg.child_window(width=scaled(340), border=True):
            create_parameter_panel(
                tag_prefix="tab_cs",
                on_change=_on_param_change,
                include_smooth=True,
                include_fillets=True
            )

            create_button_row(
                tag_prefix="cs",
                on_update=_update_plot,
                on_reset=_reset_and_update,
                include_export=True,
                on_export=_export_curve,
                export_formats=[".sldcrv", ".dxf"]
            )

            # Zoom toggle
            dpg.add_spacer(height=10)
            dpg.add_button(
                label="Zoom In",
                tag="btn_zoom_cs",
                callback=_toggle_zoom,
                width=scaled(80)
            )

            create_output_panel(
                tag_prefix="tab_cs",
                tab_type="circular_spline"
            )

            create_info_text("tab_cs", "Click Update to compute circular spline gear.")

        # Right panel - Plot
        with dpg.child_window(width=-1, border=True):
            _create_plot()


def _create_plot():
    """Create the visualization plot."""
    with dpg.plot(
        label="Circular Spline Gear",
        tag="tab_cs_plot",
        width=-1,
        height=-1,
        equal_aspects=True,
        anti_aliased=True
    ):
        dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)

        dpg.add_plot_axis(dpg.mvXAxis, label="X (mm)", tag="tab_cs_x")
        with dpg.plot_axis(dpg.mvYAxis, label="Y (mm)", tag="tab_cs_y"):
            pass


def _update_plot():
    """Update the plot with current parameters."""
    global _last_result

    params = AppState.read_from_widgets("tab_cs")
    smooth_val = AppState.get_smooth()
    fillet_add = AppState.get_fillet_add()
    fillet_ded = AppState.get_fillet_ded()

    update_info_text("tab_cs", "Computing circular spline...", color=(255, 200, 100))

    # First compute conjugate profile
    conj = compute_conjugate_profile(params)
    if "error" in conj:
        update_info_text("tab_cs", f"Conjugate error: {conj['error']}", color=(255, 100, 100))
        return

    # Smooth the profile
    smoothed = smooth_conjugate_profile(conj, s=smooth_val)
    if "error" in smoothed:
        update_info_text("tab_cs", f"Smoothing error: {smoothed['error']}", color=(255, 100, 100))
        return

    # Build full circular spline
    rp_c = conj.get("rp_c", 25.5)
    smoothed_flank = smoothed.get("smoothed_flank", [])

    result = build_full_circular_spline(
        params, smoothed_flank, rp_c,
        r_fillet_add=fillet_add, r_fillet_ded=fillet_ded
    )

    if "error" in result:
        update_info_text("tab_cs", f"Build error: {result['error']}", color=(255, 100, 100))
        return

    _last_result = result
    _last_result["rp_c"] = rp_c

    # Clear and redraw
    _clear_plot_series()

    y_axis = "tab_cs_y"
    chain = result.get("chain_xy", [])

    if not chain:
        update_info_text("tab_cs", "No chain data generated", color=(255, 100, 100))
        return

    # Draw gear outline
    x_data = [p[0] for p in chain]
    y_data = [p[1] for p in chain]

    # Close the loop
    x_data.append(chain[0][0])
    y_data.append(chain[0][1])

    dpg.add_line_series(
        x_data, y_data,
        label="Circular Spline",
        tag="series_cs_chain",
        parent=y_axis
    )
    dpg.bind_item_theme("series_cs_chain", "theme_line_circspline")

    # Draw reference circles
    _draw_reference_circles(result, rp_c, y_axis)

    # Set view (only fit on first update)
    global _first_update
    if _zoomed:
        _set_zoomed_view(rp_c)
    elif _first_update:
        dpg.fit_axis_data("tab_cs_x")
        dpg.fit_axis_data("tab_cs_y")
        _first_update = False

    # Update outputs
    _update_outputs(result, rp_c)

    update_info_text("tab_cs", f"Circular spline computed: {len(chain)} points", color=(100, 255, 100))

    from dpg_app.main import update_point_count
    update_point_count(len(chain))


def _clear_plot_series():
    """Clear existing series."""
    tags = ["series_cs_chain", "series_cs_add", "series_cs_pitch", "series_cs_ded"]
    for tag in tags:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)


def _draw_reference_circles(result, rp_c, y_axis):
    """Draw reference circles."""
    # Use values from result dict (computed in equations.py)
    r_add = result.get("r_add", rp_c)
    r_ded = result.get("r_ded", rp_c)

    n_pts = 180
    for r, tag, theme, label in [
        (r_add, "series_cs_add", "theme_line_addendum", "Addendum"),
        (rp_c, "series_cs_pitch", "theme_line_pitch", "Pitch"),
        (r_ded, "series_cs_ded", "theme_line_dedendum", "Dedendum"),
    ]:
        angles = [2 * math.pi * i / n_pts for i in range(n_pts + 1)]
        x_pts = [r * math.cos(a) for a in angles]
        y_pts = [r * math.sin(a) for a in angles]

        dpg.add_line_series(
            x_pts, y_pts,
            label=label,
            tag=tag,
            parent=y_axis
        )
        dpg.bind_item_theme(tag, theme)


def _set_zoomed_view(rp_c):
    """Set zoomed view for ~4 teeth."""
    z_c = int(AppState.get_param("z_c"))
    tooth_angle = 2 * math.pi / z_c
    half_view = rp_c * 2 * tooth_angle

    dpg.set_axis_limits("tab_cs_x", -half_view * 0.5, half_view * 1.5)
    dpg.set_axis_limits("tab_cs_y", rp_c - half_view * 0.5, rp_c + half_view * 0.5)


def _update_outputs(result, rp_c):
    """Update output panel."""
    chain = result.get("chain_xy", [])
    ha = AppState.get_param("ha")
    hf = AppState.get_param("hf")

    values = {
        "s": result.get("s", 0),
        "t": result.get("t", 0),
        "rp_c": rp_c,
        "r_add": rp_c - ha,
        "r_ded": rp_c + hf,
        "ha": ha,
        "hf": hf,
        "z_c": int(AppState.get_param("z_c")),
        "chain_points": len(chain),
    }
    update_output_values("tab_cs", values)


def _toggle_zoom():
    """Toggle zoom state."""
    global _zoomed
    _zoomed = not _zoomed

    btn_text = "Zoom Out" if _zoomed else "Zoom In"
    dpg.configure_item("btn_zoom_cs", label=btn_text)

    if _last_result:
        rp_c = _last_result.get("rp_c", 25.5)
        if _zoomed:
            _set_zoomed_view(rp_c)
        else:
            dpg.fit_axis_data("tab_cs_x")
            dpg.fit_axis_data("tab_cs_y")


def _export_curve():
    """Export circular spline curve."""
    if _last_result is None:
        update_info_text("tab_cs", "No data to export. Click Update first.", color=(255, 200, 100))
        return

    chain = _last_result.get("chain_xy", [])
    if not chain:
        return

    show_export_dialog("circular_spline", chain, [".sldcrv", ".dxf"], closed=True)


def _on_param_change():
    """Called when parameter changes."""
    _update_plot()


def _reset_and_update():
    """Reset and update."""
    AppState.reset_to_defaults()
    _update_plot()
