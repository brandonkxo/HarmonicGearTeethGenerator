"""
Tab 2.5 - Radial Modification Method

Overlays flexspline and circular spline for interference detection and modification.
"""

import dearpygui.dearpygui as dpg
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from equations import (
    compute_conjugate_profile, smooth_conjugate_profile,
    build_deformed_flexspline, build_full_circular_spline,
    build_modified_deformed_flexspline
)

from dpg_app.app_state import AppState, scaled
from dpg_app.widgets.parameter_panel import create_parameter_panel, create_button_row
from dpg_app.widgets.output_panel import (
    create_output_panel, update_output_values, create_info_text, update_info_text,
    create_legend
)
from dpg_app.export_manager import show_export_dialog
from dpg_app.themes import COLORS

# Module-level state
_last_fs = None
_last_cs = None
_last_modified = None
_d_max = 0.0
_interference_pts = []
_show_deformed = True
_modification_applied = False
_first_update = True


def create_tab_radial_modification():
    """Create the Tab 2.5 content."""
    with dpg.group(horizontal=True):
        # Left panel - Parameters
        with dpg.child_window(width=scaled(340), border=True):
            create_parameter_panel(
                tag_prefix="tab_ov",
                on_change=_on_param_change,
                include_smooth=True,
                include_fillets=False
            )

            create_button_row(
                tag_prefix="ov",
                on_update=_update_plot,
                on_reset=_reset_and_update
            )

            # Action buttons
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            dpg.add_button(
                label="Show Undeformed",
                tag="btn_deform_ov",
                callback=_toggle_deformed,
                width=scaled(120)
            )

            dpg.add_spacer(height=10)
            dpg.add_button(
                label="Calculate Radial Modification",
                tag="btn_calc_mod",
                callback=_calculate_modification,
                width=-1
            )

            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Export Modified",
                    tag="btn_export_mod",
                    callback=_export_modified,
                    width=scaled(120),
                    enabled=False
                )
                dpg.add_button(
                    label="Export Wave Gen",
                    tag="btn_export_wg",
                    callback=_export_wave_gen,
                    width=scaled(120)
                )

            create_output_panel(
                tag_prefix="tab_ov",
                tab_type="radial_modification"
            )

            # Legend
            create_legend("tab_ov", [
                ("Flexspline", COLORS["flexspline"]),
                ("Circular Spline", COLORS["circular_spline"]),
                ("Modified Flex", COLORS["modified"]),
                ("Interference", COLORS["interference"]),
            ])

            create_info_text("tab_ov", "Click Update to compute overlay.")

        # Right panel - Plot
        with dpg.child_window(width=-1, border=True):
            _create_plot()


def _create_plot():
    """Create the visualization plot."""
    with dpg.plot(
        label="Radial Modification Overlay",
        tag="tab_ov_plot",
        width=-1,
        height=-1,
        equal_aspects=True,
        anti_aliased=True
    ):
        dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)

        dpg.add_plot_axis(dpg.mvXAxis, label="X (mm)", tag="tab_ov_x")
        with dpg.plot_axis(dpg.mvYAxis, label="Y (mm)", tag="tab_ov_y"):
            pass


def _update_plot():
    """Update the overlay plot."""
    global _last_fs, _last_cs, _modification_applied

    params = AppState.read_from_widgets("tab_ov")
    smooth_val = AppState.get_smooth()
    fillet_add = AppState.get_fillet_add()
    fillet_ded = AppState.get_fillet_ded()

    update_info_text("tab_ov", "Computing overlay...", color=(255, 200, 100))

    # Build deformed flexspline
    if _show_deformed:
        fs_result = build_deformed_flexspline(params)
    else:
        from equations import build_full_flexspline
        fs_result = build_full_flexspline(params)

    if "error" in fs_result:
        update_info_text("tab_ov", f"Flexspline error: {fs_result['error']}", color=(255, 100, 100))
        return

    # Build circular spline
    conj = compute_conjugate_profile(params)
    if "error" in conj:
        update_info_text("tab_ov", f"Conjugate error: {conj['error']}", color=(255, 100, 100))
        return

    smoothed = smooth_conjugate_profile(conj, s=smooth_val)
    if "error" in smoothed:
        update_info_text("tab_ov", f"Smoothing error: {smoothed['error']}", color=(255, 100, 100))
        return

    rp_c = conj.get("rp_c", 25.5)
    smoothed_flank = smoothed.get("smoothed_flank", [])

    cs_result = build_full_circular_spline(
        params, smoothed_flank, rp_c,
        n_ded_arc=100,  # Match Tab 2.4 for smoother arcs
        r_fillet_add=fillet_add, r_fillet_ded=fillet_ded
    )
    if "error" in cs_result:
        update_info_text("tab_ov", f"Circular spline error: {cs_result['error']}", color=(255, 100, 100))
        return

    _last_fs = fs_result
    _last_cs = cs_result
    _last_cs["rp_c"] = rp_c
    _modification_applied = False

    # Clear and redraw
    _clear_plot_series()
    _draw_overlay()

    # Set view (only fit on first update)
    global _first_update
    if _first_update:
        dpg.fit_axis_data("tab_ov_x")
        dpg.fit_axis_data("tab_ov_y")
        _first_update = False

    # Update outputs
    _update_outputs()

    fs_pts = len(fs_result.get("chain_xy", []))
    cs_pts = len(cs_result.get("chain_xy", []))
    update_info_text("tab_ov", f"Overlay computed: FS={fs_pts}, CS={cs_pts} points", color=(100, 255, 100))


def _draw_overlay():
    """Draw both gears on the plot."""
    y_axis = "tab_ov_y"

    # Draw flexspline
    if _last_fs:
        chain = _last_fs.get("chain_xy", [])
        if chain:
            x_data = [p[0] for p in chain] + [chain[0][0]]
            y_data = [p[1] for p in chain] + [chain[0][1]]

            dpg.add_line_series(
                x_data, y_data,
                label="Flexspline",
                tag="series_ov_fs",
                parent=y_axis
            )
            dpg.bind_item_theme("series_ov_fs", "theme_line_flexspline")

    # Draw circular spline
    if _last_cs:
        chain = _last_cs.get("chain_xy", [])
        if chain:
            x_data = [p[0] for p in chain] + [chain[0][0]]
            y_data = [p[1] for p in chain] + [chain[0][1]]

            dpg.add_line_series(
                x_data, y_data,
                label="Circular Spline",
                tag="series_ov_cs",
                parent=y_axis
            )
            dpg.bind_item_theme("series_ov_cs", "theme_line_circspline")

    # Draw modified flexspline if calculated
    if _modification_applied and _last_modified:
        chain = _last_modified.get("chain_xy", [])
        if chain:
            x_data = [p[0] for p in chain] + [chain[0][0]]
            y_data = [p[1] for p in chain] + [chain[0][1]]

            dpg.add_line_series(
                x_data, y_data,
                label="Modified Flex",
                tag="series_ov_mod",
                parent=y_axis
            )
            dpg.bind_item_theme("series_ov_mod", "theme_line_modified")

    # Draw interference points
    if _interference_pts:
        dpg.add_scatter_series(
            [p[0] for p in _interference_pts],
            [p[1] for p in _interference_pts],
            label="Interference",
            tag="series_ov_interf",
            parent=y_axis
        )
        dpg.bind_item_theme("series_ov_interf", "theme_line_interference")


def _clear_plot_series():
    """Clear all series."""
    tags = ["series_ov_fs", "series_ov_cs", "series_ov_mod", "series_ov_interf"]
    for tag in tags:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)


def _calculate_modification():
    """Calculate radial modification to eliminate interference."""
    global _last_modified, _d_max, _interference_pts, _modification_applied

    if _last_fs is None or _last_cs is None:
        update_info_text("tab_ov", "Click Update first to compute overlay.", color=(255, 200, 100))
        return

    params = AppState.read_from_widgets("tab_ov")

    update_info_text("tab_ov", "Step 1: Detecting interference...", color=(255, 200, 100))

    # Simple interference detection: find where flexspline penetrates circular spline
    # This is a simplified version - the actual implementation would be more sophisticated
    fs_chain = _last_fs.get("chain_xy", [])
    cs_chain = _last_cs.get("chain_xy", [])
    rp_c = _last_cs.get("rp_c", 25.5)

    # Find penetration by checking radial distance
    _interference_pts = []
    max_penetration = 0.0

    for fx, fy in fs_chain:
        fs_r = math.sqrt(fx * fx + fy * fy)
        # For circular spline (internal gear), interference is when fs_r > cs_r at same angle
        # Simplified: check if flexspline point is outside the circular spline pitch
        if fs_r > rp_c:
            penetration = fs_r - rp_c
            if penetration > max_penetration:
                max_penetration = penetration
            _interference_pts.append((fx, fy))

    _d_max = max_penetration

    update_info_text("tab_ov", f"Step 2: Max penetration d_max = {_d_max:.4f} mm", color=(255, 200, 100))

    if _d_max > 0:
        update_info_text("tab_ov", "Step 3: Applying radial modification...", color=(255, 200, 100))

        # Build modified flexspline
        mod_result = build_modified_deformed_flexspline(params, _d_max)

        if "error" in mod_result:
            update_info_text("tab_ov", f"Modification error: {mod_result['error']}", color=(255, 100, 100))
            return

        _last_modified = mod_result
        _modification_applied = True

        # Enable export button
        dpg.configure_item("btn_export_mod", enabled=True)

        # Redraw
        _clear_plot_series()
        _draw_overlay()

        update_info_text(
            "tab_ov",
            f"Modification complete: d_max = {_d_max:.4f} mm, {len(_interference_pts)} interference points",
            color=(100, 255, 100)
        )
    else:
        update_info_text("tab_ov", "No interference detected.", color=(100, 255, 100))

    _update_outputs()


def _update_outputs():
    """Update output panel."""
    values = {
        "z_f": int(AppState.get_param("z_f")),
        "rp": _last_fs.get("rp", 0) if _last_fs else 0,
        "z_c": int(AppState.get_param("z_c")),
        "rp_c": _last_cs.get("rp_c", 0) if _last_cs else 0,
        "d_max": _d_max,
        "interference_count": len(_interference_pts),
    }
    update_output_values("tab_ov", values)


def _toggle_deformed():
    """Toggle deformed view."""
    global _show_deformed
    _show_deformed = not _show_deformed

    btn_text = "Show Undeformed" if _show_deformed else "Show Deformed"
    dpg.configure_item("btn_deform_ov", label=btn_text)

    _update_plot()


def _export_modified():
    """Export modified flexspline."""
    if _last_modified is None:
        update_info_text("tab_ov", "No modified data. Run calculation first.", color=(255, 200, 100))
        return

    chain = _last_modified.get("chain_xy", [])
    if not chain:
        return

    show_export_dialog("flexspline_modified", chain, [".sldcrv", ".dxf"], closed=True)


def _export_wave_gen():
    """Export wave generator geometry."""
    # Simplified - just export the deformation curve
    params = AppState.read_from_widgets("tab_ov")
    rm = _last_fs.get("rm", 24.5) if _last_fs else 24.5
    w0 = params["w0"]

    # Generate wave generator outline
    n_pts = 360
    wg_pts = []
    for i in range(n_pts):
        phi = 2 * math.pi * i / n_pts
        rho = rm + w0 * math.cos(2 * phi)
        x = rho * math.cos(phi)
        y = rho * math.sin(phi)
        wg_pts.append((x, y))

    show_export_dialog("wave_generator", wg_pts, [".sldcrv", ".dxf"], closed=True)


def _on_param_change():
    """Called when parameter changes."""
    _update_plot()


def _reset_and_update():
    """Reset and update."""
    global _modification_applied, _d_max, _interference_pts, _last_modified
    _modification_applied = False
    _d_max = 0.0
    _interference_pts = []
    _last_modified = None
    dpg.configure_item("btn_export_mod", enabled=False)

    AppState.reset_to_defaults()
    _update_plot()
