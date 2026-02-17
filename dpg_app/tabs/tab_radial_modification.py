"""
Tab 2.5 - Gear Overlay

Overlays flexspline and circular spline for visualization.
"""

import dearpygui.dearpygui as dpg
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from equations import (
    compute_conjugate_profile, smooth_conjugate_profile,
    build_deformed_flexspline, build_full_circular_spline,
    compute_profile, eq14_rho, eq21_mu, eq23_phi1, eq27_psi, eq29_transform
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
_last_smoothed_flank = None
_last_rp_c = None
_show_deformed = True
_debug_mode = False
_first_update = True
_debug_fs_tooth = None  # Store debug tooth data for dmax calculation
_debug_cs_tooth = None
_trimmed_fs_tooth = None  # Full tooth after dmax trim (if applied)
_last_dmax = 0.0  # Store last calculated dmax


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

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Show Undeformed",
                    tag="btn_deform_ov",
                    callback=_toggle_deformed,
                    width=scaled(120)
                )
                dpg.add_button(
                    label="Debug Single Tooth",
                    tag="btn_debug_tooth",
                    callback=_toggle_debug_mode,
                    width=scaled(120)
                )

            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Calculate dmax",
                    tag="btn_calc_dmax",
                    callback=_calculate_dmax,
                    width=scaled(120)
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
    global _last_fs, _last_cs, _last_smoothed_flank, _last_rp_c, _trimmed_fs_tooth
    _trimmed_fs_tooth = None  # Reset trim when updating

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

    # Store for debug mode
    _last_smoothed_flank = smoothed_flank
    _last_rp_c = rp_c

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


def _clear_plot_series():
    """Clear all series."""
    tags = ["series_ov_fs", "series_ov_cs", "series_debug_fs", "series_debug_cs"]
    for tag in tags:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)


def _update_outputs():
    """Update output panel."""
    values = {
        "z_f": int(AppState.get_param("z_f")),
        "rp": _last_fs.get("rp", 0) if _last_fs else 0,
        "z_c": int(AppState.get_param("z_c")),
        "rp_c": _last_cs.get("rp_c", 0) if _last_cs else 0,
    }
    update_output_values("tab_ov", values)


def _toggle_deformed():
    """Toggle deformed view."""
    global _show_deformed
    _show_deformed = not _show_deformed

    btn_text = "Show Undeformed" if _show_deformed else "Show Deformed"
    dpg.configure_item("btn_deform_ov", label=btn_text)

    _update_plot()


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


def _toggle_debug_mode():
    """Toggle debug single tooth view."""
    global _debug_mode, _trimmed_fs_tooth
    _debug_mode = not _debug_mode

    btn_text = "Show Full Gears" if _debug_mode else "Debug Single Tooth"
    dpg.configure_item("btn_debug_tooth", label=btn_text)

    if _debug_mode:
        _trimmed_fs_tooth = None  # Reset trim when entering debug mode
        _draw_debug_tooth()
    else:
        _trimmed_fs_tooth = None  # Reset trim when leaving debug mode
        _clear_plot_series()
        _draw_overlay()
        dpg.fit_axis_data("tab_ov_x")
        dpg.fit_axis_data("tab_ov_y")


def _draw_debug_tooth():
    """Draw single tooth of each gear without fillets for debugging."""
    global _debug_fs_tooth, _debug_cs_tooth

    if _last_fs is None or _last_smoothed_flank is None:
        update_info_text("tab_ov", "Click Update first to compute gears.", color=(255, 200, 100))
        return

    _clear_plot_series()
    _debug_fs_tooth = None
    _debug_cs_tooth = None

    params = AppState.read_from_widgets("tab_ov")
    y_axis = "tab_ov_y"

    # === Flexspline single tooth (deformed, no fillets) ===
    profile = compute_profile(params)
    if "error" not in profile:
        rm = profile["rm"]
        w0 = params["w0"]
        ds = profile["ds"]
        hf = params["hf"]

        # Raw tooth profile segments (no fillets)
        pts_AB = profile["pts_AB"]
        pts_BC = profile["pts_BC"]
        pts_CD = profile["pts_CD"]

        # Combine into right flank (addendum to dedendum)
        right_flank = list(pts_AB) + list(pts_BC) + list(pts_CD)
        # Mirror for left flank
        left_flank = [(-x, y) for x, y in reversed(right_flank)]

        # Transform function for deformed coordinates (phi = 0 for first tooth)
        def tooth_point_deformed(xr, yr):
            phi = 0  # First tooth
            rho = eq14_rho(phi, rm, w0)
            mu = eq21_mu(phi, w0, rm)
            phi1 = eq23_phi1(phi, w0, rm)
            gamma = phi1  # phi2 = 0
            psi = eq27_psi(mu, gamma)
            return eq29_transform(xr, yr, psi, rho, gamma)

        # Build deformed tooth outline
        fs_tooth = []
        for x, y in left_flank:
            fs_tooth.append(tooth_point_deformed(x, y))
        for x, y in right_flank:
            fs_tooth.append(tooth_point_deformed(x, y))

        # Store for dmax calculation
        _debug_fs_tooth = fs_tooth

        # Close the tooth
        if fs_tooth:
            x_data = [p[0] for p in fs_tooth] + [fs_tooth[0][0]]
            y_data = [p[1] for p in fs_tooth] + [fs_tooth[0][1]]

            dpg.add_line_series(
                x_data, y_data,
                label="Flexspline Tooth",
                tag="series_debug_fs",
                parent=y_axis
            )
            dpg.bind_item_theme("series_debug_fs", "theme_line_flexspline")

    # === Circular spline single tooth (no fillets) ===
    if _last_smoothed_flank and _last_rp_c:
        rp_c = _last_rp_c

        # Right flank from smoothed data (already in tooth-local coords)
        right_flank_cs = list(_last_smoothed_flank)
        # Mirror for left flank
        left_flank_cs = [(-x, y) for x, y in reversed(right_flank_cs)]

        # Transform from tooth-local to global (first tooth at angle 0)
        def local_to_global_cs(x_loc, y_loc):
            r = rp_c + y_loc
            theta = x_loc / rp_c  # tooth 0, no offset
            return r * math.sin(theta), r * math.cos(theta)

        # Build tooth outline
        cs_tooth = []
        for x, y in left_flank_cs:
            cs_tooth.append(local_to_global_cs(x, y))
        for x, y in right_flank_cs:
            cs_tooth.append(local_to_global_cs(x, y))

        # Store for dmax calculation
        _debug_cs_tooth = cs_tooth

        if cs_tooth:
            x_data = [p[0] for p in cs_tooth] + [cs_tooth[0][0]]
            y_data = [p[1] for p in cs_tooth] + [cs_tooth[0][1]]

            dpg.add_line_series(
                x_data, y_data,
                label="Circular Spline Tooth",
                tag="series_debug_cs",
                parent=y_axis
            )
            dpg.bind_item_theme("series_debug_cs", "theme_line_circspline")

    # Fit view to debug data
    dpg.fit_axis_data("tab_ov_x")
    dpg.fit_axis_data("tab_ov_y")

    update_info_text("tab_ov", "Debug: Single tooth view (no fillets)", color=(100, 200, 255))


def _calculate_dmax():
    """Calculate maximum interference dmax.

    Measures X-axis distance from circular spline tooth points
    to the flexspline tooth boundary (all segments).
    """
    global _last_dmax

    if _debug_fs_tooth is None or _debug_cs_tooth is None:
        update_info_text("tab_ov", "Click 'Debug Single Tooth' first.", color=(255, 200, 100))
        return

    if not _debug_mode:
        update_info_text("tab_ov", "Enable debug mode first.", color=(255, 200, 100))
        return

    fs_points = _debug_fs_tooth
    cs_points = _debug_cs_tooth

    if not fs_points or not cs_points:
        update_info_text("tab_ov", "No tooth data available.", color=(255, 100, 100))
        return

    # Split FS tooth into left and right flanks based on X coordinate
    fs_right = [(x, y) for x, y in fs_points if x > 0]
    fs_left = [(x, y) for x, y in fs_points if x < 0]

    # Get Y range of FS tooth
    fs_y_min = min(p[1] for p in fs_points)
    fs_y_max = max(p[1] for p in fs_points)

    def interpolate_x_at_y(flank_points, target_y):
        """Find X value on flank at given Y by linear interpolation."""
        if not flank_points:
            return None
        # Sort by Y
        sorted_pts = sorted(flank_points, key=lambda p: p[1])
        # Find bracketing points
        for i in range(len(sorted_pts) - 1):
            y1, y2 = sorted_pts[i][1], sorted_pts[i + 1][1]
            if y1 <= target_y <= y2 or y2 <= target_y <= y1:
                x1, x2 = sorted_pts[i][0], sorted_pts[i + 1][0]
                if abs(y2 - y1) < 1e-9:
                    return (x1 + x2) / 2
                t = (target_y - y1) / (y2 - y1)
                return x1 + t * (x2 - x1)
        return None

    # Calculate dmax: find maximum X-penetration of CS into FS tooth
    dmax = 0.0
    interference_count = 0

    for cs_x, cs_y in cs_points:
        # Only check points within FS tooth Y range
        if cs_y < fs_y_min or cs_y > fs_y_max:
            continue

        # Get FS boundary at this Y level
        fs_left_x = interpolate_x_at_y(fs_left, cs_y)
        fs_right_x = interpolate_x_at_y(fs_right, cs_y)

        if fs_left_x is None or fs_right_x is None:
            continue

        # Check for interference:
        # CS point penetrates if it's inside the FS tooth boundary

        # For the right side (positive X): CS penetrates if X < FS right boundary
        if cs_x > 0 and cs_x < fs_right_x:
            penetration = fs_right_x - cs_x
            if penetration > dmax:
                dmax = penetration
            interference_count += 1

        # For the left side (negative X): CS penetrates if X > FS left boundary
        if cs_x < 0 and cs_x > fs_left_x:
            penetration = cs_x - fs_left_x
            if penetration > dmax:
                dmax = penetration
            interference_count += 1

    _last_dmax = dmax

    # Show popup dialog with result
    _show_dmax_popup(dmax, interference_count)


def _show_dmax_popup(dmax: float, interference_count: int):
    """Show popup dialog with dmax result and trim option."""
    # Clean up existing popup
    if dpg.does_item_exist("dmax_popup"):
        dpg.delete_item("dmax_popup")

    window_width = scaled(350)
    window_height = scaled(180)

    with dpg.window(
        label="dmax Calculation Result",
        tag="dmax_popup",
        modal=True,
        width=window_width,
        height=window_height,
        pos=(dpg.get_viewport_width() // 2 - window_width // 2,
             dpg.get_viewport_height() // 2 - window_height // 2),
        no_resize=True,
        no_collapse=True,
        on_close=lambda: dpg.delete_item("dmax_popup")
    ):
        dpg.add_spacer(height=10)

        if dmax > 0:
            dpg.add_text(f"dmax = {dmax:.4f} mm", color=(255, 200, 100))
            dpg.add_text(f"({interference_count} interference points)", color=(180, 180, 180))
            dpg.add_spacer(height=15)
            dpg.add_text("Do you want to trim the flexspline by dmax?")
            dpg.add_spacer(height=15)

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Yes, Trim",
                    callback=lambda: _apply_trim(dmax),
                    width=scaled(100)
                )
                dpg.add_spacer(width=20)
                dpg.add_button(
                    label="No",
                    callback=lambda: dpg.delete_item("dmax_popup"),
                    width=scaled(80)
                )
        else:
            dpg.add_text("dmax = 0", color=(100, 255, 100))
            dpg.add_text("No interference detected.", color=(180, 180, 180))
            dpg.add_spacer(height=20)
            dpg.add_button(
                label="OK",
                callback=lambda: dpg.delete_item("dmax_popup"),
                width=-1
            )


def _apply_trim(dmax: float):
    """Apply dmax trim to the flexspline tooth.

    Moves all tooth points inward (reduces X magnitude) by dmax.
    """
    global _trimmed_fs_tooth

    if _debug_fs_tooth is None:
        dpg.delete_item("dmax_popup")
        return

    # Trim tooth: move points inward by dmax
    # For right flank (positive X): subtract dmax from X
    # For left flank (negative X): add dmax to X
    trimmed = []
    for x, y in _debug_fs_tooth:
        if x > 0:
            trimmed.append((x - dmax, y))
        elif x < 0:
            trimmed.append((x + dmax, y))
        else:
            trimmed.append((x, y))  # Points on centerline stay

    _trimmed_fs_tooth = trimmed

    # Close popup
    dpg.delete_item("dmax_popup")

    # Redraw debug view with trimmed tooth
    _redraw_debug_with_trimmed_tooth()

    update_info_text("tab_ov", f"Flexspline trimmed by dmax = {dmax:.4f} mm", color=(100, 255, 100))


def _redraw_debug_with_trimmed_tooth():
    """Redraw debug tooth view using trimmed flexspline tooth."""
    if _trimmed_fs_tooth is None:
        return

    _clear_plot_series()
    y_axis = "tab_ov_y"

    # === Draw trimmed flexspline tooth ===
    fs_tooth = _trimmed_fs_tooth
    if fs_tooth:
        x_data = [p[0] for p in fs_tooth] + [fs_tooth[0][0]]
        y_data = [p[1] for p in fs_tooth] + [fs_tooth[0][1]]

        dpg.add_line_series(
            x_data, y_data,
            label="Flexspline Tooth (Trimmed)",
            tag="series_debug_fs",
            parent=y_axis
        )
        dpg.bind_item_theme("series_debug_fs", "theme_line_flexspline")

    # === Redraw circular spline tooth (unchanged) ===
    if _last_smoothed_flank and _last_rp_c:
        rp_c = _last_rp_c
        right_flank_cs = list(_last_smoothed_flank)
        left_flank_cs = [(-x, y) for x, y in reversed(right_flank_cs)]

        def local_to_global_cs(x_loc, y_loc):
            r = rp_c + y_loc
            theta = x_loc / rp_c
            return r * math.sin(theta), r * math.cos(theta)

        cs_tooth = []
        for x, y in left_flank_cs:
            cs_tooth.append(local_to_global_cs(x, y))
        for x, y in right_flank_cs:
            cs_tooth.append(local_to_global_cs(x, y))

        if cs_tooth:
            x_data = [p[0] for p in cs_tooth] + [cs_tooth[0][0]]
            y_data = [p[1] for p in cs_tooth] + [cs_tooth[0][1]]

            dpg.add_line_series(
                x_data, y_data,
                label="Circular Spline Tooth",
                tag="series_debug_cs",
                parent=y_axis
            )
            dpg.bind_item_theme("series_debug_cs", "theme_line_circspline")

    dpg.fit_axis_data("tab_ov_x")
    dpg.fit_axis_data("tab_ov_y")


def _on_param_change():
    """Called when parameter changes."""
    _update_plot()


def _reset_and_update():
    """Reset and update."""
    AppState.reset_to_defaults()
    _update_plot()
