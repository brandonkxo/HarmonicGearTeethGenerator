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
    build_dmax_deformed_flexspline,
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
_debug_fs_addendum = None  # AB segment + addendum arc (for dmax_y calculation)
_debug_segment_info = None  # Store segment lengths for index-based trim
_trimmed_fs_tooth = None  # Full tooth after dmax trim (if applied)
_last_dmax_x = 0.0  # Store last calculated dmax in X direction
_last_dmax_y = 0.0  # Store last calculated dmax in Y direction
_dmax_applied_to_full = False  # Track if dmax was applied to full gear view
_applied_dmax_x = 0.0  # dmax_x value applied to full gear
_applied_dmax_y = 0.0  # dmax_y value applied to full gear


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

            # Main view buttons (Show Undeformed, Debug Single Tooth, Export)
            with dpg.group(horizontal=True, tag="main_view_buttons"):
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
                    width=scaled(140)
                )

            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True, tag="secondary_buttons"):
                dpg.add_button(
                    label="Export Wave Gen",
                    tag="btn_export_wg",
                    callback=_export_wave_gen,
                    width=scaled(120)
                )
                dpg.add_button(
                    label="Export dmax FS",
                    tag="btn_export_dmax_fs",
                    callback=_export_dmax_flexspline,
                    width=scaled(120),
                    show=False
                )

            # Debug mode buttons (Calculate dmax, Show Full Gears) - initially hidden
            with dpg.group(horizontal=True, tag="debug_mode_buttons", show=False):
                dpg.add_button(
                    label="Calculate dmax",
                    tag="btn_calc_dmax",
                    callback=_calculate_dmax,
                    width=scaled(120)
                )
                dpg.add_button(
                    label="Show Full Gears",
                    tag="btn_show_full_gears",
                    callback=_toggle_debug_mode,
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
    global _dmax_applied_to_full, _last_dmax_x, _last_dmax_y
    _trimmed_fs_tooth = None  # Reset trim when updating
    _dmax_applied_to_full = False  # Reset dmax applied state
    _last_dmax_x = 0.0  # Reset stored dmax values
    _last_dmax_y = 0.0

    # Hide export dmax flexspline button when plot is reset
    if dpg.does_item_exist("btn_export_dmax_fs"):
        dpg.configure_item("btn_export_dmax_fs", show=False)

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

    # If switching to undeformed and dmax was applied, ask user if they want to apply dmax
    if _show_deformed and _dmax_applied_to_full and (_applied_dmax_x > 0 or _applied_dmax_y > 0):
        # User is switching from deformed (with dmax) to undeformed
        _show_apply_dmax_to_undeformed_dialog()
        return

    _show_deformed = not _show_deformed

    btn_text = "Show Undeformed" if _show_deformed else "Show Deformed"
    dpg.configure_item("btn_deform_ov", label=btn_text)

    _update_plot()


def _show_apply_dmax_to_undeformed_dialog():
    """Show dialog asking if user wants to apply dmax to undeformed flex spline."""
    # Clean up existing dialog
    if dpg.does_item_exist("apply_dmax_undeformed_dialog"):
        dpg.delete_item("apply_dmax_undeformed_dialog")

    window_width = scaled(380)
    window_height = scaled(200)

    with dpg.window(
        label="Apply dmax to Undeformed Flexspline",
        tag="apply_dmax_undeformed_dialog",
        modal=True,
        width=window_width,
        height=window_height,
        pos=(dpg.get_viewport_width() // 2 - window_width // 2,
             dpg.get_viewport_height() // 2 - window_height // 2),
        no_resize=True,
        no_collapse=True,
        on_close=lambda: _on_apply_dmax_undeformed_close(False)
    ):
        dpg.add_spacer(height=10)
        dpg.add_text("Apply current dmax to undeformed flex spline?")
        dpg.add_spacer(height=10)

        # Show current dmax values
        if _applied_dmax_x > 0:
            dpg.add_text(f"  dmax_x = {_applied_dmax_x:.4f} mm", color=(255, 200, 100))
        else:
            dpg.add_text("  dmax_x = 0 (no X trim)", color=(120, 120, 120))

        if _applied_dmax_y > 0:
            dpg.add_text(f"  dmax_y = {_applied_dmax_y:.4f} mm", color=(255, 200, 100))
        else:
            dpg.add_text("  dmax_y = 0 (no Y trim)", color=(120, 120, 120))

        dpg.add_spacer(height=20)

        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Yes",
                callback=lambda: _on_apply_dmax_undeformed_close(True),
                width=scaled(100)
            )
            dpg.add_spacer(width=20)
            dpg.add_button(
                label="No",
                callback=lambda: _on_apply_dmax_undeformed_close(False),
                width=scaled(100)
            )


def _on_apply_dmax_undeformed_close(apply_dmax: bool):
    """Handle apply dmax to undeformed dialog close."""
    global _show_deformed, _dmax_applied_to_full

    # Close dialog
    if dpg.does_item_exist("apply_dmax_undeformed_dialog"):
        dpg.delete_item("apply_dmax_undeformed_dialog")

    # Switch to undeformed view
    _show_deformed = False
    btn_text = "Show Deformed"
    dpg.configure_item("btn_deform_ov", label=btn_text)

    if apply_dmax:
        # Keep dmax applied state and draw undeformed with dmax
        _draw_undeformed_with_dmax()
        # Show export dmax flexspline button
        dpg.configure_item("btn_export_dmax_fs", show=True)
    else:
        # Clear dmax applied state and draw normal undeformed
        _dmax_applied_to_full = False
        # Hide export dmax flexspline button
        dpg.configure_item("btn_export_dmax_fs", show=False)
        _update_plot()


def _draw_undeformed_with_dmax():
    """Draw undeformed flexspline with dmax applied."""
    global _last_fs

    params = AppState.read_from_widgets("tab_ov")
    fillet_add = AppState.get_fillet_add()
    fillet_ded = AppState.get_fillet_ded()
    smooth_val = AppState.get_smooth()

    update_info_text("tab_ov", "Computing undeformed with dmax...", color=(255, 200, 100))

    _clear_plot_series()
    y_axis = "tab_ov_y"

    # Build undeformed flexspline with dmax applied
    from equations import build_dmax_full_flexspline
    fs_result = build_dmax_full_flexspline(
        params,
        dmax_x=_applied_dmax_x,
        dmax_y=_applied_dmax_y,
        r_fillet_add=fillet_add,
        r_fillet_ded=fillet_ded
    )

    if "error" in fs_result:
        update_info_text("tab_ov", f"Flexspline error: {fs_result['error']}", color=(255, 100, 100))
        return

    _last_fs = fs_result

    # Draw flexspline
    chain = fs_result.get("chain_xy", [])
    if chain:
        x_data = [p[0] for p in chain] + [chain[0][0]]
        y_data = [p[1] for p in chain] + [chain[0][1]]

        dpg.add_line_series(
            x_data, y_data,
            label="Flexspline (dmax applied)",
            tag="series_ov_fs",
            parent=y_axis
        )
        dpg.bind_item_theme("series_ov_fs", "theme_line_flexspline")

    # Draw circular spline (unchanged)
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

    dpg.fit_axis_data("tab_ov_x")
    dpg.fit_axis_data("tab_ov_y")

    # Build status message
    msg_parts = []
    if _applied_dmax_x > 0:
        msg_parts.append(f"dmax_x={_applied_dmax_x:.4f}")
    if _applied_dmax_y > 0:
        msg_parts.append(f"dmax_y={_applied_dmax_y:.4f}")

    fs_pts = len(fs_result.get("chain_xy", []))
    cs_pts = len(_last_cs.get("chain_xy", [])) if _last_cs else 0
    msg = f"Undeformed with dmax: {', '.join(msg_parts)} mm | FS={fs_pts}, CS={cs_pts} pts"
    update_info_text("tab_ov", msg, color=(100, 255, 100))


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


def _export_dmax_flexspline():
    """Export undeformed flexspline with dmax applied."""
    if _last_fs is None:
        update_info_text("tab_ov", "No flexspline data to export.", color=(255, 200, 100))
        return

    chain = _last_fs.get("chain_xy", [])
    if not chain:
        update_info_text("tab_ov", "No flexspline chain data.", color=(255, 200, 100))
        return

    show_export_dialog("dmax_flexspline", chain, [".sldcrv", ".dxf"], closed=True)


def _toggle_debug_mode():
    """Toggle debug single tooth view."""
    global _debug_mode, _trimmed_fs_tooth

    if _debug_mode:
        # Leaving debug mode - check if we should apply dmax
        if _last_dmax_x > 0 or _last_dmax_y > 0:
            _show_apply_dmax_dialog()
        else:
            _exit_debug_mode_without_dmax()
    else:
        # Entering debug mode
        _debug_mode = True
        _trimmed_fs_tooth = None  # Reset trim when entering debug mode

        # Hide main view buttons, show debug mode buttons
        dpg.configure_item("main_view_buttons", show=False)
        dpg.configure_item("secondary_buttons", show=False)
        dpg.configure_item("debug_mode_buttons", show=True)

        _draw_debug_tooth()


def _exit_debug_mode_without_dmax():
    """Exit debug mode and show full gears without dmax applied."""
    global _debug_mode, _trimmed_fs_tooth, _dmax_applied_to_full
    _debug_mode = False
    _dmax_applied_to_full = False
    _trimmed_fs_tooth = None

    # Show main view buttons, hide debug mode buttons
    dpg.configure_item("main_view_buttons", show=True)
    dpg.configure_item("secondary_buttons", show=True)
    dpg.configure_item("debug_mode_buttons", show=False)
    # Hide export dmax button (no dmax applied)
    dpg.configure_item("btn_export_dmax_fs", show=False)

    _clear_plot_series()
    _draw_overlay()
    dpg.fit_axis_data("tab_ov_x")
    dpg.fit_axis_data("tab_ov_y")


def _show_apply_dmax_dialog():
    """Show dialog asking if user wants to apply dmax to full flex spline."""
    # Clean up existing dialog
    if dpg.does_item_exist("apply_dmax_dialog"):
        dpg.delete_item("apply_dmax_dialog")

    window_width = scaled(380)
    window_height = scaled(200)

    with dpg.window(
        label="Apply dmax to Full Gear",
        tag="apply_dmax_dialog",
        modal=True,
        width=window_width,
        height=window_height,
        pos=(dpg.get_viewport_width() // 2 - window_width // 2,
             dpg.get_viewport_height() // 2 - window_height // 2),
        no_resize=True,
        no_collapse=True,
        on_close=lambda: _on_apply_dmax_dialog_close(False)
    ):
        dpg.add_spacer(height=10)
        dpg.add_text("Apply current dmax to full flex spline?")
        dpg.add_spacer(height=10)

        # Show current dmax values
        if _last_dmax_x > 0:
            dpg.add_text(f"  dmax_x = {_last_dmax_x:.4f} mm", color=(255, 200, 100))
        else:
            dpg.add_text("  dmax_x = 0 (no X trim)", color=(120, 120, 120))

        if _last_dmax_y > 0:
            dpg.add_text(f"  dmax_y = {_last_dmax_y:.4f} mm", color=(255, 200, 100))
        else:
            dpg.add_text("  dmax_y = 0 (no Y trim)", color=(120, 120, 120))

        dpg.add_spacer(height=20)

        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Yes",
                callback=lambda: _on_apply_dmax_dialog_close(True),
                width=scaled(100)
            )
            dpg.add_spacer(width=20)
            dpg.add_button(
                label="No",
                callback=lambda: _on_apply_dmax_dialog_close(False),
                width=scaled(100)
            )


def _on_apply_dmax_dialog_close(apply_dmax: bool):
    """Handle apply dmax dialog close."""
    global _debug_mode, _trimmed_fs_tooth, _dmax_applied_to_full
    global _applied_dmax_x, _applied_dmax_y

    # Close dialog
    if dpg.does_item_exist("apply_dmax_dialog"):
        dpg.delete_item("apply_dmax_dialog")

    _debug_mode = False
    _trimmed_fs_tooth = None

    # Show main view buttons, hide debug mode buttons
    dpg.configure_item("main_view_buttons", show=True)
    dpg.configure_item("secondary_buttons", show=True)
    dpg.configure_item("debug_mode_buttons", show=False)
    # Hide export dmax button (only shown for undeformed with dmax)
    dpg.configure_item("btn_export_dmax_fs", show=False)

    if apply_dmax:
        _dmax_applied_to_full = True
        _applied_dmax_x = _last_dmax_x
        _applied_dmax_y = _last_dmax_y
        _draw_overlay_with_dmax()
    else:
        _dmax_applied_to_full = False
        _clear_plot_series()
        _draw_overlay()

    dpg.fit_axis_data("tab_ov_x")
    dpg.fit_axis_data("tab_ov_y")


def _draw_overlay_with_dmax():
    """Draw overlay with dmax applied to flexspline."""
    global _last_fs

    params = AppState.read_from_widgets("tab_ov")
    fillet_add = AppState.get_fillet_add()
    fillet_ded = AppState.get_fillet_ded()

    _clear_plot_series()
    y_axis = "tab_ov_y"

    # Build flexspline with dmax applied
    fs_result = build_dmax_deformed_flexspline(
        params,
        dmax_x=_applied_dmax_x,
        dmax_y=_applied_dmax_y,
        r_fillet_add=fillet_add,
        r_fillet_ded=fillet_ded
    )

    if "error" in fs_result:
        update_info_text("tab_ov", f"Flexspline error: {fs_result['error']}", color=(255, 100, 100))
        return

    _last_fs = fs_result

    # Draw flexspline
    chain = fs_result.get("chain_xy", [])
    if chain:
        x_data = [p[0] for p in chain] + [chain[0][0]]
        y_data = [p[1] for p in chain] + [chain[0][1]]

        dpg.add_line_series(
            x_data, y_data,
            label="Flexspline (dmax applied)",
            tag="series_ov_fs",
            parent=y_axis
        )
        dpg.bind_item_theme("series_ov_fs", "theme_line_flexspline")

    # Draw circular spline (unchanged)
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

    # Build status message
    msg_parts = []
    if _applied_dmax_x > 0:
        msg_parts.append(f"dmax_x={_applied_dmax_x:.4f}")
    if _applied_dmax_y > 0:
        msg_parts.append(f"dmax_y={_applied_dmax_y:.4f}")

    fs_pts = len(fs_result.get("chain_xy", []))
    cs_pts = len(_last_cs.get("chain_xy", [])) if _last_cs else 0
    msg = f"dmax applied: {', '.join(msg_parts)} mm | FS={fs_pts}, CS={cs_pts} pts"
    update_info_text("tab_ov", msg, color=(100, 255, 100))


def _draw_debug_tooth():
    """Draw single tooth of each gear without fillets for debugging."""
    global _debug_fs_tooth, _debug_cs_tooth, _debug_fs_addendum, _debug_segment_info

    if _last_fs is None or _last_smoothed_flank is None:
        update_info_text("tab_ov", "Click Update first to compute gears.", color=(255, 200, 100))
        return

    _clear_plot_series()
    _debug_fs_tooth = None
    _debug_cs_tooth = None
    _debug_fs_addendum = None
    _debug_segment_info = None

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

        # Store segment lengths for index-based dmax_y application
        n_AB = len(pts_AB)
        n_BC = len(pts_BC)
        n_CD = len(pts_CD)
        n_flank = n_AB + n_BC + n_CD

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

        # Store segment info for index-based trim
        # fs_tooth structure: [left_flank (CD_rev + BC_rev + AB_rev mirrored)] + [right_flank (AB + BC + CD)]
        # AB indices in left_flank: from (n_CD + n_BC) to (n_flank - 1) (these are mirrored/reversed)
        # AB indices in right_flank: from 0 to (n_AB - 1)
        # In combined fs_tooth:
        #   left AB: indices n_CD + n_BC to n_flank - 1
        #   right AB: indices n_flank to n_flank + n_AB - 1
        _debug_segment_info = {
            "n_AB": n_AB,
            "n_BC": n_BC,
            "n_CD": n_CD,
            "n_flank": n_flank,
            "left_ab_start": n_CD + n_BC,
            "left_ab_end": n_flank,  # exclusive
            "right_ab_start": n_flank,
            "right_ab_end": n_flank + n_AB  # exclusive
        }

        # Store addendum segment (AB) for dmax_y calculation
        # This is the top part of the tooth (highest Y values)
        ab_right = [tooth_point_deformed(x, y) for x, y in pts_AB]
        ab_left = [tooth_point_deformed(-x, y) for x, y in pts_AB]
        _debug_fs_addendum = {"right": ab_right, "left": ab_left}

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
    """Calculate maximum interference dmax_x and dmax_y.

    dmax_x: X-axis penetration from CS into FS (all segments)
    dmax_y: Y-axis penetration from CS into FS addendum (AB segment)
    """
    global _last_dmax_x, _last_dmax_y

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
        sorted_pts = sorted(flank_points, key=lambda p: p[1])
        for i in range(len(sorted_pts) - 1):
            y1, y2 = sorted_pts[i][1], sorted_pts[i + 1][1]
            if y1 <= target_y <= y2 or y2 <= target_y <= y1:
                x1, x2 = sorted_pts[i][0], sorted_pts[i + 1][0]
                if abs(y2 - y1) < 1e-9:
                    return (x1 + x2) / 2
                t = (target_y - y1) / (y2 - y1)
                return x1 + t * (x2 - x1)
        return None

    def interpolate_y_at_x(flank_points, target_x):
        """Find Y value on flank at given X by linear interpolation."""
        if not flank_points:
            return None
        sorted_pts = sorted(flank_points, key=lambda p: p[0])
        for i in range(len(sorted_pts) - 1):
            x1, x2 = sorted_pts[i][0], sorted_pts[i + 1][0]
            if x1 <= target_x <= x2 or x2 <= target_x <= x1:
                y1, y2 = sorted_pts[i][1], sorted_pts[i + 1][1]
                if abs(x2 - x1) < 1e-9:
                    return (y1 + y2) / 2
                t = (target_x - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)
        return None

    # === Calculate dmax_x: X-penetration of CS into FS tooth ===
    dmax_x = 0.0
    interference_count_x = 0

    for cs_x, cs_y in cs_points:
        if cs_y < fs_y_min or cs_y > fs_y_max:
            continue

        fs_left_x = interpolate_x_at_y(fs_left, cs_y)
        fs_right_x = interpolate_x_at_y(fs_right, cs_y)

        if fs_left_x is None or fs_right_x is None:
            continue

        # Right side: CS penetrates if X < FS right boundary
        if cs_x > 0 and cs_x < fs_right_x:
            penetration = fs_right_x - cs_x
            if penetration > dmax_x:
                dmax_x = penetration
            interference_count_x += 1

        # Left side: CS penetrates if X > FS left boundary
        if cs_x < 0 and cs_x > fs_left_x:
            penetration = cs_x - fs_left_x
            if penetration > dmax_x:
                dmax_x = penetration
            interference_count_x += 1

    # === Calculate dmax_y: Y-penetration of CS into FS addendum ===
    dmax_y = 0.0
    interference_count_y = 0

    if _debug_fs_addendum is not None:
        ab_right = _debug_fs_addendum["right"]
        ab_left = _debug_fs_addendum["left"]

        # Combine addendum points and find the addendum "top" boundary
        # The addendum is the highest Y region of the tooth
        all_addendum = ab_right + ab_left

        if all_addendum:
            # Get X range of addendum
            add_x_min = min(p[0] for p in all_addendum)
            add_x_max = max(p[0] for p in all_addendum)

            for cs_x, cs_y in cs_points:
                # Only check CS points within addendum X range
                if cs_x < add_x_min or cs_x > add_x_max:
                    continue

                # Find FS addendum Y at this X position
                if cs_x >= 0:
                    fs_add_y = interpolate_y_at_x(ab_right, cs_x)
                else:
                    fs_add_y = interpolate_y_at_x(ab_left, cs_x)

                if fs_add_y is None:
                    continue

                # CS penetrates addendum if CS_Y > FS addendum Y (CS is above FS)
                if cs_y > fs_add_y:
                    penetration = cs_y - fs_add_y
                    if penetration > dmax_y:
                        dmax_y = penetration
                    interference_count_y += 1

    _last_dmax_x = dmax_x
    _last_dmax_y = dmax_y

    # Show popup dialog with results
    _show_dmax_popup(dmax_x, interference_count_x, dmax_y, interference_count_y)


def _show_dmax_popup(dmax_x: float, count_x: int, dmax_y: float, count_y: int):
    """Show popup dialog with dmax results and trim options."""
    # Clean up existing popup
    if dpg.does_item_exist("dmax_popup"):
        dpg.delete_item("dmax_popup")

    window_width = scaled(400)
    window_height = scaled(280)

    with dpg.window(
        label="dmax Calculation Results",
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

        has_interference = dmax_x > 0 or dmax_y > 0

        # Display dmax_x result
        if dmax_x > 0:
            dpg.add_text(f"dmax_x = {dmax_x:.4f} mm", color=(255, 200, 100))
            dpg.add_text(f"  ({count_x} interference points, all segments)", color=(180, 180, 180))
        else:
            dpg.add_text("dmax_x = 0 (no X interference)", color=(100, 255, 100))

        dpg.add_spacer(height=5)

        # Display dmax_y result
        if dmax_y > 0:
            dpg.add_text(f"dmax_y = {dmax_y:.4f} mm", color=(255, 200, 100))
            dpg.add_text(f"  ({count_y} interference points, addendum only)", color=(180, 180, 180))
        else:
            dpg.add_text("dmax_y = 0 (no Y interference)", color=(100, 255, 100))

        dpg.add_spacer(height=15)

        if has_interference:
            dpg.add_text("Select trim options:")
            dpg.add_spacer(height=10)

            # Checkbox for dmax_x trim
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="",
                    tag="chk_trim_x",
                    default_value=dmax_x > 0
                )
                if dmax_x > 0:
                    dpg.add_text(f"Trim X by dmax_x ({dmax_x:.4f} mm)")
                else:
                    dpg.add_text("Trim X by dmax_x (no interference)", color=(120, 120, 120))

            dpg.add_spacer(height=5)

            # Checkbox for dmax_y trim
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="",
                    tag="chk_trim_y",
                    default_value=dmax_y > 0
                )
                if dmax_y > 0:
                    dpg.add_text(f"Trim Y by dmax_y ({dmax_y:.4f} mm)")
                else:
                    dpg.add_text("Trim Y by dmax_y (no interference)", color=(120, 120, 120))

            dpg.add_spacer(height=20)

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Submit",
                    callback=lambda: _apply_trim_options(dmax_x, dmax_y),
                    width=scaled(100)
                )
                dpg.add_spacer(width=20)
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item("dmax_popup"),
                    width=scaled(80)
                )
        else:
            dpg.add_text("No interference detected.", color=(100, 255, 100))
            dpg.add_spacer(height=20)
            dpg.add_button(
                label="OK",
                callback=lambda: dpg.delete_item("dmax_popup"),
                width=-1
            )


def _apply_trim_options(dmax_x: float, dmax_y: float):
    """Apply selected trim options to the flexspline tooth.

    dmax_x: Moves tooth points inward in X (reduces X magnitude)
    dmax_y: Lowers the addendum line and trims AB points above it
    """
    global _trimmed_fs_tooth

    if _debug_fs_tooth is None:
        dpg.delete_item("dmax_popup")
        return

    # Read checkbox values
    trim_x = dpg.get_value("chk_trim_x") if dpg.does_item_exist("chk_trim_x") else False
    trim_y = dpg.get_value("chk_trim_y") if dpg.does_item_exist("chk_trim_y") else False

    if not trim_x and not trim_y:
        dpg.delete_item("dmax_popup")
        update_info_text("tab_ov", "No trim options selected.", color=(180, 180, 180))
        return

    # First apply X trim to all points
    working_points = []
    for x, y in _debug_fs_tooth:
        new_x = x
        if trim_x and dmax_x > 0:
            if x > 0:
                new_x = x - dmax_x
            elif x < 0:
                new_x = x + dmax_x
        working_points.append((new_x, y))

    # Apply Y trim by lowering addendum and trimming AB points above it
    if trim_y and dmax_y > 0 and _debug_segment_info is not None:
        # Find original addendum Y (max Y of tooth)
        original_addendum_y = max(p[1] for p in working_points)
        new_addendum_y = original_addendum_y - dmax_y

        # Get AB index ranges
        left_ab_start = _debug_segment_info["left_ab_start"]
        left_ab_end = _debug_segment_info["left_ab_end"]
        right_ab_start = _debug_segment_info["right_ab_start"]
        right_ab_end = _debug_segment_info["right_ab_end"]

        # Build trimmed tooth:
        # - Keep non-AB points as-is
        # - For AB points, keep only those below new addendum line
        # - Add horizontal addendum line connecting left and right AB at new Y

        # Extract left flank (before right flank starts)
        n_flank = _debug_segment_info["n_flank"]
        left_flank_pts = working_points[:n_flank]
        right_flank_pts = working_points[n_flank:]

        # Process left flank: keep points, but trim AB points above new addendum
        # Left flank structure: CD_rev + BC_rev + AB_rev (mirrored)
        # AB is at indices left_ab_start to left_ab_end
        trimmed_left = []
        left_ab_rightmost = None  # The point where we'll connect the addendum line
        for i, (x, y) in enumerate(left_flank_pts):
            if i >= left_ab_start and i < left_ab_end:
                # This is an AB point - keep if below new addendum
                if y <= new_addendum_y:
                    trimmed_left.append((x, y))
                    # Track the last (rightmost in terms of index) AB point we keep
                    left_ab_rightmost = (x, y)
                else:
                    # Point is above new addendum - find intersection if needed
                    if left_ab_rightmost is None and i > left_ab_start:
                        # Interpolate to find intersection with new addendum line
                        prev_x, prev_y = left_flank_pts[i - 1]
                        if prev_y <= new_addendum_y < y or y < new_addendum_y <= prev_y:
                            t = (new_addendum_y - prev_y) / (y - prev_y) if abs(y - prev_y) > 1e-9 else 0
                            intersect_x = prev_x + t * (x - prev_x)
                            trimmed_left.append((intersect_x, new_addendum_y))
                            left_ab_rightmost = (intersect_x, new_addendum_y)
            else:
                trimmed_left.append((x, y))

        # If we didn't find an intersection point, add one at the last kept AB point's X
        if left_ab_rightmost is None and len(trimmed_left) > 0:
            # Use the last point of left flank
            left_ab_rightmost = trimmed_left[-1]

        # Process right flank: AB is at start (indices 0 to n_AB-1)
        n_AB = _debug_segment_info["n_AB"]
        trimmed_right = []
        right_ab_leftmost = None  # The point where we'll connect the addendum line

        # First, find where AB intersects the new addendum line (going from top down)
        for i in range(n_AB):
            x, y = right_flank_pts[i]
            if y <= new_addendum_y:
                # First point at or below new addendum
                if right_ab_leftmost is None and i > 0:
                    # Interpolate to find intersection
                    prev_x, prev_y = right_flank_pts[i - 1]
                    if prev_y > new_addendum_y >= y:
                        t = (new_addendum_y - prev_y) / (y - prev_y) if abs(y - prev_y) > 1e-9 else 0
                        intersect_x = prev_x + t * (x - prev_x)
                        right_ab_leftmost = (intersect_x, new_addendum_y)
                        trimmed_right.append((intersect_x, new_addendum_y))
                trimmed_right.append((x, y))
            elif right_ab_leftmost is not None:
                # Already past the addendum line, keep remaining points
                trimmed_right.append((x, y))
            # else: point is above new addendum and we haven't crossed yet - skip

        # Add remaining right flank points (BC + CD)
        for i in range(n_AB, len(right_flank_pts)):
            trimmed_right.append(right_flank_pts[i])

        # Build final trimmed tooth with addendum line
        trimmed = []

        # Add trimmed left flank
        trimmed.extend(trimmed_left)

        # Add horizontal addendum line from left AB to right AB
        if left_ab_rightmost is not None and right_ab_leftmost is not None:
            # Interpolate points along the addendum line
            x_left = left_ab_rightmost[0]
            x_right = right_ab_leftmost[0]
            n_addendum_pts = 10
            for j in range(1, n_addendum_pts):
                frac = j / n_addendum_pts
                x_add = x_left + frac * (x_right - x_left)
                trimmed.append((x_add, new_addendum_y))

        # Add trimmed right flank
        trimmed.extend(trimmed_right)

        _trimmed_fs_tooth = trimmed
    else:
        _trimmed_fs_tooth = working_points

    # Close popup
    dpg.delete_item("dmax_popup")

    # Redraw debug view with trimmed tooth
    _redraw_debug_with_trimmed_tooth()

    # Build status message
    msg_parts = []
    if trim_x and dmax_x > 0:
        msg_parts.append(f"X by {dmax_x:.4f}")
    if trim_y and dmax_y > 0:
        msg_parts.append(f"Y by {dmax_y:.4f}")

    msg = "Flexspline trimmed: " + ", ".join(msg_parts) + " mm"
    update_info_text("tab_ov", msg, color=(100, 255, 100))


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
