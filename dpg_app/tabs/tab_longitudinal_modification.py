"""
Tab 2.6 - Longitudinal Modification

Implements longitudinal spline modification based on the paper's Figure 12/15.
Discretizes flexspline along longitudinal direction and calculates radial
modification amounts for each section using the rack method.

Key concept:
- The flexspline cup deforms radially, but deformation varies along the cup length
- Maximum deformation at the open end (wave generator contact)
- Zero deformation at the closed cup bottom
- Each longitudinal section needs different radial modification to compensate
- Uses rack method: walks deformed FS tooth through CS engagement to measure interference
"""

import dearpygui.dearpygui as dpg
import numpy as np
import math
import os
import sys

# Add parent directory to path for equations import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from equations import (
    compute_profile, compute_conjugate_profile, smooth_conjugate_profile,
    eq14_rho, eq21_mu, eq23_phi1, eq27_psi, eq29_transform
)

from dpg_app.app_state import AppState, scaled
from dpg_app.widgets.output_panel import (
    create_info_text, update_info_text
)
from dpg_app.themes import COLORS

# Path to reference images
REF_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Ref Images")

# Track loaded texture
_longmod_texture = None

# Module state for storing calculation results
_last_calculation = None


def _load_reference_texture():
    """Load the longitudinal modification reference image texture."""
    global _longmod_texture

    if _longmod_texture is not None:
        return _longmod_texture

    image_path = os.path.join(REF_IMAGES_DIR, "LongModRef.png")
    if not os.path.exists(image_path):
        print(f"Reference image not found: {image_path}")
        return None

    try:
        width, height, channels, data = dpg.load_image(image_path)

        # Create texture registry if it doesn't exist
        if not dpg.does_item_exist("longmod_texture_registry"):
            dpg.add_texture_registry(tag="longmod_texture_registry")

        texture_tag = "texture_longmod_ref"
        if not dpg.does_item_exist(texture_tag):
            dpg.add_static_texture(
                width=width,
                height=height,
                default_value=data,
                tag=texture_tag,
                parent="longmod_texture_registry"
            )

        _longmod_texture = (texture_tag, width, height)
        return _longmod_texture
    except Exception as e:
        print(f"Error loading longitudinal modification reference: {e}")
        return None


def _show_reference_popup():
    """Show the longitudinal modification reference image in a popup."""
    # Clean up existing window
    if dpg.does_item_exist("longmod_ref_window"):
        dpg.delete_item("longmod_ref_window")

    texture_info = _load_reference_texture()
    if texture_info is None:
        return

    texture_tag, img_width, img_height = texture_info

    # Scale image to fit reasonably on screen
    max_width = scaled(900)
    max_height = scaled(600)

    scale = min(max_width / img_width, max_height / img_height, 1.0)
    display_width = int(img_width * scale)
    display_height = int(img_height * scale)

    # Create popup window
    window_width = display_width + scaled(20)
    window_height = display_height + scaled(60)

    with dpg.window(
        label="Longitudinal Spline Modification Reference (Figure 15)",
        tag="longmod_ref_window",
        modal=True,
        width=window_width,
        height=window_height,
        pos=(dpg.get_viewport_width() // 2 - window_width // 2,
             dpg.get_viewport_height() // 2 - window_height // 2),
        no_resize=False,
        no_collapse=True,
        on_close=lambda: dpg.delete_item("longmod_ref_window")
    ):
        dpg.add_image(texture_tag, width=display_width, height=display_height)
        dpg.add_spacer(height=5)
        dpg.add_button(
            label="Close",
            callback=lambda: dpg.delete_item("longmod_ref_window"),
            width=-1
        )


def create_tab_longitudinal_modification():
    """Create the Tab 2.6 content."""
    with dpg.group(horizontal=True):
        # Left panel - Parameters
        with dpg.child_window(width=scaled(340), border=True):
            dpg.add_text("Longitudinal Modification", color=(180, 180, 255))
            dpg.add_spacer(height=10)

            dpg.add_text(
                "Calculates radial modification along the flexspline length. "
                "The cup wall bows opposite to flex direction (Fig. 12), causing "
                "interference on both sides of the main section. Teeth must be "
                "shortened away from the main engagement section.",
                wrap=scaled(320),
                color=(150, 150, 150)
            )

            dpg.add_spacer(height=15)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Reference image button
            dpg.add_button(
                label="Show Reference (Fig. 15)",
                tag="btn_show_longmod_ref",
                callback=_show_reference_popup,
                width=-1
            )

            dpg.add_spacer(height=15)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Geometry Parameters
            dpg.add_text("Geometry Parameters", color=(180, 180, 255))
            dpg.add_spacer(height=5)

            # Distance from cup bottom to start of teeth
            with dpg.group(horizontal=True):
                dpg.add_text("l\u2080 (mm):", color=(150, 150, 150))
                dpg.add_input_float(
                    tag="input_l0",
                    default_value=10.0,
                    width=scaled(100),
                    format="%.2f",
                    step=0.5,
                    min_value=0.1,
                    min_clamped=True
                )
            dpg.add_text("  Distance from cup bottom to start of teeth", color=(100, 100, 100), wrap=scaled(320))

            dpg.add_spacer(height=5)

            # Distance from cup bottom to main section (midpoint of teeth)
            with dpg.group(horizontal=True):
                dpg.add_text("l\u1d62 main (mm):", color=(150, 150, 150))
                dpg.add_input_float(
                    tag="input_li_main",
                    default_value=15.0,
                    width=scaled(100),
                    format="%.2f",
                    step=0.5,
                    min_value=0.1,
                    min_clamped=True
                )
            dpg.add_text("  Distance from cup bottom to main section", color=(100, 100, 100), wrap=scaled(320))

            dpg.add_spacer(height=5)

            # Derived tooth length (read-only display)
            with dpg.group(horizontal=True):
                dpg.add_text("Tooth length:", color=(150, 150, 150))
                dpg.add_text("--", tag="display_tooth_length", color=(200, 200, 100))
            dpg.add_text("  = 2 \u00d7 (l\u1d62 - l\u2080)", color=(100, 100, 100), wrap=scaled(320))

            dpg.add_spacer(height=5)

            # Number of sections
            with dpg.group(horizontal=True):
                dpg.add_text("Sections n:", color=(150, 150, 150))
                dpg.add_input_int(
                    tag="input_n_sections",
                    default_value=16,
                    width=scaled(100),
                    min_value=4,
                    max_value=64,
                    min_clamped=True,
                    max_clamped=True
                )
            dpg.add_text("  Number of discrete sections", color=(100, 100, 100), wrap=scaled(320))

            dpg.add_spacer(height=15)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Deformation Parameters
            dpg.add_text("Deformation Parameters", color=(180, 180, 255))
            dpg.add_spacer(height=5)

            # Main section deformation coefficient
            with dpg.group(horizontal=True):
                dpg.add_text("Main coeff k\u2080:", color=(150, 150, 150))
                dpg.add_input_float(
                    tag="input_k0",
                    default_value=1.0,
                    width=scaled(100),
                    format="%.3f",
                    step=0.1,
                    min_value=0.1,
                    min_clamped=True
                )
            dpg.add_text("  Deformation coefficient at main section", color=(100, 100, 100), wrap=scaled(320))

            dpg.add_spacer(height=5)

            # w0 from main parameters (read-only display)
            with dpg.group(horizontal=True):
                dpg.add_text("w\u2080 (from Tab 2.5):", color=(150, 150, 150))
                dpg.add_text("--", tag="display_w0", color=(200, 200, 100))
            dpg.add_text("  Max radial deformation from parameters", color=(100, 100, 100), wrap=scaled(320))

            dpg.add_spacer(height=15)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Polynomial fitting
            dpg.add_text("Output Options", color=(180, 180, 255))
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="",
                    tag="chk_poly_fit",
                    default_value=True
                )
                dpg.add_text("Fit polynomial (for machining)", color=(150, 150, 150))

            with dpg.group(horizontal=True):
                dpg.add_text("  Degree:", color=(100, 100, 100))
                dpg.add_input_int(
                    tag="input_poly_degree",
                    default_value=4,
                    width=scaled(60),
                    min_value=1,
                    max_value=6,
                    min_clamped=True,
                    max_clamped=True
                )

            dpg.add_spacer(height=15)

            # Action buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Calculate",
                    tag="btn_calc_longmod",
                    callback=_calculate_modification,
                    width=scaled(100)
                )
                dpg.add_button(
                    label="Reset",
                    tag="btn_reset_longmod",
                    callback=_reset_parameters,
                    width=scaled(100)
                )

            dpg.add_spacer(height=10)
            dpg.add_separator()

            # Results display
            dpg.add_spacer(height=10)
            dpg.add_text("Results", color=(180, 180, 255))
            dpg.add_spacer(height=5)

            with dpg.child_window(tag="longmod_results_panel", height=scaled(120), border=False):
                dpg.add_text("", tag="longmod_result_1", color=(200, 200, 200))
                dpg.add_text("", tag="longmod_result_2", color=(200, 200, 200))
                dpg.add_text("", tag="longmod_result_3", color=(200, 200, 200))
                dpg.add_text("", tag="longmod_result_4", color=(200, 200, 200))
                dpg.add_text("", tag="longmod_result_5", color=(200, 200, 200))

            create_info_text("tab_longmod", "Click 'Calculate' to compute modification profile.")

        # Right panel - Plot
        with dpg.child_window(width=-1, border=True):
            _create_plot()


def _create_plot():
    """Create the visualization plot."""
    with dpg.plot(
        label="Longitudinal Modification Profile",
        tag="tab_longmod_plot",
        width=-1,
        height=-1,
        anti_aliased=True
    ):
        dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)

        dpg.add_plot_axis(dpg.mvXAxis, label="l\u1d62 - Distance from cup bottom (mm)", tag="tab_longmod_x")
        with dpg.plot_axis(dpg.mvYAxis, label="Radial Modification (mm)", tag="tab_longmod_y"):
            pass


def _generate_deformed_fs_tooth(params, w_value):
    """Generate a deformed flexspline tooth profile for a given w (deformation).

    Returns the tooth outline points and addendum points for interference calculation.
    """
    # Create modified params with the specified w value
    modified_params = params.copy()
    modified_params["w0"] = w_value

    profile = compute_profile(modified_params)
    if "error" in profile:
        return None, None

    rm = profile["rm"]

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
        rho = eq14_rho(phi, rm, w_value)
        mu = eq21_mu(phi, w_value, rm)
        phi1 = eq23_phi1(phi, w_value, rm)
        gamma = phi1  # phi2 = 0
        psi = eq27_psi(mu, gamma)
        return eq29_transform(xr, yr, psi, rho, gamma)

    # Build deformed tooth outline
    fs_tooth = []
    for x, y in left_flank:
        fs_tooth.append(tooth_point_deformed(x, y))
    for x, y in right_flank:
        fs_tooth.append(tooth_point_deformed(x, y))

    # Build addendum points for dmax_y calculation
    ab_right = [tooth_point_deformed(x, y) for x, y in pts_AB]
    ab_left = [tooth_point_deformed(-x, y) for x, y in pts_AB]
    fs_addendum = {"right": ab_right, "left": ab_left}

    return fs_tooth, fs_addendum


def _generate_cs_tooth(params):
    """Generate a circular spline tooth profile.

    Returns the tooth outline points.
    """
    smooth_val = AppState.get_smooth()

    conj = compute_conjugate_profile(params)
    if "error" in conj:
        return None, None

    smoothed = smooth_conjugate_profile(conj, s=smooth_val)
    if "error" in smoothed:
        return None, None

    rp_c = conj.get("rp_c", 25.5)
    smoothed_flank = smoothed.get("smoothed_flank", [])

    if not smoothed_flank:
        return None, None

    # Right flank from smoothed data (already in tooth-local coords)
    right_flank_cs = list(smoothed_flank)
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

    return cs_tooth, rp_c


def _calculate_interference(fs_tooth, fs_addendum, cs_tooth):
    """Calculate interference between FS and CS teeth.

    Returns (dmax_x, dmax_y) - the maximum penetration in X and Y directions.
    """
    if not fs_tooth or not cs_tooth:
        return 0.0, 0.0

    # Split FS tooth into left and right flanks based on X coordinate
    fs_right = [(x, y) for x, y in fs_tooth if x > 0]
    fs_left = [(x, y) for x, y in fs_tooth if x < 0]

    # Get Y range of FS tooth
    fs_y_min = min(p[1] for p in fs_tooth)
    fs_y_max = max(p[1] for p in fs_tooth)

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

    # Calculate dmax_x: X-penetration of CS into FS tooth
    dmax_x = 0.0
    for cs_x, cs_y in cs_tooth:
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

        # Left side: CS penetrates if X > FS left boundary
        if cs_x < 0 and cs_x > fs_left_x:
            penetration = cs_x - fs_left_x
            if penetration > dmax_x:
                dmax_x = penetration

    # Calculate dmax_y: Y-penetration of CS into FS addendum
    dmax_y = 0.0
    if fs_addendum is not None:
        ab_right = fs_addendum["right"]
        ab_left = fs_addendum["left"]
        all_addendum = ab_right + ab_left

        if all_addendum:
            add_x_min = min(p[0] for p in all_addendum)
            add_x_max = max(p[0] for p in all_addendum)

            for cs_x, cs_y in cs_tooth:
                if cs_x < add_x_min or cs_x > add_x_max:
                    continue

                if cs_x >= 0:
                    fs_add_y = interpolate_y_at_x(ab_right, cs_x)
                else:
                    fs_add_y = interpolate_y_at_x(ab_left, cs_x)

                if fs_add_y is None:
                    continue

                # CS penetrates addendum if CS_Y > FS addendum Y
                if cs_y > fs_add_y:
                    penetration = cs_y - fs_add_y
                    if penetration > dmax_y:
                        dmax_y = penetration

    return dmax_x, dmax_y


def _calculate_modification():
    """Calculate the longitudinal modification amounts using rack method.

    For each longitudinal section:
    1. Calculate the deformation coefficient k_i at that section
    2. Generate deformed FS tooth with w_i = w0 * k_i
    3. Calculate interference with CS tooth
    4. The interference is the required modification
    """
    global _last_calculation

    # Read parameters
    l0 = dpg.get_value("input_l0")  # Distance from cup bottom to start of teeth
    li_main = dpg.get_value("input_li_main")  # Distance from cup bottom to main section
    n_sections = dpg.get_value("input_n_sections")
    k0 = dpg.get_value("input_k0")
    fit_poly = dpg.get_value("chk_poly_fit")
    poly_degree = dpg.get_value("input_poly_degree")

    # Get base parameters from AppState
    params = AppState.read_from_widgets("tab_ov")
    w0 = params["w0"]

    # Calculate derived values
    tooth_length = 2 * (li_main - l0)
    l_end = l0 + tooth_length

    # Update displays
    if dpg.does_item_exist("display_w0"):
        dpg.set_value("display_w0", f"{w0:.4f} mm")
    if dpg.does_item_exist("display_tooth_length"):
        dpg.set_value("display_tooth_length", f"{tooth_length:.2f} mm")

    # Validate inputs
    if li_main <= l0:
        update_info_text(
            "tab_longmod",
            "Error: Main section l\u1d62 must be greater than l\u2080.",
            color=(255, 100, 100)
        )
        return

    if l0 <= 0 or li_main <= 0:
        update_info_text(
            "tab_longmod",
            "Error: Distances must be positive.",
            color=(255, 100, 100)
        )
        return

    update_info_text("tab_longmod", "Computing with rack method...", color=(255, 200, 100))

    # Generate CS tooth once (doesn't change between sections)
    cs_tooth, rp_c = _generate_cs_tooth(params)
    if cs_tooth is None:
        update_info_text(
            "tab_longmod",
            "Error: Could not generate circular spline tooth.",
            color=(255, 100, 100)
        )
        return

    # Calculate section positions
    section_positions = np.linspace(l0, l_end, n_sections)

    # For each section, calculate the deformation and interference
    # k_i = k0 * (li / li_main) - deformation coefficient varies with position
    # At li_main: k = k0 (full deformation, design point)
    # At l0: k = k0 * (l0 / li_main) < k0 (less deformation)
    # At l_end: k = k0 * (l_end / li_main) > k0 (more deformation)

    k_values = []
    modification_mm = []
    dmax_x_values = []
    dmax_y_values = []

    # Calculate interference at main section (baseline - should be ~0 or the design interference)
    w_main = w0 * k0
    fs_main, fs_add_main = _generate_deformed_fs_tooth(params, w_main)
    if fs_main is None:
        update_info_text(
            "tab_longmod",
            "Error: Could not generate flexspline tooth at main section.",
            color=(255, 100, 100)
        )
        return

    dmax_x_main, dmax_y_main = _calculate_interference(fs_main, fs_add_main, cs_tooth)
    baseline_interference = max(dmax_x_main, dmax_y_main)

    for li in section_positions:
        # Calculate k_i for this section
        # k varies linearly with position, with k = k0 at li_main
        k_i = k0 * (li / li_main)
        k_values.append(k_i)

        # Calculate w_i for this section
        w_i = w0 * k_i

        # Generate FS tooth with this deformation
        fs_tooth, fs_addendum = _generate_deformed_fs_tooth(params, w_i)

        if fs_tooth is None:
            # If generation fails, use interpolated value
            dmax_x_values.append(0.0)
            dmax_y_values.append(0.0)
            modification_mm.append(0.0)
            continue

        # Calculate interference
        dmax_x, dmax_y = _calculate_interference(fs_tooth, fs_addendum, cs_tooth)
        dmax_x_values.append(dmax_x)
        dmax_y_values.append(dmax_y)

        # The modification needed is the interference relative to baseline
        # Negative because we need to reduce the tooth (remove material)
        total_interference = max(dmax_x, dmax_y)
        mod = -(total_interference - baseline_interference)
        modification_mm.append(mod)

    k_values = np.array(k_values)
    modification_mm = np.array(modification_mm)
    dmax_x_values = np.array(dmax_x_values)
    dmax_y_values = np.array(dmax_y_values)

    # Store calculation results
    _last_calculation = {
        "positions": section_positions,
        "k_values": k_values,
        "modification_mm": modification_mm,
        "dmax_x_values": dmax_x_values,
        "dmax_y_values": dmax_y_values,
        "l0": l0,
        "li_main": li_main,
        "l_end": l_end,
        "tooth_length": tooth_length,
        "k0": k0,
        "w0": w0,
        "baseline_interference": baseline_interference
    }

    # Clear existing plot series
    _clear_plot_series()

    y_axis = "tab_longmod_y"

    # Plot discrete modification values (in mm)
    dpg.add_stem_series(
        section_positions.tolist(),
        modification_mm.tolist(),
        label="Section Modification",
        tag="series_longmod_discrete",
        parent=y_axis
    )
    dpg.bind_item_theme("series_longmod_discrete", "theme_line_flexspline")

    # Polynomial fit if requested
    poly_coeffs = None
    if fit_poly and n_sections > poly_degree:
        try:
            # Fit polynomial (in mm)
            poly_coeffs = np.polyfit(section_positions, modification_mm, poly_degree)
            poly_func = np.poly1d(poly_coeffs)

            # Generate smooth curve for plotting
            x_smooth = np.linspace(l0, l_end, 100)
            y_smooth = poly_func(x_smooth)

            dpg.add_line_series(
                x_smooth.tolist(),
                y_smooth.tolist(),
                label=f"Polynomial Fit (deg {poly_degree})",
                tag="series_longmod_poly",
                parent=y_axis
            )
            dpg.bind_item_theme("series_longmod_poly", "theme_line_circspline")

            _last_calculation["poly_coeffs"] = poly_coeffs

        except Exception as e:
            print(f"Polynomial fit failed: {e}")

    # Add reference line at zero
    dpg.add_line_series(
        [l0, l_end],
        [0, 0],
        label="Zero Reference",
        tag="series_longmod_zero",
        parent=y_axis
    )

    # Add vertical line at main section
    mod_range = max(abs(modification_mm.min()), abs(modification_mm.max())) * 1.2
    if mod_range < 0.001:
        mod_range = 0.01  # Minimum range in mm
    dpg.add_line_series(
        [li_main, li_main],
        [-mod_range, mod_range],
        label=f"Main Section (l\u1d62={li_main})",
        tag="series_longmod_main",
        parent=y_axis
    )

    # Fit axes
    dpg.fit_axis_data("tab_longmod_x")
    dpg.fit_axis_data("tab_longmod_y")

    # Update results display
    _update_results_display(section_positions, modification_mm, k_values, poly_coeffs, poly_degree)

    # Update status
    max_mod = np.max(np.abs(modification_mm))
    update_info_text(
        "tab_longmod",
        f"Computed {n_sections} sections. Max modification: {max_mod:.4f} mm",
        color=(100, 255, 100)
    )


def _update_results_display(positions, modifications, k_values, poly_coeffs, poly_degree):
    """Update the results panel with calculation summary."""
    # Result 1: Max modification (most negative)
    mod_min = np.min(modifications)
    mod_max = np.max(modifications)
    if dpg.does_item_exist("longmod_result_1"):
        dpg.set_value("longmod_result_1", f"Mod range: {mod_min:.4f} to {mod_max:.4f} mm")

    # Result 2: Baseline interference at main section
    baseline = _last_calculation.get("baseline_interference", 0) if _last_calculation else 0
    if dpg.does_item_exist("longmod_result_2"):
        dpg.set_value("longmod_result_2", f"Baseline dmax: {baseline:.4f} mm")

    # Result 3: Modification at start of teeth (li=l0)
    if dpg.does_item_exist("longmod_result_3"):
        dpg.set_value("longmod_result_3", f"At tooth start: {modifications[0]:.4f} mm")

    # Result 4: Modification at end of teeth
    if dpg.does_item_exist("longmod_result_4"):
        dpg.set_value("longmod_result_4", f"At tooth end: {modifications[-1]:.4f} mm")

    # Result 5: k value range
    k_min = np.min(k_values)
    k_max = np.max(k_values)
    if dpg.does_item_exist("longmod_result_5"):
        dpg.set_value("longmod_result_5", f"k range: {k_min:.3f} to {k_max:.3f}")


def _clear_plot_series():
    """Clear all plot series."""
    tags = ["series_longmod_discrete", "series_longmod_poly", "series_longmod_zero", "series_longmod_main"]
    for tag in tags:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)


def _reset_parameters():
    """Reset parameters to defaults."""
    if dpg.does_item_exist("input_l0"):
        dpg.set_value("input_l0", 10.0)
    if dpg.does_item_exist("input_li_main"):
        dpg.set_value("input_li_main", 15.0)
    if dpg.does_item_exist("input_n_sections"):
        dpg.set_value("input_n_sections", 16)
    if dpg.does_item_exist("input_k0"):
        dpg.set_value("input_k0", 1.0)
    if dpg.does_item_exist("chk_poly_fit"):
        dpg.set_value("chk_poly_fit", True)
    if dpg.does_item_exist("input_poly_degree"):
        dpg.set_value("input_poly_degree", 4)

    # Clear results and displays
    if dpg.does_item_exist("display_tooth_length"):
        dpg.set_value("display_tooth_length", "--")
    for i in range(1, 6):
        tag = f"longmod_result_{i}"
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, "")

    # Clear plot
    _clear_plot_series()

    update_info_text(
        "tab_longmod",
        "Parameters reset to defaults.",
        color=(100, 255, 100)
    )
