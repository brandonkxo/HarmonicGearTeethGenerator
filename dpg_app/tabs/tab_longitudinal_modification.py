"""
Tab 2.6 - Longitudinal Modification

Implements longitudinal spline modification based on the paper's Figure 15.
Discretizes flexspline along longitudinal direction and calculates radial
modification amounts for each section based on varying deformation coefficients.

Key concept:
- The flexspline cup deforms radially, but deformation varies along the cup length
- Maximum deformation at the open end (wave generator contact)
- Zero deformation at the closed cup bottom
- Each longitudinal section needs different radial modification to compensate
"""

import dearpygui.dearpygui as dpg
import numpy as np
import os

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


def _calculate_modification():
    """Calculate the longitudinal modification amounts."""
    global _last_calculation

    # Read parameters
    l0 = dpg.get_value("input_l0")  # Distance from cup bottom to start of teeth
    li_main = dpg.get_value("input_li_main")  # Distance from cup bottom to main section
    n_sections = dpg.get_value("input_n_sections")
    k0 = dpg.get_value("input_k0")
    fit_poly = dpg.get_value("chk_poly_fit")
    poly_degree = dpg.get_value("input_poly_degree")

    # Get w0 from AppState
    w0 = AppState.get_param("w0")

    # Calculate derived values
    # Tooth length = 2 × (li_main - l0)
    # Main section is at the midpoint of the tooth region
    tooth_length = 2 * (li_main - l0)
    l_end = l0 + tooth_length  # End of teeth

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

    update_info_text("tab_longmod", "Computing modification profile...", color=(255, 200, 100))

    # Calculate section positions
    # Sections span from l0 (start of teeth) to l_end (end of teeth)
    # l_i = distance from cup bottom to section i
    section_positions = np.linspace(l0, l_end, n_sections)

    # Calculate modification based on Figure 12 from the paper:
    # The cup wall bows in the opposite direction on the non-flex side,
    # causing interference on BOTH sides of the main engagement section.
    #
    # Definitions from paper:
    # - l0: distance from cup bottom to start of teeth
    # - li: distance from cup bottom to section i
    # - li_main: distance from cup bottom to main section (midpoint of teeth)
    # - tooth_length = 2 × (li_main - l0)
    #
    # Key insight:
    # - At main section (li = li_main): Zero modification (design point)
    # - Away from main section (either direction): The cup bowing causes
    #   the teeth to effectively protrude more, creating interference
    # - Therefore, we need NEGATIVE modification (reduce tooth height)
    #   on both sides of the main section
    #
    # The modification follows a parabolic relationship:
    # - Centered at main section (li_main)
    # - Magnitude increases with distance from main section
    # - Always negative (tooth needs to be shorter)

    # Calculate distance from main section for each section
    # delta_l = li - li_main (can be positive or negative)
    delta_l = section_positions - li_main

    # Half tooth length (distance from main section to either end)
    half_tooth = li_main - l0

    # Normalize by half tooth length
    normalized_dist = delta_l / half_tooth

    # Modification is proportional to (distance from main section)^2
    # Using a parabolic model: modification = -w0 * k0 * (normalized_dist)^2
    # This gives zero at li_main and increasingly negative values away from it
    modification_mm = -w0 * k0 * (normalized_dist ** 2)

    # Calculate effective k values for display (shows how much "extra" deformation)
    # k_effective represents the interference factor at each section
    k_values = k0 * (1 + np.abs(normalized_dist))

    # Store calculation results
    _last_calculation = {
        "positions": section_positions,
        "k_values": k_values,
        "modification_mm": modification_mm,
        "l0": l0,
        "li_main": li_main,
        "l_end": l_end,
        "tooth_length": tooth_length,
        "k0": k0,
        "w0": w0
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
            x_smooth = np.linspace(0, L_tooth, 100)
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
    if dpg.does_item_exist("longmod_result_1"):
        dpg.set_value("longmod_result_1", f"Max reduction: {mod_min:.4f} mm")

    # Result 2: At main section (should be ~0)
    # Find index closest to li_main
    li_main = _last_calculation["li_main"] if _last_calculation else 0
    main_idx = np.argmin(np.abs(positions - li_main))
    if dpg.does_item_exist("longmod_result_2"):
        dpg.set_value("longmod_result_2", f"At main section (l\u2080): {modifications[main_idx]:.4f} mm")

    # Result 3: Modification at start of teeth (li=l0)
    l0 = _last_calculation["l0"] if _last_calculation else 0
    if dpg.does_item_exist("longmod_result_3"):
        dpg.set_value("longmod_result_3", f"At tooth start (l\u1d62=l\u2080): {modifications[0]:.4f} mm")

    # Result 4: Modification at end of teeth
    if dpg.does_item_exist("longmod_result_4"):
        dpg.set_value("longmod_result_4", f"At tooth end: {modifications[-1]:.4f} mm")

    # Result 5: Polynomial coefficients if available
    if dpg.does_item_exist("longmod_result_5"):
        if poly_coeffs is not None:
            # Format polynomial: highest degree first
            terms = []
            for i, c in enumerate(poly_coeffs):
                power = poly_degree - i
                if abs(c) > 1e-6:
                    if power == 0:
                        terms.append(f"{c:.3f}")
                    elif power == 1:
                        terms.append(f"{c:.3f}x")
                    else:
                        terms.append(f"{c:.3f}x^{power}")
            poly_str = " + ".join(terms[:3])  # Show first 3 terms
            if len(terms) > 3:
                poly_str += " + ..."
            dpg.set_value("longmod_result_5", f"Poly: {poly_str}")
        else:
            dpg.set_value("longmod_result_5", "")


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
