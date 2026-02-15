"""
Theme and styling definitions for the DearPyGui application.

Provides a professional dark theme optimized for engineering software.
"""

import dearpygui.dearpygui as dpg
import os

# Color palette - Professional dark engineering theme
COLORS = {
    # Background colors
    "background": (30, 30, 35, 255),
    "panel_bg": (40, 40, 48, 255),
    "child_bg": (35, 35, 42, 255),
    "popup_bg": (45, 45, 55, 255),

    # Text colors
    "text": (220, 220, 220, 255),
    "text_dim": (150, 150, 150, 255),
    "text_disabled": (100, 100, 100, 255),

    # Accent colors
    "primary": (80, 140, 200, 255),
    "primary_hover": (100, 160, 220, 255),
    "primary_active": (60, 120, 180, 255),
    "secondary": (100, 180, 100, 255),
    "warning": (200, 180, 80, 255),
    "error": (200, 80, 80, 255),

    # Border and separator
    "border": (60, 60, 70, 255),
    "separator": (80, 80, 90, 255),

    # Input fields
    "input_bg": (50, 50, 60, 255),
    "input_border": (70, 70, 85, 255),

    # Gear visualization colors
    "flexspline": (34, 102, 204, 255),        # Blue
    "circular_spline": (204, 51, 51, 255),    # Red
    "modified": (0, 204, 204, 255),           # Cyan
    "interference": (255, 204, 0, 255),       # Yellow

    # Tooth segment colors
    "segment_AB": (220, 80, 80, 255),         # Red - convex
    "segment_BC": (80, 140, 220, 255),        # Blue - tangent
    "segment_CD": (80, 200, 80, 255),         # Green - concave

    # Reference line colors (semi-transparent)
    "addendum": (150, 150, 255, 200),
    "pitch": (100, 200, 100, 200),
    "dedendum": (255, 150, 150, 200),

    # Plot colors
    "plot_bg": (25, 25, 30, 255),
    "plot_border": (60, 60, 70, 255),
    "grid": (50, 50, 60, 255),
    "axis": (120, 120, 130, 255),
}


def setup_fonts(dpi_scale: float = 1.0):
    """Configure fonts for the application."""
    font_size = int(14 * dpi_scale)

    with dpg.font_registry():
        # Try to load Consolas for consistency with original app
        font_paths = [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/consolab.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]

        default_font = None
        for path in font_paths:
            if os.path.exists(path):
                try:
                    default_font = dpg.add_font(path, font_size)
                    break
                except Exception:
                    continue

        if default_font:
            dpg.bind_font(default_font)


def create_themes():
    """Create all application themes."""

    # Main window theme
    with dpg.theme(tag="theme_main"):
        with dpg.theme_component(dpg.mvAll):
            # Window colors
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, COLORS["background"])
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, COLORS["child_bg"])
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, COLORS["popup_bg"])

            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text, COLORS["text"])
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, COLORS["text_disabled"])

            # Borders
            dpg.add_theme_color(dpg.mvThemeCol_Border, COLORS["border"])
            dpg.add_theme_color(dpg.mvThemeCol_Separator, COLORS["separator"])

            # Frame (input fields, combo boxes)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, COLORS["input_bg"])
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (60, 60, 75, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (70, 70, 85, 255))

            # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_Button, COLORS["primary"])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, COLORS["primary_hover"])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, COLORS["primary_active"])

            # Headers (collapsing headers, tree nodes)
            dpg.add_theme_color(dpg.mvThemeCol_Header, (50, 50, 60, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (60, 60, 75, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (70, 70, 85, 255))

            # Tabs
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (45, 45, 55, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, COLORS["primary_hover"])
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, COLORS["primary"])
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, (40, 40, 50, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, (60, 100, 150, 255))

            # Checkboxes and sliders
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, COLORS["primary"])
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, COLORS["primary"])
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, COLORS["primary_hover"])

            # Title bar
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (35, 35, 42, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (45, 45, 55, 255))

            # Menu
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (35, 35, 42, 255))

            # Scrollbar
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (30, 30, 38, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (60, 60, 75, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (80, 80, 95, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (100, 100, 115, 255))

            # Styling
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4)

            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 4, 4)

        # Plot-specific styling
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, COLORS["plot_bg"])
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, COLORS["plot_border"])
            dpg.add_theme_color(dpg.mvPlotCol_FrameBg, COLORS["child_bg"])
            dpg.add_theme_color(dpg.mvPlotCol_Line, COLORS["flexspline"])
            dpg.add_theme_color(dpg.mvPlotCol_AxisText, COLORS["axis"])
            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid, COLORS["grid"])

    # Line series themes for different visualization elements
    _create_line_theme("theme_line_AB", COLORS["segment_AB"], 2.0)
    _create_line_theme("theme_line_BC", COLORS["segment_BC"], 2.0)
    _create_line_theme("theme_line_CD", COLORS["segment_CD"], 2.0)
    _create_line_theme("theme_line_flexspline", COLORS["flexspline"], 2.0)
    _create_line_theme("theme_line_circspline", COLORS["circular_spline"], 2.0)
    _create_line_theme("theme_line_modified", COLORS["modified"], 2.0)
    _create_line_theme("theme_line_interference", COLORS["interference"], 3.0)
    _create_line_theme("theme_line_mirror", (128, 128, 128, 200), 1.5)

    # Reference line themes (dashed style via thinner weight)
    _create_line_theme("theme_line_addendum", COLORS["addendum"], 1.5)
    _create_line_theme("theme_line_pitch", COLORS["pitch"], 1.5)
    _create_line_theme("theme_line_dedendum", COLORS["dedendum"], 1.5)

    # Button themes
    _create_button_theme("theme_button_update", COLORS["primary"])
    _create_button_theme("theme_button_reset", (100, 100, 110, 255))
    _create_button_theme("theme_button_export", COLORS["secondary"])
    _create_button_theme("theme_button_danger", COLORS["error"])

    # Input validation themes
    with dpg.theme(tag="theme_input_error"):
        with dpg.theme_component(dpg.mvInputFloat):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (80, 40, 40, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, COLORS["error"])

    with dpg.theme(tag="theme_input_valid"):
        with dpg.theme_component(dpg.mvInputFloat):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, COLORS["input_bg"])
            dpg.add_theme_color(dpg.mvThemeCol_Border, COLORS["input_border"])


def _create_line_theme(tag: str, color: tuple, weight: float):
    """Helper to create a line series theme."""
    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, color)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, weight)


def _create_button_theme(tag: str, color: tuple):
    """Helper to create a button theme."""
    hover = tuple(min(c + 25, 255) for c in color[:3]) + (255,)
    active = tuple(max(c - 20, 0) for c in color[:3]) + (255,)

    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, color)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hover)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, active)


def get_color(name: str) -> tuple:
    """Get a color by name from the palette."""
    return COLORS.get(name, (255, 255, 255, 255))
