"""
DearPyGui Application - Harmonic Drive DCT Tooth Calculator

Entry point for the redesigned GUI using DearPyGui.
"""

import dearpygui.dearpygui as dpg
import sys
import os
import ctypes

def set_windows_dark_titlebar():
    """Enable dark mode for the window title bar on Windows 10/11."""
    if sys.platform != "win32":
        return
    try:
        hwnd = ctypes.windll.user32.GetActiveWindow()
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        value = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(value), ctypes.sizeof(value)
        )
    except Exception:
        pass

# Get DPI scale factor and enable DPI awareness on Windows
_dpi_scale = 1.0
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
        # Get the DPI scale factor
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        dc = user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(dc, 88)  # LOGPIXELSX
        user32.ReleaseDC(0, dc)
        _dpi_scale = dpi / 96.0  # 96 is the default DPI
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

# Add parent directory to path for equations import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpg_app.themes import create_themes, setup_fonts
from dpg_app.app_state import AppState, set_dpi_scale
from dpg_app.config_manager import show_save_dialog, show_load_dialog
from dpg_app.tabs.tab_flexspline_tooth import create_tab_flexspline_tooth
from dpg_app.tabs.tab_conjugate_tooth import create_tab_conjugate_tooth
from dpg_app.tabs.tab_flexspline_full import create_tab_flexspline_full
from dpg_app.tabs.tab_circular_spline import create_tab_circular_spline
from dpg_app.tabs.tab_radial_modification import create_tab_radial_modification
from dpg_app.tabs.tab_longitudinal_modification import create_tab_longitudinal_modification


def create_menu_bar():
    """Create the application menu bar."""
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(
                label="Save Config...",
                callback=lambda: show_save_dialog(),
                shortcut="Ctrl+S"
            )
            dpg.add_menu_item(
                label="Load Config...",
                callback=lambda: show_load_dialog(),
                shortcut="Ctrl+O"
            )
            dpg.add_separator()
            dpg.add_menu_item(
                label="Exit",
                callback=lambda: dpg.stop_dearpygui()
            )

        with dpg.menu(label="View"):
            dpg.add_menu_item(
                label="Fit All Axes",
                callback=fit_current_plot,
                shortcut="Home"
            )
            dpg.add_menu_item(
                label="Reset Parameters",
                callback=lambda: AppState.reset_to_defaults(),
                shortcut="Ctrl+R"
            )


def create_status_bar():
    """Create the bottom status bar."""
    with dpg.group(horizontal=True, tag="status_bar"):
        dpg.add_text("Ready", tag="status_text")
        dpg.add_spacer(width=30)
        dpg.add_text("Points: 0", tag="point_count")
        dpg.add_spacer(width=30)
        dpg.add_text("Mode: --", tag="mode_text")


def fit_current_plot():
    """Fit the current tab's plot to its data."""
    current_tab = dpg.get_value("main_tabs")
    # Map tab to its plot axes
    tab_axes = {
        "tab_21": ("tab21_x", "tab21_y"),
        "tab_22": ("tab22_x", "tab22_y"),
        "tab_flexspline": ("tab_fs_x", "tab_fs_y"),
        "tab_circular": ("tab_cs_x", "tab_cs_y"),
        "tab_overlay": ("tab_ov_x", "tab_ov_y"),
    }
    if current_tab in tab_axes:
        x_axis, y_axis = tab_axes[current_tab]
        if dpg.does_item_exist(x_axis):
            dpg.fit_axis_data(x_axis)
        if dpg.does_item_exist(y_axis):
            dpg.fit_axis_data(y_axis)


def setup_keyboard_shortcuts():
    """Register global keyboard shortcuts."""
    with dpg.handler_registry(tag="global_handlers"):
        dpg.add_key_press_handler(dpg.mvKey_F5, callback=update_current_tab)
        dpg.add_key_press_handler(dpg.mvKey_Home, callback=lambda: fit_current_plot())


def update_current_tab():
    """Trigger update on the currently active tab."""
    current_tab = dpg.get_value("main_tabs")
    callback_map = {
        "tab_21": "btn_update_tab21",
        "tab_22": "btn_update_tab22",
        "tab_flexspline": "btn_update_fs",
        "tab_circular": "btn_update_cs",
        "tab_overlay": "btn_update_ov",
    }
    if current_tab in callback_map:
        btn_tag = callback_map[current_tab]
        if dpg.does_item_exist(btn_tag):
            # Simulate button click
            callback = dpg.get_item_callback(btn_tag)
            if callback:
                callback(btn_tag, None, None)


def update_status(text: str):
    """Update the status bar text."""
    if dpg.does_item_exist("status_text"):
        dpg.set_value("status_text", text)


def update_point_count(count: int):
    """Update the point count display."""
    if dpg.does_item_exist("point_count"):
        dpg.set_value("point_count", f"Points: {count:,}")


def update_mode(mode: str):
    """Update the mode display."""
    if dpg.does_item_exist("mode_text"):
        dpg.set_value("mode_text", f"Mode: {mode}")


def create_main_window():
    """Create the main application window with all tabs."""
    with dpg.window(tag="primary_window"):
        create_menu_bar()

        # Main tab bar
        with dpg.tab_bar(tag="main_tabs"):
            with dpg.tab(label="2.1 Flexspline Tooth", tag="tab_21"):
                create_tab_flexspline_tooth()

            with dpg.tab(label="2.2 Conjugate Profile", tag="tab_22"):
                create_tab_conjugate_tooth()

            with dpg.tab(label="2.3 Flexspline", tag="tab_flexspline"):
                create_tab_flexspline_full()

            with dpg.tab(label="2.4 Circular Spline", tag="tab_circular"):
                create_tab_circular_spline()

            with dpg.tab(label="2.5 Radial Modification", tag="tab_overlay"):
                create_tab_radial_modification()

            with dpg.tab(label="2.6 Longitudinal Mod", tag="tab_longmod"):
                create_tab_longitudinal_modification()

        dpg.add_separator()
        create_status_bar()


def main():
    """Main entry point for the application."""
    dpg.create_context()

    # Set DPI scale for other modules to use
    set_dpi_scale(_dpi_scale)

    # Setup fonts first (must be before viewport creation)
    setup_fonts(_dpi_scale)

    # Create themes
    create_themes()

    # Initialize application state
    AppState.initialize()

    # Create viewport (scale dimensions for high-DPI displays)
    dpg.create_viewport(
        title="Harmonic Drive - DCT Tooth Calculator",
        width=int(1400 * _dpi_scale),
        height=int(900 * _dpi_scale),
        min_width=int(1000 * _dpi_scale),
        min_height=int(700 * _dpi_scale)
    )

    # Set application icon
    icon_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Ref Images")
    icon_path = os.path.join(icon_dir, "app_icon.ico")
    if os.path.exists(icon_path):
        dpg.set_viewport_small_icon(icon_path)
        dpg.set_viewport_large_icon(icon_path)

    # Create main window
    create_main_window()

    # Apply main theme
    dpg.bind_theme("theme_main")

    # Set primary window
    dpg.set_primary_window("primary_window", True)

    # Setup keyboard shortcuts
    setup_keyboard_shortcuts()

    # Setup and show viewport
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Enable dark title bar on Windows
    set_windows_dark_titlebar()

    # Update status
    update_status("Ready")

    # Main loop
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
