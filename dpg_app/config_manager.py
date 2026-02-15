"""
Configuration save/load manager.

Handles JSON configuration file management with file explorer-style dialogs.
"""

import dearpygui.dearpygui as dpg
import json
import os
from typing import List, Optional

from dpg_app.app_state import AppState, PARAM_TOOLTIPS
from equations import PARAM_ORDER

# Configuration directory (relative to main script)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def ensure_config_dir():
    """Ensure the config directory exists."""
    os.makedirs(CONFIG_DIR, exist_ok=True)


def list_configs() -> List[str]:
    """List all available configuration files."""
    ensure_config_dir()
    try:
        files = [f[:-5] for f in os.listdir(CONFIG_DIR) if f.endswith(".json")]
        return sorted(files)
    except Exception:
        return []


def save_config(name: str) -> bool:
    """
    Save current parameters to a named JSON config file.

    Returns True on success, False on failure.
    """
    ensure_config_dir()

    # Sanitize filename
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
    if not safe_name:
        return False

    # Collect current parameter values
    params = AppState.read_from_widgets()

    config_data = {
        "name": name,
        "params": params,
        "smooth": AppState.get_smooth(),
        "fillet_add": AppState.get_fillet_add(),
        "fillet_ded": AppState.get_fillet_ded(),
    }

    filepath = os.path.join(CONFIG_DIR, f"{safe_name}.json")
    try:
        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def load_config(name: str) -> bool:
    """
    Load parameters from a saved JSON config file.

    Returns True on success, False on failure.
    """
    filepath = os.path.join(CONFIG_DIR, f"{name}.json")
    try:
        with open(filepath, "r") as f:
            config_data = json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return False

    # Apply loaded values
    params = config_data.get("params", {})
    AppState.update_params(params)

    smooth = config_data.get("smooth", 0.001)
    AppState.set_smooth(smooth)
    AppState.set_fillet_add(config_data.get("fillet_add", 0.15))
    AppState.set_fillet_ded(config_data.get("fillet_ded", 0.1))

    return True


def delete_config(name: str) -> bool:
    """
    Delete a configuration file.

    Returns True on success, False on failure.
    """
    filepath = os.path.join(CONFIG_DIR, f"{name}.json")
    try:
        os.remove(filepath)
        return True
    except Exception as e:
        print(f"Error deleting config: {e}")
        return False


def _refresh_config_list(listbox_tag: str):
    """Refresh the config listbox with current files."""
    configs = list_configs()
    dpg.configure_item(listbox_tag, items=configs)


def show_save_dialog():
    """Show the save configuration dialog."""
    if dpg.does_item_exist("save_dialog"):
        dpg.delete_item("save_dialog")

    configs = list_configs()

    def on_save():
        name = dpg.get_value("save_name_input")
        if not name or not name.strip():
            dpg.configure_item("save_status", default_value="Please enter a name", color=(255, 100, 100))
            return

        # Check for overwrite
        safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
        if safe_name in list_configs():
            # Show confirmation
            dpg.configure_item("save_status", default_value=f"Overwriting '{safe_name}'...", color=(255, 200, 100))

        if save_config(name):
            dpg.configure_item("save_status", default_value=f"Saved '{safe_name}'", color=(100, 255, 100))
            _refresh_config_list("save_config_list")
            # Close dialog after brief delay
            dpg.split_frame()
            dpg.delete_item("save_dialog")
        else:
            dpg.configure_item("save_status", default_value="Save failed!", color=(255, 100, 100))

    def on_delete():
        selected = dpg.get_value("save_config_list")
        if not selected:
            dpg.configure_item("save_status", default_value="Select a config to delete", color=(255, 200, 100))
            return

        if delete_config(selected):
            dpg.configure_item("save_status", default_value=f"Deleted '{selected}'", color=(100, 255, 100))
            _refresh_config_list("save_config_list")
            dpg.set_value("save_name_input", "")
        else:
            dpg.configure_item("save_status", default_value="Delete failed!", color=(255, 100, 100))

    def on_select():
        selected = dpg.get_value("save_config_list")
        if selected:
            dpg.set_value("save_name_input", selected)

    with dpg.window(
        label="Save Configuration",
        tag="save_dialog",
        modal=True,
        width=450,
        height=500,
        pos=(dpg.get_viewport_width() // 2 - 225, dpg.get_viewport_height() // 2 - 250),
        no_resize=False,
        no_collapse=True,
    ):
        dpg.add_text("Existing Configurations:", color=(180, 180, 255))
        dpg.add_spacer(height=5)

        dpg.add_listbox(
            items=configs,
            tag="save_config_list",
            width=-1,
            num_items=12,
            callback=lambda: on_select()
        )

        dpg.add_spacer(height=10)
        dpg.add_text("File name:")
        dpg.add_input_text(
            tag="save_name_input",
            width=-1,
            hint="Enter configuration name...",
            on_enter=True,
            callback=lambda: on_save()
        )

        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Save", callback=on_save, width=100)
            dpg.add_button(label="Delete", callback=on_delete, width=100)
            dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("save_dialog"), width=100)

        dpg.add_spacer(height=5)
        dpg.add_text("", tag="save_status", color=(150, 150, 150))


def show_load_dialog():
    """Show the load configuration dialog."""
    if dpg.does_item_exist("load_dialog"):
        dpg.delete_item("load_dialog")

    configs = list_configs()

    if not configs:
        # Show simple message
        with dpg.window(
            label="Load Configuration",
            tag="load_dialog",
            modal=True,
            width=300,
            height=120,
            pos=(dpg.get_viewport_width() // 2 - 150, dpg.get_viewport_height() // 2 - 60),
            no_resize=True,
            no_collapse=True,
        ):
            dpg.add_text("No saved configurations found.")
            dpg.add_spacer(height=20)
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item("load_dialog"), width=-1)
        return

    def on_load():
        selected = dpg.get_value("load_config_list")
        if not selected:
            dpg.configure_item("load_status", default_value="Select a configuration", color=(255, 200, 100))
            return

        if load_config(selected):
            dpg.configure_item("load_status", default_value=f"Loaded '{selected}'", color=(100, 255, 100))
            dpg.split_frame()
            dpg.delete_item("load_dialog")
        else:
            dpg.configure_item("load_status", default_value="Load failed!", color=(255, 100, 100))

    def on_delete():
        selected = dpg.get_value("load_config_list")
        if not selected:
            dpg.configure_item("load_status", default_value="Select a config to delete", color=(255, 200, 100))
            return

        if delete_config(selected):
            dpg.configure_item("load_status", default_value=f"Deleted '{selected}'", color=(100, 255, 100))
            _refresh_config_list("load_config_list")
        else:
            dpg.configure_item("load_status", default_value="Delete failed!", color=(255, 100, 100))

    with dpg.window(
        label="Load Configuration",
        tag="load_dialog",
        modal=True,
        width=450,
        height=450,
        pos=(dpg.get_viewport_width() // 2 - 225, dpg.get_viewport_height() // 2 - 225),
        no_resize=False,
        no_collapse=True,
    ):
        dpg.add_text("Select a Configuration:", color=(180, 180, 255))
        dpg.add_spacer(height=5)

        dpg.add_listbox(
            items=configs,
            tag="load_config_list",
            width=-1,
            num_items=14,
        )

        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Load", callback=on_load, width=100)
            dpg.add_button(label="Delete", callback=on_delete, width=100)
            dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("load_dialog"), width=100)

        dpg.add_spacer(height=5)
        dpg.add_text("", tag="load_status", color=(150, 150, 150))
