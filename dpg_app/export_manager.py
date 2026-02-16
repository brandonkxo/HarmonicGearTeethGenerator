"""
Export manager for curve and data export.

Handles export to .sldcrv (SolidWorks), .dxf (AutoCAD), and .txt formats.
"""

import dearpygui.dearpygui as dpg
import os
from typing import List, Tuple, Optional
import math
import tkinter as tk
from tkinter import filedialog

from dpg_app.app_state import scaled


def filter_duplicate_points(points: List[Tuple[float, float]], tol: float = 1e-9) -> List[Tuple[float, float]]:
    """
    Remove consecutive duplicate points within tolerance.

    Returns filtered list and count of removed duplicates.
    """
    if not points:
        return [], 0

    filtered = [points[0]]
    removed = 0

    for p in points[1:]:
        last = filtered[-1]
        dist = math.sqrt((p[0] - last[0])**2 + (p[1] - last[1])**2)
        if dist > tol:
            filtered.append(p)
        else:
            removed += 1

    return filtered, removed


def write_sldcrv(filepath: str, points: List[Tuple[float, float]]) -> bool:
    """
    Write points to SolidWorks curve format (.sldcrv).

    Format: x,y,0 per line (CSV with z=0)
    """
    try:
        with open(filepath, "w") as f:
            for x, y in points:
                f.write(f"{x},{y},0\n")
        return True
    except Exception as e:
        print(f"Error writing SLDCRV: {e}")
        return False


def write_dxf_polyline(filepath: str, points: List[Tuple[float, float]], closed: bool = True) -> bool:
    """
    Write points as DXF polyline.

    Creates ASCII DXF with a single 2D polyline entity.
    """
    try:
        with open(filepath, "w") as f:
            # Header section
            f.write("0\nSECTION\n2\nHEADER\n")
            f.write("9\n$ACADVER\n1\nAC1009\n")  # AutoCAD R12 format
            f.write("0\nENDSEC\n")

            # Entities section
            f.write("0\nSECTION\n2\nENTITIES\n")

            # Polyline header
            f.write("0\nPOLYLINE\n")
            f.write("8\n0\n")  # Layer 0
            f.write("66\n1\n")  # Vertices follow
            f.write("70\n" + ("1" if closed else "0") + "\n")  # Closed flag

            # Vertices
            for x, y in points:
                f.write("0\nVERTEX\n")
                f.write("8\n0\n")  # Layer 0
                f.write(f"10\n{x}\n")  # X
                f.write(f"20\n{y}\n")  # Y
                f.write("30\n0.0\n")  # Z

            # End polyline
            f.write("0\nSEQEND\n")

            # End entities and file
            f.write("0\nENDSEC\n")
            f.write("0\nEOF\n")

        return True
    except Exception as e:
        print(f"Error writing DXF: {e}")
        return False


def write_txt(filepath: str, data: str) -> bool:
    """Write text data to file."""
    try:
        with open(filepath, "w") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"Error writing TXT: {e}")
        return False


def export_points(filepath: str, points: List[Tuple[float, float]], closed: bool = True) -> Tuple[bool, int, int]:
    """
    Export points to file based on extension.

    Returns (success, point_count, duplicates_removed).
    """
    # Filter duplicates
    filtered, removed = filter_duplicate_points(points)

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".sldcrv":
        success = write_sldcrv(filepath, filtered)
    elif ext == ".dxf":
        success = write_dxf_polyline(filepath, filtered, closed)
    else:
        success = False

    return success, len(filtered), removed


def show_export_dialog(
    default_name: str,
    points: List[Tuple[float, float]],
    formats: List[str] = None,
    closed: bool = True,
    callback = None
):
    """
    Show native file export dialog.

    Args:
        default_name: Default filename without extension
        points: List of (x, y) points to export
        formats: List of format extensions (default: [".sldcrv", ".dxf"])
        closed: Whether the polyline should be closed (for DXF)
        callback: Optional callback(success, filepath, point_count, removed) after export
    """
    if formats is None:
        formats = [".sldcrv", ".dxf"]

    # Default export directory
    default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
    os.makedirs(default_dir, exist_ok=True)

    # Build file type tuples for tkinter
    filetypes = []
    for fmt in formats:
        label = fmt.upper().replace(".", "") + " files"
        filetypes.append((label, f"*{fmt}"))
    filetypes.append(("All files", "*.*"))

    # Create hidden tkinter root window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Bring dialog to front

    # Show native save dialog
    filepath = filedialog.asksaveasfilename(
        initialdir=default_dir,
        initialfile=default_name + formats[0],
        defaultextension=formats[0],
        filetypes=filetypes,
        title="Export Curve"
    )

    root.destroy()

    if not filepath:
        # User cancelled
        if callback:
            callback(False, "", 0, 0)
        return

    # Ensure correct extension
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in formats:
        filepath = filepath + formats[0]

    success, count, removed = export_points(filepath, points, closed)

    if success:
        msg = f"Exported {count} points to {os.path.basename(filepath)}"
        if removed > 0:
            msg += f" ({removed} duplicates removed)"
        print(msg)

        if callback:
            callback(True, filepath, count, removed)
    else:
        print(f"Export failed: {filepath}")
        if callback:
            callback(False, filepath, 0, 0)


def quick_export(filepath: str, points: List[Tuple[float, float]], closed: bool = True) -> Tuple[bool, str]:
    """
    Quick export without dialog.

    Returns (success, message).
    """
    success, count, removed = export_points(filepath, points, closed)

    if success:
        msg = f"Exported {count} points"
        if removed > 0:
            msg += f" ({removed} duplicates removed)"
        return True, msg
    else:
        return False, "Export failed"
