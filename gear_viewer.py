"""
Harmonic Drive DCT-Tooth Calculator — GUI
Tabbed interface:
  Tab 2.1 — Flexspline tooth profile (Eqs 1-6)
  Tab 2.2 — Conjugate circular spline tooth profile (placeholder)
"""
import math
import tkinter as tk
from tkinter import ttk

from equations import (DEFAULTS, PARAM_LABELS, PARAM_ORDER, PITCH_RADIUS,
                       compute_profile)

# ── Viewport settings ──────────────────────────────────────────────
HALF_VIEW_MM = 1.5
PLOT_PX = 500
PX_PER_MM = PLOT_PX / (2 * HALF_VIEW_MM)
MARGIN = 50
CANVAS_W = PLOT_PX + MARGIN + 30
CANVAS_H = PLOT_PX + MARGIN + 30
TICK_INTERVAL = 0.2


def mm_to_canvas(x_mm: float, y_mm: float) -> tuple[float, float]:
    """Cartesian mm (origin center, y-up) -> canvas pixels."""
    ox = MARGIN + PLOT_PX / 2
    oy = PLOT_PX / 2
    return ox + x_mm * PX_PER_MM, oy - y_mm * PX_PER_MM


def draw_axes(c: tk.Canvas):
    ox, oy = mm_to_canvas(0, 0)
    c.create_line(MARGIN, oy, MARGIN + PLOT_PX, oy, fill="grey70")
    c.create_line(ox, 0, ox, PLOT_PX, fill="grey70")
    v = -HALF_VIEW_MM
    while v <= HALF_VIEW_MM + 1e-9:
        px, _ = mm_to_canvas(v, 0)
        c.create_line(px, oy - 3, px, oy + 3, fill="grey70")
        c.create_text(px, oy + 14, text=f"{v:.1f}", font=("Consolas", 7))
        _, py = mm_to_canvas(0, v)
        c.create_line(ox - 3, py, ox + 3, py, fill="grey70")
        c.create_text(ox - 22, py, text=f"{v:.1f}", font=("Consolas", 7),
                      anchor="e")
        v += TICK_INTERVAL
    c.create_text(MARGIN + PLOT_PX / 2, PLOT_PX + 30,
                  text="X_R (mm)", font=("Consolas", 9, "bold"))
    c.create_text(14, PLOT_PX / 2, text="Y_R\n(mm)",
                  font=("Consolas", 9, "bold"))


# ── Drawing helpers ────────────────────────────────────────────────

def draw_segment(c: tk.Canvas, pts: list, color: str, label: str):
    if len(pts) < 2:
        return
    coords = []
    for x, y in pts:
        px, py = mm_to_canvas(x, y)
        coords.extend([px, py])
    c.create_line(*coords, fill=color, width=2, smooth=False)
    mx, my = pts[len(pts) // 2]
    px, py = mm_to_canvas(mx, my)
    c.create_text(px + 12, py, text=label, font=("Consolas", 7),
                  fill=color, anchor="w")


def draw_polyline(c: tk.Canvas, pts: list, color: str):
    if len(pts) < 2:
        return
    coords = []
    for x, y in pts:
        px, py = mm_to_canvas(x, y)
        coords.extend([px, py])
    c.create_line(*coords, fill=color, width=1, dash=(3, 3))


# ── Tab 2.1: Flexspline Tooth Profile ─────────────────────────────

class Tab21:
    """Section 2.1 — Double-circular-arc common-tangent flexspline profile."""

    def __init__(self, parent: ttk.Frame):
        self.frame = parent

        # Left: parameters
        left = ttk.Frame(parent, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Flexspline Parameters (mm)",
                  font=("Consolas", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 8))

        self.entries: dict[str, tk.StringVar] = {}
        row = 1
        for key in PARAM_ORDER:
            ttk.Label(left, text=PARAM_LABELS[key],
                      font=("Consolas", 9)).grid(row=row, column=0,
                                                  sticky="w", padx=(0, 6))
            var = tk.StringVar(value=str(DEFAULTS[key]))
            ent = ttk.Entry(left, textvariable=var, width=10,
                            font=("Consolas", 9))
            ent.grid(row=row, column=1, pady=2)
            self.entries[key] = var
            row += 1

        ttk.Button(left, text="Update", command=self.redraw).grid(
            row=row, column=0, columnspan=2, pady=10)
        row += 1

        self.info_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.info_var, font=("Consolas", 8),
                  foreground="grey30", wraplength=200, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w")

        # Right: canvas
        self.canvas = tk.Canvas(parent, width=CANVAS_W, height=CANVAS_H,
                                bg="white")
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)

        self.redraw()

    def _read_params(self) -> dict | None:
        params = {}
        for key, var in self.entries.items():
            try:
                params[key] = float(var.get())
            except ValueError:
                self.info_var.set(f"Invalid number for {key}")
                return None
        return params

    def redraw(self):
        c = self.canvas
        c.delete("all")
        draw_axes(c)

        params = self._read_params()
        if params is None:
            return

        result = compute_profile(params)
        if "error" in result:
            self.info_var.set(f"Error: {result['error']}")
            return

        ds = result["ds"]
        hf = params["hf"]
        ha = params["ha"]

        for y_val, color, label in [
            (ds + hf + ha, "#ccccff", "Addendum"),
            (ds + hf,      "#ccffcc", "Pitch"),
            (ds,           "#ffcccc", "Dedendum"),
        ]:
            px1, py = mm_to_canvas(-HALF_VIEW_MM, y_val)
            px2, _  = mm_to_canvas(HALF_VIEW_MM, y_val)
            c.create_line(px1, py, px2, py, fill=color, dash=(4, 3))
            c.create_text(px2 - 4, py - 8, text=label,
                          font=("Consolas", 7), fill=color, anchor="e")

        draw_segment(c, result["pts_AB"], "red",   "AB (convex)")
        draw_segment(c, result["pts_BC"], "blue",  "BC (tangent)")
        draw_segment(c, result["pts_CD"], "green", "CD (concave)")

        for lbl, x, y, col in [
            ("O\u2081", result["x1_R"], result["y1_R"], "red"),
            ("O\u2082", result["x2_R"], result["y2_R"], "green"),
        ]:
            px, py = mm_to_canvas(x, y)
            r = 3
            c.create_oval(px - r, py - r, px + r, py + r, fill=col)
            c.create_text(px + 8, py - 8, text=lbl,
                          font=("Consolas", 8, "bold"), fill=col)

        all_pts = result["pts_AB"] + result["pts_BC"] + result["pts_CD"]
        mirrored = [(-x, y) for x, y in all_pts]
        draw_polyline(c, mirrored, "grey50")

        deg = math.degrees
        self.info_var.set(
            f"rp = {result['rp']:.2f} mm (pitch radius)\n"
            f"ds = {result['ds']:.4f} mm\n"
            f"\u03b1  = {deg(result['alpha']):.2f}\u00b0\n"
            f"\u03b4  = {deg(result['delta']):.2f}\u00b0\n"
            f"h1 = {result['h1']:.4f} mm\n"
            f"l1 = {result['l1']:.4f}  l2 = {result['l2']:.4f}\n"
            f"l3 = {result['l3']:.4f}"
        )


# ── Tab 2.2: Conjugate Circular Spline ────────────────────────────

class Tab22:
    """Section 2.2 — Conjugate circular spline tooth profile (placeholder)."""

    def __init__(self, parent: ttk.Frame):
        self.frame = parent

        ttk.Label(parent, text="Section 2.2 — Conjugate Circular Spline\n"
                  "(Not yet implemented)",
                  font=("Consolas", 11), foreground="grey50",
                  justify="center").pack(expand=True)


# ── Main App ───────────────────────────────────────────────────────

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Harmonic Drive — DCT Tooth Calculator")
        root.resizable(False, False)

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        tab1_frame = ttk.Frame(notebook)
        tab2_frame = ttk.Frame(notebook)

        notebook.add(tab1_frame, text=" 2.1 Flexspline Profile ")
        notebook.add(tab2_frame, text=" 2.2 Conjugate Circular Spline ")

        self.tab21 = Tab21(tab1_frame)
        self.tab22 = Tab22(tab2_frame)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
