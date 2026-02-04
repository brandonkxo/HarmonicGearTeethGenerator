"""
Harmonic Drive DCT-Tooth Calculator — GUI
Tabbed interface:
  Tab 2.1 — Flexspline tooth profile (Eqs 1-6)
  Tab 2.2 — Conjugate circular spline tooth profile (placeholder)
"""
import math
import os
import tkinter as tk
from tkinter import ttk, filedialog

from equations import (DEFAULTS, PARAM_LABELS, PARAM_ORDER, PITCH_RADIUS,
                       compute_profile, compute_conjugate_profile,
                       smooth_conjugate_profile, build_single_tooth_outline)

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
    """Section 2.2 — Conjugate circular spline tooth profile."""

    def __init__(self, parent: ttk.Frame):
        self.frame = parent

        # Left: parameters
        left = ttk.Frame(parent, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Conjugate Profile Parameters (mm)",
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

        # Smoothing factor control
        ttk.Label(left, text="Smoothing (s)",
                  font=("Consolas", 9)).grid(row=row, column=0,
                                              sticky="w", padx=(0, 6))
        self.smooth_var = tk.StringVar(value="0.001")
        ttk.Entry(left, textvariable=self.smooth_var, width=10,
                  font=("Consolas", 9)).grid(row=row, column=1, pady=2)
        row += 1

        # Segment visibility toggles
        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1
        ttk.Label(left, text="Show segments",
                  font=("Consolas", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.seg_vars: dict[str, tk.BooleanVar] = {}
        self.seg_colors = {"AB": "red", "BC": "blue", "CD": "green"}
        self.seg_labels = {
            "AB": "AB (convex)",
            "BC": "BC (tangent)",
            "CD": "CD (concave)",
        }
        for seg_key in ("AB", "BC", "CD"):
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(left, text=self.seg_labels[seg_key],
                                variable=var, font=("Consolas", 9),
                                fg=self.seg_colors[seg_key],
                                command=self.redraw)
            cb.grid(row=row, column=0, columnspan=2, sticky="w")
            self.seg_vars[seg_key] = var
            row += 1

        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Update", command=self.redraw).pack(
            side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text="Reset", command=self.reset_params).pack(
            side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text="Export TXT", command=self.export_txt).pack(
            side=tk.LEFT)
        row += 1

        self.info_var = tk.StringVar(value="Click Update to compute.")
        ttk.Label(left, textvariable=self.info_var, font=("Consolas", 8),
                  foreground="grey30", wraplength=200, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w")

        self._last_result = None
        self._last_params = None

        # Right: canvas
        self.canvas = tk.Canvas(parent, width=CANVAS_W, height=CANVAS_H,
                                bg="white")
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)

    def reset_params(self):
        for key, var in self.entries.items():
            var.set(str(DEFAULTS[key]))
        self.smooth_var.set("0.001")
        self._last_result = None
        self._last_params = None
        self.canvas.delete("all")
        draw_axes(self.canvas)
        self.info_var.set("Reset to defaults.")

    def sync_from(self, tab21: Tab21):
        for key in PARAM_ORDER:
            if key in tab21.entries and key in self.entries:
                self.entries[key].set(tab21.entries[key].get())
        self._last_result = None
        self._last_params = None

    def _read_params(self) -> dict | None:
        params = {}
        for key, var in self.entries.items():
            try:
                params[key] = float(var.get())
            except ValueError:
                self.info_var.set(f"Invalid number for {key}")
                return None
        return params

    def export_txt(self):
        if self._last_result is None:
            self.info_var.set("No data — click Update first.")
            return
        raw_roots = self._last_result.get("raw_roots", [])
        if not raw_roots:
            self.info_var.set("No root data to export.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="conjugate_roots.txt",
        )
        if not path:
            return

        header = f"{'phi_rad':>14s}  {'phi_deg':>10s}  {'l':>14s}  {'x_g':>14s}  {'y_local':>14s}"
        sep = "-" * len(header)

        with open(path, "w") as f:
            for seg_key in ("AB", "BC", "CD"):
                seg_pts = [(phi, l_zero, xg, y_local)
                           for phi, l_zero, xg, y_local, s in raw_roots
                           if s == seg_key]
                seg_pts.sort(key=lambda r: r[0])
                f.write(f"=== Segment {seg_key}  ({len(seg_pts)} points) ===\n")
                f.write(header + "\n")
                f.write(sep + "\n")
                for phi, l_zero, xg, y_local in seg_pts:
                    f.write(f"{phi:14.8f}  {math.degrees(phi):10.4f}  "
                            f"{l_zero:14.8f}  {xg:14.8f}  {y_local:14.8f}\n")
                f.write("\n")

        total = len(raw_roots)
        self.info_var.set(f"Exported {total} pts\n→ {os.path.basename(path)}")

    def redraw(self):
        c = self.canvas
        c.delete("all")
        draw_axes(c)

        params = self._read_params()
        if params is None:
            return

        try:
            s_val = float(self.smooth_var.get())
        except ValueError:
            self.info_var.set("Invalid smoothing value")
            return

        # Recompute only if params changed; toggle-only redraws reuse cache
        if self._last_result is None or self._last_params != (params, s_val):
            self.info_var.set("Computing...")
            self.frame.update_idletasks()

            result = compute_conjugate_profile(params)
            if "error" in result:
                self.info_var.set(f"Error: {result['error']}")
                return

            smooth_conjugate_profile(result, s=s_val)
            self._last_result = result
            self._last_params = (params, s_val)
        else:
            result = self._last_result

        seg_branches = result.get("seg_branches", {})

        seg_counts = {}
        for seg_key in ("AB", "BC", "CD"):
            raw = seg_branches.get(seg_key, [])
            seg_counts[seg_key] = len(raw)

            if not self.seg_vars[seg_key].get():
                continue

            # Raw envelope points as colored dots
            color = self.seg_colors[seg_key]
            for x, y in raw:
                px, py = mm_to_canvas(x, y)
                c.create_oval(px - 2, py - 2, px + 2, py + 2,
                              fill=color, outline="")

        # Unified B-spline flank
        flank = result.get("smoothed_flank", [])
        if len(flank) >= 2:
            draw_segment(c, flank, "#222222", "Flank")
            mirrored = [(-x, y) for x, y in flank]
            draw_polyline(c, mirrored, "grey50")

        total_raw = sum(seg_counts.values())
        self.info_var.set(
            f"rp_c = {result['rp_c']:.2f} mm\n"
            f"AB: {seg_counts['AB']}  BC: {seg_counts['BC']}  "
            f"CD: {seg_counts['CD']}  ({total_raw} total)\n"
            f"Smoothing s = {s_val}"
        )


# ── Tab 3: Flexspline Full Geometry ────────────────────────────────

class TabFlexspline:
    """Flexspline — single tooth on pitch circle with G2 root blends."""

    FS_CANVAS = 600   # canvas pixel size
    FS_MARGIN = 50

    def __init__(self, parent: ttk.Frame):
        self.frame = parent

        # Left: parameters (synced from Tab 2.1)
        left = ttk.Frame(parent, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Flexspline Tooth on Circle",
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

        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Update", command=self.redraw).pack(
            side=tk.LEFT, padx=(0, 6))
        self._zoomed = False
        self._zoom_btn = ttk.Button(btn_frame, text="Zoom In",
                                    command=self._toggle_zoom)
        self._zoom_btn.pack(side=tk.LEFT)
        row += 1

        self.info_var = tk.StringVar(value="Click Update to draw tooth.")
        ttk.Label(left, textvariable=self.info_var, font=("Consolas", 8),
                  foreground="grey30", wraplength=200, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w")

        # Right: canvas
        cw = self.FS_CANVAS + self.FS_MARGIN + 30
        ch = self.FS_CANVAS + self.FS_MARGIN + 30
        self.canvas = tk.Canvas(parent, width=cw, height=ch, bg="white")
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)

    def sync_from(self, tab21: Tab21):
        for key in PARAM_ORDER:
            if key in tab21.entries and key in self.entries:
                self.entries[key].set(tab21.entries[key].get())

    def _toggle_zoom(self):
        self._zoomed = not self._zoomed
        self._zoom_btn.config(text="Zoom Out" if self._zoomed else "Zoom In")
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

        params = self._read_params()
        if params is None:
            return

        result = build_single_tooth_outline(params)
        if "error" in result:
            self.info_var.set(f"Error: {result['error']}")
            return

        rp = result["rp"]
        rm = result["rm"]
        ds = result["ds"]
        ha = result["ha"]
        hf = result["hf"]
        tooth_xy = result["tooth_xy"]

        # Reference radii
        r_ded = rm + ds             # dedendum circle radius
        r_pit = rm + ds + hf        # pitch circle radius (== rp)
        r_add = rm + ds + hf + ha   # addendum circle radius

        if not tooth_xy:
            self.info_var.set("No tooth points generated.")
            return

        plot_px = self.FS_CANVAS
        margin = self.FS_MARGIN

        if self._zoomed:
            # ── Zoomed viewport: tight bbox around tooth + padding ──
            xs = [p[0] for p in tooth_xy]
            ys = [p[1] for p in tooth_xy]
            pad = (ha + hf) * 0.8
            x_min, x_max = min(xs) - pad, max(xs) + pad
            y_min, y_max = min(ys) - pad, max(ys) + pad
            view_w = x_max - x_min
            view_h = y_max - y_min
            half_view = max(view_w, view_h) / 2.0
            cx_view = (x_min + x_max) / 2.0
            cy_view = (y_min + y_max) / 2.0
        else:
            # ── Full viewport: centered at origin, 120% of pitch diameter ──
            half_view = rp * 1.2
            cx_view = 0.0
            cy_view = 0.0

        px_per_mm = plot_px / (2.0 * half_view)

        # Canvas origin (where cx_view, cy_view maps to canvas center)
        ox = margin + plot_px / 2.0
        oy = plot_px / 2.0

        def to_px(x_mm, y_mm):
            return (ox + (x_mm - cx_view) * px_per_mm,
                    oy - (y_mm - cy_view) * px_per_mm)

        # ── Draw axes ──
        # Horizontal axis (y == cy_view line)
        ax_y_px = oy - (0 - cy_view) * px_per_mm
        if 0 <= ax_y_px <= plot_px:
            c.create_line(margin, ax_y_px, margin + plot_px, ax_y_px,
                          fill="grey70")
        # Vertical axis (x == cx_view line)
        ax_x_px = ox + (0 - cx_view) * px_per_mm
        if margin <= ax_x_px <= margin + plot_px:
            c.create_line(ax_x_px, 0, ax_x_px, plot_px, fill="grey70")

        # Choose a nice tick interval based on half_view
        # Aim for roughly 8-12 ticks across the full range
        raw_step = (2 * half_view) / 10.0
        # Snap to a "nice" number: 1, 2, 5, 10, 20, 50 ...
        mag = 10 ** math.floor(math.log10(max(raw_step, 1e-12)))
        candidates = [mag, 2 * mag, 5 * mag, 10 * mag]
        tick_step = min(candidates, key=lambda s: abs(s - raw_step))
        tick_fmt = f"{{:.{max(0, -math.floor(math.log10(max(tick_step, 1e-12))))}f}}"

        # Visible mm range
        v_lo = cx_view - half_view
        v_hi = cx_view + half_view

        # Snap starting tick to a multiple of tick_step
        v_start = math.floor(v_lo / tick_step) * tick_step

        # Draw tick marks and labels
        v = v_start
        while v <= v_hi + 1e-9:
            # X-axis ticks (drawn on the horizontal axis line)
            px_t, _ = to_px(v, 0)
            if margin <= px_t <= margin + plot_px and 0 <= ax_y_px <= plot_px:
                c.create_line(px_t, ax_y_px - 3, px_t, ax_y_px + 3,
                              fill="grey70")
                c.create_text(px_t, ax_y_px + 14,
                              text=tick_fmt.format(v), font=("Consolas", 7))
            v += tick_step

        v_lo_y = cy_view - half_view
        v_hi_y = cy_view + half_view
        v = math.floor(v_lo_y / tick_step) * tick_step
        while v <= v_hi_y + 1e-9:
            # Y-axis ticks (drawn on the vertical axis line)
            _, py_t = to_px(0, v)
            if 0 <= py_t <= plot_px and margin <= ax_x_px <= margin + plot_px:
                c.create_line(ax_x_px - 3, py_t, ax_x_px + 3, py_t,
                              fill="grey70")
                c.create_text(ax_x_px - 22, py_t,
                              text=tick_fmt.format(v), font=("Consolas", 7),
                              anchor="e")
            v += tick_step

        # Axis labels
        c.create_text(margin + plot_px / 2, plot_px + 30,
                      text="X (mm)", font=("Consolas", 9, "bold"))
        c.create_text(14, plot_px / 2, text="Y\n(mm)",
                      font=("Consolas", 9, "bold"))

        # ── Draw reference circles ──
        # When zoomed, only draw arcs spanning the visible angular range
        if self._zoomed:
            theta_span = 2.0 * half_view / rp * 1.5
            theta_center = math.atan2(cx_view, cy_view)
            n_arc = 200
            for radius, color, label in [
                (r_add, "#9999ff", "Addendum"),
                (r_pit, "#66cc66", "Pitch"),
                (r_ded, "#ff9999", "Dedendum"),
            ]:
                arc_coords = []
                for i in range(n_arc + 1):
                    th = theta_center - theta_span/2 + theta_span * i / n_arc
                    ax = radius * math.sin(th)
                    ay = radius * math.cos(th)
                    px, py = to_px(ax, ay)
                    arc_coords.extend([px, py])
                if len(arc_coords) >= 4:
                    c.create_line(*arc_coords, fill=color, dash=(4, 3), width=1)
                lx, ly = arc_coords[-2], arc_coords[-1]
                c.create_text(lx + 4, ly, text=label,
                              font=("Consolas", 7), fill=color, anchor="w")
        else:
            n_circ = 360
            for radius, color, label in [
                (r_add, "#9999ff", "Addendum"),
                (r_pit, "#66cc66", "Pitch"),
                (r_ded, "#ff9999", "Dedendum"),
            ]:
                circ_coords = []
                for i in range(n_circ + 1):
                    th = 2 * math.pi * i / n_circ
                    cx_mm = radius * math.sin(th)
                    cy_mm = radius * math.cos(th)
                    px, py = to_px(cx_mm, cy_mm)
                    circ_coords.extend([px, py])
                if len(circ_coords) >= 4:
                    c.create_line(*circ_coords, fill=color, dash=(4, 3), width=1)
                lx, ly = to_px(0, radius)
                c.create_text(lx + 4, ly - 8, text=label,
                              font=("Consolas", 7), fill=color, anchor="w")

        # ── Draw tooth outline ──
        if len(tooth_xy) >= 2:
            coords = []
            for X, Y in tooth_xy:
                px, py = to_px(X, Y)
                coords.extend([px, py])
            c.create_line(*coords, fill="#222222", width=2, smooth=False)

        # ── Close the tip: straight line from last point to first ──
        if len(tooth_xy) >= 2:
            px1, py1 = to_px(tooth_xy[-1][0], tooth_xy[-1][1])
            px2, py2 = to_px(tooth_xy[0][0], tooth_xy[0][1])
            c.create_line(px1, py1, px2, py2, fill="#222222", width=2)

        # ── Info ──
        self.info_var.set(
            f"rp = {rp:.2f} mm   rm = {rm:.4f} mm\n"
            f"r_add = {r_add:.4f}  r_ded = {r_ded:.4f}\n"
            f"Tooth points: {len(tooth_xy)}\n"
            f"ha = {ha:.3f}  hf = {hf:.3f}  ds = {ds:.4f}"
        )


# ── Tab 4: Circular Spline Full Geometry ──────────────────────────

class TabCircularSpline:
    """Full circular spline gear — conjugate tooth profile patterned around the ring."""

    def __init__(self, parent: ttk.Frame):
        self.frame = parent
        ttk.Label(parent, text="Circular Spline — full geometry (coming soon)",
                  font=("Consolas", 10)).pack(padx=20, pady=20)


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
        tab3_frame = ttk.Frame(notebook)
        tab4_frame = ttk.Frame(notebook)

        notebook.add(tab1_frame, text=" 2.1 Flexspline Tooth ")
        notebook.add(tab2_frame, text=" 2.2 Conjugate Circular Spline Tooth ")
        notebook.add(tab3_frame, text=" Flexspline ")
        notebook.add(tab4_frame, text=" Circular Spline ")

        self.tab21 = Tab21(tab1_frame)
        self.tab22 = Tab22(tab2_frame)
        self.tab_fs = TabFlexspline(tab3_frame)
        self.tab_cs = TabCircularSpline(tab4_frame)

        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._notebook = notebook
        self._tab2_frame = tab2_frame
        self._tab3_frame = tab3_frame
        self._tab4_frame = tab4_frame

    def _on_tab_changed(self, event):
        selected = self._notebook.select()
        if selected == str(self._tab2_frame):
            self.tab22.sync_from(self.tab21)
        elif selected == str(self._tab3_frame):
            self.tab_fs.sync_from(self.tab21)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
