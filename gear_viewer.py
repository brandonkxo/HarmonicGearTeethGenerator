"""
Harmonic Drive S-Tooth Calculator — starting with a simple GUI.
Displays a red 80 mm diameter circle on a Cartesian viewport.
"""
import tkinter as tk

# Viewport: ±60 mm per axis (enough to show an 80 mm circle comfortably)
HALF_VIEW_MM = 60
PLOT_PX = 500
PX_PER_MM = PLOT_PX / (2 * HALF_VIEW_MM)
MARGIN = 40
CANVAS_W = PLOT_PX + MARGIN + 20
CANVAS_H = PLOT_PX + MARGIN + 20
TICK_INTERVAL = 20  # mm


def mm_to_canvas(x_mm: float, y_mm: float) -> tuple[float, float]:
    """Cartesian mm (origin center, y-up) -> canvas pixels."""
    ox = MARGIN + PLOT_PX / 2
    oy = PLOT_PX / 2
    return ox + x_mm * PX_PER_MM, oy - y_mm * PX_PER_MM


def draw_axes(c: tk.Canvas):
    ox, oy = mm_to_canvas(0, 0)
    # axes
    c.create_line(MARGIN, oy, MARGIN + PLOT_PX, oy, fill="black")
    c.create_line(ox, 0, ox, PLOT_PX, fill="black")
    # ticks + labels
    for v in range(-HALF_VIEW_MM, HALF_VIEW_MM + 1, TICK_INTERVAL):
        # x ticks
        px, _ = mm_to_canvas(v, 0)
        c.create_line(px, oy - 4, px, oy + 4, fill="black")
        c.create_text(px, oy + 14, text=str(v), font=("Consolas", 8))
        # y ticks
        _, py = mm_to_canvas(0, v)
        c.create_line(ox - 4, py, ox + 4, py, fill="black")
        c.create_text(ox - 18, py, text=str(v), font=("Consolas", 8), anchor="e")
    # axis labels
    c.create_text(MARGIN + PLOT_PX / 2, PLOT_PX + 30, text="X (mm)",
                  font=("Consolas", 10, "bold"))
    c.create_text(12, PLOT_PX / 2, text="Y\n(mm)",
                  font=("Consolas", 10, "bold"))


def draw_circle(c: tk.Canvas, diameter_mm: float = 80.0):
    r = diameter_mm / 2
    x1, y1 = mm_to_canvas(-r, r)
    x2, y2 = mm_to_canvas(r, -r)
    c.create_oval(x1, y1, x2, y2, outline="red", width=2)


def main():
    root = tk.Tk()
    root.title("Harmonic Drive — Gear Teeth Calculator")
    canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg="white")
    canvas.pack(padx=10, pady=10)
    draw_axes(canvas)
    draw_circle(canvas, diameter_mm=80.0)
    root.mainloop()


if __name__ == "__main__":
    main()
