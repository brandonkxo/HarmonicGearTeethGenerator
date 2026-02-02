"""
Equations 1-6 (and supporting Eqs 7-13) from:
  Liu et al., "A Novel Rapid Design Framework for Tooth Profile of
  Double-Circular-Arc Common-Tangent Flexspline in Harmonic Reducers"
  Machines 2025, 13, 535.

Computes the double-circular-arc common-tangent flexspline tooth profile
consisting of three segments: convex arc AB, tangent line BC, concave arc CD.
"""
import math


# ── Default parameter values from paper (Section 3.5) ─────────────
# Module m = 0.5 mm, z_f = 100, z_c = 102
DEFAULTS = {
    "m":  0.5,
    "z_f": 100,     # number of flexspline teeth (fixed, not optimized)
    "r1": 0.685,
    "c1": 0.332,
    "e1": 0.155,
    "r2": 0.785,
    "c2": 0.330,
    "e2": 0.134,
    "ha": 0.275,    # addendum height — reasonable starting point
    "hf": 0.375,    # dedendum height — reasonable starting point
    "s":  0.5,      # s = mu_s * m * z_f = 0.01 * 0.5 * 100
    "t":  0.3,      # t = mu_t * s = 0.6 * 0.5
}

# Pitch radius (fixed, never changes during optimization)
#   rp = m * z_f / 2
PITCH_RADIUS = DEFAULTS["m"] * DEFAULTS["z_f"] / 2.0  # 25.0 mm

# Display labels for GUI
PARAM_LABELS = {
    "m":   "Module m",
    "z_f": "Teeth z_f",
    "r1":  "Convex radius r\u2081",
    "c1":  "O\u2081 x-offset c\u2081",
    "e1":  "O\u2081 y-offset e\u2081",
    "r2":  "Concave radius r\u2082",
    "c2":  "O\u2082 x-offset c\u2082",
    "e2":  "O\u2082 y-offset e\u2082",
    "ha":  "Addendum h\u2090",
    "hf":  "Dedendum h\u1DA0",
    "s":   "Ring wall thick. s",
    "t":   "Cup wall thick. t",
}

# Ordered list of parameter keys for consistent UI ordering
PARAM_ORDER = ["m", "z_f", "r1", "c1", "e1", "r2", "c2", "e2", "ha", "hf", "s", "t"]


def compute_profile(params: dict) -> dict:
    """
    Given flexspline parameters, compute the three profile segments
    and return all intermediate values + point lists.

    Parameters (all in mm):
        m   - module
        r1  - convex arc radius
        c1  - O1 distance to Y_R axis
        e1  - O1 distance to pitch circle tangent
        r2  - concave arc radius
        c2  - O2 distance to Y_R axis
        e2  - O2 distance to pitch circle tangent
        ha  - addendum height
        hf  - dedendum height
        s   - tooth ring wall thickness
        t   - flexspline cup wall thickness

    Returns dict with computed values and point lists, or {"error": msg}.
    """
    m   = params["m"]
    z_f = params["z_f"]
    rp  = m * z_f / 2.0     # pitch radius (fixed)
    r1 = params["r1"]
    c1 = params["c1"]
    e1 = params["e1"]
    r2 = params["r2"]
    c2 = params["c2"]
    e2 = params["e2"]
    ha = params["ha"]
    hf = params["hf"]
    s  = params["s"]
    t  = params["t"]

    # ── Equation 1 ─────────────────────────────────────────────────
    # ds: distance from dedendum circle to neutral layer
    ds = s - t / 2.0

    # ── Equation 3 ─────────────────────────────────────────────────
    # Convex arc center O1 coordinates and start angle alpha
    arg1 = (ha + e1) / r1
    if abs(arg1) > 1.0:
        return {"error": f"Eq3 arcsin domain: (ha+e1)/r1 = {arg1:.4f}"}
    alpha = math.asin(arg1)
    x1_R = -c1
    y1_R = ds + hf - e1

    # ── Equation 7 (partial) ───────────────────────────────────────
    # Concave arc center O2 coordinates
    x2_R = m * math.pi / 2.0 + c2
    y2_R = ds + hf + e2

    # ── Equation 8 ─────────────────────────────────────────────────
    # Distance between arc centers O1 and O2
    d = math.sqrt((y1_R - y2_R) ** 2 + (x1_R - x2_R) ** 2)

    # ── Equation 9 ─────────────────────────────────────────────────
    arg_eps = (r1 + r2) / d
    if abs(arg_eps) > 1.0:
        return {"error": f"Eq9 arccos domain: (r1+r2)/d = {arg_eps:.4f}"}
    epsilon = math.acos(arg_eps)

    # ── Equation 10 ────────────────────────────────────────────────
    # Note: paper uses arctan (not atan2)
    denom = x1_R - x2_R
    if abs(denom) < 1e-15:
        return {"error": "Eq10: O1 and O2 have same x-coordinate."}
    sigma = math.atan((y1_R - y2_R) / denom)

    # ── Equation 11 ────────────────────────────────────────────────
    delta = epsilon + sigma

    # ── Equation 3 (continued) ─────────────────────────────────────
    # Arc length of convex segment AB
    l1 = r1 * (alpha - delta)
    if l1 < 0:
        return {"error": f"l1 < 0 ({l1:.4f}). Parameters produce invalid geometry."}

    # ── Equation 12 ────────────────────────────────────────────────
    # Y-coordinates of tangent points B and C
    y_B = r1 * math.sin(delta) + ds + hf - e1
    y_C = ds + hf + e2 - r2 * math.sin(delta)

    # ── Equation 13 ────────────────────────────────────────────────
    h1 = y_B - y_C

    # ── Equation 5 ─────────────────────────────────────────────────
    cos_d = math.cos(delta)
    if abs(cos_d) < 1e-12:
        return {"error": "Eq5: cos(delta) ~ 0, degenerate tangent."}
    l2 = l1 + h1 / cos_d

    # ── Equation 7 (continued) ─────────────────────────────────────
    # Total arc length including concave segment CD
    arg2 = (e2 + hf) / r2
    if abs(arg2) > 1.0:
        return {"error": f"Eq7 arcsin domain: (e2+hf)/r2 = {arg2:.4f}"}
    l3 = l2 + r2 * (math.asin(arg2) - delta)

    # ── Sample points along each segment ───────────────────────────
    N_AB = 60
    N_BC = 30
    N_CD = 60

    # Equation 2: Convex arc AB
    pts_AB = []
    for i in range(N_AB + 1):
        ll = l1 * i / N_AB
        x = r1 * math.cos(alpha - ll / r1) + x1_R
        y = r1 * math.sin(alpha - ll / r1) + y1_R
        pts_AB.append((x, y))

    # Equation 4: Tangent line BC
    pts_BC = []
    l_bc_len = l2 - l1
    for i in range(N_BC + 1):
        ll = l1 + l_bc_len * i / N_BC
        x = r1 * math.cos(delta) + x1_R + (ll - l1) * math.sin(delta)
        y = r1 * math.sin(delta) + y1_R - (ll - l1) * math.cos(delta)
        pts_BC.append((x, y))

    # Equation 6: Concave arc CD
    pts_CD = []
    l_cd_len = l3 - l2
    if l_cd_len > 0:
        for i in range(N_CD + 1):
            ll = l2 + l_cd_len * i / N_CD
            angle = delta + (ll - l2) / r2
            x = -r2 * math.cos(angle) + x2_R
            y = -r2 * math.sin(angle) + y2_R
            pts_CD.append((x, y))

    return {
        "rp": rp,
        "ds": ds,
        "alpha": alpha,
        "delta": delta,
        "x1_R": x1_R,
        "y1_R": y1_R,
        "x2_R": x2_R,
        "y2_R": y2_R,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "h1": h1,
        "pts_AB": pts_AB,
        "pts_BC": pts_BC,
        "pts_CD": pts_CD,
    }
