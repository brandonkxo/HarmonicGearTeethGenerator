"""
Equations 1-6 (and supporting Eqs 7-13) from:
  Liu et al., "A Novel Rapid Design Framework for Tooth Profile of
  Double-Circular-Arc Common-Tangent Flexspline in Harmonic Reducers"
  Machines 2025, 13, 535.

Computes the double-circular-arc common-tangent flexspline tooth profile
consisting of three segments: convex arc AB, tangent line BC, concave arc CD.
"""
import math
import numpy as np
from scipy.interpolate import splprep, splev


# ── Default parameter values from paper (Section 3.5) ─────────────
# Module m = 0.5 mm, z_f = 100, z_c = 102
DEFAULTS = {
    "m":  0.5,
    "z_f": 100,     # number of flexspline teeth (fixed, not optimized)
    "z_c": 102,     # number of circular spline teeth
    "w0":  0.5,     # max radial deformation omega_0 (mm)
    "r1": 0.685,
    "c1": 0.332,
    "e1": 0.155,
    "r2": 0.785,
    "c2": 0.330,
    "e2": 0.134,
    "ha": 0.275,    # addendum height — reasonable starting point
    "hf": 0.375,    # dedendum height — reasonable starting point
    "mu_s": 0.01,   # ring wall coefficient: s = mu_s * m * z_f
    "mu_t": 0.6,    # cup wall coefficient: t = mu_t * s
}

# Pitch radius (fixed, never changes during optimization)
#   rp = m * z_f / 2
PITCH_RADIUS = DEFAULTS["m"] * DEFAULTS["z_f"] / 2.0  # 25.0 mm

# Display labels for GUI
PARAM_LABELS = {
    "m":   "Module m",
    "z_f": "Teeth z_f",
    "z_c": "Teeth z_c",
    "w0":  "Max deform \u03c9\u2080",
    "r1":  "Convex radius r\u2081",
    "c1":  "O\u2081 x-offset c\u2081",
    "e1":  "O\u2081 y-offset e\u2081",
    "r2":  "Concave radius r\u2082",
    "c2":  "O\u2082 x-offset c\u2082",
    "e2":  "O\u2082 y-offset e\u2082",
    "ha":  "Addendum h\u2090",
    "hf":  "Dedendum h\u1DA0",
    "mu_s": "Coeff μₛ (ring)",
    "mu_t": "Coeff μₜ (cup)",
}

# Ordered list of parameter keys for consistent UI ordering
PARAM_ORDER = ["m", "z_f", "z_c", "w0", "r1", "c1", "e1", "r2", "c2", "e2", "ha", "hf", "mu_s", "mu_t"]


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
    # Compute s and t from coefficient inputs
    mu_s = params["mu_s"]
    mu_t = params["mu_t"]
    s = mu_s * m * z_f
    t = mu_t * s

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

    # ── Derived: neutral layer radius ────────────────────────────────
    # From geometry: rp = rm + ds + hf  →  rm = rp - ds - hf
    rm = rp - ds - hf

    return {
        "rp": rp,
        "rm": rm,
        "ds": ds,
        "s": s,
        "t": t,
        "alpha": alpha,
        "delta": delta,
        "x1_R": x1_R,
        "y1_R": y1_R,
        "x2_R": x2_R,
        "y2_R": y2_R,
        "r1": r1,
        "r2": r2,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "h1": h1,
        "pts_AB": pts_AB,
        "pts_BC": pts_BC,
        "pts_CD": pts_CD,
    }


# ══════════════════════════════════════════════════════════════════
# Section 2.2 — Neutral layer deformation (Eqs 14-20)
# ══════════════════════════════════════════════════════════════════
#
# These describe how the flexspline neutral layer deforms under
# the cosine-type wave generator.  φ is the angle of a point on
# the undeformed neutral layer measured from the wave generator
# major axis.

def eq14_rho(phi: float, rm: float, w0: float) -> float:
    """Eq 14: Radial vector of deformed neutral layer point.

    ρ = rm + ω₀·cos(2φ)
    """
    return rm + w0 * math.cos(2.0 * phi)


def eq15_radial_arc_change(omega: float, dphi: float) -> float:
    """Eq 15: Radial change of a unit arc element.

    A'B' - AB = (rm + ω)dφ - rm·dφ = ω·dφ
    """
    return omega * dphi


def eq16_omega(rho: float, rm: float) -> float:
    """Eq 16: Radial displacement (definition).

    ω = ρ - rm
    """
    return rho - rm


def eq17_omega(phi: float, w0: float) -> float:
    """Eq 17: Radial displacement (Eq 14 substituted into Eq 16).

    ω = ω₀·cos(2φ)
    """
    return w0 * math.cos(2.0 * phi)


def eq18_tangential_arc_change(dv: float) -> float:
    """Eq 18: Tangential change of a unit arc element.

    A''B'' - A'B' = (v + dv) - v = dv
    """
    return dv


def eq19_neutral_layer_invariance(omega: float, dphi: float, dv: float) -> float:
    """Eq 19: Neutral layer invariance condition.

    ω·dφ + dv = 0

    Returns the residual (should be zero when satisfied).
    """
    return omega * dphi + dv


def eq20_v(phi: float, w0: float) -> float:
    """Eq 20: Tangential displacement (integral of Eq 19 using Eq 17).

    v = -∫ω dφ = -½·ω₀·sin(2φ)
    """
    return -0.5 * w0 * math.sin(2.0 * phi)


# ══════════════════════════════════════════════════════════════════
# Section 2.2 — Angular relationships (Eqs 21-27)
# ══════════════════════════════════════════════════════════════════
#
# These define the angular relationships needed for the coordinate
# transformation from the flexspline frame {O_R} to the fixed
# circular spline frame {O_G}.

def eq21_mu(phi: float, w0: float, rm: float) -> float:
    """Eq 21: Normal deformation angle.

    μ = arctan(ρ̇/ρ) ≈ -(1/rm)·(dω/dφ) = (2ω₀/rm)·sin(2φ)

    where ρ̇ = dρ/dφ = -2ω₀·sin(2φ), and the approximation
    holds because μ is small and rm >> ω.
    """
    return (2.0 * w0 / rm) * math.sin(2.0 * phi)


def eq22_arc_length_invariance(phi: float, phi1: float, rm: float, w0: float) -> float:
    """Eq 22: Neutral layer arc length invariance (residual form).

    rm·φ = ∫₀^φ₁ √(ρ² + ρ̇²) dφ ≈ ∫₀^φ₁ ρ dφ

    Returns rm·φ - ∫₀^φ₁ ρ dφ  (should be ~0 when φ₁ is correct).
    Uses numerical integration (trapezoidal) for verification.
    """
    N = 200
    dphi = phi1 / N
    integral = 0.0
    for i in range(N + 1):
        p = i * dphi
        rho = eq14_rho(p, rm, w0)
        weight = 0.5 if (i == 0 or i == N) else 1.0
        integral += weight * rho * dphi
    return rm * phi - integral


def eq23_phi1(phi: float, w0: float, rm: float) -> float:
    """Eq 23: Angle of deformed endpoint relative to wave generator.

    φ₁ = φ - ω₀·sin(2φ) / (2·rm)

    Derived from Eq 22 by evaluating the integral analytically.
    """
    return phi - w0 * math.sin(2.0 * phi) / (2.0 * rm)


def eq24_transmission_ratio(z_f: float, z_c: float) -> float:
    """Eq 24: Transmission ratio relationship.

    z_f / z_c = φ₂ / φ

    Returns the ratio z_f / z_c.
    """
    return z_f / z_c


def eq25_phi2(phi: float, z_f: float, z_c: float) -> float:
    """Eq 25: Wave generator rotation angle.

    φ₂ = (z_f / z_c) · φ
    """
    return (z_f / z_c) * phi


def eq26_gamma(phi1: float, phi2: float) -> float:
    """Eq 26: Deformation angle of flexspline's deformed endpoint.

    γ = φ₁ - φ₂
    """
    return phi1 - phi2


def eq27_psi(mu: float, gamma: float) -> float:
    """Eq 27: Angle between Y_G axis and Y_R axis.

    ϕ = μ + γ
    """
    return mu + gamma


# ══════════════════════════════════════════════════════════════════
# Section 2.2 — Coordinate transform & envelope (Eqs 28-30)
# ══════════════════════════════════════════════════════════════════
#
# Transform flexspline tooth profile points from the local tooth
# frame {O_R - X_R - Y_R} into the fixed circular spline frame
# {O_G - X_G - Y_G}, then apply the envelope condition to find
# the conjugate circular spline tooth profile.

def eq28_transform_matrix(psi: float, rho: float, gamma: float):
    """Eq 28: 3x3 homogeneous transformation matrix M.

    M = [ cos(ϕ)   sin(ϕ)   ρ·sin(γ) ]
        [-sin(ϕ)   cos(ϕ)   ρ·cos(γ) ]
        [   0         0         1     ]

    Returns the 3x3 matrix as a tuple of tuples.
    """
    cp = math.cos(psi)
    sp = math.sin(psi)
    tx = rho * math.sin(gamma)
    ty = rho * math.cos(gamma)
    return (
        (cp,  sp, tx),
        (-sp, cp, ty),
        (0.0, 0.0, 1.0),
    )


def eq29_transform(xr: float, yr: float,
                   psi: float, rho: float, gamma: float) -> tuple[float, float]:
    """Eq 29: Transform a point from flexspline frame to circular spline frame.

    x_g =  x_r·cos(ϕ) + y_r·sin(ϕ) + ρ·sin(γ)
    y_g = -x_r·sin(ϕ) + y_r·cos(ϕ) + ρ·cos(γ)
    """
    cp = math.cos(psi)
    sp = math.sin(psi)
    xg = xr * cp + yr * sp + rho * math.sin(gamma)
    yg = -xr * sp + yr * cp + rho * math.cos(gamma)
    return xg, yg


def eq30_envelope_condition(dxg_dl: float, dyg_dphi: float,
                            dyg_dl: float, dxg_dphi: float) -> float:
    """Eq 30: Envelope conjugate condition.

    ∂x_g/∂l · ∂y_g/∂φ  -  ∂y_g/∂l · ∂x_g/∂φ  =  0

    Returns the residual (zero on the conjugate profile).
    The four partial derivatives are computed externally via
    finite differences.
    """
    return dxg_dl * dyg_dphi - dyg_dl * dxg_dphi


# ══════════════════════════════════════════════════════════════════
# Eq 30 solver building blocks
# ══════════════════════════════════════════════════════════════════

def _profile_point_at_l(l: float, prof: dict) -> tuple[float, float]:
    """Evaluate Eqs 2/4/6 at a single arc-length value *l*.

    Returns (x_r, y_r) in the tooth-local frame {O_R}.
    *prof* is the dict returned by compute_profile().
    """
    alpha = prof["alpha"]
    delta = prof["delta"]
    x1_R  = prof["x1_R"]
    y1_R  = prof["y1_R"]
    x2_R  = prof["x2_R"]
    y2_R  = prof["y2_R"]
    r1    = prof["r1"]
    r2    = prof["r2"]
    l1    = prof["l1"]
    l2    = prof["l2"]

    if l <= l1:
        # Eq 2 — convex arc AB
        angle = alpha - l / r1
        x = r1 * math.cos(angle) + x1_R
        y = r1 * math.sin(angle) + y1_R
    elif l <= l2:
        # Eq 4 — tangent line BC
        x = r1 * math.cos(delta) + x1_R + (l - l1) * math.sin(delta)
        y = r1 * math.sin(delta) + y1_R - (l - l1) * math.cos(delta)
    else:
        # Eq 6 — concave arc CD
        angle = delta + (l - l2) / r2
        x = -r2 * math.cos(angle) + x2_R
        y = -r2 * math.sin(angle) + y2_R
    return x, y


def _xg_yg_at(l: float, phi: float,
              prof: dict, params: dict) -> tuple[float, float]:
    """Compute (x_g, y_g) in the circular-spline frame {O_G} for a
    flexspline tooth profile point at arc-length *l* and wave-generator
    angle *phi*.

    Chains: Eqs 2/4/6 → Eqs 14,21,23,25,26,27 → Eq 29.
    """
    rm = prof["rm"]
    w0 = params["w0"]
    z_f = params["z_f"]
    z_c = params["z_c"]

    # tooth profile point in {O_R}
    xr, yr = _profile_point_at_l(l, prof)

    # angular quantities (all functions of phi only)
    rho   = eq14_rho(phi, rm, w0)
    mu    = eq21_mu(phi, w0, rm)
    phi1  = eq23_phi1(phi, w0, rm)
    phi2  = eq25_phi2(phi, z_f, z_c)
    gamma = eq26_gamma(phi1, phi2)
    psi   = eq27_psi(mu, gamma)

    # Eq 29 coordinate transform
    return eq29_transform(xr, yr, psi, rho, gamma)


def eq30_residual_at(l: float, phi: float,
                     prof: dict, params: dict,
                     eps_l: float = 1e-5,
                     eps_phi: float = 1e-5) -> float:
    """Evaluate the Eq 30 envelope residual at (l, phi) using central
    finite differences.

    Returns:
        ∂x_g/∂l · ∂y_g/∂φ  −  ∂y_g/∂l · ∂x_g/∂φ

    The step sizes *eps_l* and *eps_phi* are small perturbations used
    for the central-difference approximation.
    """
    l3 = prof["l3"]

    # ── ∂/∂l  (perturb l, hold phi fixed) ──
    l_plus  = min(l + eps_l, l3)
    l_minus = max(l - eps_l, 0.0)
    dl = l_plus - l_minus

    xg_lp, yg_lp = _xg_yg_at(l_plus,  phi, prof, params)
    xg_lm, yg_lm = _xg_yg_at(l_minus, phi, prof, params)
    dxg_dl = (xg_lp - xg_lm) / dl
    dyg_dl = (yg_lp - yg_lm) / dl

    # ── ∂/∂φ  (perturb phi, hold l fixed) ──
    xg_pp, yg_pp = _xg_yg_at(l, phi + eps_phi, prof, params)
    xg_pm, yg_pm = _xg_yg_at(l, phi - eps_phi, prof, params)
    dxg_dphi = (xg_pp - xg_pm) / (2.0 * eps_phi)
    dyg_dphi = (yg_pp - yg_pm) / (2.0 * eps_phi)

    return eq30_envelope_condition(dxg_dl, dyg_dphi, dyg_dl, dxg_dphi)


def _filter_phi_jump(seg_roots: list[tuple], pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep only the engagement run of a segment's conjugate points.

    Walks through the phi-sorted points and computes consecutive phi
    steps.  When a step more than doubles the previous step, everything
    from that point onward is the exit/disengagement sweep and is cut.

    Parameters:
        seg_roots – raw root tuples sorted by phi (phi is index 0)
        pts       – corresponding (x_g, y_local) list, same order

    Returns the trimmed pts list.
    """
    if len(seg_roots) < 3:
        return pts

    prev_step = abs(seg_roots[1][0] - seg_roots[0][0])
    for i in range(2, len(seg_roots)):
        step = abs(seg_roots[i][0] - seg_roots[i - 1][0])
        if prev_step > 0 and step > 2.0 * prev_step:
            return pts[:i]
        prev_step = step
    return pts


def compute_conjugate_profile(params: dict,
                              N_phi: int = 720,
                              N_l: int = 1000) -> dict:
    """Solve Eq 30 over a (phi, l) grid to find the conjugate circular-
    spline tooth profile.

    For each discrete phi value, the Eq 30 residual is evaluated along
    the full arc-length range [0, l3].  Sign changes indicate zero
    crossings; linear interpolation pinpoints each root.  The
    corresponding (x_g, y_g) is computed via Eq 29 and shifted into
    tooth-local coordinates by subtracting the circular-spline pitch
    radius from y_g.

    Returns dict with:
        conjugate_pts  – list of (x, y) in tooth-local CS frame
        branches       – pts grouped into continuous curves
        rp_c           – circular-spline pitch radius
        n_pts / n_branches – counts
    Or {'error': msg} on failure.
    """
    prof = compute_profile(params)
    if "error" in prof:
        return prof

    m   = params["m"]
    z_c = params["z_c"]
    l3  = prof["l3"]

    rp_c = m * z_c / 2.0  # circular-spline pitch radius

    # Grid spacings
    phi_min = -math.pi / 2.0
    phi_max =  math.pi / 2.0
    dphi_grid = (phi_max - phi_min) / N_phi
    dl_grid   = l3 / N_l

    # Collect (phi, l_zero, x_local, y_local, segment) for every root
    raw_roots: list[tuple[float, float, float, float, str]] = []

    # Bounds for filtering spurious roots
    ha = params["ha"]
    hf = params["hf"]
    margin = 0.5 * (ha + hf)
    y_lo = -(hf + margin)
    y_hi =  (ha + margin)
    x_bound = m * math.pi

    l1 = prof["l1"]
    l2 = prof["l2"]

    for i in range(N_phi + 1):
        phi = phi_min + i * dphi_grid

        # Evaluate residual along l for this phi
        residuals = []
        for j in range(N_l + 1):
            lv = j * dl_grid
            residuals.append(eq30_residual_at(lv, phi, prof, params))

        # Scan for sign changes → linear interpolation
        for j in range(N_l):
            r0 = residuals[j]
            r1 = residuals[j + 1]
            if r0 * r1 < 0.0:
                frac = abs(r0) / (abs(r0) + abs(r1))
                l_zero = (j + frac) * dl_grid
                xg, yg = _xg_yg_at(l_zero, phi, prof, params)
                y_local = yg - rp_c

                # Filter spurious roots outside physical tooth bounds
                if not (y_lo <= y_local <= y_hi and abs(xg) <= x_bound):
                    continue

                # Tag by originating tooth segment
                if l_zero <= l1:
                    seg = "AB"
                elif l_zero <= l2:
                    seg = "BC"
                else:
                    seg = "CD"
                raw_roots.append((phi, l_zero, xg, y_local, seg))

    if not raw_roots:
        return {"error": "No conjugate points found — check parameters."}

    # ── Build segment-keyed branches (sorted by phi) ──
    seg_branches: dict[str, list[tuple[float, float]]] = {"AB": [], "BC": [], "CD": []}
    conjugate_pts = []

    for seg_key in ("AB", "BC", "CD"):
        seg_pts = [r for r in raw_roots if r[4] == seg_key]
        seg_pts.sort(key=lambda r: r[0])  # sort by phi
        pts = [(r[2], r[3]) for r in seg_pts]
        seg_branches[seg_key] = _filter_phi_jump(seg_pts, pts)
        conjugate_pts.extend(seg_branches[seg_key])

    # Legacy 'branches' list (all non-empty segments)
    branches = [pts for pts in seg_branches.values() if len(pts) >= 2]

    return {
        "conjugate_pts": conjugate_pts,
        "branches": branches,
        "seg_branches": seg_branches,
        "raw_roots": raw_roots,
        "rp_c": rp_c,
        "s": prof["s"],
        "t": prof["t"],
        "n_pts": len(conjugate_pts),
        "n_branches": len(branches),
    }


# ══════════════════════════════════════════════════════════════════
# Spline smoothing for conjugate profile branches
# ══════════════════════════════════════════════════════════════════

def smooth_branch(pts: list[tuple[float, float]],
                  s: float = 0.001,
                  num_out: int = 200) -> list[tuple[float, float]]:
    """Fit a parametric smoothing spline to a single branch.

    Parameters:
        pts     – list of (x, y) points (must have >= 4 points)
        s       – smoothing factor (larger = smoother, 0 = interpolate)
        num_out – number of resampled output points

    Returns a list of (x, y) resampled along the smooth spline,
    or the original pts if the branch is too short or fitting fails.
    """
    if len(pts) < 4:
        return pts
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    try:
        tck, _ = splprep([x, y], s=s, k=3)
        u_new = np.linspace(0.0, 1.0, num_out)
        xs, ys = splev(u_new, tck)
        return list(zip(xs.tolist(), ys.tolist()))
    except Exception:
        return pts


def smooth_conjugate_profile(result: dict,
                             s: float = 0.001,
                             num_out: int = 200) -> dict:
    """Concatenate all segment points into one flank and fit a single B-spline.

    The combined points are sorted by y descending (addendum → dedendum)
    to form a continuous traversal of the tooth flank, then fitted with
    one cubic B-spline for C2 continuity across the entire profile.

    Adds to *result*:
        smoothed_flank       – single unified spline curve [(x,y), ...]
        smoothed_seg_branches – per-segment splines (kept for raw dot overlay)
    """
    if "error" in result:
        return result

    # ── Per-segment splines (for backwards compat / overlay) ──
    smoothed_seg: dict[str, list[tuple[float, float]]] = {}
    for seg_key, pts in result.get("seg_branches", {}).items():
        smoothed_seg[seg_key] = smooth_branch(pts, s=s, num_out=num_out)
    result["smoothed_seg_branches"] = smoothed_seg

    # ── Unified flank: concatenate all segments, sort, fit one spline ──
    all_pts = []
    for seg_key in ("AB", "BC", "CD"):
        all_pts.extend(result.get("seg_branches", {}).get(seg_key, []))

    # Sort by y descending (tip to root) to get a continuous traversal
    all_pts.sort(key=lambda p: -p[1])

    result["smoothed_flank"] = smooth_branch(all_pts, s=s, num_out=num_out)

    # Legacy list form
    result["smoothed_branches"] = [
        smooth_branch(b, s=s, num_out=num_out) for b in result["branches"]
    ]
    return result


# ══════════════════════════════════════════════════════════════════
# Single tooth outline on pitch circle (for Tab 3 — Flexspline)
# ══════════════════════════════════════════════════════════════════

def _cubic_bezier(p0, p1, p2, p3, n=12):
    """Sample n+1 points from a cubic Bezier defined by 4 control points."""
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        x = u**3*p0[0] + 3*u**2*t*p1[0] + 3*u*t**2*p2[0] + t**3*p3[0]
        y = u**3*p0[1] + 3*u**2*t*p1[1] + 3*u*t**2*p2[1] + t**3*p3[1]
        pts.append((x, y))
    return pts


def _g2_blend(p_flank, tan_flank, curv_flank,
              p_line, tan_line, blend_len=None):
    """Build a cubic Bezier that matches G0+G1 at both ends and
    approximates G2 at the flank end.

    Parameters:
        p_flank    – (x,y) endpoint on the flank curve
        tan_flank  – (tx,ty) unit tangent at flank endpoint (pointing into blend)
        curv_flank – scalar curvature at flank endpoint (1/radius, signed)
        p_line     – (x,y) endpoint on the root line
        tan_line   – (tx,ty) unit tangent at line endpoint (pointing into blend)
        blend_len  – characteristic length for control-arm sizing

    Returns list of (x,y) sampled from the Bezier.
    """
    if blend_len is None:
        dx = p_line[0] - p_flank[0]
        dy = p_line[1] - p_flank[1]
        blend_len = math.sqrt(dx*dx + dy*dy)
    if blend_len < 1e-12:
        return [p_flank, p_line]

    # Arm length: ~1/3 of chord for nice G2 match
    arm = blend_len / 3.0

    # If curvature is significant, adjust the flank arm to better match G2
    if abs(curv_flank) > 1e-6:
        # For a cubic Bezier, curvature at t=0 is:
        #   kappa = (2/3) * |n x (P1-P0)| / |P1-P0|^2
        # We solve for arm length that yields the desired curvature
        desired_arm = max(arm, 2.0 / (3.0 * abs(curv_flank) + 1e-12))
        arm_flank = min(desired_arm, blend_len * 0.45)
    else:
        arm_flank = arm

    arm_line = arm

    cp0 = p_flank
    cp1 = (p_flank[0] + arm_flank * tan_flank[0],
           p_flank[1] + arm_flank * tan_flank[1])
    cp3 = p_line
    cp2 = (p_line[0] + arm_line * tan_line[0],
           p_line[1] + arm_line * tan_line[1])

    return _cubic_bezier(cp0, cp1, cp2, cp3, n=16)


def _estimate_tangent_and_curvature(pts, end="last"):
    """Estimate unit tangent and curvature at the start or end of a point list.

    end="last"  → tangent points outward from the last point (away from curve)
    end="first" → tangent points outward from the first point
    """
    if len(pts) < 3:
        if len(pts) == 2:
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            mag = math.sqrt(dx*dx + dy*dy)
            if mag < 1e-15:
                return (0, -1), 0.0
            t = (dx/mag, dy/mag)
            if end == "last":
                return t, 0.0
            else:
                return (-t[0], -t[1]), 0.0
        return (0, -1), 0.0

    if end == "last":
        p0 = pts[-3]
        p1 = pts[-2]
        p2 = pts[-1]
    else:
        p2 = pts[0]
        p1 = pts[1]
        p0 = pts[2]

    # Tangent from p1→p2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    mag = math.sqrt(dx*dx + dy*dy)
    if mag < 1e-15:
        return (0, -1), 0.0
    tan = (dx/mag, dy/mag)

    # Curvature via three-point circle
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    d = 2 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
    if abs(d) < 1e-15:
        return tan, 0.0
    # radius of circumscribed circle
    ab = math.sqrt((bx-ax)**2 + (by-ay)**2)
    bc = math.sqrt((cx-bx)**2 + (cy-by)**2)
    ca = math.sqrt((ax-cx)**2 + (ay-cy)**2)
    area = abs(d) / 2
    R = (ab * bc * ca) / (4 * area) if area > 1e-15 else 1e12
    curv = 1.0 / R

    return tan, curv


def build_single_tooth_outline(params: dict) -> dict:
    """Build the complete outline of one flexspline tooth placed on the
    pitch circle, including G2 blends at the root.

    Returns dict with:
        tooth_xy     – list of (X, Y) in Cartesian coords on the pitch circle
        right_flank  – right flank points (local coords before polar transform)
        left_flank   – left flank points (local coords)
        rp, rm, ds   – radii for reference circles
        ha, hf       – addendum / dedendum heights
    Or {"error": msg} on failure.
    """
    result = compute_profile(params)
    if "error" in result:
        return result

    m   = params["m"]
    z_f = params["z_f"]
    rp  = m * z_f / 2.0
    ha  = params["ha"]
    hf  = params["hf"]
    ds  = result["ds"]
    rm  = result["rm"]

    # ── Right flank: A→B→C→D (addendum to dedendum) ──
    right_flank = list(result["pts_AB"]) + list(result["pts_BC"]) + list(result["pts_CD"])

    # ── Left flank: mirror of right, reversed (dedendum to addendum) ──
    left_flank = [(-x, y) for x, y in reversed(right_flank)]

    # ── Assemble outline: right flank + left flank (open at root) ──
    local_outline = list(right_flank) + list(left_flank)

    # ── Transform local (x, y) → polar → Cartesian on pitch circle ──
    tooth_xy = []
    for x_loc, y_loc in local_outline:
        r = rm + y_loc
        theta = x_loc / rp
        X = r * math.sin(theta)
        Y = r * math.cos(theta)
        tooth_xy.append((X, Y))

    return {
        "tooth_xy": tooth_xy,
        "local_outline": local_outline,
        "right_flank": right_flank,
        "left_flank": left_flank,
        "split": len(right_flank),
        "rp": rp,
        "rm": rm,
        "ds": ds,
        "ha": ha,
        "hf": hf,
    }


def build_full_flexspline(params: dict, n_ded_arc: int = 8) -> dict:
    """Pattern the flexspline tooth around the full pitch circle.

    Produces one continuous polyline chain:
      ... left_flank_i → tip_i → right_flank_i → ded_arc → left_flank_{i+1} → ...

    The dedendum arc connects the right flank bottom (D) of tooth i
    to the left flank bottom (D') of tooth i+1 along the dedendum circle.

    Returns dict with:
        chain_xy  – list of (X, Y) forming the continuous outline
        rp, rm, ds, ha, hf – geometry references
    Or {"error": msg} on failure.
    """
    result = compute_profile(params)
    if "error" in result:
        return result

    m   = params["m"]
    z_f = int(params["z_f"])
    rp  = m * z_f / 2.0
    ha  = params["ha"]
    hf  = params["hf"]
    ds  = result["ds"]
    rm  = result["rm"]

    # Single tooth flanks in local coords
    right_flank = list(result["pts_AB"]) + list(result["pts_BC"]) + list(result["pts_CD"])
    left_flank = [(-x, y) for x, y in reversed(right_flank)]

    # Dedendum radius
    r_ded = rm + ds

    # Angular pitch
    pitch_angle = 2.0 * math.pi / z_f

    def local_to_polar(x_loc, y_loc, tooth_offset_angle):
        """Transform local tooth coords to Cartesian on the circle."""
        r = rm + y_loc
        theta = x_loc / rp + tooth_offset_angle
        return r * math.sin(theta), r * math.cos(theta)

    # Angular positions of D (right flank bottom) and D' (left flank bottom)
    # in the local tooth frame (before adding tooth offset)
    pt_D = right_flank[-1]     # bottom of right flank
    pt_Dp = left_flank[0]      # bottom of left flank
    theta_D  = pt_D[0] / rp    # angular offset of D from tooth center
    theta_Dp = pt_Dp[0] / rp   # angular offset of D' from tooth center

    chain_xy = []

    for i in range(z_f):
        angle_i = i * pitch_angle

        # Left flank: D' → A' (dedendum up to addendum)
        for x_loc, y_loc in left_flank:
            chain_xy.append(local_to_polar(x_loc, y_loc, angle_i))

        # Addendum line: A' (last of left_flank) → A (first of right_flank)
        pt_Ap = left_flank[-1]
        pt_A = right_flank[0]
        for j in range(1, n_ded_arc + 1):
            frac = j / n_ded_arc
            x_loc = pt_Ap[0] + frac * (pt_A[0] - pt_Ap[0])
            y_loc = pt_Ap[1] + frac * (pt_A[1] - pt_Ap[1])
            chain_xy.append(local_to_polar(x_loc, y_loc, angle_i))

        # Right flank: A → D (addendum down to dedendum)
        for x_loc, y_loc in right_flank:
            chain_xy.append(local_to_polar(x_loc, y_loc, angle_i))

        # Dedendum arc: D of tooth i → D' of tooth i+1
        next_i = (i + 1) % z_f
        angle_next = next_i * pitch_angle
        theta_start = theta_D + angle_i
        theta_end   = theta_Dp + angle_next

        # Handle wrap-around
        if theta_end < theta_start:
            theta_end += 2.0 * math.pi

        for j in range(1, n_ded_arc + 1):
            frac = j / n_ded_arc
            th = theta_start + frac * (theta_end - theta_start)
            chain_xy.append((r_ded * math.sin(th), r_ded * math.cos(th)))

    return {
        "chain_xy": chain_xy,
        "rp": rp,
        "rm": rm,
        "ds": ds,
        "s": result["s"],
        "t": result["t"],
        "ha": ha,
        "hf": hf,
        "z_f": z_f,
    }


def build_deformed_flexspline(params: dict, n_ded_arc: int = 8) -> dict:
    """Pattern the flexspline around the DEFORMED neutral layer.

    Uses the paper's coordinate transform (Eqs 14, 21, 23, 27, 29) to show
    how the flexspline actually looks when flexed by the elliptical
    wave generator.

    Each tooth is treated as a rigid profile in its local frame and placed
    using the paper's transform with φ₂ = 0 (wave generator at reference).

    Returns dict with:
        chain_xy  – list of (X, Y) forming the deformed outline
        rp, rm, w0, ds, ha, hf, z_f – geometry references
    Or {"error": msg} on failure.
    """
    result = compute_profile(params)
    if "error" in result:
        return result

    z_f = int(params["z_f"])
    w0  = params["w0"]
    rp  = params["m"] * z_f / 2.0
    rm  = result["rm"]
    ds  = result["ds"]

    # Single tooth flanks in local coords
    right_flank = list(result["pts_AB"]) + list(result["pts_BC"]) + list(result["pts_CD"])
    left_flank = [(-x, y) for x, y in reversed(right_flank)]

    # Angular pitch
    pitch_angle = 2.0 * math.pi / z_f

    def tooth_point_global(xr, yr, phi):
        """Transform local tooth point to global using paper's Eq 29.

        Uses φ₂ = 0 (wave generator at reference position) so γ = φ₁.
        """
        # Paper-based deformation quantities
        rho  = eq14_rho(phi, rm, w0)          # Eq 14: deformed neutral layer radius
        mu   = eq21_mu(phi, w0, rm)           # Eq 21: normal deformation angle
        phi1 = eq23_phi1(phi, w0, rm)         # Eq 23: deformed endpoint angle

        # For deformed flexspline shape, set wave generator rotation φ₂ = 0
        gamma = phi1                          # γ = φ₁ - φ₂, with φ₂ = 0
        psi   = eq27_psi(mu, gamma)           # Eq 27: ψ = μ + γ

        # Rigidly place the local tooth point using Eq 29
        return eq29_transform(xr, yr, psi, rho, gamma)

    chain_xy = []

    pt_D  = right_flank[-1]   # bottom of right flank (local)
    pt_Dp = left_flank[0]     # bottom of left flank (local)

    for i in range(z_f):
        phi      = i * pitch_angle
        next_i   = (i + 1) % z_f
        phi_next = next_i * pitch_angle

        # Left flank: D' → A' (dedendum up to addendum)
        for xr, yr in left_flank:
            chain_xy.append(tooth_point_global(xr, yr, phi))

        # Addendum line: A' (last of left_flank) → A (first of right_flank)
        pt_Ap = left_flank[-1]
        pt_A = right_flank[0]
        for j in range(1, n_ded_arc + 1):
            frac = j / n_ded_arc
            x_loc = pt_Ap[0] + frac * (pt_A[0] - pt_Ap[0])
            y_loc = pt_Ap[1] + frac * (pt_A[1] - pt_Ap[1])
            chain_xy.append(tooth_point_global(x_loc, y_loc, phi))

        # Right flank: A → D (addendum down to dedendum)
        for xr, yr in right_flank:
            chain_xy.append(tooth_point_global(xr, yr, phi))

        # Dedendum arc: D of tooth i → D' of tooth i+1
        # Get the actual transformed endpoints
        xD,  yD  = tooth_point_global(pt_D[0],  pt_D[1],  phi)
        xDp, yDp = tooth_point_global(pt_Dp[0], pt_Dp[1], phi_next)

        # Linear interpolation in Cartesian space for smooth connection
        # This guarantees the arc connects exactly to the flank endpoints
        for j in range(1, n_ded_arc + 1):
            frac = j / n_ded_arc
            x_arc = xD + frac * (xDp - xD)
            y_arc = yD + frac * (yDp - yD)
            chain_xy.append((x_arc, y_arc))

    return {
        "chain_xy": chain_xy,
        "rp": rp,
        "rm": rm,
        "w0": w0,
        "ds": ds,
        "s": result["s"],
        "t": result["t"],
        "ha": params["ha"],
        "hf": params["hf"],
        "z_f": z_f,
    }


def build_modified_deformed_flexspline(params: dict, d_max: float, n_ded_arc: int = 8) -> dict:
    """Pattern the flexspline around the DEFORMED neutral layer with radial modification.

    This is the same as build_deformed_flexspline, but with the tooth profile
    shifted inward by d_max on both flanks (Section 3.2 radial modification).

    The modification shifts both sides of the tooth profile inward by d_max
    in the X-direction:
      - Right flank (x > 0): x_new = x - d_max
      - Left flank (x < 0): x_new = x + d_max

    Parameters:
        params    – gear parameters dict
        d_max     – maximum interference distance to shift inward (mm)
        n_ded_arc – number of interpolation points per dedendum arc

    Returns dict with:
        chain_xy  – list of (X, Y) forming the modified deformed outline
        d_max     – the applied modification amount
        rp, rm, w0, ds, ha, hf, z_f – geometry references
    Or {"error": msg} on failure.
    """
    result = compute_profile(params)
    if "error" in result:
        return result

    z_f = int(params["z_f"])
    w0  = params["w0"]
    rp  = params["m"] * z_f / 2.0
    rm  = result["rm"]
    ds  = result["ds"]

    # Single tooth flanks in local coords - MODIFIED by d_max
    right_flank_orig = list(result["pts_AB"]) + list(result["pts_BC"]) + list(result["pts_CD"])

    # Apply radial modification: shift right flank inward (subtract d_max from x)
    right_flank = [(x - d_max, y) for x, y in right_flank_orig]

    # Left flank: mirror of modified right flank
    left_flank = [(-x, y) for x, y in reversed(right_flank)]

    # Angular pitch
    pitch_angle = 2.0 * math.pi / z_f

    def tooth_point_global(xr, yr, phi):
        """Transform local tooth point to global using paper's Eq 29."""
        rho  = eq14_rho(phi, rm, w0)
        mu   = eq21_mu(phi, w0, rm)
        phi1 = eq23_phi1(phi, w0, rm)
        gamma = phi1
        psi   = eq27_psi(mu, gamma)
        return eq29_transform(xr, yr, psi, rho, gamma)

    chain_xy = []

    pt_D  = right_flank[-1]
    pt_Dp = left_flank[0]

    for i in range(z_f):
        phi      = i * pitch_angle
        next_i   = (i + 1) % z_f
        phi_next = next_i * pitch_angle

        # Left flank
        for xr, yr in left_flank:
            chain_xy.append(tooth_point_global(xr, yr, phi))

        # Addendum line: A' (last of left_flank) → A (first of right_flank)
        pt_Ap = left_flank[-1]
        pt_A = right_flank[0]
        for j in range(1, n_ded_arc + 1):
            frac = j / n_ded_arc
            x_loc = pt_Ap[0] + frac * (pt_A[0] - pt_Ap[0])
            y_loc = pt_Ap[1] + frac * (pt_A[1] - pt_Ap[1])
            chain_xy.append(tooth_point_global(x_loc, y_loc, phi))

        # Right flank
        for xr, yr in right_flank:
            chain_xy.append(tooth_point_global(xr, yr, phi))

        # Dedendum arc
        xD,  yD  = tooth_point_global(pt_D[0],  pt_D[1],  phi)
        xDp, yDp = tooth_point_global(pt_Dp[0], pt_Dp[1], phi_next)

        for j in range(1, n_ded_arc + 1):
            frac = j / n_ded_arc
            x_arc = xD + frac * (xDp - xD)
            y_arc = yD + frac * (yDp - yD)
            chain_xy.append((x_arc, y_arc))

    return {
        "chain_xy": chain_xy,
        "d_max": d_max,
        "rp": rp,
        "rm": rm,
        "w0": w0,
        "ds": ds,
        "s": result["s"],
        "t": result["t"],
        "ha": params["ha"],
        "hf": params["hf"],
        "z_f": z_f,
    }


def build_full_circular_spline(params: dict,
                                smoothed_flank: list[tuple[float, float]],
                                rp_c: float,
                                n_ded_arc: int = 8) -> dict:
    """Pattern the circular spline conjugate tooth around the full pitch circle.

    Uses the pre-computed smoothed_flank (one side of the conjugate tooth
    in tooth-local coords where y_local = y_g - rp_c) and mirrors it to
    form a complete tooth, then repeats z_c times with dedendum arc
    connections.

    Parameters:
        params         – gear parameters dict
        smoothed_flank – list of (x, y_local) for one flank, sorted
                         addendum→dedendum (y descending)
        rp_c           – circular spline pitch radius
        n_ded_arc      – number of interpolation points per dedendum arc

    Returns dict with:
        chain_xy  – list of (X, Y) forming the continuous outline
        rp_c, ha, hf, z_c – geometry references
    Or {"error": msg} on failure.
    """
    if len(smoothed_flank) < 2:
        return {"error": "Smoothed flank too short to pattern."}

    z_c = int(params["z_c"])
    m   = params["m"]
    z_f = params["z_f"]
    ha  = params["ha"]
    hf  = params["hf"]
    # Compute s and t from coefficient inputs
    mu_s = params["mu_s"]
    mu_t = params["mu_t"]
    s = mu_s * m * z_f
    t = mu_t * s
    ds  = s - t / 2.0

    # Right flank: addendum (top) → dedendum (bottom), y descending
    right_flank = list(smoothed_flank)

    # Left flank: mirror and reverse (dedendum → addendum)
    left_flank = [(-x, y) for x, y in reversed(right_flank)]

    # Dedendum radius for circular spline
    # In local coords, dedendum is at y_local = -hf (below pitch line)
    # On the circle: r = rp_c + y_local, so r_ded = rp_c - hf
    # But the actual dedendum of the circular spline depends on ds:
    # The bottom of the flank gives us the actual dedendum y
    y_ded = right_flank[-1][1]  # most negative y in the flank
    r_ded = rp_c + y_ded

    # Angular pitch
    pitch_angle = 2.0 * math.pi / z_c

    def local_to_polar(x_loc, y_loc, tooth_offset_angle):
        r = rp_c + y_loc
        theta = x_loc / rp_c + tooth_offset_angle
        return r * math.sin(theta), r * math.cos(theta)

    # Angular positions of flank bottom endpoints
    pt_D = right_flank[-1]     # bottom of right flank
    pt_Dp = left_flank[0]      # bottom of left flank
    theta_D  = pt_D[0] / rp_c
    theta_Dp = pt_Dp[0] / rp_c

    chain_xy = []

    for i in range(z_c):
        angle_i = i * pitch_angle

        # Left flank: D' → A' (dedendum up to addendum)
        for x_loc, y_loc in left_flank:
            chain_xy.append(local_to_polar(x_loc, y_loc, angle_i))

        # Tip is implicit (A' connects to A in the polyline)

        # Right flank: A → D (addendum down to dedendum)
        for x_loc, y_loc in right_flank:
            chain_xy.append(local_to_polar(x_loc, y_loc, angle_i))

        # Dedendum arc: D of tooth i → D' of tooth i+1
        next_i = (i + 1) % z_c
        angle_next = next_i * pitch_angle
        theta_start = theta_D + angle_i
        theta_end   = theta_Dp + angle_next

        if theta_end < theta_start:
            theta_end += 2.0 * math.pi

        for j in range(1, n_ded_arc + 1):
            frac = j / n_ded_arc
            th = theta_start + frac * (theta_end - theta_start)
            chain_xy.append((r_ded * math.sin(th), r_ded * math.cos(th)))

    # Reference radii
    y_add = right_flank[0][1]   # most positive y (addendum tip)
    r_add = rp_c + y_add

    return {
        "chain_xy": chain_xy,
        "rp_c": rp_c,
        "r_add": r_add,
        "r_ded": r_ded,
        "ha": ha,
        "hf": hf,
        "z_c": z_c,
    }
