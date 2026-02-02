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
    "s":   "Ring wall thick. s",
    "t":   "Cup wall thick. t",
}

# Ordered list of parameter keys for consistent UI ordering
PARAM_ORDER = ["m", "z_f", "z_c", "w0", "r1", "c1", "e1", "r2", "c2", "e2", "ha", "hf", "s", "t"]


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

    # ── Derived: neutral layer radius ────────────────────────────────
    # From geometry: rp = rm + ds + hf  →  rm = rp - ds - hf
    rm = rp - ds - hf

    return {
        "rp": rp,
        "rm": rm,
        "ds": ds,
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
                     eps_l: float = 1e-7,
                     eps_phi: float = 1e-7) -> float:
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
