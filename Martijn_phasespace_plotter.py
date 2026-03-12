# -*- coding: utf-8 -*-

"""

Created on Wed Feb 25 16:49:44 2026

@author: madriaans

"""

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# -------------------------

# 1) Beam generation

# -------------------------

def generate_beam(N,sigma_r = 20e-9,sigma_angle=3e-5,sigma_delta=1e-4):

    #sigma_r = 20e-9        # 20 nm

    #sigma_angle = 3e-5     # 1 mrad

    #sigma_delta = 1/1e4     # 1 eV / 1 keV

    x  = np.random.normal(0, sigma_r, N)

    xp = np.random.normal(0, sigma_angle, N)

    y  = np.random.normal(0, sigma_r, N)

    yp = np.random.normal(0, sigma_angle, N)/1

    d  = np.random.normal(0, sigma_delta, N)

    return np.vstack([x, xp, y, yp, d])


# -------------------------

# 2) Linear lens + aberrations

# -------------------------

def multipole_lens(X,f=5e-3,Cs=None,Cc=None,C5=None):

    #f  = 3e-3     # 5 mm focal length

    if Cs==None:

        Cs = 2/f**3    # 2 mm

    if Cc==None:

        Cc = -2/f #1*100*2e-3     # 2 mm

    if C5 ==None:

        C5 = 2/f**5

    x, xp, y, yp, d = X

    r2 = xp**2 + yp**2

    r4 = r2**2

    xp -= x/f

    yp -= y/f

    xp -= Cs * r2 * x

    yp -= Cs * r2 * y

    xp -= C5 * r4 * x

    yp -= C5 * r4 * y

    xp -= 1 *x**2 #deflection coma

    xp -= (Cc) * x * d

    yp -= (Cc) * y * d

    return np.vstack([x, xp, y, yp, d])

# -------------------------

# 3) Drift

# -------------------------

def drift(X, z):

    x, xp, y, yp, d = X

    x = x + z * xp

    y = y + z * yp

    return np.vstack([x, xp, y, yp, d])


# -------------------------

# 4) Space-charge diffusion

# -------------------------

def drift_with_space_charge_diffusion(X, z, beam_current=1e-12, energy_eV=1000):

    x, xp, y, yp, d = X

    # Drift

    x = x + z * xp

    y = y + z * yp

    # Constants

    e = 1.602e-19

    eps0 = 8.854e-12

    me = 9.11e-31

    E = energy_eV * e

    v = np.sqrt(2 * E / me)

    sigma_r = np.std(x)

    area = np.pi * sigma_r**2

    n = beam_current / (e * v * area)

    K = e**4 / ((4*np.pi*eps0)**2 * E**2)

    dtheta2_dz = K * n

    sigma_theta = np.sqrt(dtheta2_dz * z)

    dtheta_x = np.random.normal(0, sigma_theta, len(xp))

    dtheta_y = np.random.normal(0, sigma_theta, len(yp))

    xp += dtheta_x

    yp += dtheta_y

    # Longitudinal diffusion

    d += 0.5 * (dtheta_x**2 + dtheta_y**2)

    return np.vstack([x, xp, y, yp, d])
 
 
def drift_with_accel_collisions_vectorized(

        X,

        L,

        U0=1000.0,          # initial acceleration voltage [V]

        U1=1000.0,          # final acceleration voltage [V]

        beam_current=0.0,

        base_neighbors=1,

        interaction_window_factor=3.0):

    """

    Vectorized drift with pairwise weak Coulomb collisions through accelerating drift.
 
    Parameters:

    -----------

    X : ndarray (5, N)

        Phase space [x, x', y', y', dE/E]

    L : float

        Drift length [m]

    U0 : float

        Initial beam voltage [V]

    U1 : float

        Final beam voltage [V]

    beam_current : float

        Beam current [A]

    base_neighbors : int

        Minimum neighbour count

    interaction_window_factor : float

        Multiples of mean longitudinal spacing for interaction window
 
    Returns:

    --------

    X_new : ndarray (5, N)

        Updated phase space

    """
 
    N = X.shape[1]
 
    # --- Pure drift for zero current ---

    if beam_current <= 0:

        x, xp, y, yp, d = X

        x = x + xp * L

        y = y + yp * L

        accel_factor = np.sqrt(U0 / U1)

        xp *= accel_factor

        yp *= accel_factor

        d  *= U0 / U1

        return np.vstack([x, xp, y, yp, d])
 
    # --- Copy state ---

    X_new = X.copy()

    x, xp, y, yp, d = X_new
 
    # --- Constants ---

    e = 1.602e-19

    eps0 = 8.854e-12

    m = 9.11e-31
 
    v0 = np.sqrt(2 * e * U0 / m)
 
    # --- Longitudinal positions ---

    mean_dt = e / beam_current

    mean_dz = v0 * mean_dt

    dz_random = np.random.exponential(mean_dz, N)

    z_positions = np.cumsum(dz_random)
 
    neighbors = max(base_neighbors, 1)

    z_cut = interaction_window_factor * mean_dz
 
    # --- Prepare indices ---

    i_indices = np.arange(N - 1)[:, None]       # shape (N-1,1)

    j_indices = i_indices + np.arange(1, neighbors+1)  # shape (N-1, neighbors)

    mask_valid = j_indices < N
 
    i_flat = i_indices[mask_valid]

    j_flat = j_indices[mask_valid]
 
    dz = z_positions[j_flat] - z_positions[i_flat]

    mask_dz = dz < z_cut

    i_final = i_flat[mask_dz]

    j_final = j_flat[mask_dz]
 
    # --- Vectorized transverse distances ---

    dx = x[i_final] - x[j_final]

    dy = y[i_final] - y[j_final]

    b = np.sqrt(dx*dx + dy*dy) + 1e-20  # impact parameter
 
    # --- Small-angle scattering ---

    F_perp = e**2 / (4 * np.pi * eps0 * b**2)

    t_flight = L / v0

    theta = F_perp * t_flight / m
 
    tx = theta * dx / b

    ty = theta * dy / b
 
    # --- Linear superposition for all collisions ---

    delta_xp = np.zeros(N)

    delta_yp = np.zeros(N)

    delta_x  = np.zeros(N)

    delta_y  = np.zeros(N)
 
    # Add contributions symmetrically

    np.add.at(delta_xp, i_final, tx)

    np.add.at(delta_xp, j_final, -tx)

    np.add.at(delta_yp, i_final, ty)

    np.add.at(delta_yp, j_final, -ty)
 
    dx_offset = tx * t_flight

    dy_offset = ty * t_flight

    np.add.at(delta_x, i_final, dx_offset)

    np.add.at(delta_x, j_final, -dx_offset)

    np.add.at(delta_y, i_final, dy_offset)

    np.add.at(delta_y, j_final, -dy_offset)
 
    # --- Apply collisions ---

    xp += delta_xp

    yp += delta_yp

    x  += delta_x

    y  += delta_y
 
    # --- Drift ---

    x += xp * L

    y += yp * L
 
    # --- Acceleration scaling ---

    accel_factor = np.sqrt(U0 / U1)

    xp *= accel_factor

    yp *= accel_factor

    d  *= U0 / U1
 
    return np.vstack([x, xp, y, yp, d])

def nominal_tof_uniform_accel(L, V0, V1):

    """

    Nominal time of flight through a drift section of length L

    with uniform axial acceleration corresponding to a linear

    potential change from V0 to V1.

    Parameters

    ----------

    L  : float

        Drift length [m]

    V0 : float

        Initial nominal beam energy [eV]  (treated as acceleration voltage)

    V1 : float

        Final nominal beam energy [eV]

    Returns

    -------

    t_nom : float

        Nominal time of flight [s]

    vz0   : float

        Initial nominal axial speed [m/s]

    vz1   : float

        Final nominal axial speed [m/s]

    az    : float

        Uniform nominal axial acceleration [m/s^2]

    """

    e = 1.602e-19

    m = 9.11e-31

    V0 = max(V0, 1e-12)

    V1 = max(V1, 1e-12)

    vz0 = np.sqrt(2 * e * V0 / m)

    vz1 = np.sqrt(2 * e * V1 / m)

    if np.isclose(V1, V0):

        t_nom = L / vz0

        az = 0.0

    else:

        # Uniform axial acceleration that produces Delta(E) = e (V1 - V0)

        az = e * (V1 - V0) / (m * L)

        # Solve L = vz0 t + 0.5 az t^2

        disc = vz0**2 + 2 * az * L

        disc = max(disc, 0.0)

        t_nom = (-vz0 + np.sqrt(disc)) / az

    return t_nom, vz0, vz1, az

def pair_collision_kick_and_offset_vectorized(

        r0,         # (3, M)

        vrel,       # (3, M)

        t_nom,

        n_time_samples=17):
 
    e = 1.602e-19

    eps0 = 8.854e-12

    m = 9.109e-31

    k_c = 1.0 / (4 * np.pi * eps0)
 
    t = np.linspace(0.0, t_nom, n_time_samples)

    tc = 0.5 * t_nom

    tau = t - tc
 
    # Relative trajectory

    r_t = r0[:, :, None] + vrel[:, :, None] * tau[None, None, :]
 
    r2 = np.sum(r_t**2, axis=0)

    r2 = np.maximum(r2, (1e-15)**2)

    r3 = r2 * np.sqrt(r2)
 
    a = (k_c * e**2 / m) * (r_t / r3[None, :, :])
 
    dv = np.trapezoid(a, t, axis=2)

    dr = np.trapezoid(a * (t_nom - t)[None, None, :], t, axis=2)
 
    return dv, dr
 
def pair_collision_analytical(r0, vrel, t_nom):

    """

    Analytical hyperbolic Coulomb collision for a pair of electrons.
 
    Parameters

    ----------

    r0 : (3,) array

        Relative position at start [m]

    vrel : (3,) array

        Relative velocity [m/s]

    t_nom : float

        Drift duration [s]
 
    Returns

    -------

    dv_i : (3,) array

        Velocity kick on particle i [m/s]

    dr_i : (3,) array

        Position offset on particle i at the end [m]

    """
 
    e = 1.602e-19

    eps0 = 8.854e-12

    m = 9.11e-31

    k_c = 1.0 / (4*np.pi*eps0)
 
    # Transverse impact parameter

    dx, dy, dz = r0

    b = np.sqrt(dx**2 + dy**2) + 1e-15

    v_mag = np.linalg.norm(vrel) + 1e-15
 
    # Scattering angle (hyperbolic)

    theta = 2 * np.arctan(k_c * e**2 / (m * b * v_mag**2))
 
    # Velocity kick in transverse plane

    dv_x = theta * dx / b

    dv_y = theta * dy / b

    dv_z = 0.0  # longitudinal kick negligible for weak space charge

    dv_i = np.array([dv_x, dv_y, dv_z])
 
    # Transverse position offset at the end

    dr_x = b * (1/np.cos(theta/2) - 1) * dx / b

    dr_y = b * (1/np.cos(theta/2) - 1) * dy / b

    dr_z = 0.0

    dr_i = np.array([dr_x, dr_y, dr_z])
 
    return dv_i, dr_i 

def drift_with_finite_drift_collisions(

    X,

    L,

    V0=1000.0,

    V1=1000.0,

    beam_current=0.0,

    base_neighbors=1,

):

    """

    Drift particles with acceleration + finite-drift analytical pairwise collisions.
 
    Parameters

    ----------

    X : ndarray (5, N)

        Particle array [x, x', y, y', dE/E]

    L : float

        Drift length [m]

    U0 : float

        Initial acceleration voltage [V]

    U1 : float

        Final acceleration voltage [V]

    beam_current : float

        Beam current [A] to determine number of nearest-neighbor collisions

    base_neighbors : int

        Minimum number of neighboring particles to include for collisions

    """
 
    # Constants

    e = 1.602e-19

    m = 9.11e-31

    eps0 = 8.854e-12

    k_e = 1.0 / (4*np.pi*eps0)
 
    # Unpack particle array

    x, xp, y, yp, d = X

    N = X.shape[1]
 
    # Nominal velocities

    v0 = np.sqrt(2*V0*e/m)

    v1 = np.sqrt(2*V1*e/m)

    t_drift = L / ((v0 + v1)/2)
 
    # Convert angles to transverse velocities

    vx = xp * v0

    vy = yp * v0

    vz = v0 * np.ones_like(vx)
 
    pos = np.stack([x, y, np.zeros_like(x)], axis=0)

    vel = np.stack([vx, vy, vz], axis=0)
 
    # Determine number of neighbor collisions

    n_collisions = max(int(beam_current / 1e-7), base_neighbors)
 
    for offset in range(1, n_collisions+1):

        i = np.arange(N - offset)

        j = i + offset
 
        # Relative positions and velocities

        r0 = pos[:, j] - pos[:, i]

        vrel = vel[:, j] - vel[:, i]
 
        # Transverse impact parameter

        b = np.linalg.norm(r0[:2,:], axis=0) + 1e-15

        v_mag = np.linalg.norm(vrel, axis=0) + 1e-15
 
        # Finite-drift velocity kick from analytical integral

        # dv_perp = (k_e * e^2) / (m * b * v) * L / sqrt(b^2 + L^2)

        dv_perp = (k_e * e**2) / (m * b * v_mag) * (t_drift * v_mag) / np.sqrt(b**2 + (t_drift * v_mag)**2)
 
        # Unit vector in transverse plane

        ux = r0[0,:] / b

        uy = r0[1,:] / b
 
        dvx = dv_perp * ux

        dvy = dv_perp * uy
 
        # Transverse displacement during drift

        dx_offset = dvx * t_drift

        dy_offset = dvy * t_drift
 
        # Update velocities

        vel[0, i] += dvx

        vel[1, i] += dvy

        vel[0, j] -= dvx

        vel[1, j] -= dvy
 
        # Update positions

        pos[0, i] += dx_offset

        pos[1, i] += dy_offset

        pos[0, j] -= dx_offset

        pos[1, j] -= dy_offset
 
    # Final drift

    pos += vel * t_drift
 
    # Convert back to angles at final energy

    xp_new = vel[0,:] / v1

    yp_new = vel[1,:] / v1
 
    # Update dE/E for acceleration

    d_new = d * V0 / V1
 
    # Return updated particle array

    X_new = np.vstack([pos[0,:], xp_new, pos[1,:], yp_new, d_new])

    return X_new

def estimate_required_neighbors(

        X,

        z_positions,

        vx,

        vy,

        vz0,

        d,

        t_nom,

        max_neighbors=40,

        tol=0.02,

        n_time_samples=17):
 
    x = X[0]

    y = X[2]

    N = len(vx)
 
    cumulative_dvx = np.zeros(N)

    cumulative_dvy = np.zeros(N)
 
    prev_sigma = 0.0
 
    for k in range(1, max_neighbors + 1):
 
        i = np.arange(0, N - k)

        j = i + k

        if len(i) == 0:

            break
 
        r0 = np.vstack([

            x[i] - x[j],

            y[i] - y[j],

            z_positions[i] - z_positions[j]

        ])
 
        vz_dev_i = 0.5 * vz0 * d[i]

        vz_dev_j = 0.5 * vz0 * d[j]
 
        vrel = np.vstack([

            vx[i] - vx[j],

            vy[i] - vy[j],

            vz_dev_i - vz_dev_j

        ])
 
        dv, _ = pair_collision_kick_and_offset_vectorized(

            r0, vrel, t_nom, n_time_samples

        )
 
        cumulative_dvx[i] += dv[0]

        cumulative_dvy[i] += dv[1]
 
        cumulative_dvx[j] -= dv[0]

        cumulative_dvy[j] -= dv[1]
 
        sigma = np.std(np.sqrt(cumulative_dvx**2 + cumulative_dvy**2))
 
        if k > 1:

            if abs(sigma - prev_sigma) / (prev_sigma + 1e-16) < tol:

                return k
 
        prev_sigma = sigma
 
    return max_neighbors

def drift_with_ordered_collisions(

        X,

        L,

        V0=100.0,

        V1=None,

        beam_current=0.0,

        base_neighbors=1,

        current_scale=1e-6,

        n_time_samples=10):

    """

    Drift with ordered pairwise Coulomb collisions and optional axial acceleration.

    Phase space layout (5 x N):

        X[0] = x

        X[1] = x'

        X[2] = y

        X[3] = y'

        X[4] = d = dE / E_nom

    What this version does:

    1) Computes the nominal time of flight through the drift for acceleration V0 -> V1

    2) Computes pairwise momentum transfer and position offset using the evolving

       relative trajectory in the colliding-pair frame

    3) Adds all pair contributions linearly, independently, on top of the nominal motion

    4) Rescales x', y', and d for the change in nominal acceleration potential

    """

    if V1 is None:

        V1 = V0

    X_new = X.copy()

    N = X_new.shape[1]

    x  = X_new[0]

    xp = X_new[1]

    y  = X_new[2]

    yp = X_new[3]

    d  = X_new[4]

    e = 1.602e-19

    # -----------------------------------------

    # Nominal accelerating drift

    # -----------------------------------------

    t_nom, vz0, vz1, az = nominal_tof_uniform_accel(L, V0, V1)

    # Under pure axial acceleration, transverse velocity is unchanged

    # so slopes scale as 1 / vz

    angle_scale = vz0 / vz1

    # If absolute dE is conserved while nominal energy changes,

    # then (dE / E_nom) rescales inversely with the nominal energy

    delta_scale = V0 / V1 if V1 != 0 else 1.0

    # Physical transverse velocities before acceleration scaling

    vx = xp * vz0

    vy = yp * vz0

    # Nominal drift offsets from time-of-flight

    x_nom = x + vx * t_nom

    y_nom = y + vy * t_nom

    # If no current, just return the accelerated nominal drift

    if beam_current == 0:

        xp_out = xp * angle_scale

        yp_out = yp * angle_scale

        d_out  = d  * delta_scale

        return np.vstack([x_nom, xp_out, y_nom, yp_out, d_out])

    # -----------------------------------------

    # Longitudinal emission ordering

    # -----------------------------------------

    mean_dt = e / beam_current

    mean_dz = vz0 * mean_dt

    dz_random = np.random.exponential(mean_dz, N)

    z_positions = np.cumsum(dz_random)

    # -----------------------------------------

    # Smooth neighbour scaling

    # -----------------------------------------

    #neighbors = base_neighbors + int(beam_current / current_scale)

    #neighbors = max(neighbors, 1)

    neighbors = estimate_required_neighbors(

        X_new,

        z_positions,

        vx,

        vy,

        vz0,

        d,

        t_nom

    )

    print("drifting using "+str(neighbors) + " collision neighbours")

    # -----------------------------------------

    # Linearly accumulated pairwise collision terms

    # -----------------------------------------

    dvx = np.zeros(N)

    dvy = np.zeros(N)

    dvz = np.zeros(N)

    dx_coll = np.zeros(N)

    dy_coll = np.zeros(N)

    # -----------------------------------------

    # Vectorized ordered collisions

    # -----------------------------------------

    dvx = np.zeros(N)

    dvy = np.zeros(N)

    dvz = np.zeros(N)
 
    dx_coll = np.zeros(N)

    dy_coll = np.zeros(N)
 
    for k in range(1, neighbors + 1):
 
        i = np.arange(0, N - k)

        j = i + k

        if len(i) == 0:

            break
 
        r0 = np.vstack([

            x[i] - x[j],

            y[i] - y[j],

            z_positions[i] - z_positions[j]

        ])
 
        vz_dev_i = 0.5 * vz0 * d[i]

        vz_dev_j = 0.5 * vz0 * d[j]
 
        vrel = np.vstack([

            vx[i] - vx[j],

            vy[i] - vy[j],

            vz_dev_i - vz_dev_j

        ])
 
        dv, dr = pair_collision_kick_and_offset_vectorized(

            r0, vrel, t_nom, n_time_samples

        )
 
        dvx[i] += dv[0]

        dvy[i] += dv[1]

        dvz[i] += dv[2]
 
        dvx[j] -= dv[0]

        dvy[j] -= dv[1]

        dvz[j] -= dv[2]
 
        dx_coll[i] += dr[0]

        dy_coll[i] += dr[1]
 
        dx_coll[j] -= dr[0]

        dy_coll[j] -= dr[1]

    # -----------------------------------------

    # Final state at the end of the accelerated section

    # -----------------------------------------

    x_out = x_nom + dx_coll

    y_out = y_nom + dy_coll

    # Final axial velocity includes nominal acceleration + summed pair kicks

    vz_out = np.maximum(vz1 + dvz, 1e-9)

    # Final slopes are physical transverse velocity divided by final axial velocity

    vx_out = vx + dvx

    vy_out = vy + dvy

    xp_out = (vx_out) / vz_out

    yp_out = (vy_out) / vz_out

    # Rescaled chromatic coordinate

    #d_out = d * delta_scale

    # -----------------------------------------

    # Boersch energy broadening

    # -----------------------------------------

    m = 9.109e-31
 
    # Total kinetic energy after collisions

    E_final = 0.5 * m * (vx_out**2 + vy_out**2 + vz_out**2)
 
    # Nominal kinetic energy after acceleration

    E_nominal = 0.5 * m * vz1**2
 
    # Chromatic coordinate relative to nominal energy

    d_out = d*delta_scale+(E_final - E_nominal) / E_nominal


    return np.vstack([x_out, xp_out, y_out, yp_out, d_out])

# -------------------------

# 5) FW50

# -------------------------

def compute_fw50(X):

    r = np.sqrt(X[0]**2 + X[2]**2)

    return np.percentile(r, 50)
 


# -------------------------

# 6) Phase-space scatter plot

# -------------------------

def plot_phase_space(X, axes=(0,1), title="Phase Space"):

    a = X[axes[0]]

    b = X[axes[1]]

    delta = X[4]

    plt.figure(figsize=(6,5))

    sc = plt.scatter(a*1e9, b*1e3, c=delta*1e3, s=1, cmap='turbo')

    plt.xlabel("x (nm)")

    plt.ylabel("x' (mrad)")

    plt.title(title)

    plt.colorbar(sc, label="Energy deviation (‰)")

    plt.tight_layout()

    plt.show()

# -------------------------------------------------

# Effective Gaussian fit model

# -------------------------------------------------

def fw50_model(z, sigma_min, alpha_eff, z0):

    return 1.349 * np.sqrt(sigma_min**2 + (alpha_eff * (z - z0))**2)


# -------------------------------------------------

# Through-focus measurement around focal plane

# -------------------------------------------------

def through_focus_fw50(

        X_focus,

        z_span=2e-3,

        n_steps=41,

        source_sigma=20e-9,

        source_alpha=1e-3,

        plot =False):

    z_steps = np.linspace(-z_span/2, z_span/2, n_steps)

    fw50_sim = []

    # Linearized around focus

    for dz in z_steps:

        Xz = drift(X_focus.copy(), dz)

        fw50_sim.append(compute_fw50(Xz))

    fw50_sim = np.array(fw50_sim)

    # -------------------------------------------

    # Fit effective Gaussian to simulation

    # -------------------------------------------

    idx_min = np.argmin(fw50_sim)

    sigma_guess = fw50_sim[idx_min] / 1.349

    alpha_guess = source_alpha

    z0_guess = z_steps[idx_min]

    popt, _ = curve_fit(

        fw50_model,

        z_steps,

        fw50_sim,

        p0=[sigma_guess, alpha_guess, z0_guess]

    )

    sigma_fit, alpha_fit, z0_fit = popt

    fw50_fit = fw50_model(z_steps, *popt)

    # -------------------------------------------

    # Analytical ideal Gaussian

    # -------------------------------------------

    fw50_analytic = 1.349 * np.sqrt(

        source_sigma**2 + (source_alpha * z_steps)**2

    )

    # -------------------------------------------

    # Plot

    # -------------------------------------------

    if plot:

        plt.figure(figsize=(7,5))

        plt.plot(z_steps*1e3, fw50_sim*1e9, label="Simulation")

        plt.plot(z_steps*1e3, fw50_analytic*1e9, "--", label="Ideal Gaussian")

        plt.plot(z_steps*1e3, fw50_fit*1e9, ":", label="Fit to Simulation")

        plt.xlabel("Defocus around focus (mm)")

        plt.ylabel("FW50 radius (nm)")

        plt.title("Through-Focus FW50 Comparison")

        plt.legend()

        plt.tight_layout()

        plt.show()

    # -------------------------------------------

    # Diagnostics

    # -------------------------------------------

    print("\n---- Effective Gaussian Fit ----")

    print(f"Minimum radius (fit): {sigma_fit*1e9:.2f} nm")

    print(f"Effective divergence: {alpha_fit*1e3:.3f} mrad")

    print(f"Focal plane shift: {z0_fit*1e6:.3f} um")

    return {

        "z": z_steps,

        "fw50_sim": fw50_sim,

        "fw50_fit": fw50_fit,

        "fw50_analytic": fw50_analytic,

        "sigma_fit": sigma_fit,

        "alpha_fit": alpha_fit,

        "z0_fit": z0_fit

    }

def plot_phase_space_scatter_clipped(

        X,

        axes=(0,1),

        title=None,

        fw50_scale=2.0,

        fw50_scale_E =3.0,

        s=0.15,

        alpha=0.5,

        c=None):
 
    data = X[list(axes), :]
 
    # --- FW50 helper ---

    def fw50(arr):

        med = np.median(arr)

        r = np.abs(arr - med)

        sorted_r = np.sort(r)

        return sorted_r[int(0.5 * len(sorted_r))]
 
    # Axis limits

    fw50_x = fw50(data[0])

    fw50_y = fw50(data[1])

    xlim = fw50_scale * fw50_x

    ylim = fw50_scale * fw50_y
 
    # Colour data

    if c is None:

        color_data = X[4] * 1e3   # energy in ‰

        c_label = "Energy deviation (‰)"

    else:

        color_data = c

        c_label = "Value"
 
    # Colour clipping range

    fw50_c = fw50(color_data)

    clim = fw50_scale_E * fw50_c
 
    # Identify in-range and out-of-range points

    in_range = np.abs(color_data) <= clim

    out_range = ~in_range
 
    plt.figure(figsize=(5,5))
 
    # Plot in-range with colormap

    sc = plt.scatter(

        data[0][in_range],

        data[1][in_range],

        c=color_data[in_range],

        s=s,

        alpha=alpha,

        cmap='turbo',

        vmin=-clim,

        vmax=clim

    )
 
    # Plot outliers in black

    if np.any(out_range):

        plt.scatter(

            data[0][out_range],

            data[1][out_range],

            c='black',

            s=s,

            alpha=alpha

        )
 
    plt.xlim(-xlim, xlim)

    plt.ylim(-ylim, ylim)
 
    plt.colorbar(sc, label=c_label)
 
    plt.xlabel(f"Axis {axes[0]}")

    plt.ylabel(f"Axis {axes[1]}")
 
    if title:

        plt.title(title)
 
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":

    N = 200000

    # Initial source

    X = generate_beam(N,sigma_r = 1e-7,sigma_angle=1.00e-2,sigma_delta=2e-2)

    Current=0*1e-10

    L = 1e-2

    #X = drift_with_ordered_collisions(X,  1e-1,         V0=1000.0,beam_current=Current)

    for number in range(6):

        fac  = 1/2**number

        X = multipole_lens(X, f=L/2,Cs=fac/L**3,C5=0)#,Cs=.1/L**3, C5=0)

        X = drift_with_ordered_collisions(X,  L,         V0=1000.0,beam_current=Current)
 
    #X = drift(X, 1e-1)  # long drift before lens

    #X = multipole_lens(X,f=L/2)

    #X = drift_with_ordered_collisions(X,  L,         V0=1000.0,beam_current=Current)

    #X = multipole_lens(X,f=L/2)3

    #X = drift_with_ordered_collisions(X,  L,         V0=1000.0,beam_current=Current)

    #X = multipole_lens(X,f=L/2)

    X_focus = drift_with_ordered_collisions(X, L-1.5e-3-178e-6-1.5e-3,         V0=1000.0,beam_current=Current)

    # Drift to focus with space-charge

    #z_to_focus = 1 / (1/3e-3 - 1/1e-1) # thin-lens formula

    #X_focus = drift_with_ordered_collisions(X, z_to_focus*1-0.00139-2e-6+0.076e-6*10,V0=1000.0,V1=100.0, beam_current=Current)
 
    

    #X_focus = drift_with_ordered_weak_collisions(X, z_to_focus*20, beam_current=0)

    #X_focus = drift(X, z_to_focus*1)

    #plot_phase_space(X_focus, axes=(0,1), title="1 keV SEM Phase Space at Focus (x, x')")

    plot_phase_space_scatter_clipped(X_focus, axes=(0,1), title="x vs x' (clipped to 2x FW50)", fw50_scale=3.0)

    results = through_focus_fw50(

        X_focus,

        z_span=L,

        n_steps=300,

        source_sigma=20e-9,

        source_alpha=1e-3

    )

    # Phase-space at focus

    """

    # -------------------------------------------------

    # Through-focus FW50 series (linearized around focus)

    # -------------------------------------------------

    z_steps = np.linspace(-3e-4, 3e-4, 60)  # 0.3 mm around focus

    fw50_series = []

    for dz in z_steps:

        # Linearized drift around focus

        Xz = drift(X_focus.copy(), dz)

        fw50_series.append(compute_fw50(Xz))

    # Analytical Gaussian for comparison

    sigma0 = 20e-9

    alpha0 = 1e-3

    fw50_analytic = 1.349 * np.sqrt(sigma0**2 + (alpha0 * z_steps)**2)

    """

    # Plot through-focus

    plt.figure(figsize=(6,5))

    plt.plot(results["z"]*1e3, results["fw50_sim"]*1e9, label="Simulated FW50")

    #plt.plot(results["z"]*1e3, results["fw50_analytic"]*1e9, '--', label="Analytical Gaussian")

    plt.plot(results["z"]*1e3, results["fw50_fit"]*1e9, '--', label="Analytical Gaussian")

    plt.xlabel("Defocus around focus (mm)")

    plt.ylabel("FW50 radius (nm)")

    plt.title("Through-Focus FW50 (linearized around focus)")

    plt.legend()

    plt.tight_layout()

    plt.show()
 