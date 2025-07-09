import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.constants import G
import matplotlib.pyplot as plt

# Constants
G_pc_yr = G.to(u.pc**3 / (u.kg * u.yr**2))  # G in pc^3 kg^-1 yr^-2
delta_t = 1000 * u.yr  # Time step
N = 10000  # Number of steps
time_values = np.arange(0, N) * delta_t.to(u.Myr).value  # Time array

# Initial conditions (example values - replace with your actual values)
m1 = 1e8 * const.M_sun
m2 = 1e8 * const.M_sun
m_gal = 1e11 * const.M_sun
r1 = np.array([100, 0, 0]) * u.pc
r2 = np.array([-100, 0, 0]) * u.pc
v1 = np.array([0, 126, 0]) * u.km/u.s
v2 = np.array([0, -126, 0]) * u.km/u.s

# Convert velocities to pc/yr
v1 = v1.to(u.pc/u.yr)
v2 = v2.to(u.pc/u.yr)

# Storage arrays
r1_sol, r2_sol = [r1.copy()], [r2.copy()]
v1_sol, v2_sol = [v1.copy()], [v2.copy()]
a_df1_sol, a_df2_sol = [], []
F_df1_sol, F_df2_sol = [], []

# ==============================
# CORRECTED DF FUNCTIONS
# ==============================
def velocity_dispersion(m_gal):
    """Velocity dispersion relative to galaxy center (origin)"""
    v_disp = (10**(2.2969)) * (m_gal / (10**11 * const.M_sun))**(0.299) * (u.km/u.s)
    return v_disp.to(u.pc/u.yr)

def coulomb_logarithm(r_mag, sigma, G, m_bh):
    """Coulomb logarithm using natural log (standard)"""
    arg = (r_mag * sigma**2) / (G * m_bh)
    return np.log(arg.to(u.dimensionless_unscaled))

def df_acceleration(r_mag, v_vec, ln_Lambda, G, m_bh):
    """DF acceleration opposite to velocity vector (galaxy frame)"""
    v_mag = np.linalg.norm(v_vec)
    if v_mag > 0:
        unit_v = v_vec / v_mag
    else:
        unit_v = np.zeros(3) * (u.pc/u.yr)
    
    # Chandrasekhar formula (magnitude)
    a_df_mag = 0.428 * ln_Lambda * (G * m_bh) / r_mag**2
    
    # Vector acceleration opposite to velocity
    return -a_df_mag * unit_v

def calculate_df(G, m_bh, r_vec, v_vec, m_gal):
    """Compute DF acceleration for one black hole"""
    sigma = velocity_dispersion(m_gal)
    r_mag = np.linalg.norm(r_vec)
    ln_Lambda = coulomb_logarithm(r_mag, sigma, G, m_bh)
    return df_acceleration(r_mag, v_vec, ln_Lambda, G, m_bh)
# ==============================

# Main simulation loop
for i in range(N):
    # Current positions and velocities
    r1_val = r1_sol[-1]
    r2_val = r2_sol[-1]
    v1_val = v1_sol[-1]
    v2_val = v2_sol[-1]
    
    # Gravitational acceleration between BHs
    r12 = r2_val - r1_val
    r12_mag = np.linalg.norm(r12)
    a1_grav = (G_pc_yr * m2 / r12_mag**3) * r12
    a2_grav = (G_pc_yr * m1 / r12_mag**3) * (-r12)
    
    # Dynamical friction (relative to galaxy center)
    a1_df = calculate_df(G_pc_yr, m1, r1_val, v1_val, m_gal)
    a2_df = calculate_df(G_pc_yr, m2, r2_val, v2_val, m_gal)
    
    # Total acceleration
    a1_total = a1_grav + a1_df
    a2_total = a2_grav + a2_df
    
    # Update velocities (Euler integration)
    v1_new = v1_val + a1_total * delta_t
    v2_new = v2_val + a2_total * delta_t
    
    # Update positions
    r1_new = r1_val + v1_new * delta_t
    r2_new = r2_val + v2_new * delta_t
    
    # Store results
    r1_sol.append(r1_new)
    r2_sol.append(r2_new)
    v1_sol.append(v1_new)
    v2_sol.append(v2_new)
    a_df1_sol.append(a1_df.value)
    a_df2_sol.append(a2_df.value)
    F_df1_sol.append((m1 * a1_df).to(u.kg * u.m / u.s**2))
    F_df2_sol.append((m2 * a2_df).to(u.kg * u.m / u.s**2))

# Convert lists to arrays for plotting
r1_sol = np.array([r.value for r in r1_sol]) * u.pc
r2_sol = np.array([r.value for r in r2_sol]) * u.pc
v1_sol = np.array([v.value for v in v1_sol]) * (u.pc/u.yr)
v2_sol = np.array([v.value for v in v2_sol]) * (u.pc/u.yr)
a_df1_sol = np.array(a_df1_sol) * (u.pc/u.yr**2)
a_df2_sol = np.array(a_df2_sol) * (u.pc/u.yr**2)

# Plotting (same as your snippet)
plt.figure(figsize=(14, 12))
time_values.shape

