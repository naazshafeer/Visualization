# %% [markdown]
# SMBH Binary Simulation with Dynamical Friction
# Features:
# - Symplectic integration with RK45
# - Toggle for dynamical friction
# - Consistent unit handling
# - Galaxy mass profile modeling

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sci
from scipy.integrate import solve_ivp
from astropy import units as u
import astropy.constants as const
import math

# Set plotting style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 12

# %% [markdown]
# ## Physical Constants and Initial Conditions

# %%
# Fundamental constants
G = const.G.to(u.pc**3 / (u.Msun * u.yr**2))  # Consistent units
m_gal = 1e11 * const.M_sun  # Galaxy mass

# Black hole masses (equal mass binary)
m1 = 4e8 * const.M_sun
m2 = 4e8 * const.M_sun

# Initial separation (430 pc)
sep = 430 * u.pc
a = 0.5 * sep  # Semi-major axis

# Initial positions
r1_initial = np.array([sep.value/2, 0, 0]) * u.pc
r2_initial = np.array([-sep.value/2, 0, 0]) * u.pc

# Initial velocities (circular orbit)
orb_v = np.sqrt(G * (m1 + m2) / a).to(u.km/u.s)
v1_initial = np.array([0, orb_v.value, 0]) * u.km/u.s
v2_initial = np.array([0, -orb_v.value, 0]) * u.km/u.s

# Center of mass
r_com = (m1 * r1_initial + m2 * r2_initial) / (m1 + m2)
v_com = (m1 * v1_initial + m2 * v2_initial) / (m1 + m2)

# Time parameters
T = 20 * u.Myr  # Simulation time
N = 500  # Number of steps
delta_t = T / N

# Dynamical friction parameters
a_0 = 2.95 
b_0 = 0.596
mstar_tot = 1e11 * u.Msun
r_eff = (a_0 * (mstar_tot / (1e6 * u.Msun))**b_0) * u.pc

# %% [markdown]
# ## Helper Functions

# %%
def velocity_disp(m_gal):
    """Velocity dispersion of the galaxy (km/s)"""
    return 10**2.2969 * (m_gal / (1e11 * const.M_sun))**0.299 * u.km/u.s

def km_s_to_pc_yr(velocity):
    """Convert km/s to pc/yr"""
    return velocity * 3.24078e-14 * 3.15576e7 * u.pc/u.yr

def pc_yr_to_km_s(velocity):
    """Convert pc/yr to km/s"""
    return velocity / (3.24078e-14 * 3.15576e7) * u.km/u.s

def coulomb_log(r, o, G, m):
    """Calculate Coulomb logarithm"""
    x = (r * o**2 / (G * m)).decompose()
    return np.log10(x.value) if x.value > 0 else 1.0

def dynamical_friction_accel(r, v_rel, m, o, G):
    """Dynamical friction acceleration"""
    lnA = coulomb_log(np.linalg.norm(r), o, G, m)
    v_mag = np.linalg.norm(v_rel)
    
    if v_mag > 1e-10:  # Avoid division by zero
        unit_v = v_rel / v_mag
        accel_mag = 0.428 * lnA * G * m / np.linalg.norm(r)**2
        return -accel_mag * unit_v
    return np.zeros(3) * u.pc/u.yr**2

# %% [markdown]
# ## ODE System

# %%
def smbh_system(t, y, m1_val, m2_val, G_val, df_on=True):
    """ODE system for SMBH binary with optional DF"""
    # Unpack state vector [r1, r2, v1, v2]
    r1 = y[0:3] * u.pc
    r2 = y[3:6] * u.pc
    v1 = y[6:9] * u.pc/u.yr
    v2 = y[9:12] * u.pc/u.yr
    
    # Center of mass
    r_com = (m1_val * r1 + m2_val * r2) / (m1_val + m2_val)
    v_com = (m1_val * v1 + m2_val * v2) / (m1_val + m2_val)
    
    # Relative position and distance
    r_rel = r2 - r1
    r_mag = np.linalg.norm(r_rel)
    
    # Gravitational accelerations
    a_grav1 = G_val * m2_val * r_rel / r_mag**3
    a_grav2 = -G_val * m1_val * r_rel / r_mag**3
    
    # Initialize DF accelerations
    a_df1 = np.zeros(3) * u.pc/u.yr**2
    a_df2 = np.zeros(3) * u.pc/u.yr**2
    
    if df_on:
        # Velocity dispersion (convert to pc/yr)
        o_val = velocity_disp(m_gal)
        o_val = km_s_to_pc_yr(o_val).value * u.pc/u.yr
        
        # Relative velocities to COM
        v1_rel = v1 - v_com
        v2_rel = v2 - v_com
        
        # Dynamical friction accelerations
        a_df1 = dynamical_friction_accel(r1 - r_com, v1_rel, m1_val, o_val, G_val)
        a_df2 = dynamical_friction_accel(r2 - r_com, v2_rel, m2_val, o_val, G_val)
    
    # Total accelerations
    a1_total = a_grav1 + a_df1
    a2_total = a_grav2 + a_df2
    
    # Derivatives: [dr1/dt, dr2/dt, dv1/dt, dv2/dt]
    return np.concatenate([
        v1.value, 
        v2.value, 
        a1_total.to(u.pc/u.yr**2).value, 
        a2_total.to(u.pc/u.yr**2).value
    ])

# %% [markdown]
# ## Simulation Execution

# %%
# Convert to numerical values in consistent units
G_val = G.value
m1_val = m1.to_value(u.Msun)
m2_val = m2.to_value(u.Msun)

# Initial state vector [r1, r2, v1, v2]
r1_val = r1_initial.to_value(u.pc)
r2_val = r2_initial.to_value(u.pc)
v1_val = km_s_to_pc_yr(v1_initial).to_value(u.pc/u.yr)
v2_val = km_s_to_pc_yr(v2_initial).to_value(u.pc/u.yr)

y0 = np.concatenate([r1_val, r2_val, v1_val, v2_val])

# Time grid (in years)
t_span = [0, T.to_value(u.yr)]
t_eval = np.linspace(t_span[0], t_span[1], N)

# Run simulation with DF ON
sol_df = solve_ivp(
    fun=lambda t, y: smbh_system(t, y, m1_val, m2_val, G_val, df_on=True),
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8,
    atol=1e-8
)

# Run simulation with DF OFF
sol_no_df = solve_ivp(
    fun=lambda t, y: smbh_system(t, y, m1_val, m2_val, G_val, df_on=False),
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8,
    atol=1e-8
)

# Extract solutions
r1_df = sol_df.y[0:3].T * u.pc
r2_df = sol_df.y[3:6].T * u.pc
r1_no_df = sol_no_df.y[0:3].T * u.pc
r2_no_df = sol_no_df.y[3:6].T * u.pc
time_myr = sol_df.t * u.yr.to(u.Myr)

# %% [markdown]
# ## Analysis and Visualization

# %%
# Separation calculations
sep_df = np.linalg.norm(r1_df - r2_df, axis=1)
sep_no_df = np.linalg.norm(r1_no_df - r2_no_df, axis=1)

# Plot separation comparison
plt.figure(figsize=(10, 6))
plt.plot(time_myr, sep_df, 'b-', label='With DF', linewidth=2)
plt.plot(time_myr, sep_no_df, 'r--', label='Without DF', linewidth=2)
plt.axhline(dynhard_rad.value, color='k', linestyle=':', label='Hardening Radius')
plt.xlabel('Time (Myr)')
plt.ylabel('Separation (pc)')
plt.title('SMBH Binary Separation Evolution')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('separation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 3D orbit plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# With DF
ax.plot(r1_df[:, 0], r1_df[:, 1], r1_df[:, 2], 'b-', label='BH1 (DF)', alpha=0.7)
ax.plot(r2_df[:, 0], r2_df[:, 1], r2_df[:, 2], 'c-', label='BH2 (DF)', alpha=0.7)

# Without DF
ax.plot(r1_no_df[:, 0], r1_no_df[:, 1], r1_no_df[:, 2], 'r--', label='BH1 (No DF)', alpha=0.5)
ax.plot(r2_no_df[:, 0], r2_no_df[:, 1], r2_no_df[:, 2], 'm--', label='BH2 (No DF)', alpha=0.5)

ax.set_xlabel('X (pc)')
ax.set_ylabel('Y (pc)')
ax.set_zlabel('Z (pc)')
ax.set_title('3D Orbits Comparison')
ax.legend()
plt.savefig('3d_orbits_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Final separation
final_sep_df = np.linalg.norm(r1_df[-1] - r2_df[-1])
final_sep_no_df = np.linalg.norm(r1_no_df[-1] - r2_no_df[-1])
print(f"Final separation with DF: {final_sep_df:.2f} pc")
print(f"Final separation without DF: {final_sep_no_df:.2f} pc")
print(f"Hardening radius: {dynhard_rad.value:.2f} pc")