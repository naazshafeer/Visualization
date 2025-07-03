import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import astropy.constants as const
import astropy.units as u
import math


G = const.G.to(u.pc**3/(u.kg*u.yr**2))  # Gravitational constant in pc³/(kg·yr²)
pc_to_km = u.pc.to(u.km)                 # 1 pc in km
yr_to_s = u.yr.to(u.s)                   # 1 yr in seconds

# Unit conversion factors
km_s_to_pc_yr = (u.km/u.s).to(u.pc/u.yr)  # Velocity conversion
pc_yr_to_km_s = 1/km_s_to_pc_yr           # Inverse conversion

# ==================================================================
# 2. PHYSICAL PARAMETERS
# ==================================================================
# Galaxy properties
m_gal = 1e11 * const.M_sun                # Galaxy mass
m_gal_val = m_gal.to(u.kg).value          # In kg

# Black hole properties
m1 = 4e8 * const.M_sun                    # BH A mass
m2 = 4e8 * const.M_sun                    # BH B mass
m1_val = m1.to(u.kg).value                # In kg
m2_val = m2.to(u.kg).value                # In kg

# Initial separation
sep = 430 * u.pc                          # Initial separation
sep_val = sep.value                       # In pc

# ==================================================================
# 3. INITIAL CONDITIONS
# ==================================================================
# Positions (pc)
r1_initial = np.array([sep_val/2, 0, 0])   # BH A initial position
r2_initial = np.array([-sep_val/2, 0, 0])  # BH B initial position

# Circular velocity (km/s)
v_circ_km_s = np.sqrt(G * (m1 + m2) / sep).to(u.km/u.s).value

# Convert to pc/yr for integration
v_circ_pc_yr = v_circ_km_s * km_s_to_pc_yr

# Velocities (pc/yr)
v1_initial = np.array([0, v_circ_pc_yr, 0])   # BH A initial velocity
v2_initial = np.array([0, -v_circ_pc_yr, 0])  # BH B initial velocity

# Initial state vector
y0 = np.concatenate([r1_initial, r2_initial, v1_initial, v2_initial])

# ==================================================================
# 4. DYNAMICAL FRICTION FUNCTIONS
# ==================================================================
def velocity_dispersion(m_gal_kg):
    """Velocity dispersion in pc/yr for given galaxy mass in kg"""
    m_ratio = m_gal_kg / (1e11 * const.M_sun.to(u.kg).value)
    v_kms = (10**2.2969) * (m_ratio)**0.299
    return v_kms * km_s_to_pc_yr  # Convert to pc/yr

def coulomb_logarithm(r_gal_pc, sigma_pc_yr, G_val, m_val):
    """Coulomb logarithm calculation"""
    x = (r_gal_pc * sigma_pc_yr**2) / (G_val * m_val)
    return math.log10(x) if x > 0 else 3.0  # Minimum reasonable value

def dynamical_friction_acceleration(r_gal_pc, v_gal_pcyr, ln_Lambda, G_val, m_val):
    """DF acceleration in pc/yr²"""
    v_mag = np.linalg.norm(v_gal_pcyr)
    if v_mag == 0:
        return np.zeros(3)
    
    unit_v = v_gal_pcyr / v_mag
    a_df_mag = 0.428 * ln_Lambda * (G_val * m_val) / r_gal_pc**2
    return -a_df_mag * unit_v  # Opposite to motion

def calculate_DF(G_val, m_gal_val, r1_pc, r2_pc, v1_pcyr, v2_pcyr):
    """Calculate DF accelerations relative to galactic center"""
    # Velocity dispersion
    sigma_pc_yr = velocity_dispersion(m_gal_val)
    
    # Distance to galactic center (origin)
    r1_gal = np.linalg.norm(r1_pc)
    r2_gal = np.linalg.norm(r2_pc)
    
    # Coulomb logarithm
    ln_Lambda1 = coulomb_logarithm(r1_gal, sigma_pc_yr, G_val, m1_val)
    ln_Lambda2 = coulomb_logarithm(r2_gal, sigma_pc_yr, G_val, m2_val)
    
    # DF accelerations
    a_df1 = dynamical_friction_acceleration(r1_gal, v1_pcyr, ln_Lambda1, G_val, m1_val)
    a_df2 = dynamical_friction_acceleration(r2_gal, v2_pcyr, ln_Lambda2, G_val, m2_val)
    
    return a_df1, a_df2

# ==================================================================
# 5. EQUATIONS OF MOTION
# ==================================================================
def dydt(t, y, G_val, m1_val, m2_val, m_gal_val, df_on=True):
    """Differential equations for the binary system"""
    # Unpack state vector
    r1 = y[0:3]    # BH1 position (pc)
    r2 = y[3:6]    # BH2 position (pc)
    v1 = y[6:9]    # BH1 velocity (pc/yr)
    v2 = y[9:12]   # BH2 velocity (pc/yr)
    
    # Vector between BHs
    r12 = r2 - r1
    r12_mag = np.linalg.norm(r12)
    
    # Gravitational accelerations
    a1_grav = (G_val * m2_val / r12_mag**3) * r12
    a2_grav = -(G_val * m1_val / r12_mag**3) * r12
    
    # Dynamical friction
    a1_df, a2_df = np.zeros(3), np.zeros(3)
    if df_on:
        a1_df, a2_df = calculate_DF(G_val, m_gal_val, r1, r2, v1, v2)
    
    # Total accelerations
    a1_total = a1_grav + a1_df
    a2_total = a2_grav + a2_df
    
    return np.concatenate([v1, v2, a1_total, a2_total])

# ==================================================================
# 6. INTEGRATION AND SIMULATION
# ==================================================================
# Time parameters (100 Myr simulation)
t_span = (0, 100e6)  # 100 Myr in years
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Run simulation with DF
print("Running simulation with dynamical friction...")
sol_df = solve_ivp(
    fun=lambda t, y: dydt(t, y, G.value, m1_val, m2_val, m_gal_val, df_on=True),
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method='DOP853',
    rtol=1e-10,
    atol=1e-10
)

# Run simulation without DF for comparison
print("Running simulation without dynamical friction...")
sol_no_df = solve_ivp(
    fun=lambda t, y: dydt(t, y, G.value, m1_val, m2_val, m_gal_val, df_on=False),
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method='DOP853',
    rtol=1e-10,
    atol=1e-10
)

# ==================================================================
# 7. ANALYSIS AND PLOTTING
# ==================================================================
def analyze_solution(sol):
    """Extract positions and separation from solution"""
    r1 = sol.y[0:3].T
    r2 = sol.y[3:6].T
    sep = np.linalg.norm(r1 - r2, axis=1)
    return r1, r2, sep

# Process results
r1_df, r2_df, sep_df = analyze_solution(sol_df)
r1_no_df, r2_no_df, sep_no_df = analyze_solution(sol_no_df)
time_myr = t_eval / 1e6  # Convert time to Myr

# Plot orbits
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(r1_no_df[:, 0], r1_no_df[:, 1], 'b-', alpha=0.7, label='BH1 (no DF)')
plt.plot(r2_no_df[:, 0], r2_no_df[:, 1], 'r-', alpha=0.7, label='BH2 (no DF)')
plt.title("Orbits Without Dynamical Friction")
plt.xlabel("X [pc]")
plt.ylabel("Y [pc]")
plt.axis('equal')
plt.legend()

plt.subplot(132)
plt.plot(r1_df[:, 0], r1_df[:, 1], 'b-', alpha=0.7, label='BH1 (with DF)')
plt.plot(r2_df[:, 0], r2_df[:, 1], 'r-', alpha=0.7, label='BH2 (with DF)')
plt.title("Orbits With Dynamical Friction")
plt.xlabel("X [pc]")
plt.ylabel("Y [pc]")
plt.axis('equal')
plt.legend()

# Plot separation
plt.subplot(133)
plt.semilogy(time_myr, sep_no_df, 'k--', label='Without DF')
plt.semilogy(time_myr, sep_df, 'r-', label='With DF')
plt.xlabel("Time [Myr]")
plt.ylabel("Separation [pc]")
plt.title("Binary Separation Evolution")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('binary_evolution.png', dpi=300)
plt.show()

# ==================================================================
# 8. FINAL DIAGNOSTICS
# ==================================================================
# Calculate final separation
final_sep = sep_df[-1]
print(f"\nFinal separation with DF: {final_sep:.2f} pc")
print(f"Initial separation: {sep_val} pc")

# Calculate energy loss
initial_energy = -G.value * m1_val * m2_val / sep_val
final_energy = -G.value * m1_val * m2_val / final_sep
energy_loss = (initial_energy - final_energy) / initial_energy
print(f"Energy loss: {energy_loss*100:.2f}%")

# Calculate merger timescale
hardening_radius = 10 * u.pc  # When stellar hardening becomes dominant
hardening_time = (final_sep**4) / (hardening_radius.value**3 * 1e7)  # Simplified estimate
print(f"Estimated time to hardening: {hardening_time:.2e} Myr")