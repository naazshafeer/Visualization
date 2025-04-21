import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import scipy as sci
from scipy import integrate
from scipy.integrate import odeint
from astropy import units as u
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import astropy.constants as const
import math


def semimajor_axis(r1_, r2_):
    '''
    This function calculated the semi-major axis. In this case, r=a
    '''
    return (np.linalg.norm(r2_ - r1_) / 2)
def distance_to_com(r_, rcom_):
    '''
    This function calculates the distance from the center of mass
    '''
    return np.linalg.norm(r_ - rcom_)
def sigma(m_gal_):
    '''
    This function calculates the velocity dispersion of the galaxy
    '''
    return (10**(2.2969)*(m_gal_/ (10**(11) * const.M_sun))**(0.299)) * (u.km/u.s)
def coulomb_log(dist_, o_, G_, m_):

    '''
    Got this from implement df, making sure that both m1, and m2 can be used here
    '''
    x = (dist_ * o_**(2)) / (G_ * m_)
    return math.log10(x.to_value(u.dimensionless_unscaled))
def DF_force(ln_A_, m_, m_gal_, r_choice_, rcom_, vfirst_, vsecond_, G_):
    '''
    Now we will only be looking at the force and then apply the acceleration later.
    Also added first and second to make sure that the negative signs are applied correctly
    '''
    F_D = 0.428 * ln_A_ * ((G_ * m_**(2))/(np.linalg.norm(r_choice_ - rcom_))**(2))

    v_r = vfirst_ - vsecond_

    v_rmag = np.linalg.norm(v_r) 

    v_rel_unit = (v_r / v_rmag)

    F_D_i = (F_D * v_rel_unit[0])
    F_D_j = (F_D * v_rel_unit[1])
    F_D_k = (F_D * v_rel_unit[2])
    
    # np.array([F_D_i.value, F_D_j.value, F_D_k.value])

    return np.array([F_D_i.to((u.kg*u.m)/(u.s**2)).value, F_D_j.to((u.kg*u.m)/ (u.s**2)).value, F_D_k.to((u.kg*u.m)/ (u.s**2)).value])
def twobodyeqn(w, t, G_, m1_, m2_, m_gal_, rcom_, r_, b_=1):
    '''
    used the two body equation from initial summer runs
    '''
    r1 = w[:3] * u.pc
    r2 = w[3:6] * u.pc
    v1 = w[6:9] * u.km / u.s
    v2 = w[9:12] * u.km / u.s
    

    r_vec = r2 - r1
    r = np.linalg.norm(r_vec)
    

    a = semimajor_axis(r1, r2)
    o = sigma(m_gal_) 

    dist1 = distance_to_com(r1, rcom_)
    dist2 = distance_to_com(r2, rcom_)

    ln_A1 = coulomb_log(dist1, o, G_, m1_)
    ln_A2 = coulomb_log(dist2, o, G_, m2_)
    

    df_force1 = DF_force(ln_A1, m1_, m_gal_, r1, rcom_, v2, v1, G_) * u.kg * u.m / u.s**2
    df_force2 = DF_force(ln_A2, m2_, m_gal_, r2, rcom_, v1, v2, G_) * u.kg * u.m / u.s**2
    

    df_accel1 = (df_force1 / m1_).to(u.km / u.s**2)
    df_accel2 = (df_force2 / m2_).to(u.km / u.s**2)
    

    dv1bydt = (G_ * m2_ * r_vec / r**3).to(u.km / u.s**2) 
    dv2bydt = (-G_ * m1_ * r_vec / r**3).to(u.km / u.s**2) 
    

    dr1bydt = v1.to(u.km / u.s)
    dr2bydt = v2.to(u.km / u.s)
    
    r_derivs = np.concatenate((dr1bydt, dr2bydt))
    derivs = np.concatenate((r_derivs, dv1bydt, dv2bydt))

    
    return derivs

