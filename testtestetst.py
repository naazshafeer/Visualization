import astropy 
from astropy import units as u
from astropy import constants as const
import numpy as np
import matplotlib.pyplot as plt

G = const.G
m1 = m2 = 4e8 * u.Msun
sep = 430 * u.pc
r1 = [0, sep.value/2, 0] * u.pc
r2 = [0, -sep.value/2, 0] * u.pc

orb_v1 = np.sqrt((G * m2 * r1) / (np.linalg.norm(r1-r2))**2).to(u.km/u.s)
print("Orbital velocity:", orb_v1)
print(np.linalg.norm(r1 - r2))
print(G)