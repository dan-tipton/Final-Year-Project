from astropy.modeling.powerlaws import Schechter1D
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import fitting
import astropy.units as u


class Schechter():

    def __init__(self, phi_star, m_star, alpha):
        self.phi_star = phi_star # in number density units, e.g., Mpc^-3
        self.m_star = m_star # characteristic magnitude
        self.alpha = alpha # faint-end slope
        self.M_sun = 4.83
        self.L_sun = 1
        self.L_star = 1e10 #3e11 # WHAT IS L⋆?: ANATOMY OF THE GALAXY LUMINOSITY FUNCTION
    
    def convert(self,lum):
        mag = self.M_sun - 2.5 * np.log10(lum / self.L_sun)
        return mag 

    def magntiude(self, mags=None):
        model = Schechter1D(phi_star=self.phi_star, m_star=self.m_star, alpha=self.alpha)
        if mags == None:
            mags = np.linspace(-24, -16, 100)
        return model(mags), mags
    
    def luminosity(self, lums=None):
        # dimensionless luminosity
        if lums == None:
            lums = np.logspace(8, 12, 100)  # luminosities from 1e8 to 1e12
        x = lums / self.L_star
        dn = phi_star * x**alpha * np.exp(-x)
        return dn, lums


phi_star = 4.3e-4 * (u.Mpc ** -3)
m_star = -20.26
alpha = -1.98
sch = Schechter(phi_star, m_star, alpha)

phi_mag, mags = sch.magntiude()
phi_lum, lums = sch.luminosity()

fig, (ax_mag, ax_lum) = plt.subplots(1, 2, figsize=(8,5))

# Plot
ax_mag.plot(mags, phi_mag)
ax_mag.set_xlabel("Magnitude")
ax_mag.set_ylabel("Number Density")
ax_mag.invert_xaxis()  # Magnitudes are brighter to the left
ax_mag.set_yscale('log')

ax_lum.plot(lums, phi_lum)
ax_lum.set_xlabel("Luminosity")
ax_lum.set_ylabel("dn/dL")
ax_lum.set_yscale('log')
ax_lum.set_xscale('log')

plt.show()

"""
# Example data
mag_data = np.array([-23, -21, -20, -19, -17])
phi_data = np.array([0.0001, 0.0005, 0.001, 0.002, 0.003])
# Initial guess
model_guess = Schechter1D(phi_star=0.001, m_star=-20, alpha=-1.0)
# Fitter
fitter = fitting.LevMarLSQFitter()
fitted_model = fitter(model_guess, mag_data, phi_data)
print(f"Fitted phi_star: {fitted_model.phi_star.value}")
print(f"Fitted m_star: {fitted_model.m_star.value}")
print(f"Fitted alpha: {fitted_model.alpha.value}")
"""