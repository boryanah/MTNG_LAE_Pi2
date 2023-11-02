import numpy as np
import Corrfunc
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
fp_dm = 'fp'
gal_type = "LAE"#'LBG'
n_gal = '1.0e-02'
#n_gal = '5.0e-04'
snapshot = 129 #94 #129
boxsize = 500.
nthreads = 32 #16
if snapshot == 129: redshift = 2.
elif snapshot == 94: redshift = 3.

# load pos
SubhaloPos = np.load(tng_dir+f"data_{fp_dm}/SubhaloPos_{fp_dm}_{snapshot:d}.npy")
SubhaloVel = np.load(tng_dir+f"data_{fp_dm}/SubhaloVel_{fp_dm}_{snapshot:d}.npy")
pos_parts = np.load(tng_dir+f"data_parts/pos_down_100000_snap_{snapshot:d}_{fp_dm}.npy")

bins = np.logspace(-1, 1.4, 21)
#bins = np.logspace(-1, 1.8, 21)
binc = 0.5*(bins[1:] + bins[:-1])

# load galaxy indices
if "arepo" in tng_dir:
    index = np.load(f"/home/boryanah/MTNG/LAE/data/index_{gal_type:s}_{n_gal}_{snapshot:d}_dm_arepo.npy")
else:
    index = np.load(f"/home/boryanah/MTNG/LAE/data/index_{gal_type:s}_{n_gal}_{snapshot:d}.npy")
    
# galaxy properties
gal_pos = SubhaloPos[index]
gal_vel = SubhaloVel[index]

# set up cosmology
h = 0.6774
cosmo = FlatLambdaCDM(H0=h*100, Om0=0.3089, Tcmb0=2.725)
H_z = cosmo.H(redshift).value
print("H(z) = ", H_z)

# adding RSDs
pos_rsd = gal_pos.copy()
pos_rsd[:, 2] += gal_vel[:, 2]*(1.+redshift)/H_z*h # Mpc/h
pos_rsd %= boxsize
np.save(f"data/{gal_type}_pos.npy", gal_pos)
np.save(f"data/{gal_type}_pos_rsd.npy", pos_rsd)
quit()

# compute correlation function
results = Corrfunc.theory.xi(boxsize, nthreads, bins, gal_pos[:, 0], gal_pos[:, 1], gal_pos[:, 2])
Xi = results['xi']

# compute correlation function
results = Corrfunc.theory.xi(boxsize, nthreads, bins, pos_parts[:, 0], pos_parts[:, 1], pos_parts[:, 2])
Xi_matter = results['xi']

# plot results
plt.figure(figsize=(9, 7))
plt.plot(binc, np.sqrt(Xi/Xi_matter))
plt.xscale('log')

# plot results
plt.figure(figsize=(9, 7))
plt.plot(binc, binc**2*Xi)
plt.xscale('log')
plt.show()
