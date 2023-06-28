import numpy as np
import Corrfunc
import matplotlib.pyplot as plt

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
fp_dm = 'fp'
gal_type = 'LBG'
n_gal = '1.0e-02'
snapshot = 129
boxsize = 500.
nthreads = 16

# load pos
SubhaloPos = np.load(tng_dir+f"data_{fp_dm}/SubhaloPos_{fp_dm}_{snapshot:d}.npy")
bins = np.logspace(-1, 1.4, 21)
binc = 0.5*(bins[1:] + bins[:-1])

# load galaxy indices
if "arepo" in tng_dir:
    index = np.load(f"/home/boryanah/MTNG/LAE/data/index_{gal_type:s}_{n_gal}_{snapshot:d}_dm_arepo.npy")
else:
    index = np.load(f"/home/boryanah/MTNG/LAE/data/index_{gal_type:s}_{n_gal}_{snapshot:d}.npy")
    
# galaxy properties
gal_pos = SubhaloPos[index]

# compute correlation function
results = Corrfunc.theory.xi(boxsize, nthreads, bins, gal_pos[:, 0], gal_pos[:, 1], gal_pos[:, 2])
Xi = results['xi']

# plot results
plt.figure(figsize=(9, 7))
plt.plot(binc, binc**2*Xi)
plt.xscale('log')
plt.show()
