"""
Plot HOD and Poisson noise
"""
import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

# colors
hexcolors_bright = ['#CC3311','#0077BB','#EE7733','limegreen','#BBBBBB','#33BBEE','#EE3377','#0099BB']

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
#gal_types = ['LBG', 'LAE']
gal_types = ['LAE']
n_gal = '1.0e-02' #'1.0e-03' '5.0e-04'
snapshots = [129, 94]
zs = [2., 3.]
p1, p2 = n_gal.split('e-0')

# definitions for the axes
left, width = 0.14, 0.85
bottom, height = 0.1, 0.25
spacing = 0.005
rect_scatter = [left, bottom + (height + spacing), width, 0.6]
rect_histx = [left, bottom, width, height]

# start with a rectangular Figure
plt.figure(figsize=(9, 10))
ax_scatter = plt.axes(rect_scatter)
#(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)

counter = 0
for i, snapshot in enumerate(snapshots):
    z = zs[i]
    z_label = f"z = {z:.1f}"
    print(z_label)
    for gal_type in gal_types:
        gal_label = "{\\rm "+f"{gal_type}s"+"}"
        print(gal_label)
        data = np.load(f"data/hod_{n_gal}_{gal_type}_{snapshot}.npz")
        hod_cent_gal = data['hod_cent_gal']
        mbinc = data['mbinc']
        hod_sat_gal = data['hod_sat_gal']
        std = data['std']
        poisson = data['poisson']

        alpha  = (poisson/std)**2
        print(alpha)
        print(mbinc)
                
        print("std = ", std)
        print("poisson = ", poisson)
        print("percentage difference = ", 100.*(std-poisson)/std)
        
        ax_scatter.plot(mbinc, hod_cent_gal, color=hexcolors_bright[counter], ls='--', label=rf"${z_label}, \ {gal_label}$")
        ax_scatter.plot(mbinc, hod_sat_gal, color=hexcolors_bright[counter], ls='-')

        ax_histx.axhline(y=1, color='black', ls='--', lw=3.5)
        ax_histx.plot(mbinc, std/poisson, color=hexcolors_bright[counter], ls='-', lw=2.5)
        
        counter += 1
label = r"$n_{\rm gal} = %s \times 10^{-%s}$"%(p1, p2)
ax_scatter.set_xscale('log')
ax_scatter.set_yscale('log')
ax_scatter.legend()
ax_scatter.set_xlim([1.e10, 2.e14])
ax_scatter.set_ylim([1.e-4, 2.e1])
ax_scatter.set_xticklabels([])
ax_scatter.text(0.1, 0.8, s=label, transform=ax_scatter.transAxes)
ax_scatter.set_ylabel(r'$\langle N_{\rm gal} \rangle$')
ymin, ymax = ax_histx.get_ylim()
ymin = np.floor(ymin*10.)/10.
ymax = np.ceil(ymax*10.)/10.
ax_histx.set_yticks(np.arange(ymin, ymax, 0.1))
ax_histx.grid(color='gray', linestyle='-', linewidth=1.)
ax_histx.minorticks_on()    
ax_histx.set_xlabel(r'$M_{\rm halo} \ [M_\odot/h]$')
ax_histx.set_ylabel(r'$\sqrt{{\rm Var}[N_{\rm sat}]}/\sqrt{\langle N_{\rm sat} \rangle}$')
ax_histx.set_ylim([0.77, 1.23])
ax_histx.set_xlim([1.e10, 2.e14])
ax_histx.set_xscale('log')
plt.savefig(f"figs/hod_{n_gal}.png", bbox_inches='tight', pad_inches=0.)
plt.show()
