"""
Plot HOD and Poisson noise
"""
import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
fp_dm = 'fp'
#gal_types = ['LBG', 'LAE']
gal_types = ['LAE']
#n_gals = ['5.0e-04', '1.0e-03']
n_gals = ['1.0e-02']
snapshots = [129, 94]
zs = [2., 3.]

# loop over snapshots
for i, snapshot in enumerate(snapshots):
    # redshift
    z = zs[i]
    z_label = f"z = {z:.1f}"
    print(z)
    
    # load other halo properties
    GrMcrit_fp = np.load(tng_dir+f'data_{fp_dm}/Group_M_TopHat200_{fp_dm}_{snapshot:d}.npy')*1.e10
    SubhaloGrNr = np.load(tng_dir+f"data_{fp_dm}/SubhaloGroupNr_{fp_dm}_{snapshot:d}.npy")

    # max halo mass
    print("max halo mass = %.1e"%GrMcrit_fp.max())

    # identify central subhalos
    _, sub_inds_cent = np.unique(SubhaloGrNr, return_index=True)

    # loop over galaxy types
    for gal_type in gal_types:
        # define label
        gal_label = "{\\rm "+f"{gal_type}s"+"}"
        print(gal_type)

        # loop over number density
        for n_gal in n_gals:
            # indices of the galaxies
            if "arepo" in tng_dir:
                index = np.load(f"data/index_{gal_type:s}_{n_gal}_{snapshot:d}_dm_arepo.npy")
            else:
                index = np.load(f"data/index_{gal_type:s}_{n_gal}_{snapshot:d}.npy")

            # which galaxies are centrals
            index_cent = np.intersect1d(index, sub_inds_cent)

            # galaxy properties
            grnr_gal = SubhaloGrNr[index]
            grnr_cent_gal = SubhaloGrNr[index_cent]

            # count unique halo repetitions
            grnr_gal_uni, cts = np.unique(grnr_gal, return_counts=True)
            count_halo = np.zeros(len(GrMcrit_fp), dtype=int)
            count_halo[grnr_gal_uni] = cts
            grnr_cent_gal_uni, cts = np.unique(grnr_cent_gal, return_counts=True)
            count_cent_halo = np.zeros(len(GrMcrit_fp), dtype=int)
            count_cent_halo[grnr_cent_gal_uni] = cts

            # save counts per halo (generally useful)
            want_save = True
            if want_save:
                np.save(tng_dir+f"data_{fp_dm}/GroupCount{gal_type:s}_{n_gal:s}_{fp_dm}_{snapshot:d}.npy", count_halo)
                np.save(tng_dir+f"data_{fp_dm}/GroupCentsCount{gal_type:s}_{n_gal:s}_{fp_dm}_{snapshot:d}.npy", count_cent_halo)

            # define mass bins
            mbins = np.logspace(10.5, 15, 41)
            mbinc = (mbins[1:]+mbins[:-1]) * 0.5

            # satellite counts
            count_sats_halo = count_halo - count_cent_halo
            print("satellite fraction", np.sum(count_sats_halo)/np.sum(count_halo))

            # histograms
            hist_gal, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=count_halo)
            hist_cent_gal, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=count_cent_halo)
            hist_halo, _ = np.histogram(GrMcrit_fp, bins=mbins)
            hod_cent_gal = hist_cent_gal/hist_halo
            hod_gal = hist_gal/hist_halo
            hod_sat_gal = hod_gal-hod_cent_gal

            # std around the HOD mean (don't worrya bout this)
            std, _, _ = stats.binned_statistic(GrMcrit_fp, count_halo-count_cent_halo, statistic='std', bins=mbins)
            poisson = np.sqrt(hod_sat_gal)
            np.savez(f"data/hod_{n_gal}_{gal_type}_{snapshot}.npz", hod_cent_gal=hod_cent_gal, mbinc=mbinc, hod_sat_gal=hod_sat_gal, std=std, poisson=poisson)
            print("std = ", std)
            print("poisson = ", poisson)
            print("percentage difference = ", 100.*(std-poisson)/std)
