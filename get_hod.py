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
hexcolors_bright = ['#CC3311','#0077BB','#EE7733','#BBBBBB','#33BBEE','#EE3377','#0099BB']

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/";fp_dm = 'fp'
#tng_dir = "/mnt/alan1/boryanah/MTNG/dm_arepo/";fp_dm = 'dm'
#gal_types = ['LBG', 'LAE']
gal_types = ['LAE']
#n_gals = ['5.0e-04', '1.0e-03']
n_gals = ['1.0e-02']
snapshots = [129, 94]
zs = [2., 3.]

# plotting parameters
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

# loop over snapshots
for i, snapshot in enumerate(snapshots):
    # redshift
    z = zs[i]
    z_label = f"z = {z:.1f}"
    print(z)
    
    # load other halo properties
    #GroupPos_fp = np.load(tng_dir+f'data_{fp_dm}/GroupPos_{fp_dm}_{snapshot:d}.npy')
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

            # save counts per halo
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

            # probability
            want_probs = False
            if want_probs:
                # whether a halo has both a central and a satellite or just a satellite
                choice_anysat = ((count_sats_halo > 0)).astype(int)
                choice_acent = ((count_cent_halo == 1)).astype(int)
                choice_anysat_acent = ((count_sats_halo > 0) & (count_cent_halo == 1)).astype(int)
                choice_nosat_acent = ((count_sats_halo == 0) & (count_cent_halo == 1)).astype(int)
                choice_anysat_nocent = ((count_sats_halo > 0) & (count_cent_halo == 0)).astype(int)
                choice_nosat_nocent = ((count_sats_halo == 0) & (count_cent_halo == 0)).astype(int)

                hist_anysat, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=choice_anysat)
                hist_acent, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=choice_acent)
                hist_anysat_acent, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=choice_anysat_acent)
                hist_nosat_acent, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=choice_nosat_acent)
                hist_anysat_nocent, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=choice_anysat_nocent)
                hist_nosat_nocent, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=choice_nosat_nocent)
                hist_norm, _ = np.histogram(GrMcrit_fp, bins=mbins)

                # probability that a halo at a given mass bin holds blah
                prob_anysat = hist_anysat/hist_norm
                prob_acent = hist_acent/hist_norm
                prob_anysat_acent = hist_anysat_acent/hist_norm
                prob_nosat_acent = hist_nosat_acent/hist_norm
                prob_anysat_nocent = hist_anysat_nocent/hist_norm
                prob_nosat_nocent = hist_nosat_nocent/hist_norm

                np.savez(f"data/{gal_type:s}_{n_gal:s}_{fp_dm}_{snapshot:d}.npz", mbinc=mbinc, prob_acent=prob_acent, prob_anysat=prob_anysat, prob_acent_given_anysat=prob_anysat_acent/prob_anysat, prob_anysat_given_acent=prob_anysat_acent/prob_acent)

            # histograms
            hist_gal, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=count_halo)
            hist_cent_gal, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=count_cent_halo)
            hist_halo, _ = np.histogram(GrMcrit_fp, bins=mbins)
            hod_cent_gal = hist_cent_gal/hist_halo
            hod_gal = hist_gal/hist_halo
            hod_sat_gal = hod_gal-hod_cent_gal

            std, _, _ = stats.binned_statistic(GrMcrit_fp, count_halo-count_cent_halo, statistic='std', bins=mbins)
            #std, _, _ = stats.binned_statistic(GrMcrit_fp, count_sats_halo*(count_sats_halo-1), statistic='mean', bins=mbins); std = np.sqrt(std) # alternative definition
            poisson = np.sqrt(hod_sat_gal)
            poiss_up, poiss_dw = hod_sat_gal + poisson, hod_sat_gal - poisson
            np.savez(f"data/hod_{n_gal}_{gal_type}_{snapshot}.npz", hod_cent_gal=hod_cent_gal, mbinc=mbinc, hod_sat_gal=hod_sat_gal, std=std, poisson=poisson)
            print("std = ", std)
            print("poisson = ", poisson)
            print("percentage difference = ", 100.*(std-poisson)/std)
