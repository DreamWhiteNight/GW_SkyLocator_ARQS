import pandas as pd
import healpy as hp
import numpy as np
import ligo.skymap
import argparse
import matplotlib.pyplot as plt

from matplotlib import rcParams
from pathlib import Path
from ligo.skymap import io, kde, postprocess
from ligo.skymap.plot.marker import reticle

import h5py

##f1 = h5py.File("/fred/oz016/Chayan/GW-SkyNet/evaluation/Bayestar_comparison/Injection_run_BNS_3_det_design_test_0_sec_SNR-8-40_NSIDE-64_ResNet-2D_Bayestar_test_2_0.hdf", "r")
#f1 = h5py.File("/fred/oz016/Chayan/GW-SkyLocator/evaluation/Adaptive_NSIDE/Negative_latency/Bayestar_comparison_post-merger/New/Injection_run_BBH_3_det_design_test_Gaussian_KDE_0.hdf", "r")
'''
f1 = h5py.File('/Users/slash/SNR_time_series_sample_files/O3_noise/BBH/bank_3/O3_noise_GW170817_BBH_3_det_parameters_3.hdf','r')
probs = f1["Probabilities"][()]
ra_preds = f1["RA_samples"][()]
dec_preds = f1["Dec_samples"][()]
ra_test = f1["RA_test"][()]
dec_test = f1["Dec_test"][()]

f1.close()

# Do this only for the new version of the code:
ra_test = np.squeeze(ra_test.T)
dec_test = np.squeeze(dec_test.T)

ra_preds = np.where(ra_preds > 2.0*np.pi, 2.0*np.pi, ra_preds)
ra_preds = np.where(ra_preds < 0.0, 0.0, ra_preds)

dec_preds = np.where(dec_preds > np.pi/2, np.pi/2, dec_preds)
dec_preds = np.where(dec_preds < -np.pi/2, -np.pi/2, dec_preds)

pts = np.stack([ra_preds, dec_preds], axis=2)

sky_posterior = []
'''
#f1 = h5py.File('/home/slash/bank_3/O3_noise_GW170817_BBH_3_det_parameters_3.hdf','r')
#f1 = h5py.File('/mnt/data/slash/Inj_Data/Inj_Data_zoo_parameters_P-P_TEST.hdf','r')
#f1 = h5py.File('/mnt/data/slash/Inj_Data/Inj_Data_zoo_SpinTaylorT4_parameters_NSBH_TEST.hdf','r')
f1 = h5py.File(f'/mnt/data/slash/Inj_Data/BAYESTAR_INJECTION/TEST/HDF_DATA/Inj_Data_Bayestar_test_Parm.hdf','r')
#f1 = h5py.File(f'/mnt/data/slash/Inj_Data/Inj_Data_Bayestar_test_Parm.hdf','r')
ra_test = 2.0 * np.pi * f1['ra_test'][()]
dec_test = np.arcsin(1.0 - 2.0 * f1['dec_test'][()])
#ra_test = f1['ra'][()]
#dec_test = f1['dec'][()]
inj_snr  = f1['Injection_SNR'][()]
snr_filter = np.logical_and(inj_snr > 0 , inj_snr < 100)

ra_test = ra_test[snr_filter]
dec_test = dec_test[snr_filter]

ra_test = np.squeeze(ra_test.T)


dec_test = np.squeeze(dec_test.T)
import random

#index = random.sample(range(12000), 2000)

#for i in range(len(ra_test)): #pts.shape[0]
#    sky_posterior.append(kde.Clustered2DSkyKDE(pts[i], trials=1, jobs=20))
    
#hpmap = []

#for i in range(len(ra_test)):
#    hpmap.append(sky_posterior[i].as_healpix())
    
#for i in range(len(ra_test)):
#    io.write_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/BNS/skymaps/Test_'+str(i)+'.fits', hpmap[i], nest=True)
num = 100
fail = 0
lis = [4,8,16]
fig = plt.figure(figsize=(5, 5))
#ax.add_lightning(2000, 20) # Add some random realizations of n samples
dir_p = 'NSBH'
#dir_p = '10_02_RESNET34_Paper'
for k in range(1):
    skymap = []

    cl = []
    cl_2 = []
    area_90 = []
    area_50 = []
    area_90_2 = []
    area_50_2 = []
    search_area = []
    search_area_2 = []

    eps = 1e-5

    for i in range(num): #pts.shape[0]-1
    #    s, metadata = io.fits.read_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/NSBH/skymaps/Gaussian_KDE/Test_3_bij_lr_schedule_'+str(i)+'.fits', nest=None)
    #    s, metadata = io.fits.read_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/Pre-merger/New/45_secs/Test_3_bij_50_epochs_lr_schedule_'+str(i)+'.fits', nest=None)

        #s, metadata = io.fits.read_sky_map('/fred/oz016/Chayan/GW-SkyLocator/evaluation/skymaps/CPU/BBH/skymaps/Gaussian_KDE/Test_new_BN_'+str(i)+'.fits', nest=None)
        #s, metadata = io.fits.read_sky_map(f'/home/slash/train_time_cross/Test_{3}_{i}.fits', nest=True)
        #s, metadata = io.fits.read_sky_map(f'/home/slash/{dir_p}/Test_{k+2}_{i}.fits', nest=True)
        try:
            t, metadata_2 = io.fits.read_sky_map(f'/home/slash/Paper_Inj/Test_2_{i}.fits',nest = True)
            s, metadata = io.fits.read_sky_map(f'/mnt/data/slash/Inj_Data/BAYESTAR_INJECTION/TEST/FITS/Bayestar_coinc{i}.fits/'+'0.fits', nest = True)
            skymap = s
            
            # Convert to probability per square degree
            nside = hp.npix2nside(len(skymap))
            deg2perpix = hp.nside2pixarea(nside, degrees=True)
            probperdeg2 = skymap / deg2perpix
            
            nside_2 = hp.npix2nside(len(t))
            deg2perpix_2 = hp.nside2pixarea(nside_2, degrees=True)
            probperdeg2_2 = t / deg2perpix_2
            event_ra = ra_test[i]
            event_de = dec_test[i]

            vmax = probperdeg2.max()
            vmin = probperdeg2.min()
            #print(np.sum(skymap))
            confidence_levels = postprocess.find_greedy_credible_levels(skymap)
            confidence_levels_2 = postprocess.find_greedy_credible_levels(t)
            cl.append(confidence_levels[hp.ang2pix(nside,-event_de+np.pi/2,event_ra, nest=True)])
            cl_2.append(confidence_levels_2[hp.ang2pix(nside_2,-event_de+np.pi/2,event_ra, nest=True)])
            #cl.append(confidence_levels[hp.ang2pix(nside,event_de % np.pi,event_ra, nest=True)])
            area_90.append(np.sum(confidence_levels <= 0.9*np.sum(skymap)) * hp.nside2pixarea(nside, degrees=True) + eps)
            area_50.append(np.sum(confidence_levels <= 0.5*np.sum(skymap)) * hp.nside2pixarea(nside, degrees=True) + eps)
            area_90_2.append(np.sum(confidence_levels_2 <= 0.9*np.sum(t)) * hp.nside2pixarea(nside_2, degrees=True) + eps)
            area_50_2.append(np.sum(confidence_levels_2 <= 0.5*np.sum(t)) * hp.nside2pixarea(nside_2, degrees=True) + eps)
            print('Done',i)
            search_area.append(np.sum(confidence_levels <= cl[i-fail]*np.sum(skymap)) * hp.nside2pixarea(nside, degrees=True) + eps)
            search_area_2.append(np.sum(confidence_levels_2 <= cl_2[i-fail]*np.sum(t)) * hp.nside2pixarea(nside_2, degrees=True) + eps)
        except Exception as e:
            print(e)
            fail += 1
            num -= 1
            continue
    from tabulate import tabulate

    table = [["90%",np.min(area_90),np.argmin(area_90)],
            ["50%",np.min(area_50),np.argmin(area_50)],
            ["Searched Area",np.min(search_area),np.argmin(search_area)]]
    table_2 = [["90%",np.min(area_90_2),np.argmin(area_90_2)],
            ["50%",np.min(area_50_2),np.argmin(area_50_2)],
            ["Searched Area",np.min(search_area_2),np.argmin(search_area_2)]]

    print(tabulate(table, headers=["Percentage", "Minimum area", "Minimum index"]))
    print(tabulate(table_2, headers=["Percentage", "Minimum area", "Minimum index"]))
    plt.hist(np.log10(area_90),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',density=True,label='90%, median='+str(round(np.median(area_90),2)))
    plt.hist(np.log10(area_50),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',density=True,label='50%, median='+str(round(np.median(area_50),2)))
    plt.hist(np.log10(search_area),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',density=True,label='Search, median='+str(round(np.median(search_area),2)))
    plt.hist(np.log10(area_90_2),50,range=(1,np.max(np.log10(search_area_2))),cumulative=True,histtype='step',density=True,label='90%, median='+str(round(np.median(area_90_2),2)),linestyle='--')
    plt.hist(np.log10(area_50_2),50,range=(1,np.max(np.log10(search_area_2))),cumulative=True,histtype='step',density=True,label='50%, median='+str(round(np.median(area_50_2),2)),linestyle='--')
    plt.hist(np.log10(search_area_2),50,range=(1,np.max(np.log10(search_area_2))),cumulative=True,histtype='step',density=True,label='Search, median='+str(round(np.median(search_area_2),2)),linestyle='--')
    '''
    plt.hist(np.log10(area_90),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',label='BAYESTAR 90%',density=True)
    plt.hist(np.log10(area_50),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',label='BAYESTAR 50%',density=True)
    #plt.hist(np.log10(search_area),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',label='90%',density=True)
    plt.hist(np.log10(area_90_2),50,range=(1,np.max(np.log10(search_area_2))),cumulative=True,histtype='step',label='SkyLocator 90%',density=True,linestyle='--')
    plt.hist(np.log10(area_50_2),50,range=(1,np.max(np.log10(search_area_2))),cumulative=True,histtype='step',label='SkyLocator 50%',density=True,linestyle='--')
    '''
    #plt.hist(np.log10(search_area_2),50,range=(1,np.max(np.log10(search_area_2))),cumulative=True,histtype='step',label='Search_median',density=True,linestyle='--')
    plt.legend(loc=4)
    plt.ylabel('Cumulative Ratio')
    plt.xlabel('Area in log(deg^2)')
    plt.savefig('Area_Test.png', dpi=400)
    #f1 = h5py.File('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/Pre-merger/New/Injection_run_BNS_3_det_design_test_45_sec_SNR-8-40_Adaptive_Gaussian_KDE_3_bij_50_epochs_lr_sch_bandw_003_stdscale.hdf', 'w')
'''
f1 = h5py.File('/fred/oz016/Chayan/GW-SkyLocator/evaluation/skymaps/CPU/BBH/Injection_run_BBH_3_det_design_test_Gaussian_KDE_new_BN.hdf', 'w')
f1.create_dataset('Area-90', data=area_90)
f1.create_dataset('Area-50', data=area_50)
f1.create_dataset('Area-Searched', data=search_area)
#f1.create_dataset('Index', data=index)

f1.close()
'''
record = open('CL_BAY.txt','w')
record.write(str(cl))
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='pp_plot')
ax.add_series(cl) # Add our data
ax.add_confidence_band(num,alpha=0.95) # Add 90% confidence band
ax.add_diagonal() # Add diagonal line`
#ax.legend(['3N+SNR','SNR_Cross+3N','SNR'])
plt.xlabel('Credible intervals')
plt.ylabel('Cumulative distribution')
plt.savefig('CL_Test.png', dpi=400)

print('Done!')
