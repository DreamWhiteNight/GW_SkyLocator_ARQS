from pycbc.detector import Detector
import numpy as np
import math
import pandas as pd
import seaborn as sns
import re
import random
import os
import time
import logging
import h5py
import healpy as hp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from astropy.utils import iers
from astropy.table import Table
from astropy import units as u

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import normflows as nf

import ligo.skymap.kde as KDE
from ligo.skymap import io
import astropy_healpix as ah
from ligo.skymap.kde import moc
from astropy.table import Table
from astropy import units as u
from ligo.skymap.core import nest2uniq, uniq2nest, uniq2order, uniq2pixarea, uniq2ang
from ligo.skymap.core import rasterize as _rasterize
from pycbc.detector import Detector
from scipy import signal

# Check for GPU availability
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#torch.cuda.set_device()
# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting script...")

# Disable Astropy IERS auto download
iers.conf.auto_download = False

def load_snr_and_injection_data(snr_file, params_file, idx_start=0, idx_end=None):
    with h5py.File(snr_file, 'r') as snr_f, h5py.File(params_file, 'r') as params_f:
        # Load SNR series
        h1_real = np.real(snr_f['h1_snr_series'][idx_start:idx_end][()])
        l1_real = np.real(snr_f['l1_snr_series'][idx_start:idx_end][()])
        v1_real = np.real(snr_f['v1_snr_series'][idx_start:idx_end][()])
        
        h1_imag = np.imag(snr_f['h1_snr_series'][idx_start:idx_end][()])
        l1_imag = np.imag(snr_f['l1_snr_series'][idx_start:idx_end][()])
        v1_imag = np.imag(snr_f['v1_snr_series'][idx_start:idx_end][()])
        mass_1 = params_f['mass1'][()]
        mass_2 = params_f['mass2'][()]
        spin_1 = params_f['spin1z'][()]
        spin_2 = params_f['spin2z'][()]
        # Load Injection_SNR and intrinsic parameters
        injection_snr = params_f['Injection_SNR'][idx_start:idx_end][()]
        ra = 2.0 * np.pi * params_f['ra'][idx_start:idx_end][()]
        dec = np.arcsin(1.0 - 2.0 * params_f['dec'][idx_start:idx_end][()])

    return h1_real, l1_real, v1_real, h1_imag, l1_imag, v1_imag, injection_snr, ra, dec, mass_1, mass_2, spin_1, spin_2


def load_snr_and_injection_data(snr_file, params_file, idx_start=0, idx_end=None):
    with h5py.File(snr_file, 'r') as snr_f, h5py.File(params_file, 'r') as params_f:
        # Load SNR series
        h1_real = np.real(snr_f['h1_snr_series'][idx_start:idx_end][()])
        l1_real = np.real(snr_f['l1_snr_series'][idx_start:idx_end][()])
        v1_real = np.real(snr_f['v1_snr_series'][idx_start:idx_end][()])
        
        h1_imag = np.imag(snr_f['h1_snr_series'][idx_start:idx_end][()])
        l1_imag = np.imag(snr_f['l1_snr_series'][idx_start:idx_end][()])
        v1_imag = np.imag(snr_f['v1_snr_series'][idx_start:idx_end][()])
        mass_1 = params_f['mass1'][()]
        mass_2 = params_f['mass2'][()]
        spin_1 = params_f['spin1z'][()]
        spin_2 = params_f['spin2z'][()]
        # Load Injection_SNR and intrinsic parameters
        injection_snr = params_f['Injection_SNR'][idx_start:idx_end][()]
        ra = 2.0 * np.pi * params_f['ra'][idx_start:idx_end][()]
        dec = np.arcsin(1.0 - 2.0 * params_f['dec'][idx_start:idx_end][()])

    return h1_real, l1_real, v1_real, h1_imag, l1_imag, v1_imag, injection_snr, ra, dec, mass_1, mass_2, spin_1, spin_2


def load_snr_and_injection_data_own(snr_file, params_file, idx_start=0, idx_end=None):
    with h5py.File(snr_file, 'r') as snr_f, h5py.File(params_file, 'r') as params_f:
        # Load SNR series
        h1_real = np.real(snr_f['h1_snr_series'][idx_start:idx_end][()])
        l1_real = np.real(snr_f['l1_snr_series'][idx_start:idx_end][()])
        v1_real = np.real(snr_f['v1_snr_series'][idx_start:idx_end][()])
        
        h1_imag = np.imag(snr_f['h1_snr_series'][idx_start:idx_end][()])
        l1_imag = np.imag(snr_f['l1_snr_series'][idx_start:idx_end][()])
        v1_imag = np.imag(snr_f['v1_snr_series'][idx_start:idx_end][()])
        mass_1 = params_f['mass1'][()]
        mass_2 = params_f['mass2'][()]
        spin_1 = params_f['spin1z'][()]
        spin_2 = params_f['spin2z'][()]
        # Load Injection_SNR and intrinsic parameters
        injection_snr = params_f['Injection_SNR'][idx_start:idx_end][()]
        ra = 2.0 * np.pi * params_f['ra_test'][idx_start:idx_end][()]
        dec = np.arcsin(1.0 - 2.0 * params_f['dec_test'][idx_start:idx_end][()])
    return h1_real, l1_real, v1_real, h1_imag, l1_imag, v1_imag, injection_snr, ra, dec, mass_1, mass_2, spin_1, spin_2
def load_snr_and_injection_data_own_ML4GW(snr_file, params_file, idx_start=0, idx_end=None):
    with h5py.File(snr_file, 'r') as snr_f, h5py.File(params_file, 'r') as params_f:
        # Load SNR series
        h1_real = np.real(snr_f['h1_snr_series'][idx_start:idx_end][()])
        l1_real = np.real(snr_f['l1_snr_series'][idx_start:idx_end][()])
        v1_real = np.real(snr_f['v1_snr_series'][idx_start:idx_end][()])
        
        h1_imag = np.imag(snr_f['h1_snr_series'][idx_start:idx_end][()])
        l1_imag = np.imag(snr_f['l1_snr_series'][idx_start:idx_end][()])
        v1_imag = np.imag(snr_f['v1_snr_series'][idx_start:idx_end][()])
        mass_1 = params_f['mass1'][()]
        mass_2 = params_f['mass2'][()]
        spin_1 = params_f['spin1z'][()]
        spin_2 = params_f['spin2z'][()]
        # Load Injection_SNR and intrinsic parameters
        #injection_snr = params_f['Injection_SNR'][idx_start:idx_end][()]
        ra = params_f['ra'][idx_start:idx_end][()]
        dec = params_f['dec'][idx_start:idx_end][()]

    return h1_real, l1_real, v1_real, h1_imag, l1_imag, v1_imag, ra, dec, mass_1, mass_2, spin_1, spin_2
# Load data for training
logging.info("Loading SNR and injection data for training...")
h1_real_1, l1_real_1, v1_real_1, h1_imag_1, l1_imag_1, v1_imag_1, injection_snr_1, ra_1, dec_1,mass_1_1, mass_1_2, spin_1_1, spin_1_2 = load_snr_and_injection_data(
    '/home/slash/O3_noise_snr_series_GW170817_BBH_3_det_0_secs_1.hdf',
    '/home/slash/O3_noise_GW170817_BBH_3_det_parameters_1.hdf',idx_end = 0)

h1_real_2, l1_real_2, v1_real_2, h1_imag_2, l1_imag_2, v1_imag_2, injection_snr_2, ra_2, dec_2, mass_2_1, mass_2_2, spin_2_1, spin_2_2 = load_snr_and_injection_data(
    '/home/slash/O3_noise_snr_series_GW170817_BBH_3_det_0_secs_2.hdf',
    '/home/slash/O3_noise_GW170817_BBH_3_det_parameters_2.hdf',idx_end = 0)
h1_real_5, l1_real_5, v1_real_5, h1_imag_5, l1_imag_5, v1_imag_5, injection_snr_5, ra_5, dec_5,mass_5_1, mass_5_2, spin_5_1, spin_5_2 = load_snr_and_injection_data(
    '/home/slash/bank_3/O3_noise_snr_series_GW170817_BBH_3_det_0_secs_3.hdf',
    '/home/slash/bank_3/O3_noise_GW170817_BBH_3_det_parameters_3.hdf',idx_end = 0)

h1_real_4, l1_real_4, v1_real_4, h1_imag_4, l1_imag_4, v1_imag_4, injection_snr_4, ra_4, dec_4, mass_4_1, mass_4_2, spin_4_1, spin_4_2 = load_snr_and_injection_data(
    '/home/slash/O3_noise_snr_series_GW170817_BBH_3_det_0_secs_4.hdf',
    '/home/slash/O3_noise_GW170817_BBH_3_det_parameters_4.hdf',idx_end = 0)

h1_real_5, l1_real_5, v1_real_5, h1_imag_5, l1_imag_5, v1_imag_5, ra_5, dec_5,mass_5_1, mass_5_2, spin_5_1, spin_5_2 = load_snr_and_injection_data_own_ML4GW(
    '/mnt/data/slash/Inj_Data/ML4GW_3.hdf',
    '/mnt/data/slash/Inj_Data/MLGW_3_parameters.hdf',idx_end = 0)
h1_real_5, l1_real_5, v1_real_5, h1_imag_5, l1_imag_5, v1_imag_5, ra_5, dec_5,mass_5_1, mass_5_2, spin_5_1, spin_5_2 = load_snr_and_injection_data_own_ML4GW(
    '/mnt/data/slash/Inj_Data/ML4GW_1.hdf',
    '/mnt/data/slash/Inj_Data/MLGW_1_parameters.hdf',idx_end = None)

h1_real_6, l1_real_6, v1_real_6, h1_imag_6, l1_imag_6, v1_imag_6, injection_snr_6, ra_6, dec_6,mass_6_1, mass_6_2, spin_6_1, spin_6_2 = load_snr_and_injection_data_own(
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_2.hdf',
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_2_parameters.hdf',idx_end = 0)

h1_real_7, l1_real_7, v1_real_7, h1_imag_7, l1_imag_7, v1_imag_7, injection_snr_7, ra_7, dec_7,mass_7_1, mass_7_2, spin_7_1, spin_7_2 = load_snr_and_injection_data_own(
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_3.hdf',
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_3_parameters.hdf',idx_end = 0)

h1_real_8, l1_real_8, v1_real_8, h1_imag_8, l1_imag_8, v1_imag_8, injection_snr_8, ra_8, dec_8,mass_8_1, mass_8_2, spin_8_1, spin_8_2 = load_snr_and_injection_data_own(
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_4.hdf',
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_4_parameters.hdf',idx_end = 0)
h1_real_9, l1_real_9, v1_real_9, h1_imag_9, l1_imag_9, v1_imag_9, injection_snr_9, ra_9, dec_9,mass_9_1, mass_9_2, spin_9_1, spin_9_2 = load_snr_and_injection_data_own(
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_3.hdf',
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_3_parameters.hdf',idx_end = 0)
h1_real_10, l1_real_10, v1_real_10, h1_imag_10, l1_imag_10, v1_imag_10, injection_snr_10, ra_10, dec_10,mass_10_1, mass_10_2, spin_10_1, spin_10_2 = load_snr_and_injection_data_own(
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_4.hdf',
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_4_parameters.hdf',idx_end = 0)
h1_real_11, l1_real_11, v1_real_11, h1_imag_11, l1_imag_11, v1_imag_11, injection_snr_11, ra_11, dec_11,mass_11_1, mass_11_2, spin_11_1, spin_11_2 = load_snr_and_injection_data_own(
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_4.hdf',
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_4_parameters.hdf',idx_end = 0)
h1_real_12, l1_real_12, v1_real_12, h1_imag_12, l1_imag_12, v1_imag_12, injection_snr_12, ra_12, dec_12,mass_12_1, mass_12_2, spin_12_1, spin_12_2 = load_snr_and_injection_data_own(
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_6.hdf',
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_6_parameters.hdf',idx_end = 0)
h1_real_13, l1_real_13, v1_real_13, h1_imag_13, l1_imag_13, v1_imag_13, injection_snr_13, ra_13, dec_13,mass_13_1, mass_13_2, spin_13_1, spin_13_2 = load_snr_and_injection_data_own(
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_7.hdf',
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_7_parameters.hdf',idx_end = 0)
h1_real_13, l1_real_13, v1_real_13, h1_imag_13, l1_imag_13, v1_imag_13, injection_snr_13, ra_13, dec_13,mass_13_1, mass_13_2, spin_13_1, spin_13_2 = load_snr_and_injection_data_own(
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_6.hdf',
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_6_parameters.hdf',idx_end = 0)
h1_real_14, l1_real_14, v1_real_14, h1_imag_14, l1_imag_14, v1_imag_14, injection_snr_14, ra_14, dec_14,mass_14_1, mass_14_2, spin_14_1, spin_14_2 = load_snr_and_injection_data_own(
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_6.hdf',
    '/mnt/data/slash/Inj_Data/Inj_Data_zoo_6_parameters.hdf',idx_end = 0)
# Concatenate data from different banks
'''
h1_real = np.concatenate([h1_real_1, h1_real_2,h1_real_3 ,h1_real_4], axis=0)
l1_real = np.concatenate([l1_real_1, l1_real_2,l1_real_3 ,l1_real_4], axis=0)
v1_real = np.concatenate([v1_real_1, v1_real_2,v1_real_3 ,v1_real_4], axis=0)

h1_imag = np.concatenate([h1_imag_1, h1_imag_2,h1_imag_3 ,h1_imag_4], axis=0)
l1_imag = np.concatenate([l1_imag_1, l1_imag_2,l1_imag_3 ,l1_imag_4], axis=0)
v1_imag = np.concatenate([v1_imag_1, v1_imag_2,v1_imag_3 ,v1_imag_4], axis=0)

injection_snr = np.concatenate([injection_snr_1, injection_snr_2,injection_snr_3 ,injection_snr_4], axis=0)

# Apply the filter: Only keep samples with Injection SNR > 12
logging.info("Filtering samples with Injection SNR > 12...")
snr_filter = injection_snr > 5

h1_real = h1_real[snr_filter]
l1_real = l1_real[snr_filter]
v1_real = v1_real[snr_filter]
h1_imag = h1_imag[snr_filter]
l1_imag = l1_imag[snr_filter]
v1_imag = v1_imag[snr_filter]

mass_1 = np.concatenate([mass_1_1, mass_2_1,mass_3_1, mass_4_1], axis=0)
mass_2 = np.concatenate([mass_1_2, mass_2_2,mass_3_2 ,mass_4_2], axis=0)
spin_1 = np.concatenate([spin_1_1, spin_2_1,spin_3_1 ,spin_4_1], axis=0)
spin_2 = np.concatenate([spin_1_2, spin_2_1,spin_3_2 ,spin_4_2], axis=0)


mass_1 = mass_1[:,None]
mass_2 = mass_2[:,None]
spin_1 = spin_1[:,None]
spin_2 = spin_2[:,None]
# Apply the same filter to the intrinsic parameters (ra, dec)
ra = np.concatenate([ra_1, ra_2,ra_3 ,ra_4], axis=0)
dec = np.concatenate([dec_1, dec_2,dec_3, dec_4], axis=0)
'''
h1_real = np.concatenate([h1_real_1, h1_real_2,h1_real_4,h1_real_5,h1_real_6,h1_real_7,h1_real_8,h1_real_9,h1_real_10,h1_real_11,h1_real_12,h1_real_13,h1_real_14], axis=0)
l1_real = np.concatenate([l1_real_1, l1_real_2,l1_real_4,l1_real_5,l1_real_6,l1_real_7,l1_real_8,l1_real_9,l1_real_10,l1_real_11,l1_real_12,l1_real_13,l1_real_14], axis=0)
v1_real = np.concatenate([v1_real_1, v1_real_2,v1_real_4,v1_real_5,v1_real_6,v1_real_7,v1_real_8,v1_real_9,v1_real_10,v1_real_11,v1_real_12,v1_real_13,v1_real_14], axis=0)

h1_imag = np.concatenate([h1_imag_1, h1_imag_2,h1_imag_4,h1_imag_5,h1_imag_6,h1_imag_7,h1_imag_8,h1_imag_9,h1_imag_10,h1_imag_11,h1_imag_12,h1_imag_13,h1_imag_14], axis=0)
l1_imag = np.concatenate([l1_imag_1, l1_imag_2,l1_imag_4,l1_imag_5,l1_imag_6,l1_imag_7,l1_imag_8,l1_imag_9,l1_imag_10,l1_imag_11,l1_imag_12,l1_imag_13,l1_imag_14], axis=0)
v1_imag = np.concatenate([v1_imag_1, v1_imag_2,v1_imag_4,v1_imag_5,v1_imag_6,v1_imag_7,v1_imag_8,v1_imag_9,v1_imag_10,v1_imag_11,v1_imag_12,l1_imag_13,l1_imag_14], axis=0)

injection_snr = np.concatenate([injection_snr_1, injection_snr_2,injection_snr_4,injection_snr_6,injection_snr_7,injection_snr_8,injection_snr_9,injection_snr_10,injection_snr_11,injection_snr_12,injection_snr_13,injection_snr_14], axis=0)

# Apply the filter: Only keep samples with Injection SNR > 12
logging.info("Filtering samples with Injection SNR > 15...")
snr_filter = np.logical_and(injection_snr > 5,injection_snr <35)
'''
h1_real = h1_real[snr_filter]
l1_real = l1_real[snr_filter]
v1_real = v1_real[snr_filter]
h1_imag = h1_imag[snr_filter]
l1_imag = l1_imag[snr_filter]
v1_imag = v1_imag[snr_filter]
'''
#snr_filter = snr_filter < 35
'''
h1_real = h1_real[snr_filter]
l1_real = l1_real[snr_filter]
v1_real = v1_real[snr_filter]
h1_imag = h1_imag[snr_filter]
l1_imag = l1_imag[snr_filter]
v1_imag = v1_imag[snr_filter]
'''
mass_1 = np.concatenate([mass_1_1, mass_2_1  ,mass_4_1,mass_6_1,mass_7_1,mass_8_1,mass_9_1,mass_10_1,mass_11_1,mass_12_1,mass_13_1,mass_14_1], axis=0)
mass_2 = np.concatenate([mass_1_2, mass_2_2  ,mass_4_2,mass_6_2,mass_7_2,mass_8_2,mass_9_2,mass_10_2,mass_11_2,mass_12_2,mass_13_2,mass_14_2], axis=0)
spin_1 = np.concatenate([spin_1_1, spin_2_1 ,spin_4_1,spin_6_1,spin_7_1,spin_8_1,spin_9_1,spin_10_1,spin_11_1,spin_12_1,spin_13_1,spin_14_1], axis=0)
spin_2 = np.concatenate([spin_1_2, spin_2_2 ,spin_4_2,spin_6_2,spin_7_2,spin_8_2,spin_9_2,spin_10_2,spin_11_2,spin_12_2,spin_13_2,spin_14_2], axis=0)

mass_1 = mass_1[:,None]
mass_2 = mass_2[:,None]
spin_1 = spin_1[:,None]
spin_2 = spin_2[:,None]

# Apply the same filter to the intrinsic parameters (ra, dec)
ra = np.concatenate([ra_1, ra_2 ,ra_4,ra_5,ra_6,ra_7,ra_8,ra_9,ra_10,ra_11,ra_12,ra_13,ra_14], axis=0)
dec = np.concatenate([dec_1, dec_2,dec_4, dec_5,dec_6,dec_7,dec_8,dec_9,dec_10,dec_11,dec_12,dec_13,dec_14], axis=0)

#snr_filter = injection_snr > 15 and injection_snr <30

#ra = ra[snr_filter]
#dec = dec[snr_filter]

'''
snr_filter = snr_filter <35

ra = ra[snr_filter]
dec = dec[snr_filter]
'''
ra = ra - np.pi
ra_x = np.cos(ra)
ra_y = np.sin(ra)

ra = ra[:, None]
ra_x = ra_x[:, None]
ra_y = ra_y[:, None]

dec = dec[:, None]

y_train = np.concatenate((ra_x, ra_y, dec), axis=1).astype('float64')
intrinsic_train = np.concatenate((mass_1, mass_2, spin_1, spin_2), axis=1)
# Expand dimensions for concatenation
h1_real = h1_real[:, :, None]
l1_real = l1_real[:, :, None]
v1_real = v1_real[:, :, None]

h1_imag = h1_imag[:, :, None]
l1_imag = l1_imag[:, :, None]
v1_imag = v1_imag[:, :, None]

logging.info("Concatenating SNR data...")
X_train_real = np.concatenate((h1_real, l1_real, v1_real), axis=2)
X_train_imag = np.concatenate((h1_imag, l1_imag, v1_imag), axis=2)


logging.info("Loading test SNR series data...")

def load_test_data(test_snr_file, test_params_file, num_test=10000):
    with h5py.File(test_snr_file, 'r') as snr_f, h5py.File(test_params_file, 'r') as params_f:
        # Load SNR series
        h1_test_real = np.real(snr_f['h1_snr_series'][0:num_test][()])
        l1_test_real = np.real(snr_f['l1_snr_series'][0:num_test][()])
        v1_test_real = np.real(snr_f['v1_snr_series'][0:num_test][()])
        
        h1_test_imag = np.imag(snr_f['h1_snr_series'][0:num_test][()])
        l1_test_imag = np.imag(snr_f['l1_snr_series'][0:num_test][()])
        v1_test_imag = np.imag(snr_f['v1_snr_series'][0:num_test][()])
        
        # Expand dimensions
        h1_test_real = h1_test_real[:, :, None]
        l1_test_real = l1_test_real[:, :, None]
        v1_test_real = v1_test_real[:, :, None]
        
        h1_test_imag = h1_test_imag[:, :, None]
        l1_test_imag = l1_test_imag[:, :, None]
        v1_test_imag = v1_test_imag[:, :, None]
        
        # Concatenate real and imaginary parts
        X_test_real = np.concatenate((h1_test_real, l1_test_real, v1_test_real), axis=2)
        X_test_imag = np.concatenate((h1_test_imag, l1_test_imag, v1_test_imag), axis=2)
        
        # Load intrinsic parameters
        ra_test = 2.0 * np.pi * params_f['ra'][0:num_test][()]
        dec_test = np.arcsin(1.0 - 2.0 * params_f['dec'][0:num_test][()])
        gps_time_test = params_f['gps_time'][0:num_test][()]
        mass_test_1 = params_f['mass1'][()]
        mass_test_2 = params_f['mass2'][()]
        spin_test_1 = params_f['spin1z'][()]
        spin_test_2 = params_f['spin2z'][()]
        
    return X_test_real, X_test_imag, h1_test_real, l1_test_real, v1_test_real, h1_test_imag, l1_test_imag, v1_test_imag, ra_test, dec_test, gps_time_test, mass_test_1, mass_test_2, spin_test_1, spin_test_2

def load_test_real_events_data(test_snr_file, test_params_file):
    with h5py.File(test_snr_file, 'r') as snr_f, h5py.File(test_params_file, 'r') as params_f:
        # Load SNR series
        h1_real = np.real(snr_f['h1_snr_series'][()][None,:])
        l1_real = np.real(snr_f['l1_snr_series'][()][None,:])
        v1_real = np.real(snr_f['v1_snr_series'][()][None,:])
        
        h1_imag = np.imag(snr_f['h1_snr_series'][()][None,:])
        l1_imag = np.imag(snr_f['l1_snr_series'][()][None,:])
        v1_imag = np.imag(snr_f['v1_snr_series'][()][None,:])
        
        # Load Injection_SNR and intrinsic parameters
        ra = 2.0 * np.pi * params_f['ra'][()]
        dec = np.arcsin(1.0 - 2.0 * params_f['dec'][()])
        gps_time_test = params_f['gps_time'][()]
        mass_1 = params_f['mass1'][()]
        mass_2 = params_f['mass2'][()]
        spin_1 = params_f['spin1z'][()]
        spin_2 = params_f['spin2z'][()]
    return h1_real, l1_real, v1_real, h1_imag, l1_imag, v1_imag, ra, dec, gps_time_test,mass_1,mass_2,spin_1,spin_2


# Load test data
#X_test_real, X_test_imag, h1_test_real, l1_test_real, v1_test_real, h1_test_imag, l1_test_imag, v1_test_imag, ra_test, dec_test, gps_time_test = load_test_data(
#    "/workspace/LIGO/GW-SkyLocator/data/O3_noise_snr_series_GW170817_BBH_3_det_0_secs_2.hdf",
#    "/workspace/LIGO/GW-SkyLocator/data/O3_noise_GW170817_BBH_3_det_parameters_2.hdf",
#    num_test=10000
#)

# Load real events test data
h1_real_test_1, l1_real_test_1, v1_real_test_1, h1_imag_test_1, l1_imag_test_1, v1_imag_test_1, ra_test_1, dec_test_1, gps_time_test_1,mass_1_1,mass_1_2,spin_1_1,spin_1_2 = load_test_real_events_data(
    '/home/slash/real_event_snr_time_series_GW200224_222234.hdf',
    '/home/slash/real_event_parameters_GW200224_222234.hdf')

h1_real_test_2, l1_real_test_2, v1_real_test_2, h1_imag_test_2, l1_imag_test_2, v1_imag_test_2, ra_test_2, dec_test_2, gps_time_test_2,mass_2_1,mass_2_2,spin_2_1, spin_2_2 = load_test_real_events_data(
    '/home/slash/real_event_snr_time_series_GW190521.hdf',
    '/home/slash/real_event_parameters_GW190521.hdf')

# Concatenate data from different banks
h1_real_test = np.concatenate([h1_real_test_1, h1_real_test_2], axis=0)
l1_real_test = np.concatenate([l1_real_test_1, l1_real_test_2], axis=0)
v1_real_test = np.concatenate([v1_real_test_1, v1_real_test_2], axis=0)

h1_imag_test = np.concatenate([h1_imag_test_1, h1_imag_test_2], axis=0)
l1_imag_test = np.concatenate([l1_imag_test_1, l1_imag_test_2], axis=0)
v1_imag_test = np.concatenate([v1_imag_test_1, v1_imag_test_2], axis=0)

# Expand dimensions for concatenation
h1_real_test = h1_real_test[:, :, None]
l1_real_test = l1_real_test[:, :, None]
v1_real_test = v1_real_test[:, :, None]

h1_imag_test = h1_imag_test[:, :, None]
l1_imag_test = l1_imag_test[:, :, None]
v1_imag_test = v1_imag_test[:, :, None]

logging.info("Concatenating SNR data...")
X_test_real = np.concatenate((h1_real_test, l1_real_test, v1_real_test), axis=2)
X_test_imag = np.concatenate((h1_imag_test, l1_imag_test, v1_imag_test), axis=2)

logging.info("Loading test intrinsic parameters...")
# Compute RA and Dec for test data

ra_test = np.concatenate([ra_test_1, ra_test_2], axis=0)
dec_test = np.concatenate([dec_test_1, dec_test_2], axis=0)

ra_test = ra_test - np.pi
ra_x_test = np.cos(ra_test)
ra_y_test = np.sin(ra_test)

ra_test = ra_test[:, None]
ra_x_test = ra_x_test[:, None]
ra_y_test = ra_y_test[:, None]

dec_test = dec_test[:, None]

gps_time_test = np.concatenate([gps_time_test_1, gps_time_test_2])
mass_1 = np.concatenate([mass_1_1, mass_2_1], axis=0)
mass_2 = np.concatenate([mass_1_2, mass_2_2], axis=0)
spin_1 = np.concatenate([spin_1_1, spin_2_1], axis=0)
spin_2 = np.concatenate([spin_1_2, spin_2_2], axis=0)

mass_1 = mass_1[:,None]
mass_2 = mass_2[:,None]
spin_1 = spin_1[:,None]
spin_2 = spin_2[:,None]
y_test = np.concatenate((ra_x_test, ra_y_test, dec_test), axis=1).astype('float64')
intrinsic_test = np.concatenate((mass_1, mass_2, spin_1, spin_2), axis=1)

# Function to standardize the real part of the SNR time series sample by sample.
logging.info("Standardizing data...")

def standardize_data(X_train_real, X_test_real):
    # Compute standard deviation along the sample dimension
    X_train_real_std = X_train_real.std(axis=1, keepdims=True)
    X_test_real_std = X_test_real.std(axis=1, keepdims=True)
    
    # Avoid division by zero
    X_train_real_std[X_train_real_std == 0] = 1.0
    X_test_real_std[X_test_real_std == 0] = 1.0
    
    X_train_real_standardized = X_train_real / X_train_real_std
    X_test_real_standardized = X_test_real / X_test_real_std
    
    return X_train_real_standardized, X_test_real_standardized

X_train_real, X_test_real = standardize_data(X_train_real, X_test_real)

# Stack real and imaginary parts
X_train = np.hstack((X_train_real, X_train_imag))
X_test = np.hstack((X_test_real, X_test_imag))

# Compute amplitude ratios and phase lags at the merger
logging.info("Computing amplitude ratios and phase lags at merger...")

def compute_metrics_at_merger(h1_real, l1_real, v1_real, h1_imag, l1_imag, v1_imag):
    # Find the indices of the peak (merger) points in each detector
    c = 299792.458
    merger_idx_h1 = np.argmax(np.abs(h1_real+h1_imag), axis=1)
    merger_idx_l1 = np.argmax(np.abs(l1_real+l1_imag), axis=1)
    merger_idx_v1 = np.argmax(np.abs(v1_real+v1_imag), axis=1)
    print(merger_idx_h1)

    # Precompute indices for gathering values at the merger
    idx_range = np.arange(h1_real.shape[0])
    # Efficiently gather values at the merger points
    h1_real_merger = h1_real[idx_range, merger_idx_h1]
    l1_real_merger = l1_real[idx_range, merger_idx_l1]
    v1_real_merger = v1_real[idx_range, merger_idx_v1]
    
    h1_imag_merger = h1_imag[idx_range, merger_idx_h1]
    l1_imag_merger = l1_imag[idx_range, merger_idx_l1]
    v1_imag_merger = v1_imag[idx_range, merger_idx_v1]

    # Compute time delays using precomputed indices
    time_diff_h1_l1 = merger_idx_h1 - merger_idx_l1
    time_diff_h1_v1 = merger_idx_h1 - merger_idx_v1 
    time_diff_l1_v1 = merger_idx_l1 - merger_idx_v1

    time_diffs = np.stack([time_diff_h1_l1 * c, time_diff_h1_v1 * c, time_diff_l1_v1 * c], axis=1)
    # Compute amplitude ratios at the merger
    amp_ratios = np.stack([(h1_real_merger+h1_imag_merger) / (l1_real_merger+l1_imag_merger), 
                           (h1_real_merger+h1_imag_merger) / (v1_real_merger+v1_imag_merger), 
                           (l1_real_merger+l1_imag_merger) / (v1_real_merger+v1_imag_merger)], axis=1)

    # Compute phase lags at the merger using the imaginary parts (Fourier transforms) (in DEG)
    phase_h1 = np.angle(h1_real_merger+h1_imag_merger)/np.pi*180
    phase_l1 = np.angle(l1_real_merger+l1_imag_merger)/np.pi*180
    phase_v1 = np.angle(v1_real_merger+v1_imag_merger)/np.pi*180
    
    t_h1_l1 = np.array([])
    t_h1_v1 = np.array([])
    t_l1_v1 = np.array([])
    time_corr = []
    # Comepute the convolve of two SNR time series
    for s in range(len(h1_real)):
        #time_cross_imag_h1_l1 = signal.correlate(h1_imag[s], l1_imag[s],mode = 'same') * (-1j)
        #time_cross_imag_h1_v1 = signal.correlate(h1_imag[s], v1_imag[s],mode = 'same') * (-1j) 
        #time_cross_imag_l1_v1 = signal.correlate(l1_imag[s], v1_imag[s],mode = 'same') * (-1j)

        '''
        time_cross_real_h1_l1 = signal.correlate(h1_real[s], l1_real[s],mode = 'same')
        time_cross_real_h1_v1 = signal.correlate(h1_real[s], v1_real[s],mode = 'same')
        time_cross_real_l1_v1 = signal.correlate(l1_real[s], v1_real[s],mode = 'same')
        '''

        time_cross_real_h1_l1 = signal.convolve(h1_real[s], l1_real[s])
        time_cross_real_h1_v1 = signal.convolve(h1_real[s], v1_real[s])
        time_cross_real_l1_v1 = signal.convolve(l1_real[s], v1_real[s])
        
        time_cross_imag_h1_l1 = signal.convolve(h1_imag[s], l1_imag[s])
        time_cross_imag_h1_v1 = signal.convolve(h1_imag[s], v1_imag[s])
        time_cross_imag_l1_v1 = signal.convolve(l1_imag[s], v1_imag[s])
        '''
        time_cross_real_h1_imag_l1 = signal.correlate(h1_real[s], l1_imag[s])
        time_cross_imag_h1_real_l1 = signal.correlate(h1_imag[s], l1_real[s])

        time_cross_real_h1_imag_v1 = signal.correlate(h1_real[s], v1_imag[s])
        time_cross_imag_h1_real_v1 = signal.correlate(h1_imag[s], v1_real[s])

        time_cross_real_l1_imag_v1 = signal.correlate(l1_real[s], v1_imag[s])
        time_cross_imag_l1_real_v1 = signal.correlate(l1_imag[s], v1_real[s])

        
        time_cross_imag_h1_l1 = (time_cross_imag_h1_l1-time_cross_imag_h1_l1.min())/(time_cross_imag_h1_l1.max()-time_cross_imag_h1_l1.min())
        time_cross_imag_h1_v1 = (time_cross_imag_h1_v1-time_cross_imag_h1_v1.min())/(time_cross_imag_h1_v1.max()-time_cross_imag_h1_v1.min())
        time_cross_imag_l1_v1 = (time_cross_imag_l1_v1-time_cross_imag_l1_v1.min())/(time_cross_imag_l1_v1.max()-time_cross_imag_l1_v1.min())
        time_cross_real_h1_imag_l1 = (time_cross_real_h1_imag_l1 - time_cross_real_h1_imag_l1.min())/(time_cross_real_h1_imag_l1.max()-time_cross_real_h1_imag_l1.min())
        time_cross_imag_h1_real_l1 = (time_cross_imag_h1_real_l1 - time_cross_imag_h1_real_l1.min())/(time_cross_imag_h1_real_l1.max()-time_cross_imag_h1_real_l1.min())
        time_cross_real_h1_imag_v1 = (time_cross_real_h1_imag_v1 - time_cross_real_h1_imag_v1.min())/(time_cross_real_h1_imag_v1.max()-time_cross_real_h1_imag_v1.min())
        time_cross_imag_h1_real_v1 = (time_cross_imag_h1_real_v1 - time_cross_imag_h1_real_v1.min())/(time_cross_imag_h1_real_v1.max()-time_cross_imag_h1_real_v1.min())
        time_cross_real_l1_imag_v1 = (time_cross_real_l1_imag_v1 - time_cross_real_l1_imag_v1.min())/(time_cross_real_l1_imag_v1.max()-time_cross_real_l1_imag_v1.min())
        time_cross_imag_l1_real_v1 = (time_cross_imag_l1_real_v1 - time_cross_imag_l1_real_v1.min())/(time_cross_imag_l1_real_v1.max()-time_cross_imag_l1_real_v1.min())
        '''
        
        '''
        time_cross_real_h1_l1 /= time_cross_real_h1_l1.std()
        time_cross_real_h1_v1 /= time_cross_real_h1_v1.std()
        time_cross_real_l1_v1 /= time_cross_real_l1_v1.std()
        time_cross_real_h1_l1 = (time_cross_real_h1_l1 - time_cross_real_h1_l1.min())/(time_cross_real_h1_l1.max()-time_cross_real_h1_l1.min())
        time_cross_real_h1_v1 = (time_cross_real_h1_v1 - time_cross_real_h1_v1.min())/(time_cross_real_h1_v1.max()-time_cross_real_h1_v1.min())
        time_cross_real_l1_v1 = (time_cross_real_l1_v1 - time_cross_real_l1_v1.min())/(time_cross_real_l1_v1.max()-time_cross_real_l1_v1.min())
        '''
        '''
        t_h1_l1 = np.concatenate((time_cross_real_h1_l1,time_cross_imag_h1_l1), axis=0)
        t_h1_v1 = np.concatenate((time_cross_real_h1_v1,time_cross_imag_h1_v1), axis=0)
        t_l1_v1 = np.concatenate((time_cross_real_l1_v1,time_cross_imag_l1_v1), axis=0)
        '''
        t_h1_l1 = time_cross_real_h1_l1
        t_h1_v1 = time_cross_real_h1_v1
        t_l1_v1 = time_cross_real_l1_v1

        Unit = np.stack((t_h1_l1,t_h1_v1,t_l1_v1))
        Unit = Unit.T
        time_corr.append(Unit)

    #print(time_cross_real_l1_v1.shape)
    #print(amp_ratios.shape)
    #time_real = np.stack((time_cross_real_h1_l1, time_cross_real_h1_v1, time_cross_real_l1_v1),axis=1)
    

    
    
    del h1_real_merger, l1_real_merger, v1_real_merger
    del h1_imag_merger, l1_imag_merger, v1_imag_merger
    del time_cross_real_h1_l1,time_cross_real_h1_v1,time_cross_real_l1_v1
    del Unit
    del time_corr
   
    return combined_metrics, combined_time

# Compute the new input features (amplitude ratios and phase lags at merger)
print((h1_real.squeeze()).shape)
metrics_train, time_train = compute_metrics_at_merger(h1_real.squeeze(), l1_real.squeeze(), v1_real.squeeze(), 
                                         h1_imag.squeeze(), l1_imag.squeeze(), v1_imag.squeeze())
metrics_test, time_test = compute_metrics_at_merger(h1_real_test.squeeze(), l1_real_test.squeeze(), v1_real_test.squeeze(), 
                                         h1_imag_test.squeeze(), l1_imag_test.squeeze(), v1_imag_test.squeeze())

# Function to standardize the metrics
def scale_labels(data_train, data_test):
    mean = data_train.mean(axis=0)
    std = data_train.std(axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    data_train_standardized = (data_train - mean) / std
    data_test_standardized = (data_test - mean) / std
    return data_train_standardized, data_test_standardized, mean, std

#metrics_train, metrics_test, metrics_mean, metrics_std = scale_labels(metrics_train, metrics_test)
#y_train, y_test, y_mean, y_std = scale_labels(y_train,y_test)
#time_train, time_test, time_mean, time_std = scale_labels(time_train, time_test)
#intrinsic_train, intrinsic_test, intrinsic_mean, intrinsic_std = scale_labels(intrinsic_train, intrinsic_test)
logging.info("Metrics standardized.")

# Convert all data to torch tensors
logging.info("Converting data to torch tensors...")
#X_train = torch.tensor(X_train, dtype=torch.float64).to(device)
#X_train_real = torch.tensor(X_train_real, dtype=torch.float64).to(device)
#X_train_imag = torch.tensor(X_train_imag, dtype=torch.float64).to(device)
y_train = torch.tensor(y_train, dtype=torch.float64)
metrics_train = torch.tensor(metrics_train, dtype=torch.float64)
time_train = torch.tensor(time_train, dtype=torch.float64)
intrinsic_train = torch.tensor(intrinsic_train, dtype = torch.float64)
#time_train = time_train.view(time_train.size(0),-1)


#X_test = torch.tensor(X_test, dtype=torch.float64).to(device)
#X_test_real = torch.tensor(X_test_real, dtype=torch.float64).to(device)
#X_test_imag = torch.tensor(X_test_imag, dtype=torch.float64).to(device)
y_test = torch.tensor(y_test, dtype=torch.float64)
metrics_test = torch.tensor(metrics_test, dtype=torch.float64)
time_test = torch.tensor(time_test, dtype=torch.float64)
intrinsic_test = torch.tensor(intrinsic_test, dtype = torch.float64)
#time_test = time_test.view(time_test.size(0),-1)
class GWDataset(Dataset):
    def __init__(self,X,metrics,z, y):
        self.X = X
        self.metrics = metrics
        #self.time = time
        self.y = y
        self.z = z

    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        return self.X[idx], self.metrics[idx],self.z[idx], self.y[idx]

# Create Dataset and DataLoader
logging.info("Creating Dataset and DataLoader...")
dataset = GWDataset(time_train,metrics_train,intrinsic_train, y_train)
batch_size = 256
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(metrics_train.shape)
class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=nn.ReLU()):
        super(ResidualUnit, self).__init__()
        self.activation = activation
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            self.activation,
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))
topk_x = 256
topk_metrics = 64
topk_intrinsic = 16
metrics_features = 64
class ResNet34Encoder(nn.Module):
    def __init__(self, input_channels,metrics_dim,intrinsic_dim):
        super(ResNet34Encoder, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Define Residual Layers
        self.layer1 = self._make_layer(32, 32, blocks=3, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=4, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=6, stride=2)
        self.layer4 = self._make_layer(128, 256, blocks=3, stride=2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        # Metrics MLP
        self.metrics_mlp = nn.Sequential(
            nn.Linear(metrics_dim, metrics_features),
            nn.ReLU(),
            nn.BatchNorm1d(metrics_features),
            nn.Linear(metrics_features, metrics_features),
            nn.ReLU(),
            nn.BatchNorm1d(metrics_features),
            nn.Linear(metrics_features, metrics_features),
            nn.ReLU(),
            nn.BatchNorm1d(metrics_features),
            nn.Linear(metrics_features, metrics_features),
            nn.ReLU(),
            nn.BatchNorm1d(metrics_features),
            nn.Linear(metrics_features, metrics_features),
            nn.BatchNorm1d(metrics_features)
        )
        '''
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64)
        )
        '''
        self.intrinsic_mlp = nn.Sequential(
            nn.Linear(intrinsic_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64)
        )
        
        # Merge and Output
        self.merge = nn.Sequential(
            nn.Sigmoid(),
            nn.LayerNorm(topk_metrics+topk_intrinsic)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualUnit(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualUnit(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self,x, metrics,intrinsic):

        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = torch.topk(x,topk_x)[0]
        '''
        metrics = self.metrics_mlp(metrics)
        metrics = torch.topk(metrics,topk_metrics)[0]
        #time = self.time_mlp(time)
        intrinsic = self.intrinsic_mlp(intrinsic)
        intrinsic = torch.topk(intrinsic,topk_intrinsic)[0]
        merged = torch.cat([metrics,intrinsic], dim=1)
        out = self.merge(merged)
        return out

class NormalizingFlowModel(nn.Module):
    def __init__(self, encoder_output_dim, num_features=4, num_layers=5):
        super(NormalizingFlowModel, self).__init__()
        
        self.encoder_output_dim = encoder_output_dim
        self.num_features = num_features
        
        latent_size = 3
        hidden_units = 256
        hidden_layers = 3

        # Set base distribution
        q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)

        flows = []
        for i in range(num_layers):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                             num_context_channels=self.encoder_output_dim)]

            flows += [nf.flows.LULinearPermute(latent_size)]

    
        # Construct flow model
        self.flow = nf.ConditionalNormalizingFlow(q0, flows).to(device)


    def forward(self, x, context):
        return self.flow.log_prob(x, context=context)

    def sample(self, num_samples, context):
        
        # Repeat the context for each sample in the batch dimension (num_samples)
        expanded_features = context.repeat(num_sample,1)  # Shape: [num_samples, context_dim]

        # Sample from the normalizing flow conditioned on the repeated features
        samples = self.flow.sample(num_samples, context=expanded_features)[0].detach().cpu().numpy()

        # Convert back to torch tensor if needed
        return torch.tensor(samples, dtype=torch.float64, device=device)


logging.info("Building ResNet34 encoder and Normalizing Flow models...")

# Instantiate the encoder
input_channels = time_train.shape[1]  # 820 (assuming 820 features)
metrics_dim = metrics_train.shape[1]  # 9 (time diffs + amp ratios + phase lags)
#time_dim = time_train.shape[-1]
#time_dim2 = time_train.shape[2]
intrinsic_dim = intrinsic_train.shape[-1]
encoder = ResNet34Encoder(input_channels,metrics_dim,intrinsic_dim)
encoder = encoder.to(device).double()
print(X_train.shape)
# Instantiate the normalizing flow
encoder_output_dim = topk_metrics+topk_intrinsic  # Fr3om ResNet and metrics MLP\t
flow_model = NormalizingFlowModel(
    encoder_output_dim=encoder_output_dim,
    num_features=13,
    num_layers=5
)

flow_model = flow_model.to(device).double()

# Verify encoder parameters
for name, param in encoder.named_parameters():
    assert param.dtype == torch.float64, f"Encoder parameter {name} is not float64."

# Verify flow model parameters
for name, param in flow_model.named_parameters():
    assert param.dtype == torch.float64, f"Flow model parameter {name} is not float64."


# Define optimizer
optimizer = optim.Adam(list(encoder.parameters()) + list(flow_model.parameters()), lr=1e-4, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, verbose=True)

# Early stopping parameters
early_stopping_patience = 10
best_val_loss = np.inf
epochs_no_improve = 0
num_epochs = 700

# Split dataset into training and validation
logging.info("Splitting dataset into training and validation sets...")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Lists to store loss history
train_losses = []
val_losses = []
logging.info("Starting training...")
'''
checkpoint = torch.load('best_model_Original_ResNet34_5_5_256_4_time.pth')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
flow_model.load_state_dict(checkpoint['flow_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
train_losses = checkpoint['train_loss']
val_losses = checkpoint['val_loss']
start = len(val_losses)
'''
for epoch in range(num_epochs):
    #epoch += start
    encoder.train()
    flow_model.train()
    running_loss = 0.0
    for batch_X, batch_metrics,batch_z, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_metrics = batch_metrics.to(device)
        batch_y = batch_y.to(device)
        batch_z = batch_z.to(device)
        optimizer.zero_grad()
        
        # Forward pass through encoder
        encoded_features = encoder(batch_X, batch_metrics,batch_z)
        
        # Compute log_prob
        log_prob = flow_model(batch_y, context=encoded_features)
        loss = -log_prob.mean()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * len(batch_metrics)
    
    epoch_loss = running_loss / train_size
    train_losses.append(epoch_loss)
    
    # Validation
    encoder.eval()
    flow_model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for val_X, val_metrics,val_z,val_y in val_loader:
            val_X = val_X.to(device)
            val_metrics = val_metrics.to(device)
            val_y = val_y.to(device)
            val_z = val_z.to(device)
            encoded_features = encoder(val_X, val_metrics,val_z)
            log_prob = flow_model(val_y, context=encoded_features)
            loss = -log_prob.mean()
            val_running_loss += loss.item() * len(val_metrics)
    
    val_epoch_loss = val_running_loss / val_size
    val_losses.append(val_epoch_loss)
    
    logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")
    
    # Step the scheduler
    scheduler.step(val_epoch_loss)
    
    # Early stopping
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        epochs_no_improve = 0
        # Save the best model
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'flow_state_dict': flow_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_losses,
            'val_loss': val_losses,
        }, 'best_model_Original_ResNet34_5_5_256_4_time.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

# Load the best model
logging.info("Loading the best model from checkpoint...")
checkpoint = torch.load('best_model_Original_ResNet34_5_5_256_4_time.pth')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
flow_model.load_state_dict(checkpoint['flow_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
train_losses = checkpoint['train_loss']
val_losses = checkpoint['val_loss']

def plot_loss_curves(train_losses, val_losses, save_path='./Loss_curve__Original_ResNet34_5_5_256_4_time.png'):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, "r--", label="Loss on training data")
    plt.plot(val_losses, "r", label="Loss on validation data")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(save_path, dpi=400)
#    plt.show()

logging.info("Plotting loss curves...")
plot_loss_curves(train_losses, val_losses)

# Save encoder and flow models
logging.info("Saving encoder and flow model weights...")
torch.save(encoder.state_dict(), './ResNet34_BBH_weights_new_PyTorch__Original_ResNet34_5_5_256_4_time.pth')
torch.save(flow_model.state_dict(), './NF_BBH_weights_new_PyTorch__Original_ResNet34_5_5_256_4_time.pth')

def bayestar_adaptive_grid_new(probdensity, flow, preds, delta, top_nside=16, rounds=8):
    """
    Create a sky map by evaluating a function on an adaptive grid.

    Perform the BAYESTAR adaptive mesh refinement scheme as described in
    Section VI of Singer & Price 2016, PRD, 93, 024013
    :doi:10.1103/PhysRevD.93.024013. This computes the sky map
    using a provided analytic function and refines the grid, dividing the
    highest 25% into subpixels and then recalculating their values. The extra
    given args and kwargs will be passed to the given probdensity function.

    Parameters
    ----------
    probdensity : callable
        Probability density function. The first argument consists of
        column-stacked array of right ascension and declination in radians.
        The return value must be a 1D array of the probability density in
        inverse steradians with the same length as the argument.
    top_nside : int
        HEALPix NSIDE resolution of initial evaluation of the sky map
    rounds : int
        Number of refinement rounds, including the initial sky map evaluation

    Returns
    -------
    skymap : astropy.table.Table
        An astropy Table with UNIQ and PROBDENSITY columns, representing
        a multi-ordered sky map
    probs : list
        List of probability densities at each refinement round
    """
    probs = []
    top_npix = ah.nside_to_npix(top_nside)
    nrefine = top_npix // 4
    cells = list(zip([0] * nrefine, [top_nside // 2] * nrefine, range(nrefine)))
    
    for iround in range(rounds - 1):
        logging.info(f'adaptive refinement round {iround+1} of {rounds-1} ...')
        # Sort cells based on probability density
        cells_sorted = sorted(cells, key=lambda p_n_i: p_n_i[0] / (p_n_i[1]**2))
        # Refine the top nrefine cells
        new_nside, new_ipix = [], []
        for _, nside, ipix in cells_sorted[-nrefine:]:
            for i in range(4):
                new_nside.append(nside * 2)
                new_ipix.append(ipix * 4 + i)
        
        theta, phi = hp.pix2ang(new_nside, new_ipix, nest=True)
        ra = phi
        ra = np.mod(ra + delta, 2.0 * np.pi)
        dec = 0.5 * np.pi - theta
        
        ra = ra - np.pi
        ra_x = np.cos(ra)
        ra_y = np.sin(ra)
        
        pixels = np.stack([ra_x, ra_y, dec], axis=1)
        pixels = torch.tensor(pixels, dtype=torch.float64).to(device)
        
        # Compute probability density
        p = probdensity(flow, pixels, preds)
        probs.append(p)
        
        # Update the refined cells with new probabilities
        cells_sorted[-nrefine:] = list(zip(p, new_nside, new_ipix))
    
    # Normalize probabilities
    post, nside, ipix = zip(*cells_sorted)
    post = np.array(post)
    nside = np.array(nside)
    ipix = np.array(ipix)
    post /= np.sum(post * hp.nside2pixarea(nside).astype(float))
    
    # Convert from NESTED to UNIQ pixel indices
    order = np.log2(nside).astype(int)
    uniq = nest2uniq(order.astype(np.int8), ipix)
    
    return Table([uniq, post], names=['UNIQ', 'PROBDENSITY'], copy=False), probs

def nf_prob_density(flow, pixels, preds):
    """
    Compute probability density using the Normalizing Flow model.

    Parameters
    ----------
    flow : NormalizingFlowModel
        The trained normalizing flow model
    pixels : torch.Tensor
        Tensor of shape [num_pixels, 3]
    preds : torch.Tensor
        Tensor of shape [batch_size, encoder_output_dim]

    Returns
    -------
    prob_density : numpy.ndarray
        Probability density values for each pixel
    """
    with torch.no_grad():
        # Repeat the context for each sample in the batch dimension (num_samples)
        expanded_features = preds.repeat(len(pixels), 1)  # Shape: [num_samples, context_dim]
        log_prob = flow(pixels, context=expanded_features)
        prob_density = torch.exp(log_prob).cpu().numpy()
    return prob_density

# Load the best model
logging.info("Loading the best model for inference...")
checkpoint = torch.load('best_model_Original_ResNet34_5_5_256_4_time.pth')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
flow_model.load_state_dict(checkpoint['flow_state_dict'])
encoder.eval()
flow_model.eval()

logging.info("Starting inference and sky map generation...")