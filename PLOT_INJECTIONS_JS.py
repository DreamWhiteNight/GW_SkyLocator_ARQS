import healpy as hp
import numpy as np
import ligo.skymap
import argparse
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from scipy.ndimage import zoom
from pathlib import Path
from ligo.skymap import io, kde, postprocess
from ligo.skymap.plot.marker import reticle
from scipy.interpolate import interp1d
import h5py
lsdr = [i for i in os.listdir('/home/slash/Real_events/shift/') if '.fits' in i]
def normalize_map(m):
    """Normalize a HEALPix map to a probability distribution."""
    m = np.copy(m)
    m[m < 0] = 0  # remove negatives if any
    total = np.sum(m)
    if total == 0:
        raise ValueError("Map has zero total probability.")
    return m / total

'''
def match_resolution(map1, map2, nside_out=None):
    """
    Match two HEALPix maps to the same NSIDE using interpolation.
    Default: upgrade both to the higher NSIDE.
    """
    nside1 = hp.get_nside(map1)
    nside2 = hp.get_nside(map2)

    if nside_out is None:
        nside_out = max(nside1, nside2)

    map1_resampled = hp.ud_grade(map1, nside_out=nside_out, order_in='NEST', order_out='NEST')
    map2_resampled = hp.ud_grade(map2, nside_out=nside_out, order_in='NEST', order_out='NEST')

    return map1_resampled, map2_resampled
'''
def match_pdf_size(p1, p2, num=None):
    """
    Interpolate two 1D PDFs to the same size without distorting distribution.

    Parameters
    ----------
    p1, p2 : arrays
        Input probability densities
    num : int or None
        Target size (default = max length)

    Returns
    -------
    x, p1_new, p2_new
    """

    if num is None:
        num = max(len(p1), len(p2))

    # Assume same domain [0,1]
    x1 = np.linspace(0, 1, len(p1))
    x2 = np.linspace(0, 1, len(p2))
    x = np.linspace(0, 1, num)

    # Interpolate
    p1_new = np.interp(x, x1, p1)
    p2_new = np.interp(x, x2, p2)

    # Remove numerical artifacts
    p1_new = np.clip(p1_new, 0, None)
    p2_new = np.clip(p2_new, 0, None)

    #  Renormalize (preserve probability distribution)
    p1_new /= np.sum(p1_new)
    p2_new /= np.sum(p2_new)
    return p1_new, p2_new



def kl_divergence(p, q, eps=1e-12):
    """Compute KL divergence KL(p || q)."""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return np.sum(p * np.log2(p / q))
def js_divergence(map1, map2, match_nside=True, nside_out=None):
    """
    Compute Jensen-Shannon divergence between two HEALPix probability maps.

    Parameters:
    - map1, map2: input HEALPix maps
    - match_nside: whether to resample to same resolution
    - nside_out: target NSIDE (optional)

    Returns:
    - JS divergence (scalar)
    """
    # Match resolution if needed
    #if match_nside:
        #p,q = match_resolution(map1, map2, nside_out=nside_out)
    #p,q = match_pdf_size(map1,map2)
    '''
    if len(map1) != len(map2):
         x_fine = range(len(map2))
         pf = interp1d(range(len(map1)),map1,kind='linear',bounds_error=False, fill_value="extrapolate")
         p = pf(x_fine)
         q = map2
         print(p,q)
    else:
         p = map1
         q = map2
    '''
    p = map1
    q = map2
    print(p,q)
    # Normalize
    #p = -p + 100
    #q = -q + 100
    plt.plot(p)
    plt.plot(q)
    plt.savefig(f'/home/slash/Paper_Inj/nside512/skymaps/Prob_{s}.png')
    plt.close()
    #p = hp.smoothing(p,sigma = np.radians(1))
    #q = hp.smoothing(q,sigma = np.radians(1))
    print('js s')
    threshold = 1e-10
    mask = (p > threshold) | (q > threshold)
    mask_p = p[mask]
    mask_q = q[mask]
    

    p_valid = mask_p
    q_valid = mask_q

    p_valid /= sum(p_valid)
    q_valid /= sum(q_valid)
    p = p_valid
    q = q_valid
    # Mixture
    m = 0.5 * (p + q)

    # JS divergence
    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    return js
num = 100
js = 0
f = open('JSD_records_512.txt','w')
f1 = h5py.File(f'/mnt/data/slash/Inj_Data/BAYESTAR_INJECTION/TEST/HDF_DATA/Inj_Data_Bayestar_test_Parm.hdf','r')
f2 = open('Area_records','w')
ra_test = 2.0 * np.pi * f1['ra_test'][()]
dec_test = np.arcsin(1.0 - 2.0 * f1['dec_test'][()])
for s in range(num):
    try:
        #name = s.split('.')
        #name = name[0].split('time_')
        #name = name[-1]
        skymap, metadata_ML = io.fits.read_sky_map(f'/home/slash/Paper_Inj/Test_2_{s}.fits', nest=None)
        pe_GW200219, metadata = io.fits.read_sky_map(f'/mnt/data/slash/Inj_Data/BAYESTAR_INJECTION/TEST/FITS/Bayestar_coinc{s}.fits/'+'0.fits', nest=None)
        nside1 = hp.npix2nside(len(skymap))
        nside2 = hp.npix2nside(len(pe_GW200219))
        print(nside2)
        nside_out = max(nside1, nside2)
        sum1 = sum(skymap)
        sum2 = sum(pe_GW200219)
        #if nside1 != nside2:
            #pe_GW200219 = hp.ud_grade(pe_GW200219,nside1,order_in = 'NEST', order_out = 'NEST')
            #num -= 1
            #continue
        print('done')
        nside2 = hp.npix2nside(len(pe_GW200219))
        deg2perpix_pe_js = hp.nside2pixarea(nside2, degrees=True)
        probperdeg2_pe_js = pe_GW200219 / deg2perpix_pe_js
        nside1 = hp.npix2nside(len(skymap))
        deg2perpix_js = hp.nside2pixarea(nside1, degrees=True)
        probperdeg2_js = skymap / deg2perpix_js
        #probperdeg2_js, probperdeg2_pe_js = match_pdf_size(skymap, pe_GW200219)
        print('done3')
        #skymap, metadata_ML = io.fits.read_sky_map('/home/slash/GW-SkyLocator_ARQS/skymap_test_new/'+s, nest=None)
        #print(name)
        #file_SNR = h5py.File(f'/home/slash/Real_events/parm/real_event_parameters_GW{name}.hdf')
        #SNR = file_SNR['Injection_SNR'][0]
        SNR = f1['Injection_SNR'][s]
        #SNR = np.round(SNR,3)
        from astropy.coordinates import SkyCoord
        # Convert to probability per square degree
        nside = hp.npix2nside(len(skymap))
        text = []
        
        deg2perpix = hp.nside2pixarea(nside, degrees=True)
        probperdeg2_ML = skymap / deg2perpix
        #skymap /= sum(skymap)
        #skymap *= sum1
        pe_GW200219 /= sum(pe_GW200219)
        pe_GW200219 *= sum2
        probperdeg2_ML /= sum(probperdeg2_ML)
        confidence_levels = 100 * postprocess.find_greedy_credible_levels(skymap)
        nside_pe = hp.npix2nside(len(pe_GW200219))
        deg2perpix_pe = hp.nside2pixarea(nside_pe, degrees=True)
        probperdeg2_pe = pe_GW200219 / deg2perpix_pe
        #onfidence_levels_old = 100 * postprocess.find_greedy_credible_levels(skymap_old)
        confidence_levels_pe = 100 * postprocess.find_greedy_credible_levels(pe_GW200219)
        #JSD_in = js_divergence(confidence_levels,confidence_levels_pe)
        #JSD_in = js_divergence(probperdeg2_ML,probperdeg2_pe)
        JSD_in = 10
        plt.close()
        print(f'JS of {s}th  = {JSD_in}')
        js += JSD_in

        ax = plt.axes(projection="astro hours mollweide")
        ax.grid()
        
        vmax_ML = probperdeg2_ML.max()
        vmin_ML = probperdeg2_ML.min()
    
        pp = np.round([50,90]).astype(int)
        ii = np.round(np.searchsorted(np.sort(confidence_levels),[50,90]) * deg2perpix).astype(int)
        for i, p in zip(ii, pp):
                    # FIXME: use Unicode symbol instead of TeX '$^2$'
                    # because of broken fonts on Scientific Linux 7.
                    text.append(str(s)+u'SkyLocator {:d}% area: {:d} deg²'.format(p, i, grouping=True))
        #ax.text(1, 1, '\n'.join(text), transform=ax.transAxes, ha='right')
        from astropy.coordinates import SkyCoord
        # Convert to probability per square degree
        ax = plt.axes(projection="astro hours mollweide")
        ax.grid()
        
        vmax = probperdeg2_pe.max()
        vmin = probperdeg2_pe.min()
        ax.imshow_hpx((probperdeg2_pe, 'ICRS'), nested=True, vmin=vmin, vmax=vmax, cmap='cylon')
        
        proxy = [plt.Rectangle((0,0),1,1,fc= 'red'),plt.Rectangle((0,0),1,1,fc= 'black')]
        plt.legend(proxy,['Bayestar','GW_SkyLocator'],loc='best',framealpha = 0)
        #prob_dist_pred_old, prob_dist_bay = match_size(skymap_old,pe_GW200219)
        #prob_dist_pred = Get_prob_dist(prob_dist_pred)
        #prob_dist_bay = Get_prob_dist(prob_dist_bay)
        #JSD_old = np.round(js_divergence(skymap_old,pe_GW200219),3)
        ii2 = np.round(np.searchsorted(np.sort(confidence_levels_pe),[50,90]) * deg2perpix_pe).astype(int)
        ax.text(0,0,f'Event: {s}',transform=ax.transAxes,ha='left',va='top')
        ax.plot_coord(SkyCoord(ra_test[s], dec_test[s], unit='rad'), 'x',markeredgecolor='blue', markersize=5)
        #contours_old = ax.contour_hpx((confidence_levels_old, 'ICRS'), nested=metadata['nest'],linewidths=1.5,levels=[50,90],colors=['yellow','yellow'])
        contours_pe = ax.contour_hpx((confidence_levels_pe, 'ICRS'), nested=metadata['nest'],linewidths=1.5,levels=[50,90],colors=['red','red'])
        contours = ax.contour_hpx((confidence_levels, 'ICRS'), nested=metadata_ML['nest'],linewidths=1.5,levels=[50,90],colors=['black','black'])
        ax.figure.savefig('/home/slash/Paper_Inj/nside512/skymaps/'+str(s)+'.png', bbox_inches="tight", facecolor='w', transparent=False, dpi=400)
        f.write(f'{s}th data JSD = {JSD_in}, snr  = {SNR}\n')
        #f2.write(u'SkyLocator {:d}% area: {:d} deg²'.format(p, i, grouping=True))
        f2.write(f'skylocator:  50 = {str(ii[0])} 90 = {str(ii[1])}, BAYESTAR: 50 = {str(ii2[0])} 90 = {str(ii2[1])} \n')
        #ax.legend('black','White')
        #plt.legend(True)
        plt.close()
        
        
        #prob_dist_pred, prob_dist_bay = match_size(skymap,pe_GW200219)
        #prob_dist_pred = Get_prob_dist(prob_dist_pred)
        #prob_dist_bay = Get_prob_dist(prob_dist_bay)
        #JSD = js_divergence(prob_dist_pred,prob_dist_bay)
        #print(name,JSD_in,JSD_old)
        del pe_GW200219
        del skymap
    except Exception as e:
           print(e)
           num -= 1
           continue
js /= num
print(js)


    #ax.figure.close()
