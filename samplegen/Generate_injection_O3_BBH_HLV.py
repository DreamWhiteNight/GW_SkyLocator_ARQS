import pycbc
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.filter.resample import highpass, lowpass
from pycbc.waveform.utils import td_taper
from pycbc.filter.resample import resample_to_delta_t
import gwpy
from gwpy.timeseries import TimeSeries
#import pycbc
from pycbc.detector import Detector
#from pycbc.waveform import get_td_waveform
#from matplotlib import pyplot as plt
from gwpy.plot import Plot
from pycbc.filter import matched_filter, resample_to_delta_t
from pycbc.noise import noise_from_psd
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.types import TimeSeries as PyCBCTimeSeries
from matplotlib import pyplot as plt
import numpy as np
import random
import h5py
import multiprocessing
import pylab
RA = []
DEC = []
mass1 = []
mass2 = []
spin1z = []
spin2z = []
Polarization = []
h1_snr_series = []
l1_snr_series = []
v1_snr_series = []
Injection_SNR = []
H_time = []
L_time = []
V_time = []
gps_time = []
def generate(i):
    ra_test = float(random.uniform(0,1)) # record this number
    dec_test = float(random.uniform(0,1)) # record this number
    pol = random.uniform(0,0.99)
    pol *= np.pi
    ra = 2.0 * np.pi * ra_test
    dec  = np.arcsin(1.0 - 2.0 * dec_test)
    spin1 = random.uniform(0,0.99)
    spin2 = random.uniform(0,0.99)
    c_phase = float(random.uniform(0,1))
    c_phase *= 2 * np.pi
    spin1x = 0.0
    spin1y = 0.0
    spin2x = 0.0
    spin2y = 0.0
    inclination = random.uniform(0,0.99)
    inclination *= np.pi
    m1 = float(np.random.randint(5,50)) + np.round(random.random(), decimals = 3)
    m2 = float(np.random.randint(5,50)) + np.round(random.random(), decimals = 3)
    distance = np.random.randint(300,1000)
    #distance = 100
    print('parameter done')
    if m1 < m2 :
        m1, m2 = m2, m1
    
    try:
        hp, hc = get_td_waveform(approximant = approximant, mass1 = m1, mass2 = m2, spin1z = spin1, spin2z = spin2, inclination = inclination,coa_phase = c_phase ,distance = distance ,f_lower = f_lower, delta_t = 1.0/2048)
        #template_length = len(hp)
        st_crop = int(-template_length/2 - hp.sample_times[0])
        ed_crop = hp.sample_times[-1] + 1/2048.0 - template_length/2
        #print(st_crop,0)
        hp.start_time += time
        hc.start_time += time
        print(len(hp))
        #hp = hp.crop(st_crop, 0)
        #hc = hc.crop(st_crop,0)
        #start_time = hp.start_time
        D = 'H1'
        E = 'L1'
        F = 'V1'
        detector_D = Detector(D)
        detector_E = Detector(E)
        detector_F = Detector(F)
        dt_h1l1 = detector_E.time_delay_from_detector(detector_D, ra, dec, time)
        #print(dt_h1l1)
        dt_h1v1 = detector_F.time_delay_from_detector(detector_D, ra, dec, time)
        '''
        fp, fc = detector.antenna_pattern(ra, dec, pol, time)
        ht = fp * hp + fc * hc
        '''
        
        #hc.start_time = hp.start_time
        ht_H1 = detector_D.project_wave(hp, hc, ra, dec, pol)
        ht_L1 = detector_E.project_wave(hp, hc, ra, dec, pol)
        ht_V1 = detector_F.project_wave(hp, hc, ra, dec, pol)
        #H_start = ht_H1.start_time
        #L_start = ht_L1.start_time
        #V_start = ht_V1.start_time
        #ht_H1.start_time
        shift_H = ht_H1.start_time
        shift_L = ht_L1.start_time
        shift_V = ht_V1.start_time
        print(ht_H1.start_time, ht_L1.start_time, ht_V1.start_time)
        #ht_H1.start_time += time
        #ht_L1.start_time += time
        #ht_V1.start_time += time
        ht_H1 = ht_H1.copy()
        ht_L1 = ht_L1.copy()
        ht_V1 = ht_V1.copy()
        print('signal done')
        '''
        st_crop = -40 - ht_H1.sample_times[0]
        ed_crop = ht_H1.sample_times[-1] - 0 + 1.0/sample_rate
        ht_H1 = ht_H1.crop(st_crop,0)
        st_crop = -40 - ht_V1.sample_times[0]
        ed_crop = ht_V1.sample_times[-1] - 0 + 1.0/sample_rate
        ht_V1 = ht_V1.crop(st_crop,0)
        st_crop = -40 - ht_L1.sample_times[0]
        ed_crop = ht_L1.sample_times[-1] - 0 + 1.0/sample_rate
        ht_L1 = ht_L1.crop(st_crop,0)
        '''
        signal_delta_f  = ht_H1.delta_f
        ht_H1 = TimeSeries(ht_H1,dt = 1.0/2048,t0 = ht_H1.start_time, channel = D)
        ht_L1 = TimeSeries(ht_L1,dt = 1.0/2048,t0 = ht_L1.start_time, channel = E)
        ht_V1 = TimeSeries(ht_V1,dt = 1.0/2048,t0 = ht_V1.start_time, channel = F)
        print('start_time_H : ',ht_H1.t0,'Length',len(ht_H1))
        print('start_time_L : ',ht_L1.t0,'Length',len(ht_L1))
        print('start_time_V : ',ht_V1.t0,'Length',len(ht_V1))

        
        '''
        pylab.loglog(psd_H1.sample_frequencies, psd_H1)
        pylab.xlim(100,512)
        pylab.ylim(1e-47,1e-45)
        pylab.savefig('psd.png')
        pylab.close()
        '''
        
        #noise_H1 = noise_from_psd(200*2048, 1.0/2048, psd_H1, seed=np.random.randint(1,100000))
        #noise_L1 = noise_from_psd(200*2048, 1.0/2048, psd_L1, seed=np.random.randint(1,100000))
        #noise_V1 = noise_from_psd(200*2048, 1.0/2048, psd_V1, seed=np.random.randint(1,100000))
        
        print('noise_done')
        

        noise_H1 = TimeSeries(noise_data_H1,t0 = ht_H1.t0.value - 2400, dt = 1.0/2048)
        noise_L1 = TimeSeries(noise_data_L1,t0 = ht_H1.t0.value - 2400, dt = 1.0/2048)
        noise_V1 = TimeSeries(noise_data_V1,t0 = ht_H1.t0.value - 2400, dt = 1.0/2048)
        '''
        signal_H1 = ht_H1.taper()
        signal_H1.plot()
        plt.savefig('template.png')
        '''
        data_H1 = noise_H1
        #signal_H1 = ht_H1.taper(side = 'left',duration = 5)
        signal_H1 = ht_H1
        data_H1 = data_H1.inject(signal_H1)
        signal_H1 = signal_H1.to_pycbc()
        
        #psd_H1_s = highpass(psd_H1_s,20)
        #psd_H1_s = psd_H1_s.crop(20,20)
        
        #psd_data_H1_s = data_H1.copy().crop(data_H1.t0.value,data_H1.t0.value+2000).to_pycbc()
        #psd_data_H1_s = data_H1.to_pycbc()
        #psd_H1_s = psd_data_H1_s.psd(4)
        data_H1 = data_H1.copy().crop(data_H1.t0.value+2300,data_H1.t0.value+2500).to_pycbc()
        data_H1 = highpass(data_H1,low).resample(1.0/2048)
        data_H1 = data_H1.crop(30,30)
        #data_H1 = data_H1.astype("float64")
        
        #psd_H1_d = psd_H1_d.astype("float64")
        #psd_H1_d = inverse_spectrum_truncation(psd_H1_d, 4*2048, low_frequency_cutoff=low)
        #data_H1 = data_H1.to_frequencyseries()
        '''
        pylab.loglog(psd_H1.sample_frequencies, psd_H1)
        pylab.xlim(50, 1000)
        pylab.ylim(1e-48,1e-43)
        pylab.savefig('psd.png')
        pylab.close()
        '''
        print('state done')
        #psd_H1_signal = interpolate(psd_H1_s, signal_delta_f)

        #signal_L1 = ht_L1.taper(side = 'left',duration = 5)
        signal_L1 = ht_L1
        data_L1 = noise_L1
        data_L1 = data_L1.inject(signal_L1)
        signal_L1 = signal_L1.copy()
        signal_L1 = signal_L1.to_pycbc()
        
        #psd_data_L1_s = data_L1.to_pycbc()
        #psd_data_L1_s = highpass(psd_data_L1_s,20)
        #psd_data_L1_s = psd_data_L1_s.crop(20,20)
        #
        #psd_L1_s = data_L1.to_pycbc().psd(4)
        data_L1 = data_L1.copy().crop(data_L1.t0.value+2300,data_L1.t0.value+2500).to_pycbc()
        data_L1 = highpass(data_L1,low).resample(1.0/2048)
        data_L1 = data_L1.crop(30,30)
        data_L1 = data_L1.astype("float64")
        
        #psd_L1_d = psd_L1_d.astype("float64")
        #psd_L1_signal = interpolate(psd_L1_s, signal_L1.delta_f)
        #psd_L1_d = inverse_spectrum_truncation(psd_L1_d, 4*2048, low_frequency_cutoff=low)
        

        #signal_V1 = ht_V1.taper(side = 'left', duration = 5)
        signal_V1 = ht_V1
        data_V1 = noise_V1
        data_V1 = data_V1.inject(signal_V1)
        signal_V1 = signal_V1.copy().to_pycbc()
        #psd_data_V1_s = data_V1.to_pycbc()
        
        #psd_data_V1_s = highpass(psd_data_V1_s,20)
        #psd_data_V1_s = psd_data_V1_s.crop(20,20)
        #psd_V1_s = psd_data_V1_s.psd(8)
        #psd_V1_s = data_V1.to_pycbc().psd(4)
        data_V1 = data_V1.copy().crop(data_V1.t0.value+2300,data_V1.t0.value+2500).to_pycbc()
        data_V1 = highpass(data_V1,low).resample(1.0/2048)
        data_V1 = data_V1.crop(30,30)
        #data_V1 = data_V1.astype("float64")
        
        psd_H1_d = interpolate(psd_H1_s, data_H1.delta_f)
        psd_L1_d = interpolate(psd_L1_s, data_L1.delta_f)
        psd_V1_d = interpolate(psd_V1_s, data_V1.delta_f)
        #psd_V1_d = psd_V1_d.astype("float64")
        #psd_V1_d = inverse_spectrum_truncation(psd_V1_d, 4*2048, low_frequency_cutoff=low)
        #data_V1 = data_V1.to_frequencyseries()
        '''
        psd_V1_signal = interpolate(psd_V1_s, signal_V1.delta_f)
        pylab.loglog(psd_H1.sample_frequencies, psd_H1)
        pylab.xlim(100,512)
        pylab.ylim(1e-47,1e-45)
        pylab.savefig('psd_V.png')
        pylab.close()
        print('state done')
        '''
        mass1_dev = 1
        mass2_dev = 1
        spin1z_dev  = 1
        spin2z_dev  = 1
        hp, hc = get_td_waveform(approximant = 'SEOBNRv4', mass1 = m1*mass1_dev, mass2 = m2*mass2_dev,spin1z = spin1, spin2z = spin2,f_lower = f_lower, delta_t = data_H1.delta_t )
        st_crop = -template_length/2 - hp.sample_times[0]
        #ed_crop = hp.sample_times[-1] + 1/2048.0 - template_length/2
        #hp = hp.crop(st_crop, 0)
        #hp = td_taper(out=template, start=template.sample_times[0], end=template.sample_times[0]+1, side="left")
        #a = time-100
        #hp = TimeSeries(hp,t0 = a ,dt=1.0/2048,channel = 'H1')
        
        #hp = hp.taper(side = 'left', duration = 5)
        #hp.plot()
        #plt.savefig('template.png')
        #plt.close()
        
        #hp = hp.to_pycbc()
        '''
        hp = hp.to_timeseries()
        hp = TimeSeries(hp, dt=hp.delta_t, t0=time)
        hp = hp.resample(2048)
        hp = hp.to_pycbc()
        print(len(hp))
        '''
        #hp.resize(len(data_H1))
        #hp = hp.cyclic_time_shift(hp.start_time)
        hp_template = hp.copy()
        print(len(hp))
        '''
        plt.plot(hp)
        plt.savefig('template.png')
        plt.close()
        '''
        hp_template_H = hp_template 
        '''
        st_crop = -40 - hp_template_H.sample_times[0]
        ed_crop = hp_template_H.sample_times[-1] - 0 + 1.0/sample_rate
        hp_template_H = hp_template_H.crop(st_crop, 0)
        '''
        #hp_template_H = hp_template_H.to_frequencyseries(delta_f = data_H1.delta_f)
        hp_template_H.resize(len(data_H1))
        hp_template_H = hp_template_H.cyclic_time_shift(hp_template_H.start_time)
        
        hp_template_L = hp_template
        #hp_template_L = hp_template_L.to_frequencyseries(delta_f = data_L1.delta_f)
        
        '''
        st_crop = -40 - hp_template_L.sample_times[0]
        ed_crop = hp_template_L.sample_times[-1] - 0 + 1.0/sample_rate
        hp_template_L = hp_template_L.crop(st_crop, 0)
        '''
        hp_template_L.resize(len(data_L1))
        hp_template_L = hp_template_L.cyclic_time_shift(hp_template_L.start_time)

        hp_template_V = hp_template
        #hp_template_V = hp_template_V.to_frequencyseries(delta_f = data_V1.delta_f)
        '''
        st_crop = -40 - hp_template_V.sample_times[0]
        ed_crop = hp_template_V.sample_times[-1] - 0 + 1.0/sample_rate
        hp_template_V = hp_template_V.crop(st_crop, 0)
        '''
        hp_template_V.resize(len(data_V1))
        hp_template_V = hp_template_V.cyclic_time_shift(hp_template_V.start_time)

        snr_series_H1 = matched_filter(hp_template_H,data_H1, psd=psd_H1_d, low_frequency_cutoff=cut_low,high_frequency_cutoff = 1024)
        print('match filter state done')
        #snr_series_H1_signal = matched_filter(hp_template_H,signal_H1, psd=psd_H1_signal, low_frequency_cutoff=30.0)
        print('len = ',len(snr_series_H1))
        snr_series_H1 = snr_series_H1.crop(55, 55)
        #snr_series_H1_signal = snr_series_H1_signal.crop(53, 12)
        snr_array_H1 = abs(np.array(snr_series_H1))
        snr_series_L1 = matched_filter(hp_template_L,data_L1, psd=psd_L1_d, low_frequency_cutoff=cut_low,high_frequency_cutoff = 1024)
        #snr_series_L1_signal = matched_filter(hp_template_L,signal_L1, psd=psd_L1_signal, low_frequency_cutoff=30.0)
        print('match filter state done')
        snr_series_L1 = snr_series_L1.crop(55, 55)
        #snr_series_L1_signal = snr_series_L1_signal.crop(53, 12)
        snr_array_L1 = abs(np.array(snr_series_L1))

        snr_series_V1 = matched_filter(hp_template_V,data_V1, psd=psd_V1_d, low_frequency_cutoff=cut_low,high_frequency_cutoff = 1024)
        #snr_series_V1_signal = matched_filter(hp_template_V,signal_V1, psd=psd_V1_signal, low_frequency_cutoff=30.0
        snr_series_V1 = snr_series_V1.crop(55, 55)
        #snr_series_V1_signal = snr_series_V1_signal.crop(53, 12)
        snr_array_V1 = abs(np.array(snr_series_V1))
        print(snr_array_V1)
        merger_L1 = np.argmax(abs(snr_series_L1))
        merger_H1 = merger_L1 + int((shift_H -shift_L)*2048)
        merger_V1 = merger_L1 + int((shift_V -shift_L)*2048)
        #snr_series_H1 = snr_series_H1[merger_H1-500:merger_H1+500]
        #snr_series_L1 = snr_series_L1[merger_L1-500:merger_L1+500]
        #snr_series_V1 = snr_series_V1[merger_V1-500:merger_V1+500]
        H1_merger_time = 1187008882.4 + dt_h1l1
        L1_merger_time = 1187008882.4 + dt_h1l1
        V1_merger_time = 1187008882.4 + dt_h1v1
        print(L1_merger_time)
        print(V1_merger_time)
        #merger_H1 = np.argmax(abs(snr_series_H1))
        snr_H1 = snr_array_H1[merger_H1]

        #merger_L1 = np.argmax(abs(snr_series_L1))
        snr_L1 = snr_array_L1[merger_L1]
        print(merger_H1)
        #merger_V1 = np.argmax(abs(snr_series_V1))
        snr_V1 = snr_array_V1[merger_V1]
        print(snr_H1, snr_L1, snr_V1)

        '''
        plt.plot(abs(snr_series_H1),label = 'H1')
        plt.plot(abs(snr_series_L1),label = 'L1')
        plt.plot(abs(snr_series_V1),label = 'V1')
        plt.legend()
        plt.savefig('NSBH.png')
        plt.close()
        '''
        
        #if True:
        
        
        print(merger_H1, merger_L1, merger_V1)

        
        snr = 0
        snr += snr_H1**2
        snr += snr_L1**2
        snr += snr_V1**2
        snr = snr**0.5
        print('SNR = ',snr)

        if snr > 8 and snr < 30:
        #if True:
            #merger_V1
            #merger_L1
            #merger_H1
            
            snr_series_H1_s = np.array(snr_series_H1)[merger_H1-int(0.1*2048)-1:merger_H1+int(0.1*2048)+1]
            snr_series_L1_s = np.array(snr_series_L1)[merger_L1-int(0.1*2048)-1:merger_L1+int(0.1*2048)+1]
            snr_series_V1_s = np.array(snr_series_V1)[merger_V1-int(0.1*2048)-1:merger_V1+int(0.1*2048)+1]
            
            print('SNR = ',snr)
            #if True:
            if len(snr_series_H1_s) == 410 and len(snr_series_L1_s) == 410 and len(snr_series_V1_s) == 410:
                print(f'Done {i}th data')
                return {
                            'snr_series_H1': snr_series_H1_s,
                            'snr_series_L1': snr_series_L1_s,
                            'snr_series_V1': snr_series_V1_s,
                            'm1': m1,
                            'm2': m2,
                            'spin1': spin1,
                            'spin2': spin2,
                            'time': time,
                            'snr': snr,
                            'ra_test': ra_test,
                            'dec_test': dec_test,
                            'H1_merger_time':H1_merger_time,
                            'V1_merger_time':V1_merger_time,
                            'L1_merger_time':L1_merger_time
                        }
            else: return None
        else:
            return None

    except Exception as e:
        print('input error',m1,m2)
        print(e)
        return None
   

'''
for k in range (len(h1_snr_series)):
    plt.plot(abs(h1_snr_series[k]),label = 'H1')
    plt.plot(abs(l1_snr_series[k]),label = 'L1')
    plt.plot(abs(v1_snr_series[k]),label = 'V1')
    plt.legend()
    plt.show()
    plt.close()
'''
'''
def process_task(numbers):
    """Each process runs a Pool to parallelize tasks"""
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(generate, numbers)
    #print(f"Results from process {multiprocessing.current_process().name}: {results}")
'''

def retry_task(n):
    """Retry the task if it fails."""
    retries = 0
    while True:
        result = generate(n)
        if result is not None:
            return result  # Return the correct result if no failure
            break
        retries += 1
        print(f"Retrying task {n} (attempt {retries})...")
        
    return None

if __name__ == "__main__":
    #approximant = 'IMRPhenomPv2_NRTidal'
    #approximant = 'IMRPhenomD'
    #approximant = 'SpinTaylorT4'
    template_length=10
    approximant = 'SEOBNRv4_opt'
    f_lower = 20
    low = 20
    high = 1024
    cut_low = 30
    delta_t = 1/2048.0
    sample_rate = 2048
    time =  1187008882.4
    #detectors = ['H1','L1','V1']
    #num = 100000
    #i = 0
    noise_data_H1 = TimeSeries.read('/mnt/data/slash/Inj_Data/background_H1.hdf5')
    noise_data_L1 = TimeSeries.read('/mnt/data/slash/Inj_Data/background_L1.hdf5')
    noise_data_V1 = TimeSeries.read('/mnt/data/slash/Inj_Data/background_V1.hdf5')
    random_start = np.random.randint(0,10000)
    noise_data_H1 = noise_data_H1.resample(2048)
    noise_data_L1 = noise_data_L1.resample(2048)
    noise_data_V1 = noise_data_V1.resample(2048)
    noise_data_H1 = noise_data_H1.copy().crop(noise_data_H1.t0.value+random_start,noise_data_H1.t0.value+random_start+2500)
    noise_data_L1 = noise_data_L1.copy().crop(noise_data_L1.t0.value+random_start,noise_data_L1.t0.value+random_start+2500)
    noise_data_V1 = noise_data_V1.copy().crop(noise_data_V1.t0.value+random_start,noise_data_V1.t0.value+random_start+2500)
    
    psd_H1_data_s = noise_data_H1.copy().crop(noise_data_H1.t0.value+0,noise_data_H1.t0.value+2300).to_pycbc()
    psd_data_L1_s = noise_data_L1.copy().crop(noise_data_L1.t0.value+0,noise_data_L1.t0.value+2300).to_pycbc()
    psd_data_V1_s = noise_data_V1.copy().crop(noise_data_V1.t0.value+0,noise_data_V1.t0.value+2300).to_pycbc()
    psd_H1_data_s = highpass(psd_H1_data_s,low).resample(1.0/2048)
    psd_data_L1_s = highpass(psd_data_L1_s,low).resample(1.0/2048)
    psd_data_V1_s = highpass(psd_data_V1_s,low).resample(1.0/2048)
    
    psd_H1_s = psd_H1_data_s.psd(8)
    psd_L1_s = psd_data_L1_s.psd(8)
    psd_V1_s = psd_data_V1_s.psd(8)
    
    #numbers = list(range(5))  # Example input
    numbers = range(0, 100000)
    #num_workers = multiprocessing.cpu_count()
    #manager = multiprocessing.Manager()
    #pool_size = multiprocessing.cpu_count()
    with multiprocessing.Pool(20) as pool:
        results = pool.map(retry_task, numbers, chunksize=1)  # Run tasks in parallel
    for i, result in enumerate(results):
        #print(f"Result {i}: {result}")
        snr_series_H1 = result['snr_series_H1']
        snr_series_V1 = result['snr_series_V1']
        snr_series_L1 = result['snr_series_L1']
        m1 = result['m1']
        m2 = result['m2']
        spin1 = result['spin1']
        spin2 = result['spin2']
        time = result['time']
        snr = result['snr']
        ra_test = result['ra_test']
        dec_test = result['dec_test']
        H1_merger_time = result['H1_merger_time']
        V1_merger_time = result['V1_merger_time']
        L1_merger_time = result['L1_merger_time']
        RA.append(ra_test)
        DEC.append(dec_test)
        mass1.append(m1)
        mass2.append(m2)
        spin1z.append(spin1)
        spin2z.append(spin2)
        gps_time.append(time)
        Injection_SNR.append(snr)
        H_time.append(H1_merger_time)
        L_time.append(L1_merger_time)
        V_time.append(V1_merger_time)
        #print(snr_series_H1)
        h1_snr_series.append(snr_series_H1)
        l1_snr_series.append(snr_series_L1)
        v1_snr_series.append(snr_series_V1)

    print(h1_snr_series)
    print(V_time)
    print(L_time)
    h1_snr_series = np.array(h1_snr_series)
    l1_snr_series = np.array(l1_snr_series)
    v1_snr_series = np.array(v1_snr_series)
    #snr_series_h1 = np.roll(snr_series_H1,3)
    #snr_series_l1 = np.roll(snr_series_L1,int(dt_h1l1 * 2048))
    #h1_snr_series.append(snr_series_H1)
    #l1_snr_series.append(snr_series_L1)
    #v1_snr_series.append(snr_series_V1)
    f = h5py.File('/mnt/data/slash/Inj_Data/BNS/Parameters/BNS_TEST_parameters_LV_7.hdf','w')
    f.create_dataset('mass1',data = np.array(mass1))
    f.create_dataset('mass2',data = np.array(mass2))
    f.create_dataset('spin1z',data = np.array(spin1z))
    f.create_dataset('spin2z',data = np.array(spin2z))
    f.create_dataset('gps_time',data = np.array(gps_time))
    f.create_dataset('Injection_SNR',data = np.array(Injection_SNR))
    f.create_dataset('ra_test',data = np.array(RA))
    f.create_dataset('dec_test',data = np.array(DEC))
    f.create_dataset('H1_merger_time',data = np.array(H_time))
    f.create_dataset('L1_merger_time',data = np.array(L_time))
    f.create_dataset('V1_merger_time',data = np.array(V_time))
    f.close()

    #f = h5py.File('/mnt/data/slash/Inj_Data/Inj_Data_zoo_NRTidal_NSBH_TEST.hdf','w')
    f = h5py.File('/mnt/data/slash/Inj_Data/BNS/BNS_TEST_LV_7.hdf','w')
    f.create_dataset('h1_snr_series',data = h1_snr_series)
    f.create_dataset('v1_snr_series',data = v1_snr_series)
    f.create_dataset('l1_snr_series',data = l1_snr_series)
    #print(results)
