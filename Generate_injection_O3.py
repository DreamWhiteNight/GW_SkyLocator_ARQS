import pycbc
from pycbc.waveform import get_td_waveform
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
gps_time = []
def generate(i):
    #set the parameters for GW waveform
    ra_test = float(random.uniform(0,1))
    dec_test = float(random.uniform(0,1)) 
    pol = random.uniform(0,0.99)
    pol *= np.pi
    ra = 2.0 * np.pi * ra_test
    dec  = np.arcsin(1.0 - 2.0 * dec_test)
    spin1 = random.uniform(-0.99,0.99)
    spin2 = random.uniform(-0.99,0.99)
    inclination = random.uniform(0,0.99)
    inclination *= np.pi
    m1 = float(np.random.randint(5,50)) + np.round(random.random(), decimals = 3)
    m2 = float(np.random.randint(5,50)) + np.round(random.random(), decimals = 3)
    distance = np.random.randint(400,650)
    if m1 < m2 :
        m1, m2 = m2, m1
    
    try:
        #Generate the waveform
        hp, hc = get_td_waveform(approximant = approximant, mass1 = m1, mass2 = m2, spin1z = spin1, spin2z = spin2, inclination = inclination ,distance = distance ,f_lower = f_lower, delta_t = 1.0/4096)
        hp.resample(1.0/2048)
        hc.resample(1.0/2048)
        #print('start_time : ',hp.start_time,'Length',len(hp))
        start_time = time + float(hp.start_time)
        D = 'H1'
        E = 'L1'
        F = 'V1'
        detector_D = Detector(D)
        detector_E = Detector(E)
        detector_F = Detector(F)
        dt_h1l1 = detector_E.time_delay_from_detector(detector_D, ra, dec, time) #caculate the time difference
        #print(dt_h1l1)
        dt_h1v1 = detector_F.time_delay_from_detector(detector_D, ra, dec, time)
        '''
        fp, fc = detector.antenna_pattern(ra, dec, pol, time)
        ht = fp * hp + fc * hc
        '''
        hc.start_time = hp.start_time
        ht_H1 = detector_D.project_wave(hp, hc, ra, dec, pol)
        ht_H1 = TimeSeries(ht_H1,dt = 1.0/2048,t0 = start_time, channel = D) # Project the waveform to different detectors
        ht_L1 = detector_E.project_wave(hp, hc, ra, dec, pol)
        ht_L1 = TimeSeries(ht_L1,dt = 1.0/2048,t0 = start_time, channel = E)
        ht_V1 = detector_E.project_wave(hp, hc, ra, dec, pol)
        ht_V1 = TimeSeries(ht_V1,dt = 1.0/2048,t0 = start_time, channel = F)
        noise_data_H1 = TimeSeries.read('/mnt/data/slash/Inj_Data/background_H1.hdf5') # read the background noise for PSD
        noise_data_L1 = TimeSeries.read('/mnt/data/slash/Inj_Data/background_L1.hdf5')
        noise_data_V1 = TimeSeries.read('/mnt/data/slash/Inj_Data/background_V1.hdf5')
        print('noise done')
        noise_data_H1 = noise_data_H1.to_pycbc()
        noise_data_L1 = noise_data_L1.to_pycbc()
        noise_data_V1 = noise_data_V1.to_pycbc()
        psd_H1 = noise_data_H1.psd(4)
        psd_L1 = noise_data_L1.psd(4)
        psd_V1 = noise_data_V1.psd(4)
        psd_H1 = interpolate(psd_H1, noise_data_H1.delta_f)
        psd_L1 = interpolate(psd_L1, noise_data_L1.delta_f)
        psd_V1 = interpolate(psd_V1, noise_data_V1.delta_f)

        flow = 30.0
        delta_f = 1.0 / 16
        flen = int(2048 / delta_f) + 1

        noise_H1 = noise_from_psd(10*2048, 1.0/2048, psd_H1, seed=np.random.randint(1,1000000)) # Generate the noise by using PSD
        noise_L1 = noise_from_psd(10*2048, 1.0/2048, psd_L1, seed=np.random.randint(1,1000000))
        noise_V1 = noise_from_psd(10*2048, 1.0/2048, psd_V1, seed=np.random.randint(1,1000000))
        noise_H1 = TimeSeries(noise_H1,dt = 1.0/2048,t0 = start_time-5,channel = D)
        signal_H1 = ht_H1.taper()
        data_H1 = noise_H1.inject(signal_H1)
        data_H1 = data_H1.to_pycbc()
        #print(len(data))
        noise_H1 = noise_H1.to_pycbc()
        psd_H1 = data_H1.psd(4)
        psd_H1 = interpolate(psd_H1, data_H1.delta_f)
        psd_H1 = inverse_spectrum_truncation(psd_H1, 4*2048, low_frequency_cutoff=20.0)
        # Inject the templates to noise
        noise_L1 = TimeSeries(noise_L1,dt = 1.0/2048,t0 = start_time-5,channel = E)
        signal_L1 = ht_L1.taper()
        data_L1 = noise_L1.inject(signal_L1)
        data_L1 = data_L1.to_pycbc()
        #print(len(data))
        noise_L1 = noise_L1.to_pycbc()
        psd_L1 = data_L1.psd(4)
        psd_L1 = interpolate(psd_L1, data_L1.delta_f)
        psd_L1 = inverse_spectrum_truncation(psd_L1, 4*2048, low_frequency_cutoff=20.0)

        noise_V1 = TimeSeries(noise_V1,dt = 1.0/2048,t0 = start_time-5,channel = F)
        signal_V1 = ht_V1.taper()
        data_V1 = noise_V1.inject(signal_V1)
        data_V1 = data_V1.to_pycbc()
        #print(len(data))
        noise_V1 = noise_V1.to_pycbc()
        psd_V1 = data_V1.psd(4)
        psd_V1 = interpolate(psd_V1, data_V1.delta_f)
        psd_V1 = inverse_spectrum_truncation(psd_V1, 4*2048, low_frequency_cutoff=20.0)

        hp, hc = get_td_waveform(approximant = approximant, mass1 = m1, mass2 = m2,f_lower = f_lower, delta_t = delta_t)

        hp.resize(len(data_H1))

        template = hp.cyclic_time_shift(float(hp.start_time))
        # Perform matched filtering
        snr_series_H1 = matched_filter(template,data_H1, psd=psd_H1, low_frequency_cutoff=20.0)
        snr_series_H1 = snr_series_H1.crop(3, 3)
        snr_array_H1 = abs(np.array(snr_series_H1))

        snr_series_L1 = matched_filter(template,data_L1, psd=psd_L1, low_frequency_cutoff=20.0)
        snr_series_L1 = snr_series_L1.crop(3, 3)
        snr_array_L1 = abs(np.array(snr_series_L1))

        snr_series_V1 = matched_filter(template,data_V1, psd=psd_V1, low_frequency_cutoff=20.0)
        snr_series_V1 = snr_series_V1.crop(3, 3)
        snr_array_V1 = abs(np.array(snr_series_V1))

        merger_H1 = np.argmax(snr_array_H1[2048:2048*3])
        snr_H1 = snr_array_H1[merger_H1+2048]

        merger_L1 = np.argmax(snr_array_L1[2048:2048*3])
        snr_L1 = snr_array_L1[merger_L1+2048]
        print(merger_H1)
        merger_V1 = np.argmax(snr_array_V1[int(2048):int(2048)*3])
        snr_V1 = snr_array_V1[merger_V1+2048]

        snr = (snr_H1**2 + snr_L1**2 + snr_V1**2)**0.5
        print(snr)
        if snr > 10 and snr < 30 :
            merger_V1 += 2048
            merger_L1 += 2048
            merger_H1 += 2048
            snr_series_H1 = np.array(snr_series_H1)[merger_H1-int(0.1*2048)-1:merger_H1+int(0.1*2048)+1]
            snr_series_L1 = np.array(snr_series_L1)[merger_L1-int(0.1*2048)-1-int(dt_h1l1 * 2048):merger_L1+int(0.1*2048)+1-int(dt_h1l1 * 2048)]
            snr_series_V1 = np.array(snr_series_V1)[merger_V1-int(0.1*2048)-1-int(dt_h1v1 * 2048):merger_V1+int(0.1*2048)+1-int(dt_h1v1 * 2048)]
            
            if len(snr_series_H1) != 410 or len(snr_series_L1) != 410 or len(snr_series_V1) != 410:
                return None
            else:
                #print(len(snr_series_H1))
                print(f'Done {i+1}th data')
                '''
                RA.append(ra_test)
                DEC.append(dec_test)
                mass1.append(m1)
                mass2.append(m2)
                spin1z.append(spin1)
                spin2z.append(spin2)
                gps_time.append(time)
                Injection_SNR.append(snr)
                #snr_series_h1 = np.roll(snr_series_H1,3)
                #snr_series_l1 = np.roll(snr_series_L1,int(dt_h1l1 * 2048))
                h1_snr_series.append(snr_series_H1)
                l1_snr_series.append(snr_series_L1)
                v1_snr_series.append(snr_series_V1)
                '''
                return {
                            'snr_series_H1': snr_series_H1,
                            'snr_series_L1': snr_series_L1,
                            'snr_series_V1': snr_series_V1,
                            'm1': m1,
                            'm2': m2,
                            'spin1': spin1,
                            'spin2': spin2,
                            'time': time,
                            'snr': snr,
                            'ra_test': ra_test,
                            'dec_test': dec_test
                        }

        else:
            return None

    except:
        print('input error',m1,m2)
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
    approximant = 'SEOBNRv4_opt'
    f_lower = 30
    delta_t = 1/2048.0

    time =  1187008882.4 #merger time
    #detectors = ['H1','L1','V1']
    #num = 50000
    #i = 0

    #numbers = list(range(5))  # Example input
    numbers = range(0, 100000)
    #num_workers = multiprocessing.cpu_count()
    #manager = multiprocessing.Manager()
    #pool_size = multiprocessing.cpu_count()
    with multiprocessing.Pool(20) as pool:
        results = pool.map(retry_task, numbers, chunksize=2)  # Run tasks in parallel
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
        RA.append(ra_test)
        DEC.append(dec_test)
        mass1.append(m1)
        mass2.append(m2)
        spin1z.append(spin1)
        spin2z.append(spin2)
        gps_time.append(time)
        Injection_SNR.append(snr)
        #print(snr_series_H1)
        h1_snr_series.append(snr_series_H1)
        l1_snr_series.append(snr_series_L1)
        v1_snr_series.append(snr_series_V1)

    h1_snr_series = np.array(h1_snr_series)
    l1_snr_series = np.array(l1_snr_series)
    v1_snr_series = np.array(v1_snr_series)
    
    #snr_series_h1 = np.roll(snr_series_H1,3)
    #snr_series_l1 = np.roll(snr_series_L1,int(dt_h1l1 * 2048))
    #h1_snr_series.append(snr_series_H1)
    #l1_snr_series.append(snr_series_L1)
    #v1_snr_series.append(snr_series_V1)

    f = h5py.File('/mnt/data/slash/Inj_Data/Inj_Data_zoo_5_parameters.hdf','w')
    f.create_dataset('mass1',data = np.array(mass1))
    f.create_dataset('mass2',data = np.array(mass2))
    f.create_dataset('spin1z',data = np.array(spin1z))
    f.create_dataset('spin2z',data = np.array(spin2z))
    f.create_dataset('gps_time',data = np.array(gps_time))
    f.create_dataset('Injection_SNR',data = np.array(Injection_SNR))
    f.create_dataset('ra_test',data = np.array(RA))
    f.create_dataset('dec_test',data = np.array(DEC))
    f.close()

    f = h5py.File('/mnt/data/slash/Inj_Data/Inj_Data_zoo_5.hdf','w')
    f.create_dataset('h1_snr_series',data = h1_snr_series)
    f.create_dataset('v1_snr_series',data = v1_snr_series)
    f.create_dataset('l1_snr_series',data = l1_snr_series)
    #print(results)

