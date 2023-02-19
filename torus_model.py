import numpy as np
import random
from random import choices
import matplotlib.pyplot as plt
import celerite
from celerite import terms
import astropy.units as u
import astropy.constants as const
import scipy
from scipy.interpolate import interp1d


def simulate_transfer_func(incl, Rin, sigma, Y=1.5, p=-1.0, ncloud=1000000, dt=0.1, seed=1, 
        return_data=True, return_cloud=False, show_plot=False):
    '''
    incl: inclination angle in degree; 0 means face on and 90 means edge on

    Rin: inner radius in pc

    sigma: half-opening angle in degree

    Y: outer to inner radius ratio

    p: radial powerlaw index

    ncloud: cloud number

    dt: time interval of the transfer function, default 1 day

    return:
        delay_t: tau in days
        delay_phi: transfer function
    '''
    np.random.seed(seed)
    random.seed(seed)

    # beta: vertical distribution
    theta_cos = np.random.uniform(0, np.cos((90-sigma)*np.pi/180), ncloud)

    theta = np.arccos(theta_cos)
    theta_degree = theta*180/np.pi

    beta_degree = 90 - theta_degree
    beta_degree = np.append(-beta_degree, beta_degree)

    # assuming gaussian distribution
    beta_prob = (1./(np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*(np.abs(beta_degree)/sigma)**2)

    beta_degree_random = choices(beta_degree, weights=beta_prob, k=len(beta_degree))
    beta = np.array(beta_degree_random) * np.pi / 180

    # phi
    phi = np.random.uniform(0, 2*np.pi, len(beta))

    # inner and outer radius
    Rout = Rin * Y
    RR = np.random.uniform(Rin, Rout, len(beta))
    R_prob = RR**p
    R_random = choices(RR, weights=R_prob, k=len(beta))

    incl = incl*np.pi/180.0
    cos_alpha = np.cos(beta)*np.cos(phi)*np.sin(incl) + np.sin(beta)*np.cos(incl)

    delay = (R_random*u.pc * (1-cos_alpha) / const.c).to(u.day).value

    delay_hist = np.histogram(delay, bins=np.arange(0, 1e5, dt), density=True)
    delay_t = (delay_hist[1][:-1] + delay_hist[1][1:]) / 2.0
    delay_phi = delay_hist[0]

    tm = int(((2.0*Rin*Y)*u.pc / const.c).to(u.day).value)

    if tm < 10:
        tmax = 10
    else:
        idx = delay_t > tm
        zero_idx = list(delay_phi[idx]).index(0)
        tmax = int(1.1*delay_t[idx][zero_idx])

    delay_phi = delay_phi[delay_t <= tmax]
    delay_t = delay_t[delay_t <= tmax]

    if show_plot:
        plt.figure()
        plt.plot(delay_t, delay_phi)
        plt.xlabel(r'$\tau$ (days)')
        plt.ylabel(r'$\phi(\tau)$')
        plt.xlim(0,)
        plt.ylim(0,)
        plt.show()
    
    if return_data:
        if return_cloud:
            return delay_t, delay_phi, R_random, beta, phi, delay
        else:
            return delay_t, delay_phi


def simulate_drw_lc(t, tau, sigma, flux_mean, seed=1):
    """    
    t: rest-frame time 
    tau: DRW damping timescale
    sigma: rms of the drw light curve
    flux_mean: mean flux of the light curve
    """
    np.random.seed(seed) 
    
    log_a = np.log(sigma**2)
    log_c = np.log(1/tau)
    kernel = terms.RealTerm(log_a=log_a, log_c=log_c)
    
    gp = celerite.GP(kernel, mean=flux_mean)
    gp.compute(t)
    
    y = gp.sample(size=1)[0]
    return y


def simulate_perfect_light_curve(tau_g, sigma_g, flux_mean, 
                         incl, Rin, sigma, Y, p, amp=1.0,
                         ncloud=1000000, dt=1.0,
                         initial_length=9000, cadence=5,
                         seed=1):
    '''
    Simulate idealized optical light curve from the DRW process. 
    Convolve the optical light curve with the torus transfer function to derive the MIR light curve.
    
    DRW parameters:
        tau_g: damping timescale
        sigma_g: rms variability
        flux_mean: mean flux
        
    Torus geomtry:
        incl: inclination angle in degree; 0 means face on and 90 means edge on
        Rin: inner radius in pc
        sigma: half-opening angle in degree
        Y: outer to inner radius ratio
        p: radial powerlaw index
        dt: time interval of the transfer function (day); a small number is preferred (e.g., 1)
        ncloud: cloud number 
        amp: MIR/optical flux ratio
        
    Light curve:   
        cadence: cadence of the simulate light
        initial_length: initial length used to generate the MIR light curve; 
        
    Returns:
        simulated light curves in the optical and MIR.
    '''
            
    # simulate transfer function
    delay_t, delay_phi = simulate_transfer_func(
                            incl, Rin, sigma, Y, p, ncloud, dt, seed, return_data=True, return_cloud=False, show_plot=False)
    tmax = max(delay_t)
    
    # simulate optical light curve
    mjd_all = np.arange(0, tmax+initial_length+dt, dt)
    flux_g_all = simulate_drw_lc(mjd_all, tau=tau_g, sigma=sigma_g, flux_mean=flux_mean, seed=seed)

    # simulate MIR light curve
    flux_w_all = amp * np.convolve(delay_phi, flux_g_all) * dt
    idx1 = len(delay_t)
    idx2 = len(flux_g_all)
            
    flux_w_valid = flux_w_all[idx1:idx2]
    mjd_w_valid = mjd_all[idx1:idx2] 
    
    lc_w = interp1d(mjd_w_valid, flux_w_valid)
    lc_g = interp1d(mjd_all, flux_g_all)
    
    mjd_g = np.arange(min(mjd_all), max(mjd_all), cadence)
    mjd_w = np.arange(min(mjd_w_valid), max(mjd_w_valid), cadence)
    
    flux_w = lc_w(mjd_w)
    flux_g = lc_g(mjd_g)
    
    return flux_g, mjd_g, flux_w, mjd_w


def mag_var_at_delta_t(mjd, w1):
    array_length = int(len(mjd)*(len(mjd)-1)/2)
    delta_t = np.zeros(array_length,)
    delta_mag2 = np.zeros(array_length,)
            
    mjd_length = len(mjd)

    for i in range(mjd_length-1):
        begin = int(i*(mjd_length-1) - (0+i-1)*i/2)
        end = int((i+1)*(mjd_length-1) - (0+i)*(i+1)/2)
        if begin < 0:
            begin = 0
        delta_t[begin:end] = np.abs(mjd[i+1:] - mjd[i])
        delta_mag2[begin:end] = (w1[i] - w1[i+1:])**2 

    return delta_t, delta_mag2


def sf(sf_result, seed, tau_g, sigma_g, flux_mean,
       incl, Rin, sigma, Y, p, amp,
       ncloud, dt, initial_length, cadence,
       bin_range = 10**np.arange(np.log10(10.0), np.log10(6000)+0.1, 0.1)):
    '''
    structure function of individual light curve
    '''

    flux_g, mjd_g, flux_w, mjd_w = simulate_perfect_light_curve(
                         tau_g, sigma_g, flux_mean,
                         incl, Rin, sigma, Y, p, amp,
                         ncloud, dt, initial_length, cadence,
                         seed)

    # MIR structure function
    delta_t, delta_mag2 = mag_var_at_delta_t(mjd_w, flux_w)

    bin_50 = scipy.stats.binned_statistic(delta_t, delta_mag2,
                                statistic='mean',
                                bins=bin_range)

    tbin_50 = scipy.stats.binned_statistic(delta_t, delta_t,
                                statistic='mean',
                                bins=bin_range)

    lag_w, sf2_w = tbin_50[0], bin_50[0]
    idx = sf2_w > 0
    sf_w = np.sqrt(sf2_w[idx])
    lag_w = lag_w[idx]
    sf2_w = sf2_w[idx]

    if sf_result != None:
        sf_result['{}'.format(seed)] = {'lag_w': lag_w, 'sf_w': sf_w, 'sf2_w': sf2_w}
    return sf_result


def ensemble_sf(band, sf_result, ensemblue_method='mean_sqrt'):
    '''
    method:
        mean_sqrt: derive mean of sf2_w, then sqrt
        sqrt_mean: derive sqrt first, then calculate mean
    '''
    lag, sf = [], []

    for i in range(len(sf_result)):
        name = str(i)
        lag.append(sf_result[name]['lag_'+band])
        if ensemblue_method == 'sqrt_mean':
            sf.append(sf_result[name]['sf_'+band])
        else:
            sf.append(sf_result[name]['sf2_'+band])

    lag_50 = np.nanmean(lag, axis=0)
    if ensemblue_method == 'sqrt_mean':
        sf_50 = np.nanmean(sf, axis=0)
    else:
        sf_50 = np.sqrt(np.nanmean(sf, axis=0))

    return lag_50, sf_50


def model_sf(tau_g, sigma_g, flux_mean,
             incl, Rin, sigma, Y, p, amp=1.0,
             ncloud=100000, dt=1.0,
             initial_length=10000, cadence=5, ensemble_num=5,
             ensemblue_method='mean_sqrt'):
    '''
    model structure function derived from N ensembles
    '''
    sf_result = {}
    params = []

    print('Start...')
    for seed in range(ensemble_num):
        if seed in np.arange(20, ensemble_num+20, 20):
            print('ensemble {}...'.format(seed))
        sf(sf_result, seed, tau_g, sigma_g, flux_mean,
           incl, Rin, sigma, Y, p, amp,
           ncloud, dt, initial_length, cadence)
    print('ensemble {}... Done'.format(ensemble_num))

    lag_w, sf_w = ensemble_sf('w', sf_result, ensemblue_method)

    sf_params = {'lag_w': lag_w, 'sf_w': sf_w}

    return sf_params


def sf_analytic(delta_t, SFinf, tau, beta):
    delta_t = np.abs(delta_t)
    y = SFinf**2 * (1 - np.exp(-(delta_t/tau)**beta))
    return np.sqrt(y)




