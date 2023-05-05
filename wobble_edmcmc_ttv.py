#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 11:30:25 2022

@author: zdebeurs

TO DO:
    - add in TTV fitter [Semi-MAJOR TASK]
    - add in Gaia simultaneous fitting [MAJOR TASK]

"""
import numpy as np
import matplotlib.pyplot as plt
import edmcmc as edm
import radvel
from IPython.display import display, Math
import pandas as pd
#from random import sample

class wobble_edmcmc(object):
    
    # Initialize the class (often empty)
    def __init__(self, t, y, yerr, labels,
                  M1, nwalkers, nlink, nburnin, nfree_params,
                  
                  # non-default widths
                  w_Mp, w_logP, w_tp, w_e, w_om, w_D1,
                  
                  
                  # params
                  K, Mp, logP, tp,e0, D1, jit1, om0, lin=0, quad=0, de_dt=0, dom_dt=0, de2_dt2=0, dom2_dt2=0, 
                  K2=0, Mp2=0, logP2=0, tp2=0, e02=0, de2_dt=0, om02=0, dom2_dt=0, 
                  D2 = 0, D3 = 0, jit2=0, jit3=0,
                  
                  # mcmc settings
                  ncores = 1,
                  
                  # priors
                  pr_K=[np.nan, np.nan], pr_Mp =[np.nan, np.nan], pr_logP=[np.nan, np.nan], pr_tp=[np.nan, np.nan], pr_e=[0,1],pr_D1=[np.nan, np.nan],
                  pr_D2=[np.nan, np.nan], pr_D3=[np.nan, np.nan],
                  pr_om=[0, 2*np.pi], pr_lin=[np.nan, np.nan], pr_quad=[np.nan, np.nan], pr_de_dt=[np.nan, np.nan],
                  pr_dom_dt=[np.nan, np.nan], pr_de2_dt2 = [np.nan, np.nan], pr_dom2_dt2 = [np.nan, np.nan],
                  pr_K2=[np.nan, np.nan],pr_Mp2 =[np.nan, np.nan], pr_logP2=[np.nan, np.nan],pr_tp2 =[np.nan, np.nan], 
                  pr_e2=[np.nan, np.nan], pr_de2_dt=[np.nan, np.nan], pr_om2=[np.nan, np.nan],pr_dom2_dt=[np.nan, np.nan], 
                  pr_jit1 = [-1,200], pr_jit2 = [-1,200], pr_jit3 = [-1,200],
                  
                  #width
                  w_K = 0, w_D2=0, w_D3=0, w_lin=0, w_quad=0, w_de_dt=0, w_dom_dt=0, w_de2_dt2=0, w_dom2_dt2=0, w_K2=0, w_Mp2=0, w_logP2=0, w_e2=0, w_tp2 = 0, w_de2_dt=0, w_om2=0, w_dom2_dt=0,
                  w_jit1 = 0, w_jit2 = 0, w_jit3 = 0,
                  # number of telescopes
                  telescopes = 0,
                  
                  # phase for initial plotting
                  phase_plot=0.5,
                  
                  # gaussian priors on on some parameters
                  gausspriors=False, gauss_theta_indexes=0,gauss_theta_priors=0, gauss_theta_err=0, use_transit_times = False,
                  transitpriors = False, transit_times_list=[], 
                  use_mass_parametrization = False, e_med_list=[], e_err = [], om_med_list = [], om_err =[]
                  
                  ):
        self.t = t
        self.y = y
        self.yerr = yerr
        self.labels = labels
        self.M1 = M1
        
        self.nwalkers = nwalkers
        self.nlink = nlink
        self.nburnin = nburnin
        self.ncores = ncores
        self.nfree_params = nfree_params
        
        self.K = K
        self.Mp = Mp
        self.logP = logP
        self.tp = tp
        self.e0 = e0
        self.de_dt = de_dt
        self.om0 = om0
        self.lin  = lin
        self.quad = quad
        self.dom_dt = dom_dt
        self.de2_dt2 = de2_dt2
        self.dom2_dt2 = dom2_dt2
        self.K2 = K2
        self.Mp2 = Mp2
        self.logP2 = logP2
        self.tp2 = tp2
        self.e02 = e02
        self.de2_dt = de2_dt
        self.om02 = om02
        self.dom2_dt = dom2_dt
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.jit1 = jit1
        self.jit2 = jit2
        self.jit3 = jit3
        self.theta = [self.Mp, self.logP, self.tp, self.e0, self.de_dt, 
                  self.om0, self.dom_dt, self.de2_dt2, self.dom2_dt2,self.Mp2, self.logP2, self.tp2,
                  self.e02, self.de2_dt, self.om02, self.dom2_dt, self.D1,
                  self.D2, self.D3, self.jit1, self.jit2, self.jit3, self.lin, self.quad]
        self.theta_priors = [pr_Mp, pr_logP, pr_tp, pr_e, pr_de_dt, 
                               pr_om, pr_dom_dt, pr_de2_dt2, pr_dom2_dt2,
                               pr_Mp2, pr_logP2, pr_tp2,
                               pr_e2, pr_de2_dt, pr_om2, pr_dom2_dt, pr_D1,
                               pr_D2, pr_D3, pr_jit1, pr_jit2, pr_jit3, pr_lin, pr_quad]
        self.width = np.array([w_Mp, w_logP, w_tp, w_e, w_de_dt, 
                               w_om, w_dom_dt, w_de2_dt2, w_dom2_dt2, w_Mp2, w_logP2, w_tp2,
                               w_e2, w_de2_dt, w_om2, w_dom2_dt, w_D1, 
                               w_D2, w_D3, w_jit1, w_jit2, w_jit3, w_lin, w_quad])
        self.telescopes = telescopes
        self.phase_plot = phase_plot
        self.gausspriors = gausspriors
        self.gauss_theta_priors= gauss_theta_priors
        self.gauss_theta_indexes = gauss_theta_indexes
        self.gauss_theta_err = gauss_theta_err
        
        self.transitpriors = transitpriors
        self.use_transit_times = use_transit_times
        self.transit_times_list = transit_times_list
        self.use_mass_parametrization = use_mass_parametrization
        self.e_med_list = e_med_list
        self.e_err = e_err
        self.om_med_list = om_med_list
        self.om_err = om_err
        
    # define a function that computed the minimum mass (Mp*sin(i)) when provided with K, e, Mstar, P
    def Mp_from_K(self):
        ''' Takes semi-amplitude (K) in ms^-1, eccentricity (e), mass of star (M1) in solar masses, Period (P) in years
        and return Minimum mass (Mp_sin_i) in Jupiter masses
        '''
        # planet 1
        K = self.K
        e0 = self.e0
        logP = self.logP
        
        # planet 2
        K2 = self.K2
        e02 = self.e02
        logP2 = self.logP2
        
        P = 10**logP
        P2 = 10**logP2
          
        M1 = self.M1
        
        # asumme edge-on orbit. Thus the mass is just the minimum mass Mp*sin(i)
        i = 90
        if K2==0.0:
            return K*((np.sqrt(1-e0**2))/28.4329) * (1/np.sin(i)) * (M1)**(1/2) * (P/365.25)**(1/3)
        else:
            return K*((np.sqrt(1-e0**2))/28.4329) * (1/np.sin(i)) * (M1)**(1/2) * (P/365.25)**(1/3),  K2*((np.sqrt(1-e02**2))/28.4329) * (1/np.sin(i)) * (M1)**(1/2) * (P2/365.25)**(1/3)
    
    def K_from_Mp(self, Mp, P, tp,  e0, om0, Mp2=0, P2=0, tp2=0,  e02=0, om02=0):
        '''Takes minimum mass (Mp_sin_i) in Jupiter masses, eccentricity, mass of star (M1) in solar masses, 
        and periods (P1, P2) in years, and returns the expected semi-amplitude (K) in m/s
        '''
        
        
        # stellar mass
        M1 = self.M1
        
        # asumme edge-on orbit. Thus the mass is just the minimum mass Mp*sin(i)
        i = 90
        if np.median(Mp2)==0.0:
            return Mp*((np.sqrt(1-e0**2))/28.4329)**(-1) * (np.sin(i)) * (M1)**(-1/2) * (P/365.25)**(-1/3), 0
        else:
            return Mp*((np.sqrt(1-e0**2))/28.4329)**(-1) * (np.sin(i)) * (M1)**(-1/2) * (P/365.25)**(-1/3),  Mp2*((np.sqrt(1-e02**2))/28.4329)**(-1) * (np.sin(i)) * (M1)**(-1/2) * (P2/365.25)**(-1/3)
  
    
    def initial_guess_plot(self):
        time_np = self.t
        rv_np = self.y
        s_rv_np = self.yerr
        telescopes = self.telescopes
        theta = self.theta
        phase_plot = self.phase_plot
        

        amp_mcmc, logperiod_mcmc, tp_mcmc, e_mcmc, de_dt_mcmc, om_mcmc,dom_dt_mcmc, de2_dt2_mcmc, dom2_dt2_mcmc, amp2_mcmc, logperiod2_mcmc, tp2_mcmc, e02_mcmc, de2_dt_mcmc, om02_mcmc,dom2_dt_mcmc, D1_mcmc,D2_mcmc,D3_mcmc, jit1_mcmc, jit2_mcmc, jit3_mcmc,  lin_mcmc, quad_mcmc = theta
        
        period_mcmc = 10**logperiod_mcmc
        period2_mcmc = 10**logperiod2_mcmc
        
        
        D_s = [D1_mcmc,D2_mcmc,D3_mcmc]
        ti_np = np.arange(np.min(time_np), np.max(time_np), 0.1)
        print(ti_np)

        fig, axes = plt.subplots(2,2, figsize=(20, 8),  gridspec_kw={'height_ratios': [3, 0.6]})
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        if amp2_mcmc==0.0:
               
            # plot the average model
            #print(e_mcmc, om1)
            y_preds = wobble_edmcmc.one_planet_model(self, ti_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                            e_mcmc, om_mcmc,0, 0, 0]), telescopes)
            y_preds_time_np = wobble_edmcmc.one_planet_model(self, time_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                            e_mcmc, om_mcmc,0, 0, 0]), telescopes)
            y_preds_time_np2 = wobble_edmcmc.lin_quad_model(time_np, np.array([lin_mcmc, quad_mcmc]))
            data_ti = {'time': ti_np,
                    'phase': (ti_np-phase_plot*period_mcmc)%period_mcmc,
                    'rvs': y_preds}
            df_di = pd.DataFrame(data=data_ti)
            df_di_sorted = df_di.sort_values(by=['phase'])

            data2 = {'time': ti_np,
                       'phase': ti_np-phase_plot*period_mcmc,
                       'y_preds': wobble_edmcmc.lin_quad_model(ti_np, np.array([lin_mcmc, quad_mcmc]))}
            df_ti2 = pd.DataFrame(data=data2)
            df_ti_sorted2 = df_ti2.sort_values(by=['phase'])
            

            # plot hat-p-2b RVs
            
            colors = ['#24ab9b', 'r', 'k']
            num_color = 0
            # plot each of the rv datasets
            for tel_name in np.unique(telescopes):
                indexes = np.where(telescopes==tel_name)
                #print(i, indexes, rv_np[indexes])
                
                # plot rvs
                ax = axes[0][0]
                ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                            yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                # plot residuals
                ax = axes[1][0]
                ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, 
                            rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],yerr=s_rv_np[indexes],
                            fmt = 'o', color = colors[num_color],label=tel_name)
                
                # plot quadratic trend data
                ax = axes[0][1]
                ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-D_s[num_color],
                            yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                
                ax = axes[1][1]
                ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                            yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                
                num_color += 1
            
            ax = axes[0][0]
            ax.plot(df_di_sorted['phase'], df_di_sorted['rvs'], color='#7829e6', linewidth=2,
                    label="e="+str(round(e_mcmc,3))+", $\omega$="+str(round(om_mcmc*180/np.pi,3))+"$^\circ$")
            ax.set_ylabel("RV (m/s)")
            ax.legend(loc="lower right")
            legend = ax.legend(loc="lower right", prop={'size': 21}, facecolor='#24ab9b')
            plt.setp(legend.get_texts(), color='k')

            # plot hat-p-2b residuals labels
            ax = axes[1][0]
            ax.set_ylabel("Residuals")
            ax.set_xlabel("Phase (days)")

            # plot quadratic trend
            ax = axes[0][1]
            ax.set_title("1st and 2nd order trend")
            ax.plot(df_ti_sorted2['phase'], df_ti_sorted2['y_preds'], color='#7829e6', linewidth=2)

            ax = axes[1][1]
            ax.set_xlabel("Phase (days)")
            
            return df_di_sorted, df_ti_sorted2
        else: # plot two planet case
            print(ti_np)      
            # plot the average model
            # t, K, P, tp, e0, om0, D1, D2, D3
            y_preds = wobble_edmcmc.one_planet_model(self, ti_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                            e_mcmc, om_mcmc,0, 0, 0]), telescopes)
            y_preds2 = wobble_edmcmc.one_planet_model(self, ti_np, np.array([amp2_mcmc, period2_mcmc, tp2_mcmc,
                                                            e02_mcmc, om02_mcmc,0, 0, 0]), telescopes)
            y_preds_time_np = wobble_edmcmc.one_planet_model(self, time_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                            e_mcmc, om_mcmc,0, 0, 0]), telescopes)
            y_preds_time_np2 = wobble_edmcmc.one_planet_model(self, time_np, np.array([amp2_mcmc, period2_mcmc, tp2_mcmc,
                                                            e02_mcmc, om02_mcmc,0, 0, 0]), telescopes)
            
            
            data_ti = {'time': ti_np,
                    'phase': ti_np%period_mcmc-phase_plot*period_mcmc,
                    'rvs': y_preds}
            df_di = pd.DataFrame(data=data_ti)
            df_di_sorted = df_di.sort_values(by=['phase']).reset_index()

            data2 = {'time': ti_np,
                    'phase': ti_np,
                    'y_preds': y_preds2}
            df_ti2 = pd.DataFrame(data=data2)
            df_ti_sorted2 = df_ti2.sort_values(by=['phase']).reset_index()


            # plot hat-p-2b RVs
            
            colors = ['#24ab9b', 'r', 'k']
            num_color = 0
            # plot each of the rv datasets
            for tel_name in np.unique(telescopes):
                indexes = np.where(telescopes==tel_name)
                #print(i, indexes, rv_np[indexes])
                
                # plot rvs
                ax = axes[0][0]
                ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                            yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                # plot residuals
                ax = axes[1][0]
                ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, 
                            rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],yerr=s_rv_np[indexes],
                            fmt = 'o', color = colors[num_color],label=tel_name)
                ymax = np.max(np.abs(rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color]))
                ax.set_ylim(-ymax-0.2*ymax, ymax+0.2*ymax)
                
                # plot quadratic trend data
                ax = axes[0][1]
                ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-D_s[num_color],
                            yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                
                ax = axes[1][1]
                ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                            yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                ymax = np.max(np.abs(rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color]))
                ax.set_ylim(-ymax-0.2*ymax, ymax+0.2*ymax)
                
                num_color += 1
            
            ax = axes[0][0]
            ax.plot(df_di_sorted['phase'], df_di_sorted['rvs'], color='#7829e6', linewidth=2,
                    label="e="+str(round(e_mcmc,3))+", $\omega$="+str(round(om_mcmc*180/np.pi,3))+"$^\circ$")
            ax.set_ylabel("RV (m/s)")
            #ax.set_title(time_chunk+eval_method)
            ax.legend(loc="lower right")
            legend = ax.legend(loc="lower right", prop={'size': 21}, facecolor='#24ab9b')
            plt.setp(legend.get_texts(), color='k')

            # plot hat-p-2b residuals labels
            ax = axes[1][0]
            ax.set_ylabel("Residuals")
            ax.set_xlabel("Phase (days)")

            # plot hat-p-2c?
            ax = axes[0][1]
            ax.plot(df_ti_sorted2['phase'], df_ti_sorted2['y_preds'], color='#7829e6', linewidth=2)
            ax.set_title("2nd Planet")

            ax = axes[1][1]
            ax.set_xlabel("Phase (days)")
        
            return df_di_sorted, df_ti_sorted2
       # model = wobble_edmcmc.loglikelihood(self, theta, t, y, yerr, nfree_params)[8]
    
    def one_planet_model(self, t, params, telescopes):
        use_mass_parametrization = self.use_mass_parametrization
        
        if use_mass_parametrization:
            Mp, P, tp, e0, om0, D1, D2, D3 = params
            K = wobble_edmcmc.K_from_Mp(self, Mp, P, tp,  e0, om0)[0]
        else:
            K, P, tp, e0, om0, D1, D2, D3 = params

        telescopes = self.telescopes
        
        rvs = K *radvel.kepler.rv_drive(t, [P, tp,e0, om0, 1.0], use_c_kepler_solver=True)
        
        unique_telescopes = np.unique(telescopes)
        if len(unique_telescopes)==3:
            D_s = [D1, D2, D3]
            for tel in range(0, len(unique_telescopes)):
                indexes_telesc = np.where(telescopes==unique_telescopes[tel])
                rvs[indexes_telesc]= rvs[indexes_telesc]+D_s[tel]
            return rvs
        elif len(unique_telescopes)==2:
            D_s = [D1, D2]
            for tel in range(0, len(unique_telescopes)):
                indexes_telesc = np.where(telescopes==unique_telescopes[tel])
                rvs[indexes_telesc]= rvs[indexes_telesc]+D_s[tel]
            return rvs
        else:
            return rvs + D1
    
    
    def lin_quad_model(t, params):
        lin, quad = params
        return lin*(t-np.min(t)) + quad*(t-np.min(t))**2
    
    def two_planet_de_dt_dom_dt(self, theta):
        t = self.t
        telescopes = self.telescopes
        use_mass_parametrization = self.use_mass_parametrization
        K2 = self.K2 # K2 will be redefined if we are not using the mass paramterization
        
        if use_mass_parametrization:
           Mp, logP, tp, e0, de_dt, om0, dom_dt, de2_dt2, dom2_dt2, Mp2, logP2, tp2, e02, de2_dt, om02, dom2_dt, D1, D2, D3, jit1, jit2, jit3, lin, quad = theta
        else:
           K, logP, tp, e0, de_dt, om0, dom_dt, de2_dt2, dom2_dt2, K2, logP2, tp2, e02, de2_dt, om02, dom2_dt, D1, D2, D3, jit1, jit2, jit3, lin, quad = theta
         
        P = 10**logP
        P2 = 10**logP2
        
        D_s = [D1, D2, D3]
        unique_telescopes = np.unique(telescopes)
        
        om_list = []
        e_list = []
        
        # check if de_dt, dom_dt, and K2 are all == 0. If so, run a stable 1 planet model with 1st and second order background trend
        if de_dt ==0.0 and dom_dt ==0.0 and K2 == 0.0 and Mp2==0 and de2_dt2==0 and dom2_dt2==0:
            if use_mass_parametrization:
                K = wobble_edmcmc.K_from_Mp(self, Mp, P, tp,  e0, om0, Mp2, P2, tp2,  e02, om02)[0]
            model = K *radvel.kepler.rv_drive(t, [P, tp,e0, om0, 1.0], use_c_kepler_solver=True) + lin*(t-np.min(t)) + quad*(t-np.min(t))**2
            # Add offsets for individual telescopes
            if len(unique_telescopes)==1:
                model = model.copy() +D1
            else:
                for tel in range(0, len(unique_telescopes)):
                    indexes_telesc = np.where(telescopes==unique_telescopes[tel])
                    model[indexes_telesc]= model[indexes_telesc].copy()+D_s[tel]
        else:
            # run the full, slower model instead
            rv_contrib_list = []
            for i in np.arange(0, len(t)):
                om = dom_dt*t[i]+om0
                e = de_dt*t[i]+e0
                om2 = dom2_dt*t[i]+om02
                e2 = de2_dt*t[i]+e02
                
                K, K2 = wobble_edmcmc.K_from_Mp(self, Mp, P, tp,  e, om, Mp2, P2, tp2,  e2, om2)
                
                om_list.append(om2)
                e_list.append(e2)
                #tp = dTp_dt*t[i]+Tp_0
                rv_contrib_p1 = K *radvel.kepler.rv_drive(np.array([t[i]]), [P, tp,e, om, 1.0], use_c_kepler_solver=True)
                rv_contrib_p2  = K2 *radvel.kepler.rv_drive(np.array([t[i]]), [P2, tp2,e2, om2, 1.0], use_c_kepler_solver=True)
                rv_contrib_lin_quad = lin*(t[i]-np.min(t)) + quad*(t[i]-np.min(t))**2
                rv_contrib_list.append(rv_contrib_p1[0]+rv_contrib_p2[0]+rv_contrib_lin_quad)
        
            model = np.array(rv_contrib_list) #+ D
            
            # Add offsets for individual telescopes
            if len(unique_telescopes)==1:
                model = model.copy() +D1
            else:
                for tel in range(0, len(unique_telescopes)):
                    indexes_telesc = np.where(telescopes==unique_telescopes[tel])
                    model[indexes_telesc]= model[indexes_telesc].copy()+D_s[tel]
                    
        return model, om_list, e_list
    
    def get_transit_time(period, w_attt, e_attt, tp, orbit_num):
        '''Calculated the time of transit given the period, angle of periastron, eccentricity and time of periastron'''
        
        tp_at_orbit = tp+period*orbit_num
        
        f = np.pi/2 - w_attt
        ea = 2*np.arctan(np.sqrt((1-e_attt)/(1+e_attt))*np.tan(f/2))
        te = period/(2*np.pi)*(ea-e_attt*np.sin(ea))
        tt = tp_at_orbit+te
        return tt
    
    def get_occult_time(period, w_attt, e_attt, tp, orbit_num):
        '''Calculated the time of scondary eclipse given the period, angle of periastron, eccentricity and time of periastron'''
        
        tp_at_orbit = tp+period*orbit_num
        
        f = -3*np.pi/2 - w_attt
        ea = 2*np.arctan(np.sqrt((1-e_attt)/(1+e_attt))*np.tan(f/2))
        te = period/(2*np.pi)*(ea-e_attt*np.sin(ea))
        tt = tp_at_orbit+te
        return tt
    
    def get_periastron_time(period, w_attt, e_attt, tt):
        '''Calculated the time of periastron passage given the period, angle of periastron, eccentricity and time of transit'''
        
        f = np.pi/2 - w_attt
        ea = 2*np.arctan(np.sqrt((1-e_attt)/(1+e_attt))*np.tan(f/2))       
        te = period/(2*np.pi)*(ea-e_attt*np.sin(ea))    
        tp = tt-te 
        return tp
    
    
    def loglikelihood(self, theta, t, y, yerr, nfree_params):
        t = self.t
        y = self.y
        yerr = self.yerr
        telescopes = self.telescopes
        use_mass_parametrization = self.use_mass_parametrization
        
        if use_mass_parametrization:
           Mp, logP, tp, e0, de_dt, om0, dom_dt, de2_dt2, dom2_dt2, Mp2, logP2, tp2, e02, de2_dt, om02, dom2_dt, D1, D2, D3, jit1, jit2, jit3, lin, quad = theta
        else:
           K, logP, tp, e0, de_dt, om0, dom_dt, de2_dt2, dom2_dt2, K2, logP2, tp2, e02, de2_dt, om02, dom2_dt, D1, D2, D3, jit1, jit2, jit3, lin, quad = theta

        P = 10**logP
        P2 = 10**logP2
        K2 = wobble_edmcmc.K_from_Mp(self, Mp, P, tp,  e0, om0, Mp2, P2, tp2,  e02, om02)[1] # K2 will be redefined if we are not using the mass paramterization

        
        D_s = [D1, D2, D3]
        jitterpars = [jit1, jit2, jit3]
        unique_telescopes = np.unique(telescopes)
        
        om_list = []
        e_list = []
        K_list = []
        K2_list =  []
        
        sigma2 = yerr**2
        # check if de_dt, dom_dt, and K2 are all == 0. If so, run a stable 1 planet model with 1st and second order background trend
        if de_dt ==0.0 and dom_dt ==0.0 and K2 == 0.0 and de2_dt2==0 and dom2_dt2==0:
            if use_mass_parametrization:
               K = wobble_edmcmc.K_from_Mp(self, Mp, P, tp,  e0, om0, Mp2, P2, tp2,  e02, om02)[0]
            
            model = K *radvel.kepler.rv_drive(t, [P, tp,e0, om0, 1.0], use_c_kepler_solver=True) + lin*(t-np.min(t)) + quad*(t-np.min(t))**2
            # Add offsets for individual telescopes
            if len(unique_telescopes)==1:
                model = model.copy() +D1
                sigma2 = sigma2.copy() +jit1**2
            else:
                for tel in range(0, len(unique_telescopes)):
                    indexes_telesc = np.where(telescopes==unique_telescopes[tel])
                    model[indexes_telesc]= model[indexes_telesc].copy()+D_s[tel]
                    sigma2[indexes_telesc] = sigma2[indexes_telesc].copy()+jitterpars[tel]**2
# =============================================================================
#         elif de2_dt2==0 and dom2_dt2==0:
#             # run the full, slower model instead that allows for de_dt, dom_dt, K2 but not de2_dt2 and dom2_dt2
#             rv_contrib_list = []
#             for i in np.arange(0, len(t)):
#                 om = dom_dt*t[i]+om0
#                 e = de_dt*t[i]+e0
#                 om2 = dom2_dt*t[i]+om02
#                 e2 = de2_dt*t[i]+e02
#                 om_list.append(om2)
#                 e_list.append(e2)
#                 #tp = dTp_dt*t[i]+Tp_0
#                 rv_contrib_p1 = K *radvel.kepler.rv_drive(np.array([t[i]]), [P, tp,e, om, 1.0], use_c_kepler_solver=True)
#                 rv_contrib_p2  = K2 *radvel.kepler.rv_drive(np.array([t[i]]), [P2, tp2,e2, om2, 1.0], use_c_kepler_solver=True)
#                 rv_contrib_lin_quad = lin*(t[i]) + quad*(t[i])**2
#                 rv_contrib_list.append(rv_contrib_p1[0]+rv_contrib_p2[0]+rv_contrib_lin_quad)
#         
#             model = np.array(rv_contrib_list) #+ D
#             
#             # Add offsets for individual telescopes
#             if len(unique_telescopes)==1:
#                 model = model.copy() +D1
#                 sigma2 = sigma2.copy()+jit1**2
#             else:
#                 for tel in range(0, len(unique_telescopes)):
#                     indexes_telesc = np.where(telescopes==unique_telescopes[tel])
#                     model[indexes_telesc]= model[indexes_telesc].copy()+D_s[tel]
#                     sigma2[indexes_telesc] = sigma2[indexes_telesc].copy()+jitterpars[tel]**2
# =============================================================================
        else: 
            # # run model with second order trends
            rv_contrib_list = []
            for i in np.arange(0, len(t)):
                om = dom_dt*t[i]+om0 +dom2_dt2*t[i]**2
                e = de_dt*t[i]+e0 + de2_dt2*t[i]**2
                om2 = dom2_dt*t[i]+om02
                e2 = de2_dt*t[i]+e02
                om_list.append(om2)
                e_list.append(e2)
                
                K, K2 = wobble_edmcmc.K_from_Mp(self, Mp, P, tp,  e, om, Mp2, P2, tp2,  e2, om2)
                K_list.append(K)
                K2_list.append(K2)
                
                #tp = dTp_dt*t[i]+Tp_0
                rv_contrib_p1 = K *radvel.kepler.rv_drive(np.array([t[i]]), [P, tp,e, om, 1.0], use_c_kepler_solver=True)
                rv_contrib_p2  = K2 *radvel.kepler.rv_drive(np.array([t[i]]), [P2, tp2,e2, om2, 1.0], use_c_kepler_solver=True)
                rv_contrib_lin_quad = lin*(t[i]-np.min(t)) + quad*(t[i]-np.min(t))**2
                rv_contrib_list.append(rv_contrib_p1[0]+rv_contrib_p2[0]+rv_contrib_lin_quad)
        
            model = np.array(rv_contrib_list) #+ D
            
            # Add offsets for individual telescopes
            if len(unique_telescopes)==1:
                model = model.copy() +D1
                sigma2 = sigma2.copy()+jit1**2
            else:
                for tel in range(0, len(unique_telescopes)):
                    indexes_telesc = np.where(telescopes==unique_telescopes[tel])
                    model[indexes_telesc]= model[indexes_telesc].copy()+D_s[tel]
                    sigma2[indexes_telesc] = sigma2[indexes_telesc].copy()+jitterpars[tel]**2
            
        #if transit_eclipse_t:
        #    transit_model = get_transit_time(period, om0, e0, tp, orbit_num=df_transit_times['Orbit number'])
        #    transit_loglikelihood = np.sum(0.5 * (transit_times - transit_model)**2 / (tr_sig)**2 + np.log(tr_sig))
        
        
        newsig = np.sqrt(sigma2)
    
        negloglikelihood = np.sum(0.5 * (y - model)**2 / (newsig)**2 + np.log(newsig))
        loglikelihood = -1*negloglikelihood
    
        # compute other goodness of fit params
        # reduced chi squared
        N = len(y)
        chisq = (1/(N-1))*np.sum((y - model)**2 / sigma2)
        chisq_k = (1/(N-nfree_params))*np.sum((y - model)**2 / sigma2)
        # Bayesian Information Criterion (BIC)
        BIC = nfree_params*np.log(N) - 2*loglikelihood    
        AIC = 2*nfree_params - 2*loglikelihood
    

        return loglikelihood, chisq, AIC, BIC, om_list, e_list, newsig, chisq_k, model, K_list, K2_list
    

    def transit_eclipse_times_loglikelihood(self, transit_times, eclipse_times, transit_orbit_nums, eclipse_orbit_nums, theta):
        transit_times = self.transit_times
        eclipse_times = self.eclipse_times
        transit_orbit_nums = self.transit_orbit_nums
        eclipse_orbit_nums = self.eclipse_orbit_nums
        transit_times_err = self.transit_times_err
        eclipse_times_err = self.eclipse_times_err
        transit_nfree_params = self.transit_nfree_params
        K2 = self.K2 # K2 will be redefined if we are not using the mass paramterization
        
        use_mass_parametrization = self.use_mass_parametrization
        
        if use_mass_parametrization:
           Mp, logP, tp, e0, de_dt, om0, dom_dt, de2_dt2, dom2_dt2, Mp2, logP2, tp2, e02, de2_dt, om02, dom2_dt, D1, D2, D3, jit1, jit2, jit3, lin, quad = theta
        else:
           K, logP, tp, e0, de_dt, om0, dom_dt, de2_dt2, dom2_dt2, K2, logP2, tp2, e02, de2_dt, om02, dom2_dt, D1, D2, D3, jit1, jit2, jit3, lin, quad = theta

        
        P = 10**logP
        tc = wobble_edmcmc.get_transit_time(P, om0, e0, tp, 0)
        
        
        t_for_transits = transit_orbit_nums*P+tc
        t_for_eclipses = wobble_edmcmc.get_eclipse_time()
        
        transit_model_list = []
        eclipse_model_list = []
        
        # check if de_dt, dom_dt, and K2 are all == 0. If so, run a stable 1 planet transit model.
        if de_dt ==0.0 and dom_dt ==0.0 and K2 == 0.0 and de2_dt2==0 and dom2_dt2==0:
            transit_model_list = wobble_edmcmc.get_transit_time(P, om0, e0, tp, transit_orbit_nums)
        else: # run an evolving orbital parameters model.
        
            t_for_transits = transit_orbit_nums*P+tc
            for i in range(0, len(transit_orbit_nums)):
                w_attt = dom_dt*t_for_transits[i]+ om0 + dom2_dt2*t_for_transits[i]**2
                e_attt = de_dt*t_for_transits[i] + e0 + de2_dt2*t_for_transits[i]**2
                transit_model_contrib = wobble_edmcmc.get_transit_time(P, w_attt, e_attt, tp, transit_orbit_nums[i])
                transit_model_list.append(transit_model_contrib)
                
        sigma2 = transit_times_err**2
        newsig = np.sqrt(sigma2)
        N = len(transit_times)
        transit_loglikelihood = -1*np.sum(0.5 * (transit_times - transit_model_list)**2 / (newsig)**2 + np.log(newsig))
        transit_chisq_k = (1/(N-transit_nfree_params))*np.sum((transit_times - transit_model_list)**2 / sigma2)
        transit_AIC = 2*transit_nfree_params - 2*transit_loglikelihood
        transit_BIC = transit_nfree_params*np.log(N) - 2*transit_loglikelihood  
                
        # compute eclipse model times
        # check if de_dt, dom_dt, and K2 are all == 0. If so, run a stable 1 planet eclipse model.
        if de_dt ==0.0 and dom_dt ==0.0 and K2 == 0.0 and de2_dt2==0 and dom2_dt2==0:
            eclipse_model_list = wobble_edmcmc.get_eclipse_time(P, w_attt, e_attt, tp, eclipse_orbit_nums)
        else: # run an evolving orbital parameters model.
            for i in range(0, len(eclipse_orbit_nums)):
                w_attt = dom_dt*t_for_eclipses[i]+ om0 + dom2_dt2*t_for_eclipses[i]**2
                e_attt = de_dt*t_for_eclipses[i] + e0 + de2_dt2*t_for_eclipses[i]**2
                eclipse_model_contrib = wobble_edmcmc.get_eclipse_time(P, w_attt, e_attt, tp, eclipse_orbit_nums[i])
                eclipse_model_list.append(eclipse_model_contrib)
        
        sigma2 = eclipse_times_err**2
        newsig = np.sqrt(sigma2)
        N = len(eclipse_times)
        eclipse_loglikelihood = -1*np.sum(0.5 * (eclipse_times - eclipse_model_list)**2 / (newsig)**2 + np.log(newsig))
        eclipse_chisq_k = (1/(N-transit_nfree_params))*np.sum((eclipse_times - eclipse_model_list)**2 / sigma2)
        eclipse_AIC = 2*transit_nfree_params - 2*eclipse_loglikelihood
        eclipse_BIC = transit_nfree_params*np.log(N) - 2*eclipse_loglikelihood  
        
        # Compute transit and eclipse joint goodness-of-fit metrics
        loglikelihood = transit_loglikelihood+eclipse_loglikelihood
        chisq_k = transit_chisq_k + eclipse_chisq_k
        AIC = transit_AIC + eclipse_AIC
        BIC = transit_BIC + eclipse_BIC
        
        return loglikelihood, chisq_k, AIC, BIC, transit_model_list, eclipse_model_list
                
    
    def log_prior(theta, self):
        theta_priors = self.theta_priors
        time_np = self.t
        
        gausspriors =self.gausspriors
        gauss_theta_priors = self.gauss_theta_priors
        gauss_theta_indexes = self.gauss_theta_indexes
        gauss_theta_err = self.gauss_theta_err
        transitpriors = self.transitpriors
        transit_times_list = self.transit_times_list
        e_med_list = self.e_med_list
        e_err = self.e_err
        om_med_list = self.om_med_list
        om_err = self.om_err
        use_mass_parametrization = self.use_mass_parametrization
        
        if use_mass_parametrization:
           Mp, logP, tp, e0, de_dt, om0, dom_dt, de2_dt2, dom2_dt2, Mp2, logP2, tp2, e02, de2_dt, om02, dom2_dt, D1, D2, D3, jit1, jit2, jit3, lin, quad = theta
        else:
           K, logP, tp, e0, de_dt, om0, dom_dt, de2_dt2, dom2_dt2, K2, logP2, tp2, e02, de2_dt, om02, dom2_dt, D1, D2, D3, jit1, jit2, jit3, lin, quad = theta

        
        # if de/dt is changing, check that the e_extrema are between 0 to 1
# =============================================================================
#         if de_dt != 0 and de2_dt2==0 and dom2_dt2==0:
#             e_extrema1 = de_dt*np.max(time_np)+e0
#             e_extrema2 = de_dt*np.min(time_np)+e0
#             om_extrema1 = dom_dt*np.max(time_np)+om0
#             om_extrema2 = dom_dt*np.min(time_np)+om0
#             #print(e_extrema1)
#             if 0.0<e_extrema1<0.8 and 0.0<e_extrema2<0.8 and 0<om_extrema1<6.28 and 0<om_extrema2<6.28:
#                 # check other priors
#                 # Check that priors are satisfied
#                 logprior = 0
#                 indexes = np.where(np.isnan(np.array(theta_priors)[:,0])==False)[0]
#                 
#                 # Check priors for those indexes
#                 for i in indexes:
#                     #print(i)
#                     if theta_priors[i][0]<theta[i]<theta_priors[i][1]:
#                         #print(theta_priors[0][i][0],theta[0][i],theta_priors[0][i][1])
#                         logprior += 0.0
#                     else:
#                         #print(theta_priors[i][0],theta[i],theta_priors[i][1])
#                         logprior += -np.inf
#             
#                 return logprior
#             else: 
#                 return -np.inf
# =============================================================================
        # if de_dt and de2_dt2 are changing, check that the e_extrema are between 0 to 1
        if de_dt != 0 or de2_dt2!=0:
            e_extrema1 = de_dt*np.max(time_np)+e0 +de2_dt2*np.max(time_np)**2 
            e_extrema2 = de_dt*np.min(time_np)+e0 +de2_dt2*np.min(time_np)**2 
            om_extrema1 = dom_dt*np.max(time_np)+om0 +dom2_dt2*np.max(time_np)**2 
            om_extrema2 = dom_dt*np.min(time_np)+om0 +dom2_dt2*np.min(time_np)**2 
            
            #print(e_extrema1)
            if 0.0<e_extrema1<0.8 and 0.0<e_extrema2<0.8 and 0<om_extrema1<6.28 and 0<om_extrema2<6.28:
                # check other priors
                # Check that priors are satisfied
                logprior = 0
                indexes = np.where(np.isnan(np.array(theta_priors)[:,0])==False)[0]
                
                # Check priors for those indexes
                for i in indexes:
                    #print(i)
                    if theta_priors[i][0]<theta[i]<theta_priors[i][1]:
                        #print(theta_priors[0][i][0],theta[0][i],theta_priors[0][i][1])
                        logprior += 0.0
                    else:
                        #print(theta_priors[i][0],theta[i],theta_priors[i][1])
                        logprior += -np.inf
            
                #return logprior
            else: 
                logprior += -np.inf        
        else:        
            # Check that priors are satisfied
            logprior = 0
            # check which priors are not NaNs
            indexes = np.where(np.isnan(np.array(theta_priors)[:,0])==False)[0]
            
            # Check priors for those indexes
            for i in indexes:
                #print(i)
                if theta_priors[i][0]<theta[i]<theta_priors[i][1]:
                    #print(theta_priors[0][i][0],theta[0][i],theta_priors[0][i][1])
                    logprior += 0.0
                else:
                    #print(theta_priors[i][0],theta[i],theta_priors[i][1])
                    logprior += -np.inf
        
            #return logprior
        
        if gausspriors:
            for i in range(0, len(gauss_theta_indexes)):
                # indicate index of the theta of interest
                t_i = gauss_theta_indexes[i]
                
                # Check whether gauss prior is on logP. If so, adjust the way the prior is implemented
                if theta[t_i] == logP:
                   thetaP =  10**theta[t_i]
                   logprior += -0.5*((thetaP-gauss_theta_priors[i])**2/gauss_theta_err[i]**2)
                else:
                    logprior += -0.5*((theta[t_i]-gauss_theta_priors[i])**2/gauss_theta_err[i]**2)
        
        if transitpriors:
            for i in range(0, len(transit_times_list)):
                # estimate e at time i 
                e_at_time = de_dt*transit_times_list[i]+e0 +de2_dt2*(transit_times_list[i])**2 
                
                # Add log prior for e
                logprior += -0.5*((e_at_time-e_med_list[i])**2/e_err**2)
                
                # estimate omega at time i 
                om_at_time = dom_dt*transit_times_list[i]+om0 +dom2_dt2*(transit_times_list[i])**2 
                
                # Add log prior for e
                logprior += -0.5*((om_at_time-om_med_list[i])**2/om_err**2)
                
                
        return logprior
    
    def log_probability(theta, self, t, y, yerr, nfree_params):
        lp = wobble_edmcmc.log_prior(theta, self)
        if not np.isfinite(lp):
            return -np.inf

        return lp + wobble_edmcmc.loglikelihood(self, theta, t, y, yerr, nfree_params)[0]
    
    
    def transit_eclipse_times_log_probability(theta, self, t, y, yerr, nfree_params, transit_times, eclipse_times, transit_orbit_nums, eclipse_orbit_nums):
        lp = wobble_edmcmc.log_prior(theta, self)
        if not np.isfinite(lp):
            return -np.inf

        return lp + wobble_edmcmc.loglikelihood(self, theta, t, y, yerr, nfree_params)[0] + wobble_edmcmc.transit_eclipse_times_loglikelihood(self, transit_times, eclipse_times, transit_orbit_nums, eclipse_orbit_nums, theta) [0] 
    
    
    # Define mcmc function that returns the samples
    def wobble_mcmc(self):
        t = self.t
        y = self.y
        yerr = self.yerr
        use_mass_parametrization = self.use_mass_parametrization
        
        if use_mass_parametrization:
            theta = [self.Mp, self.logP, self.tp, self.e0, self.de_dt, 
                      self.om0, self.dom_dt, self.de2_dt2, self.dom2_dt2, self.Mp2, self.logP2, self.tp2,
                      self.e02, self.de2_dt, self.om02, self.dom2_dt, self.D1,
                      self.D2, self.D3, self.jit1, self.jit2, self.jit3,
                      self.lin, self.quad]
        else:
            theta = [self.K, self.logP, self.tp, self.e0, self.de_dt, 
                  self.om0, self.dom_dt, self.de2_dt2, self.dom2_dt2, self.K2, self.logP2, self.tp2,
                  self.e02, self.de2_dt, self.om02, self.dom2_dt, self.D1,
                  self.D2, self.D3, self.jit1, self.jit2, self.jit3,
                  self.lin, self.quad]
            
        width = self.width

        nwalkers = self.nwalkers
        nlink = self.nlink
        nburnin = self.nburnin
        ncores = self.ncores
        nfree_params = self.nfree_params
        use_transit_times = self.use_transit_times
        
    
        if use_transit_times:
            transit_times = self.transit_times
            eclipse_times = self.eclipse_times
            transit_orbit_nums = self.transit_orbit_nums
            eclipse_orbit_nums = self.eclipse_orbit_nums
            out = edm.edmcmc(wobble_edmcmc.transit_eclipse_times_log_probability, theta, width,
                     args=(self,t,y,yerr, nfree_params,
                           transit_times, eclipse_times, transit_orbit_nums, eclipse_orbit_nums),
                     nwalkers=nwalkers, nlink = nlink, nburnin=nburnin, ncores=ncores)
        else:
            out = edm.edmcmc(wobble_edmcmc.log_probability, theta, width,
                     args=(self,t,y,yerr, nfree_params),
                     nwalkers=nwalkers, nlink = nlink, nburnin=nburnin, ncores=ncores)
        
        print(np.median(out.flatchains[:,0]), '+/-', np.std(out.flatchains[:,0]), ';    ', np.median(out.flatchains[:,1]), '+/-', np.std(out.flatchains[:,1]))
        return out
    
    def metrics(self, out, nburnin):
        t = self.t
        y = self.y
        yerr = self.yerr
        nfree_params = self.nfree_params
        labels = self.labels
        telescopes = self.telescopes
        K2 = self.K2 # K2 will be redefined if we are not using the mass paramterization
        use_mass_parametrization = self.use_mass_parametrization
        
        flat_samples = out.get_chains(nthin =5, nburnin = nburnin, flat=True)
        
        if use_mass_parametrization:
            Mp, logP, tp, e0, de_dt, om0, dom_dt, de2_dt2, dom2_dt2, Mp2, logP2, tp2, e02, de2_dt, om02, dom2_dt, D1, D2, D3, jit1, jit2, jit3, lin, quad = np.percentile(flat_samples[:,:], [50], axis=0).tolist()[0]
        else: 
            K, logP, tp, e0, de_dt, om0, dom_dt, de2_dt2, dom2_dt2, K2, logP2, tp2, e02, de2_dt, om02, dom2_dt, D1, D2, D3, jit1, jit2, jit3, lin, quad = np.percentile(flat_samples[:,:], [50], axis=0).tolist()[0]

        
        P = 10**logP
        P2 = 10**logP2
        K2 = wobble_edmcmc.K_from_Mp(self, Mp, P, tp,  e0, om0, Mp2, P2, tp2,  e02, om02)[1] # K2 will be redefined if we are not using the mass paramterization
        D_s = [D1, D2, D3]
        jitterpars = [jit1, jit2, jit3]
        unique_telescopes = np.unique(telescopes)
        
        om_list = [] 
        e_list = []
        K_list = []
        K2_list = []
        
        # check if de_dt, dom_dt, and K2 are all == 0. If so, run a stable 1 planet model with 1st and second order background trend
        if de_dt ==0.0 and dom_dt ==0.0 and K2 == 0.0:
            if use_mass_parametrization:
               K = wobble_edmcmc.K_from_Mp(self, Mp, P, tp,  e0, om0, Mp2, P2, tp2,  e02, om02)[0]
               
            
            model = K *radvel.kepler.rv_drive(t, [P, tp,e0, om0, 1.0], use_c_kepler_solver=True) + lin*(t-np.min(t)) + quad*(t-np.min(t))**2
            K_list.append(K)
            sigma2 = yerr**2
            # Add offsets and jitterpars for individual telescopes
            if len(unique_telescopes)==1:
                model = model.copy() +D1
                sigma2 = sigma2.copy() + jit1**2
            else:
                for tel in range(0, len(unique_telescopes)):
                    indexes_telesc = np.where(telescopes==unique_telescopes[tel])
                    model[indexes_telesc]= model[indexes_telesc].copy()+D_s[tel]
                    sigma2[indexes_telesc] = sigma2[indexes_telesc].copy()+jitterpars[tel]**2
# =============================================================================
#         elif de2_dt2==0 and dom2_dt2==0:
#             # run the full, slower model instead that allows for de_dt, dom_dt, K2 but not de2_dt2 and dom2_dt2
#             rv_contrib_list = []
#             for i in np.arange(0, len(t)):
#                 om = dom_dt*t[i]+om0
#                 e = de_dt*t[i]+e0
#                 om2 = dom2_dt*t[i]+om02
#                 e2 = de2_dt*t[i]+e02
#                 om_list.append(om)
#                 e_list.append(e)
#                 #tp = dTp_dt*t[i]+Tp_0
#                 rv_contrib_p1 = K *radvel.kepler.rv_drive(np.array([t[i]]), [P, tp,e, om, 1.0], use_c_kepler_solver=True)
#                 rv_contrib_p2  = K2 *radvel.kepler.rv_drive(np.array([t[i]]), [P2, tp2,e2, om2, 1.0], use_c_kepler_solver=True)
#                 rv_contrib_lin_quad = lin*(t[i]) + quad*(t[i])**2
#                 rv_contrib_list.append(rv_contrib_p1[0]+rv_contrib_p2[0]+rv_contrib_lin_quad)
#         
#             model = np.array(rv_contrib_list) #+ D
#             sigma2 = yerr**2
#             # Add offsets for individual telescopes
#             if len(unique_telescopes)==1:
#                 model = model.copy() +D1
#                 sigma2 = sigma2.copy() + jit1**2
#             else:
#                 for tel in range(0, len(unique_telescopes)):
#                     indexes_telesc = np.where(telescopes==unique_telescopes[tel])
#                     model[indexes_telesc]= model[indexes_telesc].copy()+D_s[tel]
#                     sigma2[indexes_telesc] = sigma2[indexes_telesc].copy()+jitterpars[tel]**2
# =============================================================================
        else:
            # run model with second order trends
            rv_contrib_list = []
            for i in np.arange(0, len(t)):
                om = dom_dt*t[i]+om0 +dom2_dt2*t[i]**2
                e = de_dt*t[i]+e0 + de2_dt2*t[i]**2
                om2 = dom2_dt*t[i]+om02
                e2 = de2_dt*t[i]+e02
                om_list.append(om)
                e_list.append(e)
                
                K, K2 = wobble_edmcmc.K_from_Mp(self, Mp, P, tp,  e, om, Mp2, P2, tp2,  e2, om2)
                K_list.append(K)
                K2_list.append(K2)
                #tp = dTp_dt*t[i]+Tp_0
                rv_contrib_p1 = K *radvel.kepler.rv_drive(np.array([t[i]]), [P, tp,e, om, 1.0], use_c_kepler_solver=True)
                rv_contrib_p2  = K2 *radvel.kepler.rv_drive(np.array([t[i]]), [P2, tp2,e2, om2, 1.0], use_c_kepler_solver=True)
                rv_contrib_lin_quad = lin*(t[i]-np.min(t)) + quad*(t[i]-np.min(t))**2
                rv_contrib_list.append(rv_contrib_p1[0]+rv_contrib_p2[0]+rv_contrib_lin_quad)
        
            model = np.array(rv_contrib_list) #+ D
            sigma2 = yerr**2
            # Add offsets for individual telescopes
            if len(unique_telescopes)==1:
                model = model.copy() +D1
                sigma2 = sigma2.copy() + jit1**2
            else:
                for tel in range(0, len(unique_telescopes)):
                    indexes_telesc = np.where(telescopes==unique_telescopes[tel])
                    model[indexes_telesc]= model[indexes_telesc].copy()+D_s[tel]
                    sigma2[indexes_telesc] = sigma2[indexes_telesc].copy()+jitterpars[tel]**2
            
        newsig = np.sqrt(sigma2)
    
        negloglikelihood = np.sum(0.5 * (y - model)**2 / (newsig)**2 + np.log(newsig))
    
        # compute other goodness of fit params
        # reduced chi squared
        N = len(y)
        chisq = (1/(N-1))*np.sum((y - model)**2 / sigma2)
        chisq_k = (1/(N-nfree_params))*np.sum((y - model)**2 / sigma2)
        # Bayesian Information Criterion (BIC)
        BIC = nfree_params*np.log(N)+ 2*negloglikelihood 
        AIC = 2*nfree_params+ 2*negloglikelihood
        
        values = [negloglikelihood, chisq_k, AIC, BIC]
        metric_labels = ["loglikelihood", "Reduced \chi^2", "AIC","BIC"]
        for i in range(0, len(metric_labels)):
            txt = "\mathrm{{{1}}} = {0:.3f}"
            txt = txt.format(values[i], metric_labels[i])
            display(Math(txt))
            
        non_nan_indexes = np.where(np.isnan(out.gelmanrubin())==False)[0]
        gelman_values = out.gelmanrubin()[non_nan_indexes]
        gelman_labels = np.array(labels)[non_nan_indexes]
        
        txt = "\mathrm{{{0}}}"
        display(Math(txt.format('Gelman\ Rubin\ Values:')))
        
        for i in range(len(gelman_values)):
            txt = "\mathrm{{{1}}} = {0:.3f}"
            txt = txt.format(gelman_values[i], gelman_labels[i])
            display(Math(txt))
        
        
        return -1*negloglikelihood, chisq, AIC, BIC, gelman_values, gelman_labels, om_list, e_list, newsig, chisq_k, K_list, K2_list
    

        
    
    