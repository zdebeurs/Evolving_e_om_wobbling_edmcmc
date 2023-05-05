#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:43:33 2022

@author: zdebeurs
"""

'''
plots needed

+- chains
+- corner plot (option to feed in params you want in corner plot s.t. you can show smaller corner plots)
+- values of median +/- in nice format
+- histogram of M1 (and M2 if applicable)
- Global fit with points color-coded based on time (show Planet 1 and Planet 2/ Planet 1 + background trends)
- Global fit with 100 fair draws (show Planet 1 and Planet 2/ Planet 1 + background trends)
- Subplots showing orbital variation over time (4 subplots)
- gif of orbit changing over time
- periodogram of residuals
- Make a two pager that has all these plots saved


'''

import matplotlib.pyplot as plt
import numpy as np
import corner
import pandas as pd
import radvel
from random import sample
from IPython.display import display, Math

#globally change font size for plots
plt.rcParams['font.size'] =17
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'



def extract_varying_params_labels(out, nburnin, labels, label_indexes=None):
    if label_indexes==None:
        label_indexes=range(0, np.shape(out.flatchains)[1])
    
    # retrieve varying chains
    flat_samples = out.get_chains(nthin =5, nburnin = nburnin, flat=True)

    varying_chains = []
    varying_labels = []
    for i in label_indexes:
        #print(labels[i],", std: ", np.std(flat_samples[:, i]))
        if np.std(flat_samples[:, i])> 1e-10*np.median(flat_samples[:, i]):
            #print(labels[i],", std: ", np.std(flat_samples[:, i]))
            varying_chains.append(flat_samples[:, i].tolist())
            varying_labels.append(labels[i])            
    varying_chains_np = np.array(varying_chains).T
    
    return varying_chains_np, varying_labels


def Mp_from_K(K, e0, P, M1=1.34, K2=0, e02=0, P2=0):
    ''' Takes semi-amplitude (K) in ms^-1, eccentricity (e), mass of star (M1) in solar masses, Period (P) in years
    and return Minimum mass (Mp_sin_i) in Jupiter masses
    '''
    
    # asumme edge-on orbit. Thus the mass is just the minimum mass Mp*sin(i)
    i = 90
    if np.median(K2)==0.0:
        return K*((np.sqrt(1-e0**2))/28.4329) * (1/np.sin(i)) * (M1)**(1/2) * (P/365.25)**(1/3)
    else:
        return K*((np.sqrt(1-e0**2))/28.4329) * (1/np.sin(i)) * (M1)**(1/2) * (P/365.25)**(1/3),  K2*((np.sqrt(1-e02**2))/28.4329) * (1/np.sin(i)) * (M1)**(1/2) * (P2/365.25)**(1/3)

def K_from_Mp(Mp, P, tp,  e0, om0, Mp2=0, P2=0, tp2=0,  e02=0, om02=0, M1=1.34):
    '''Takes minimum mass (Mp_sin_i) in Jupiter masses, eccentricity, mass of star (M1) in solar masses, 
    and periods (P1, P2) in years, and returns the expected semi-amplitude (K) in m/s
    '''
    
    
    # asumme edge-on orbit. Thus the mass is just the minimum mass Mp*sin(i)
    i = 90
    if np.median(Mp2)==0.0:
        return Mp*((np.sqrt(1-e0**2))/28.4329)**(-1) * (np.sin(i)) * (M1)**(-1/2) * (P/365.25)**(-1/3), 0
    else:
        return Mp*((np.sqrt(1-e0**2))/28.4329)**(-1) * (np.sin(i)) * (M1)**(-1/2) * (P/365.25)**(-1/3),  Mp2*((np.sqrt(1-e02**2))/28.4329)**(-1) * (np.sin(i)) * (M1)**(-1/2) * (P2/365.25)**(-1/3)



def K_hist(out, nburnin):
    flat_samples = out.get_chains(nthin =5, nburnin = nburnin, flat=True)
    
    Mp, logP, tp, e0, de_dt, om0, dom_dt, de2_dt2, dom2_dt2, Mp2, logP2, tp2, e02, de2_dt, om02, dom2_dt, D1, D2, D3, jit1, jit2, jit3, lin, quad = flat_samples.T
    P = 10**logP
    P2 = 10**logP2
    
    if np.median(Mp2) == 0.0:
        fig, ax = plt.subplots(1,1, figsize=(8,4))
        K1 = K_from_Mp(Mp, P, tp, e0, om0)[0]
        ax.hist(K1, bins=20, color='#24ab9b')
        ax.set_xlabel('K (m/s')
        
        return K1
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        print(K_from_Mp(Mp, P, tp, e0, om0))
        K1, K2 = K_from_Mp(Mp, P, tp, e0, om0, Mp2, P2, tp2,  e02, om02)
        ax[0].hist(K1, bins=20, color='#24ab9b')
        ax[0].set_xlabel('K (m/s')
        ax[1].hist(K2, bins=20, color='#24ab9b')
        ax[1].set_xlabel('K (m/s')
        return K1, K2
    
    
def chains_plot(out, labels, nburnin, varying_only=False):
    '''
    plots the chains of the mcmc
    if varying_only = True, only the parameters with non-zero width will be plotted.
    The default is to plot all.
    '''
    
    if varying_only==True:
        varying_chains_np, varying_labels = extract_varying_params_labels(out, nburnin, labels)
        
        fig, axes = plt.subplots(np.shape(varying_chains_np)[1], figsize=(10, 1.5*np.shape(varying_chains_np)[1]), sharex=True)
        for i in range(len(varying_labels)):
            ax = axes[i]
            ax.plot(varying_chains_np[:, i], "k", alpha=0.3)
            ax.set_xlim(0, len(varying_chains_np))
            ax.set_ylabel(varying_labels[i], size=14)
            ax.yaxis.set_label_coords(-0.1, 0.5)
    else:
        samples = out.flatchains
        fig, axes = plt.subplots(np.shape(samples)[1], figsize=(10, 1.5*np.shape(samples)[1]), sharex=True)
        for i in range(len(labels)):
            ax = axes[i]
            ax.plot(samples[:, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i], size=14)
            ax.yaxis.set_label_coords(-0.1, 0.5)
        
def corner_plot(out, nburnin, labels, eval_method, label_indexes=None, 
                fig=None, color=None, annote_text='', txt_x=1, txt_y=0.99,
                legend_fontsize=30):
    '''
    Produces the corner plot for the MCMC. 
    corner_plot will only plot the params with non-zero width (i.e. were allowed to vary during the mcmc)
    To plot a subset of the parameters, label_indexes can be specified corresponding to those parameters.
    The default is to plot all varying parameters.
    '''
    
    if label_indexes==None:
        label_indexes=range(0, np.shape(out.flatchains)[1])
    
    varying_chains_np, varying_labels = extract_varying_params_labels(out, nburnin, labels, label_indexes)
    
    # global variable - set font-size to 14
    plt.rcParams['font.size'] =14
    
    if fig==None:
        # plot corner plot
        fig = corner.corner(
            varying_chains_np,#[8000000:,],
            color=color,
            labels=varying_labels,
            quantiles=[0.159, 0.5, 0.841],
            show_titles=True,
            title_fmt=".2E",
            title_kwargs=dict(fontsize=13),
            label_kwargs=dict(fontsize=22),
            labelpad = 0.13)
        fig.suptitle(eval_method, fontsize=legend_fontsize, horizontalalignment='left', color=color)
        fig.text(txt_x, txt_y, annote_text, ha='left', va='top', fontsize=legend_fontsize, color=color)
    else:
        corner.corner(
            varying_chains_np,#[8000000:,],
            fig=fig,
            color=color,
            labels=varying_labels,
            quantiles=[0.159, 0.5, 0.841],
            show_titles=True,
            title_fmt=".2E",
            title_kwargs=dict(fontsize=13),
            label_kwargs=dict(fontsize=22),
            labelpad = 0.13)
        fig.suptitle(eval_method, fontsize=legend_fontsize, horizontalalignment='left', color=color)
        fig.text(txt_x, txt_y, annote_text, ha='left', va='top', fontsize=legend_fontsize, color=color)
    
    # revert global font-size to 17
    plt.rcParams['font.size'] =17
    
    return fig

def corner_plot_from_file(varying_chains_np, varying_labels, eval_method, label_indexes=None, 
                fig=None, color=None, annote_text='', txt_x=1, txt_y=0.99,
                legend_fontsize=30):
    '''
    Produces the corner plot for the MCMC. 
    corner_plot will only plot the params with non-zero width (i.e. were allowed to vary during the mcmc)
    To plot a subset of the parameters, label_indexes can be specified corresponding to those parameters.
    The default is to plot all varying parameters.
    '''
    
    # global variable - set font-size to 14
    plt.rcParams['font.size'] =14
    
    if fig==None:
        # plot corner plot
        fig = corner.corner(
            varying_chains_np,#[8000000:,],
            color=color,
            labels=varying_labels,
            quantiles=[0.159, 0.5, 0.841],
            show_titles=True,
            title_fmt=".2E",
            title_kwargs=dict(fontsize=13),
            label_kwargs=dict(fontsize=22),
            labelpad = 0.13)
        fig.suptitle(eval_method, fontsize=legend_fontsize, horizontalalignment='left', color=color)
        fig.text(txt_x, txt_y, annote_text, ha='left', va='top', fontsize=legend_fontsize, color=color)
    else:
        corner.corner(
            varying_chains_np,#[8000000:,],
            fig=fig,
            color=color,
            labels=varying_labels,
            quantiles=[0.159, 0.5, 0.841],
            show_titles=True,
            title_fmt=".2E",
            title_kwargs=dict(fontsize=13),
            label_kwargs=dict(fontsize=22),
            labelpad = 0.13)
        fig.suptitle(eval_method, fontsize=legend_fontsize, horizontalalignment='left', color=color)
        fig.text(txt_x, txt_y, annote_text, ha='left', va='top', fontsize=legend_fontsize, color=color)
    
    # revert global font-size to 17
    plt.rcParams['font.size'] =17
    
    return fig
    
def plus_minus_formatter(med, lower_b, upper_b):
    return '%.3f_{-%.3f}^{+%.3f}'%(med, lower_b, upper_b)

def display_mcmc_percentiles(out, labels,nburnin, loglikelihood, chisq, AIC, BIC, gelman_values, gelman_labels, eval_method, csv_name,csv_name_p, label_indexes=None):  
    if label_indexes==None:
        label_indexes=range(0, np.shape(out.flatchains)[1])
        
    varying_chains_np, varying_labels = extract_varying_params_labels(out, nburnin, labels, label_indexes)
    
    mcmc_label_list = []
    mcmc_value_list = []
    mcmc_diff_l_list = []
    mcmc_diff_u_list = []
    for i in range(len(varying_labels)):
        # computes mass from amplitude, e, and period
        if varying_labels[i]=="Amplitude":
            mcmc = np.percentile(varying_chains_np[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            
            mcmc_label_list.append(varying_labels[i]+'_mcmc')
            mcmc_value_list.append(mcmc[1])
            mcmc_diff_l_list.append(q[0])
            mcmc_diff_u_list.append(q[1])
            txt = "\mathrm{{{3}}} = {0:.3e}_{{-{1:.3e}}}^{{{2:.3e}}} \ \mathrm{{{4}}}"
            txt = txt.format(mcmc[1], q[0], q[1], varying_labels[i], "m/s")
            display(Math(txt))
            
            #also print the corresponding mass
            amp_mcmcs = np.percentile(varying_chains_np[:, 0],  [16, 50, 84])
            e_mcmcs = np.percentile(varying_chains_np[:, 3],  [16, 50, 84])
            period_mcmcs = 10**(np.percentile(varying_chains_np[:, 1],  [16, 50, 84]))
            mcmc = Mp_from_K(amp_mcmcs, e_mcmcs, period_mcmcs, 1.34)
            q = np.diff(mcmc)
            
            mcmc_label_list.append('Mp_mcmc')
            mcmc_value_list.append(mcmc[1])
            mcmc_diff_l_list.append(q[0])
            mcmc_diff_u_list.append(q[1])
            txt = "\mathrm{{{3}}} = {0:.3e}_{{-{1:.3e}}}^{{{2:.3e}}} \ \mathrm{{{4}}}"
            txt = txt.format(mcmc[1],q[0], q[1], "M_p", "M_{jup}")
            display(Math(txt))
        if varying_labels[i]=="Amplitude2":
            mcmc = np.percentile(varying_chains_np[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            
            mcmc_label_list.append(varying_labels[i]+'_mcmc')
            mcmc_value_list.append(mcmc[1])
            mcmc_diff_l_list.append(q[0])
            mcmc_diff_u_list.append(q[1])
            txt = "\mathrm{{{3}}} = {0:.3e}_{{-{1:.3e}}}^{{{2:.3e}}} \ \mathrm{{{4}}}"
            txt = txt.format(mcmc[1], q[0], q[1], varying_labels[i], "m/s")
            display(Math(txt))
            
            #also print the corresponding mass
            amp_mcmcs = np.percentile(varying_chains_np[:, 0],  [16, 50, 84])
            e_mcmcs = np.percentile(varying_chains_np[:, 3],  [16, 50, 84])
            period_mcmcs = 10**(np.percentile(varying_chains_np[:, 1],  [16, 50, 84]))
            mcmc = Mp_from_K(amp_mcmcs, e_mcmcs, period_mcmcs, 1.34)
            q = np.diff(mcmc)
            
            mcmc_label_list.append('Mp2_mcmc')
            mcmc_value_list.append(mcmc[1])
            mcmc_diff_l_list.append(q[0])
            mcmc_diff_u_list.append(q[1])
            txt = "\mathrm{{{3}}} = {0:.3e}_{{-{1:.3e}}}^{{{2:.3e}}} \ \mathrm{{{4}}}"
            txt = txt.format(mcmc[1],q[0], q[1], "M_p", "M_{jup}")
            display(Math(txt))
        # print the period in additon to logP
        if varying_labels[i]=="LogP" or varying_labels[i]=="LogP2":
            mcmc = np.percentile(varying_chains_np[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            
            mcmc_label_list.append(varying_labels[i]+'_mcmc')
            mcmc_value_list.append(mcmc[1])
            mcmc_diff_l_list.append(q[0])
            mcmc_diff_u_list.append(q[1])
            txt = "\mathrm{{{3}}} = {0:.3e}_{{-{1:.3e}}}^{{{2:.3e}}} \ \mathrm{{{4}}}"
            txt = txt.format(mcmc[1], q[0], q[1], varying_labels[i], "m/s")
            display(Math(txt))
            
            # also print the corresponding P
            mcmc = 10**(np.percentile(varying_chains_np[:, i],  [16, 50, 84]))
            q = np.diff(mcmc)
            if varying_labels[i]=="LogP":
                mcmc_label_list.append('P_mcmc')
            else:
                mcmc_label_list.append('P2_mcmc')
            mcmc_value_list.append(mcmc[1])
            mcmc_diff_l_list.append(q[0])
            mcmc_diff_u_list.append(q[1])
            
            txt = "\mathrm{{{3}}} = {0:.6e}_{{-{1:.6e}}}^{{{2:.6e}}} \ \mathrm{{{4}}}"
            txt = txt.format(mcmc[1],q[0], q[1], "P", "days")
            display(Math(txt))
        # converts omega to degrees from radians
        elif varying_labels[i]=="om0" or varying_labels[i]=="om02" or varying_labels[i]=="dom/dt" or varying_labels[i]=="dom/dt_2" or varying_labels[i]=="dom^2/dt^2":
            mcmc = np.percentile(varying_chains_np[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            
            mcmc_label_list.append(varying_labels[i]+'_mcmc')
            mcmc_value_list.append(mcmc[1])
            mcmc_diff_l_list.append(q[0])
            mcmc_diff_u_list.append(q[1])
            
            txt = "\mathrm{{{3}}}  = {0:.3e}_{{-{1:.3e}}}^{{{2:.3e}}} \ \mathrm{{{4}}}"
            txt = txt.format(mcmc[1]*180/np.pi, q[0]*180/np.pi, q[1]*180/np.pi, varying_labels[i], "degrees")
            display(Math(txt))
        # prints the other parameters    
        else:
            mcmc = np.percentile(varying_chains_np[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            
            mcmc_label_list.append(varying_labels[i]+'_mcmc')
            mcmc_value_list.append(mcmc[1])
            mcmc_diff_l_list.append(q[0])
            mcmc_diff_u_list.append(q[1])
            
            txt = "\mathrm{{{3}}} = {0:.3e}_{{-{1:.3e}}}^{{{2:.3e}}}"
            txt = txt.format(mcmc[1], q[0], q[1], 
                             
                             
                             
                             varying_labels[i])
            display(Math(txt))
            
    #return mcmc_label_list
    # write results to csv for science
    data = {'Model': eval_method,
            'MCMC Values:':  ' '}
    # Add mcmc values to dictionary
    for i in range(len(mcmc_label_list)):
        data[mcmc_label_list[i]] = [[mcmc_value_list[i], mcmc_diff_l_list[i], mcmc_diff_u_list[i]]]
    data['Goodness of fit metrics:']= ' '
    data['Reduced \chi^2']=chisq
    data['loglikelihood']=loglikelihood
    data['BIC']= BIC
    data['AIC']= AIC
    data['Gelman\ Rubin\ Values:']= ' '
    # Add gelman rubin values to dictionary
    for i in range(len(gelman_labels)):
        data[gelman_labels[i]] = [gelman_values[i]]
    
    df = pd.DataFrame(data = data, index=[0])
    df.T.to_csv(csv_name)
    
    # write results to csv for paper formatting
    data_p = {'Model': eval_method,
            'MCMC Values:':  ' '}
    
    # define lists of variables that need to be in a special format
    sci_notation_label_list = ["de_dt_mcmc", "de^2/dt^2_mcmc", "dom_dt_mcmc", "dom^2/dt^2_mcmc","de/dt_2_mcmc","dom/dt_2_mcmc", "lin_mcmc", "quad_mcmc"]
    omega_conversion_label_list = ["om0_mcmc","dom/dt_mcmc", "dom/dt_2_mcmc","dom^2/dt^2_mcmc","dom/dt_2_mcmc_mcmc"]

    
    # Add mcmc values to dictionary
    for i in range(len(mcmc_label_list)):
        # check if params needs to be displayed in scientific notation
        if mcmc_label_list[i] in sci_notation_label_list:
            # check if param is in radians and needs to be converted to degrees
            if mcmc_label_list[i] not in omega_conversion_label_list:
                error_mean, value_mean = '%.1e'%(mcmc_diff_l_list[i]), '%.1e'%(mcmc_value_list[i])
                #print(mcmc_label_list[i], error_mean, value_mean, np.format_float_positional(float(error_mean), trim='-').split("."))
                
                num_decimals_err, num_decimals_val = len(np.format_float_positional(float(error_mean), trim='-').split(".")[1]),  len(np.format_float_positional(float(value_mean), trim='-').split(".")[1])
                num_decimals = num_decimals_err + 1 - num_decimals_val
                stringf = '%.'+str(num_decimals)+'e_{-%.1e}^{+%.1e}'
                #print(mcmc_label_list[i],value_mean, num_decimals_err, num_decimals_val, num_decimals, stringf%(mcmc_value_list[i], mcmc_diff_l_list[i], mcmc_diff_u_list[i]))
                data_p[mcmc_label_list[i]] = stringf%(mcmc_value_list[i], mcmc_diff_l_list[i], mcmc_diff_u_list[i])
            else: #param is in radians and needs to be converted to degrees
                error_mean, value_mean = '%.1e'%(mcmc_diff_l_list[i]*180/np.pi), '%.1e'%(mcmc_value_list[i]*180/np.pi)
                # Check if the value has at least one decimal (this is only a problem when omega =np.pi exactly)
                if len(np.format_float_positional(float(value_mean), trim='-').split("."))<2:
                    num_decimals_err, num_decimals_val = len(np.format_float_positional(float(error_mean), trim='-').split(".")[1]),  0
                else:
                    num_decimals_err, num_decimals_val = len(np.format_float_positional(float(error_mean), trim='-').split(".")[1]),  len(np.format_float_positional(float(value_mean), trim='-').split(".")[1])
                
                num_decimals = num_decimals_err + 1 - num_decimals_val
                stringf = '%.'+str(num_decimals)+'e_{-%.1e}^{+%.1e}'
                #print(mcmc_label_list[i], error_mean,value_mean, num_decimals_err, num_decimals_val, num_decimals, stringf%(mcmc_value_list[i], mcmc_diff_l_list[i], mcmc_diff_u_list[i]))
                data_p[mcmc_label_list[i]] = stringf%(mcmc_value_list[i]*180/np.pi, mcmc_diff_l_list[i]*180/np.pi, mcmc_diff_u_list[i]*180/np.pi)
        # params needs to not be in scientific notation
        else:
            # check if param is in radians and needs to be converted to degrees
            if mcmc_label_list[i] not in omega_conversion_label_list:
                if mcmc_diff_l_list[i]>=100:
                    error_mean = '%.3e'%(mcmc_diff_l_list[i])
                elif mcmc_diff_l_list[i]>=10:
                    error_mean = '%.2e'%(mcmc_diff_l_list[i])
                else:
                    error_mean = '%.1e'%(mcmc_diff_l_list[i])
                #print(mcmc_label_list[i], error_mean, np.format_float_positional(float(error_mean), trim='-').split("."), mcmc_value_list[i], mcmc_diff_l_list[i], mcmc_diff_u_list[i])
                if len(np.format_float_positional(float(error_mean), trim='-').split("."))<2:
                    num_decimals = 0
                else:    
                    num_decimals = len(np.format_float_positional(float(error_mean), trim='-').split(".")[1])
                    
                stringf = '%.'+str(num_decimals)+'f_{-%.1e}^{+%.1e}'
                #print(mcmc_label_list[i], error_mean,num_decimals_err, num_decimals_val, num_decimals, stringf%(mcmc_value_list[i], mcmc_diff_l_list[i], mcmc_diff_u_list[i]))
                data_p[mcmc_label_list[i]] = stringf%(mcmc_value_list[i], mcmc_diff_l_list[i], mcmc_diff_u_list[i])
            else: #param is in radians and needs to be converted to degrees
                if mcmc_diff_l_list[i]>=100:
                    error_mean = '%.3e'%(mcmc_diff_l_list[i])
                elif mcmc_diff_l_list[i]>=10:
                    error_mean = '%.2e'%(mcmc_diff_l_list[i])
                else:
                    error_mean = '%.1e'%(mcmc_diff_l_list[i])
                    
                #print(mcmc_label_list[i], error_mean, np.format_float_positional(float(error_mean), trim='-').split("."), mcmc_value_list[i], mcmc_diff_l_list[i], mcmc_diff_u_list[i])
                
                if len(np.format_float_positional(float(error_mean), trim='-').split("."))<2:
                    num_decimals = 0
                else:    
                    num_decimals = len(np.format_float_positional(float(error_mean), trim='-').split(".")[1])
                stringf = '%.'+str(num_decimals)+'f_{-%.1e}^{+%.1e}'
                #print(mcmc_label_list[i], error_mean,num_decimals_err, num_decimals_val, num_decimals, stringf%(mcmc_value_list[i], mcmc_diff_l_list[i], mcmc_diff_u_list[i]))
                data_p[mcmc_label_list[i]] = stringf%(mcmc_value_list[i]*180/np.pi, mcmc_diff_l_list[i]*180/np.pi, mcmc_diff_u_list[i]*180/np.pi)

        # convert omega (radians) to omega (degrees)
        #if mcmc_label_list[i]== "om0_mcmc" or mcmc_label_list[i]== "om02_mcmc" or mcmc_label_list[i]=='dom/dt_mcmc' or mcmc_label_list[i]=="dom/dt_2_mcmc" or mcmc_label_list[i]== "dom^2/dt^2_mcmc":
        #    data_p[mcmc_label_list[i]] = ['%.3e_{-%.3e}^{+%.3e}'%(mcmc_value_list[i]*180/np.pi, mcmc_diff_l_list[i]*180/np.pi,mcmc_diff_u_list[i]*180/np.pi)]
        #else:   
        #    data_p[mcmc_label_list[i]] = ['%.3e_{-%.3e}^{+%.3e}'%(mcmc_value_list[i], mcmc_diff_l_list[i], mcmc_diff_u_list[i])]
    data_p['Goodness of fit metrics:']= ' '
    data_p['Reduced \chi^2']='%.3f'%(chisq)
    data_p['loglikelihood']='%.3f'%(loglikelihood)
    data_p['BIC']= '%.3f'%(BIC)
    data_p['AIC']= '%.3f'%(AIC)
    data_p['Gelman\ Rubin\ Values:']= ' '
    # Add gelman rubin values to dictionary
    for i in range(len(gelman_labels)):
        data_p[gelman_labels[i]] = ['%.3f'%(gelman_values[i])]
    
    df_p = pd.DataFrame(data = data_p, index=[0])
    df_p.T.to_csv(csv_name_p)
    
    return df, varying_labels, varying_chains_np, df_p


## START FIXING CODE HERE ######
    
def one_planet_model(t, theta, telescopes, use_mass_parametrization=False):
    if use_mass_parametrization:
        Mp, P, tp, e0, om0, D1, D2, D3 = theta
        K = K_from_Mp(Mp, P, tp,  e0, om0, M1=1.34)[0]
    else:
        K, P, tp, e0, om0, D1, D2, D3 = theta

    # RVs with no offset
    model = K *radvel.kepler.rv_drive(t, [P, tp,e0, om0, 1.0], use_c_kepler_solver=True)
    
    # Add offsets for individual telescopes
    D_s = [D1, D2, D3]
    unique_telescopes = np.unique(telescopes)
    if D1 == 0 and D2== 0 and D3 ==0:
        model = model.copy()
    elif len(unique_telescopes)==1:
        model = model.copy() +D1
    else:
        for tel in range(0, len(unique_telescopes)):
            indexes_telesc = np.where(telescopes==unique_telescopes[tel])
            model[indexes_telesc]= model[indexes_telesc].copy()+D_s[tel]
            
    return model

def lin_quad_model(time, theta):
    lin, quad  = theta
    return lin*(time-np.min(time)) + quad*(time-np.min(time))**2

def fair_draws_plot(time_np, rv_np, s_rv_np, out, num_draws, nburnin, telescopes, time_chunk, eval_method,
                    amp_mcmc=0, period_mcmc=0, tp_mcmc=0,D_mcmc=0, e1=0, om1=0, lin_mcmc=0, quad_mcmc=0,
                    D1_mcmc=0,D2_mcmc=0,D3_mcmc=0, 
                    loglikelihood=0, chisq=0, AIC=0, BIC=0,
                    phase_plot=0.5, legendfontsize=21, residuals=True, uncertanties=False,
                    err_inflate = False, use_mass_parametrization=False):
    flat_samples = out.get_chains(nthin =5, nburnin = nburnin, flat=True)
    
    if use_mass_parametrization:
        Mp_mcmc, logperiod_mcmc, tp_mcmc, e_mcmc, de_dt_mcmc, om_mcmc,dom_dt_mcmc, de2_dt2_mcmc, dom2_dt2_mcmc, Mp2_mcmc, logperiod2_mcmc, tp2_mcmc, e02_mcmc, de2_dt_mcmc, om02_mcmc,dom2_dt_mcmc, D1_mcmc,D2_mcmc,D3_mcmc, jit1_mcmc, jit2_mcmc, jit3_mcmc,  lin_mcmc, quad_mcmc = np.percentile(flat_samples[:, :], [50], axis=0)[0]
        Mp_mcmc_l, logperiod_mcmc_l, tp_mcmc_l, e_mcmc_l, de_dt_mcmc_l, om_mcmc_l,dom_dt_mcmc_l, de2_dt2_mcmc_l, dom2_dt2_mcmc_l, Mp2_mcmc_l, logperiod2_mcmc_l, tp2_mcmc_l, e02_mcmc_l, de2_dt_mcmc_l, om02_mcmc_l,dom2_dt_mcmc_l, D1_mcmc_l,D2_mcmc_l,D3_mcmc_l, jit1_mcmc_l, jit2_mcmc_l, jit3_mcmc_l,  lin_mcmc_l, quad_mcmc_l = np.percentile(flat_samples[:, :], [16], axis=0)[0]
        Mp_mcmc_u, logperiod_mcmc_u, tp_mcmc_u, e_mcmc_u, de_dt_mcmc_u, om_mcmc_u,dom_dt_mcmc_u, de2_dt2_mcmc_u, dom2_dt2_mcmc_u, Mp2_mcmc_u, logperiod2_mcmc_u, tp2_mcmc_u, e02_mcmc_u, de2_dt_mcmc_u, om02_mcmc_u,dom2_dt_mcmc_u, D1_mcmc_u,D2_mcmc_u,D3_mcmc_u, jit1_mcmc_u, jit2_mcmc_u, jit3_mcmc_u,  lin_mcmc_u, quad_mcmc_u = np.percentile(flat_samples[:, :], [84], axis=0)[0]
        
        amp_mcmc, amp2_mcmc = K_from_Mp(Mp_mcmc, 10**logperiod_mcmc, tp_mcmc,  e_mcmc, om_mcmc,  Mp2=Mp2_mcmc, P2=10**logperiod2_mcmc, tp2=tp2_mcmc,  e02=e02_mcmc, om02=om02_mcmc, M1=1.34)
        amp_mcmc_l, amp2_mcmc_l = K_from_Mp(Mp_mcmc_l, 10**logperiod_mcmc_l, tp_mcmc_l,  e_mcmc_l, om_mcmc_l,  Mp2=Mp2_mcmc_l, P2=10**logperiod2_mcmc_l, tp2=tp2_mcmc_l,  e02=e02_mcmc_l, om02=om02_mcmc_l, M1=1.34)
        amp_mcmc_u, amp2_mcmc_u = K_from_Mp(Mp_mcmc_u, 10**logperiod_mcmc_u, tp_mcmc_u,  e_mcmc_u, om_mcmc_u,  Mp2=Mp2_mcmc_u, P2=10**logperiod2_mcmc_u, tp2=tp2_mcmc_u,  e02=e02_mcmc_u, om02=om02_mcmc_u, M1=1.34)
    else:
        amp_mcmc, logperiod_mcmc, tp_mcmc, e_mcmc, de_dt_mcmc, om_mcmc,dom_dt_mcmc, de2_dt2_mcmc, dom2_dt2_mcmc, amp2_mcmc, logperiod2_mcmc, tp2_mcmc, e02_mcmc, de2_dt_mcmc, om02_mcmc,dom2_dt_mcmc, D1_mcmc,D2_mcmc,D3_mcmc, jit1_mcmc, jit2_mcmc, jit3_mcmc,  lin_mcmc, quad_mcmc = np.percentile(flat_samples[:, :], [50], axis=0)[0]
        amp_mcmc_l, logperiod_mcmc_l, tp_mcmc_l, e_mcmc_l, de_dt_mcmc_l, om_mcmc_l,dom_dt_mcmc_l, de2_dt2_mcmc_l, dom2_dt2_mcmc_l, amp2_mcmc_l, logperiod2_mcmc_l, tp2_mcmc_l, e02_mcmc_l, de2_dt_mcmc_l, om02_mcmc_l,dom2_dt_mcmc_l, D1_mcmc_l,D2_mcmc_l,D3_mcmc_l, jit1_mcmc_l, jit2_mcmc_l, jit3_mcmc_l,  lin_mcmc_l, quad_mcmc_l = np.percentile(flat_samples[:, :], [16], axis=0)[0]
        amp_mcmc_u, logperiod_mcmc_u, tp_mcmc_u, e_mcmc_u, de_dt_mcmc_u, om_mcmc_u,dom_dt_mcmc_u, de2_dt2_mcmc_u, dom2_dt2_mcmc_u, amp2_mcmc_u, logperiod2_mcmc_u, tp2_mcmc_u, e02_mcmc_u, de2_dt_mcmc_u, om02_mcmc_u,dom2_dt_mcmc_u, D1_mcmc_u,D2_mcmc_u,D3_mcmc_u, jit1_mcmc_u, jit2_mcmc_u, jit3_mcmc_u,  lin_mcmc_u, quad_mcmc_u = np.percentile(flat_samples[:, :], [84], axis=0)[0]
        
    e_err = np.max([np.abs(e_mcmc_l-e_mcmc), np.abs(e_mcmc_u-e_mcmc)])
    om_err =  np.max([np.abs(om_mcmc_l-om_mcmc), np.abs(om_mcmc_u-om_mcmc)])
    
    
    period_mcmc = 10**logperiod_mcmc
    period2_mcmc = 10**logperiod2_mcmc
    
    if e1==0:
        e1 = e_mcmc
        om1 = om_mcmc
    
    D_s = [D1_mcmc,D2_mcmc,D3_mcmc]
    jit_s = [jit1_mcmc, jit2_mcmc, jit3_mcmc]
    
    if use_mass_parametrization:
        Mp_dr, logP_dr, tp_dr, e0_dr, de_dt_dr, om0_dr,dom_dt_dr,de2_dt2_dr, dom2_dt2_dr, Mp2_dr, logP2_dr, tp2_dr, e02_dr, de2_dt_dr, om02_dr,dom2_dt_dr, D1_dr,D2_dr,D3_dr, jit1_dr, jit2_dr, jit3_dr, lin_dr, quad_dr= np.array(sample(flat_samples.tolist(), num_draws)).T
        #K_dr, K2_dr = K_from_Mp(Mp_dr, 10**logP_dr, tp_dr,  e0_dr, om0_dr,  Mp2=Mp2_dr, P2=10**logP2_dr, tp2=tp2_dr,  e02=e02_dr, om02=om02_dr, M1=1.34)
        K2_dr = K_from_Mp(Mp2_dr, 10**logP2_dr, tp2_dr,  e02_dr, om02_dr, M1=1.34)[0]
    else:
       K_dr, logP_dr, tp_dr, e0_dr, de_dt_dr, om0_dr,dom_dt_dr,de2_dt2_dr, dom2_dt2_dr, K2_dr, logP2_dr, tp2_dr, e02_dr, de2_dt_dr, om02_dr,dom2_dt_dr, D1_dr,D2_dr,D3_dr, jit1_dr, jit2_dr, jit3_dr, lin_dr, quad_dr= np.array(sample(flat_samples.tolist(), num_draws)).T
       
   
    P_dr = 10**logP_dr
    P2_dr = 10**logP2_dr

    if residuals: # plot residuals
        fig, axes = plt.subplots(2,2, figsize=(20, 8),  gridspec_kw={'height_ratios': [3, 0.6]})
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        
        txt_str = "Red. $\chi^2$ = %.3f \nBIC = %.3f \nAIC = %.3f "%(chisq, BIC, AIC)
        axes[0][0].text(3.5, -1100,txt_str, 
                        bbox = {'facecolor': '#fab002', 'alpha': 0.8, 'pad': 8}, 
                        multialignment='left', size=round(legendfontsize*1.3))
    else: # do not plot residuals
        fig, axes = plt.subplots(1,2, figsize=(20, 6.5))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        txt_str = "Red. $\chi^2$ = %.3f \nBIC = %.3f \nAIC = %.3f "%(chisq, BIC, AIC)
        axes[0].text(3.5, -1100,txt_str, 
                        bbox = {'facecolor': '#fab002', 'alpha': 0.8, 'pad': 8}, 
                        multialignment='left', size=round(legendfontsize*1.3))
        
    if np.median(K2_dr)==0.0:
        # compute the average model
        print(e1, om1)
        ti_np = np.arange(np.min(time_np), np.max(time_np), 0.1)
        y_preds = one_planet_model(ti_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds_time_np = one_planet_model(time_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds_time_np2 = lin_quad_model(time_np, np.array([lin_mcmc, quad_mcmc]))
        data_ti = {'phase': (ti_np-phase_plot*period_mcmc)%period_mcmc,
                'rvs': y_preds}
        df_di = pd.DataFrame(data=data_ti)
        df_di_sorted = df_di.sort_values(by=['phase'])

        data2 = {'phase': ti_np-phase_plot*period_mcmc,
                 'y_preds': lin_quad_model(ti_np, np.array([lin_mcmc, quad_mcmc]))}
        df_ti2 = pd.DataFrame(data=data2)
        df_ti_sorted2 = df_ti2.sort_values(by=['phase'])
        
        # plot n random draws
        for i in range(num_draws):
            if use_mass_parametrization:
                Mp_dr_d = Mp_dr[i]
            else:
                K_dr_d = K_dr[i]
                
            P_dr_d = P_dr[i]
            tp_dr_d = tp_dr[i]
            e0_dr_d = e0_dr[i]
            de_dt_dr_d = de_dt_dr[i]
            om0_dr_d = om0_dr[i]
            dom_dt_dr_d = dom_dt_dr[i]
            de2_dt2_dr_d = de2_dt2_dr[i]
            dom2_dt2_dr_d = dom2_dt2_dr[i]
            D1_dr_d = D1_dr[i]
            D2_dr_d = D2_dr[i]
            D3_dr_d = D3_dr[i]
            lin_dr_d = lin_dr[i]
            quad_dr_d = quad_dr[i]
            

            # compute median e and median om
            med_e_d = np.median(de_dt_dr_d*time_np+e0_dr_d+de2_dt2_dr_d*time_np**2)
            med_om_d = np.median(dom_dt_dr_d*time_np+om0_dr_d+dom2_dt2_dr_d*time_np**2)
            
            if use_mass_parametrization:
                # Compute K_dr based on Mp, P, tp, and median e and median omega
                K_dr_d = K_from_Mp(Mp_dr_d, P_dr_d, tp_dr_d,  med_e_d, med_om_d, M1=1.34)[0]

            #print(avg_e_mcmc, avg_om_mcmc)


            #ti_np = np.arange(np.min(time_np), np.max(time_np), 0.1)
            planet_y_preds_ti_np = one_planet_model(ti_np, np.array([K_dr_d, P_dr_d, tp_dr_d,
                                                        med_e_d, med_om_d,0, 0,0]), telescopes)
            planet_y_preds_time_np = one_planet_model(time_np, np.array([K_dr_d, P_dr_d, tp_dr_d,
                                                        med_e_d, med_om_d,0, 0,0]), telescopes)
            lin_quad_y_preds_time_np = lin_quad_model(time_np, np.array([lin_dr_d, quad_dr_d]))+D1_dr_d-D1_mcmc

            planet_data_ti = {'phase': (ti_np-phase_plot*P_dr_d)%P_dr_d,
                'rvs': planet_y_preds_ti_np}#+D2_dr_d+D3_dr_d}
            df_planet_ti = pd.DataFrame(data=planet_data_ti)
            df_planet_ti_sorted = df_planet_ti.sort_values(by=['phase'])
            #avg_phase, avg_rvs = average_y_preds(df_di_sorted)

            lin_quad_data_ti = {'phase': ti_np,
                   'y_preds': lin_quad_model(ti_np, np.array([lin_dr_d, quad_dr_d]))+D1_dr_d-D1_mcmc}
            df_lin_quad_data_ti = pd.DataFrame(data=lin_quad_data_ti)
            df_lin_quad_data_ti_sorted = df_lin_quad_data_ti.sort_values(by=['phase'])

            
            if residuals: # plot residuals
                # plot hat-p-2b
                ax = axes[0][0]
                ax.plot(df_planet_ti_sorted['phase'], df_planet_ti_sorted['rvs'], color='#cebce8', alpha=0.3)
                
                # plot hat-p-2b resids
                ax = axes[1][0]
                ax.plot(df_planet_ti_sorted['phase'], df_planet_ti_sorted['rvs']-df_di_sorted['rvs'], color='#cebce8', alpha=0.3)
    
                # plot quadratic trend
                ax = axes[0][1]
                ax.plot(df_lin_quad_data_ti_sorted['phase'], df_lin_quad_data_ti_sorted['y_preds'], color='#cebce8', alpha=0.3)
                
                # plot quadratic trend resids
                ax = axes[1][1]
                ax.plot(df_lin_quad_data_ti_sorted['phase'], df_lin_quad_data_ti_sorted['y_preds']-df_ti_sorted2['y_preds'], color='#cebce8', alpha=0.3)
            else: # do not plot residuals
                # plot hat-p-2b
                ax = axes[0]
                ax.plot(df_planet_ti_sorted['phase'], df_planet_ti_sorted['rvs'], color='#cebce8', alpha=0.3)
    
                # plot quadratic trend
                ax = axes[1]
                ax.plot(df_lin_quad_data_ti_sorted['phase'], df_lin_quad_data_ti_sorted['y_preds'], color='#cebce8', alpha=0.3)
                


        # plot hat-p-2b RVs
        
        colors = ['#24ab9b', 'r', 'k']
        num_color = 0
        # plot each of the rv datasets
        for tel_name in np.unique(telescopes):
            indexes = np.where(telescopes==tel_name)
            #print(i, indexes, rv_np[indexes])
            
            inflated_s_rv_np = np.sqrt(s_rv_np[indexes].copy()**2+jit_s[num_color]**2)
            
            
            if residuals:# plot residuals
                # plot rvs
                ax = axes[0][0]
                if err_inflate: # inflate errors from jitter estimate
                    ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                            yerr=inflated_s_rv_np, fmt = 'o', color = colors[num_color],label=tel_name)
                else: # do not inflate errors
                    ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                            yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                
                # plot residuals
                ax = axes[1][0]
                if err_inflate: # inflate errors from jitter estimate
                    ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, 
                                rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                                yerr=inflated_s_rv_np, fmt = 'o', color = colors[num_color],label=tel_name)
                else:  # do not inflate errors
                    ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, 
                                rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],yerr=s_rv_np[indexes],
                                fmt = 'o', color = colors[num_color],label=tel_name)
                
                # plot quadratic trend data
                ax = axes[0][1]
                if err_inflate: # inflate errors from jitter estimate
                    ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-D_s[num_color],
                                yerr=inflated_s_rv_np, fmt = 'o', color = colors[num_color],label=tel_name)
                else: # do not inflate errors
                    ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-D_s[num_color],
                                yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                    
                ax = axes[1][1]
                if err_inflate: # inflate errors from jitter estimate
                    ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                                yerr=inflated_s_rv_np, fmt = 'o', color = colors[num_color],label=tel_name)
                else:
                    ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                                yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                
                num_color += 1
            else: # do not plot residuals
                # plot rvs
                ax = axes[0]
                if err_inflate: # inflate errors from jitter estimate
                    ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                                yerr=inflated_s_rv_np, fmt = 'o', color = colors[num_color],label=tel_name)
                else:
                    ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                                yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                
                # plot quadratic trend data
                ax = axes[1]
                if err_inflate: # inflate errors from jitter estimate
                    ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-D_s[num_color],
                                yerr=inflated_s_rv_np, fmt = 'o', color = colors[num_color],label=tel_name)
                else: # do not inflate errors
                    ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-D_s[num_color],
                                yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                
                num_color += 1
        
        if residuals: # plot residuals
            ax = axes[0][0]
            
            # plot the average model
            if uncertanties:
                ax.plot(df_di_sorted['phase'], df_di_sorted['rvs'], color='#7829e6', linewidth=2,
                        label="e= %.4f $\pm$ %.4f, $\omega$= %.2f $\pm$ %.2f $^\circ$"%(e1, e_err,om1*180/np.pi, om_err*180/np.pi))
            else:
                ax.plot(df_di_sorted['phase'], df_di_sorted['rvs'], color='#7829e6', linewidth=2,
                        label="e= %.3f, $\omega$= %.3f $^\circ$"%(e1,om1*180/np.pi))
            ax.set_ylabel("RV (m/s)")
            ax.set_title(time_chunk+eval_method)
            
    
            # plot hat-p-2b residuals labels
            ax = axes[1][0]
            ax.plot([np.min(df_di_sorted['phase']), np.max(df_di_sorted['phase'])], [0,0], color='#7829e6', linewidth=2 )
            ax.set_ylabel("Residuals")
            ax.set_xlabel("Phase (days)")
    
            # plot quadratic trend
            ax = axes[0][1]
            ax.plot(df_ti_sorted2['phase'], df_ti_sorted2['y_preds'], color='#7829e6', linewidth=2)
            ax.set_title(time_chunk+" Quadratic background trend")
            ax.legend(loc="upper right")
            legend = ax.legend(loc="upper right", prop={'size': legendfontsize}, facecolor='#24ab9b')
            plt.setp(legend.get_texts(), color='k')
    
            ax = axes[1][1]
            ax.plot([np.min(df_ti_sorted2['phase']), np.max(df_ti_sorted2['phase'])], [0,0], color='#7829e6', linewidth=2 )
            ax.set_xlabel("Phase (days)")
        else: # do not plot residuals
            ax = axes[0]
            if uncertanties:
                ax.plot(df_di_sorted['phase'], df_di_sorted['rvs'], color='#7829e6', linewidth=2,
                        label="e= %.4f $\pm$ %.4f, $\omega$= %.2f $\pm$ %.2f $^\circ$"%(e1, e_err,om1*180/np.pi, om_err*180/np.pi))
            else:
                ax.plot(df_di_sorted['phase'], df_di_sorted['rvs'], color='#7829e6', linewidth=2,
                        label="e= %.3f, $\omega$= %.3f $^\circ$"%(e1,om1*180/np.pi))            
            ax.set_ylabel("RV (m/s)")
            ax.set_title(time_chunk+eval_method)
            #ax.legend(loc="lower right")
            #legend = ax.legend(loc="lower right", prop={'size': legendfontsize}, facecolor='#24ab9b')
            #plt.setp(legend.get_texts(), color='k')
            ax.set_xlabel("Phase (days)")
    
            # plot quadratic trend
            ax = axes[1]
            ax.plot(df_ti_sorted2['phase'], df_ti_sorted2['y_preds'], color='#7829e6', linewidth=2)
            ax.set_title(time_chunk+" Quadratic background trend")
            ax.set_xlabel("Phase (days)")
            
        
    else: # plot two planet case
        #print("You should maybe write the code to plot two planets hehehe")
        # compute the average model
        print(e1, om1)
        ti_np = np.arange(np.min(time_np), np.max(time_np), 0.1)
        y_preds = one_planet_model(ti_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds2 = one_planet_model(ti_np, np.array([amp2_mcmc, period2_mcmc, tp2_mcmc,
                                                        e02_mcmc, om02_mcmc,0, 0, 0]), telescopes)
        y_preds_time_np = one_planet_model(time_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds_time_np2 = one_planet_model(time_np, np.array([amp2_mcmc, period2_mcmc, tp2_mcmc,
                                                        e02_mcmc, om02_mcmc,0, 0, 0]), telescopes)
        
        
        data_ti = {'phase': (ti_np-phase_plot*period_mcmc)%period_mcmc,
                'rvs': y_preds}
        df_di = pd.DataFrame(data=data_ti)
        df_di_sorted = df_di.sort_values(by=['phase'])

        data2 = {'phase': ti_np,
                   'y_preds': y_preds2}
        df_ti2 = pd.DataFrame(data=data2)
        df_ti_sorted2 = df_ti2.sort_values(by=['phase'])
        
        
        for i in range(num_draws):
            if use_mass_parametrization:
                Mp_dr_d = Mp_dr[i]
                Mp2_dr_d = Mp2_dr[i]
            else:
                K_dr_d = K_dr[i]
                K2_dr_d = K2_dr[i]
                
            #K_dr_d = K_dr[i]
            
            P_dr_d = P_dr[i]
            tp_dr_d = tp_dr[i]
            e0_dr_d = e0_dr[i]
            de_dt_dr_d = de_dt_dr[i]
            om0_dr_d = om0_dr[i]
            dom_dt_dr_d = dom_dt_dr[i]
            de2_dt2_dr_d = de2_dt2_dr[i]
            dom2_dt2_dr_d = dom2_dt2_dr[i]
            
           
            P2_dr_d = P2_dr[i]
            tp2_dr_d = tp2_dr[i]
            e02_dr_d = e02_dr[i]
            de2_dt_dr_d = de2_dt_dr[i]
            om02_dr_d = om02_dr[i]
            dom2_dt_dr_d = dom2_dt_dr[i]
            
            
            D1_dr_d = D1_dr[i]
            D2_dr_d = D2_dr[i]
            D3_dr_d = D3_dr[i]
            lin_dr_d = lin_dr[i]
            quad_dr_d = quad_dr[i]

            # compute median e and median om
            med_e_d = np.median(de_dt_dr_d*time_np + e0_dr_d + de2_dt2_dr_d*time_np**2)
            med_om_d = np.median(dom_dt_dr_d*time_np + om0_dr_d + dom2_dt2_dr_d*time_np**2)
            # compute median e and median om
            med_e2_d = np.median(de2_dt_dr_d*time_np+e02_dr_d)
            med_om2_d = np.median(dom2_dt_dr_d*time_np+om02_dr_d)

            #print(avg_e_mcmc, avg_om_mcmc)
            
            if use_mass_parametrization:
                # Compute K_dr based on Mp, P, tp, and median e and median omega
                K_dr_d = K_from_Mp(Mp_dr_d, P_dr_d, tp_dr_d,  med_e_d, med_om_d, M1=1.34)[0]
                K2_dr_d = K_from_Mp(Mp_dr_d, P_dr_d, tp_dr_d,  med_e_d, med_om_d, Mp2=Mp2_dr_d, P2=P2_dr_d, tp2=tp2_dr_d,  e02=med_e2_d, om02=med_om2_d, M1=1.34)[1]


            ti_np = np.arange(np.min(time_np), np.max(time_np), 0.1)
            planet_y_preds_ti_np = one_planet_model(ti_np, np.array([K_dr_d, P_dr_d, tp_dr_d,
                                                        med_e_d, med_om_d,0, 0,0]), telescopes)
            planet_y_preds_time_np = one_planet_model(time_np, np.array([K_dr_d, P_dr_d, tp_dr_d,
                                                        med_e_d, med_om_d,0, 0,0]), telescopes)
            planet2_y_preds_ti_np = one_planet_model(ti_np, np.array([K2_dr_d, P2_dr_d, tp2_dr_d,
                                                        med_e2_d, med_om2_d,0, 0,0]), telescopes)
            planet2_y_preds_time_np = one_planet_model(time_np, np.array([K2_dr_d, P2_dr_d, tp2_dr_d,
                                                        med_e2_d, med_om2_d,0, 0,0]), telescopes)
            #lin_quad_y_preds_time_np = lin_quad_model(time_np, np.array([lin_dr_d, quad_dr_d]))+D1_dr_d-D1_mcmc

            planet_data_ti = {'phase': (ti_np-phase_plot*P_dr_d)%P_dr_d,
                'rvs': planet_y_preds_ti_np}#+D2_dr_d+D3_dr_d}
            df_planet_ti = pd.DataFrame(data=planet_data_ti)
            df_planet_ti_sorted = df_planet_ti.sort_values(by=['phase'])
            
            planet2_data_ti = {'phase': ti_np,
                'planet2_rvs': planet2_y_preds_ti_np}#+D2_dr_d+D3_dr_d}
            df_planet2_ti = pd.DataFrame(data=planet2_data_ti)
            df_planet2_ti_sorted = df_planet2_ti.sort_values(by=['phase'])
            #avg_phase, avg_rvs = average_y_preds(df_di_sorted)

            #lin_quad_data_ti = {'phase': ti_np-phase_plot*P_dr_d,
            #       'y_preds': lin_quad_model(ti_np, np.array([lin_dr_d, quad_dr_d]))}
            #df_lin_quad_data_ti = pd.DataFrame(data=lin_quad_data_ti)
            #df_lin_quad_data_ti_sorted = df_lin_quad_data_ti.sort_values(by=['phase'])

            if residuals:# plot residuals
                # plot hat-p-2b
                ax = axes[0][0]
                ax.plot(df_planet_ti_sorted['phase'], df_planet_ti_sorted['rvs'], color='#cebce8', alpha=0.3)
                
                # plot hat-p-2b resids
                ax = axes[1][0]
                ax.plot(df_planet_ti_sorted['phase'], df_planet_ti_sorted['rvs']-df_di_sorted["rvs"], color='#cebce8', alpha=0.3)
    
                # plot hat-p-2c
                ax = axes[0][1]
                ax.plot(df_planet2_ti_sorted['phase'], df_planet2_ti_sorted['planet2_rvs'], color='#cebce8', alpha=0.3)
                
                # plot hat-p-2c resids
                ax = axes[1][1]
                ax.plot(df_planet2_ti_sorted['phase'], df_planet2_ti_sorted['planet2_rvs']-df_ti_sorted2["y_preds"], color='#cebce8', alpha=0.3)
                
            else: # don't plot residuals
                # plot hat-p-2b
                ax = axes[0]
                ax.plot(df_planet_ti_sorted['phase'], df_planet_ti_sorted['rvs'], color='#cebce8', alpha=0.3)
    
                # plot hat-p-2c
                ax = axes[1]
                ax.plot(df_planet2_ti_sorted['phase'],  df_planet2_ti_sorted['planet2_rvs'], color='#cebce8', alpha=0.3)


        # plot hat-p-2b RVs
        
        colors = ['#24ab9b', 'r', 'k']
        num_color = 0
        # plot each of the rv datasets
        for tel_name in np.unique(telescopes):
            indexes = np.where(telescopes==tel_name)
            #print(i, indexes, rv_np[indexes])
            
            inflated_s_rv_np = np.sqrt(s_rv_np[indexes].copy()**2+jit_s[num_color]**2)
            
            
            if residuals:# plot residuals
                # plot rvs
                ax = axes[0][0]
                if err_inflate: # inflate errors from jitter estimate
                    ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                                yerr=inflated_s_rv_np, fmt = 'o', color = colors[num_color],label=tel_name)
                else:
                    ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                                yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                
                # plot residuals
                ax = axes[1][0]
                if err_inflate: # inflate errors from jitter estimate
                    ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, 
                                rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],yerr=inflated_s_rv_np,
                                fmt = 'o', color = colors[num_color],label=tel_name)
                else:
                    ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, 
                                rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],yerr=s_rv_np[indexes],
                                fmt = 'o', color = colors[num_color],label=tel_name)
                ymax = np.max(np.abs(rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color]))
                ax.set_ylim(-ymax-0.2*ymax, ymax+0.2*ymax)
                
                # plot quadratic trend data
                ax = axes[0][1]
                if err_inflate: # inflate errors from jitter estimate
                    ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-D_s[num_color],
                                yerr=inflated_s_rv_np, fmt = 'o', color = colors[num_color],label=tel_name)
                else:
                    ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-D_s[num_color],
                                yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                
                ax = axes[1][1]
                if err_inflate: # inflate errors from jitter estimate
                    ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                                yerr=inflated_s_rv_np, fmt = 'o', color = colors[num_color],label=tel_name)
                else:
                    ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                                yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                ymax = np.max(np.abs(rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color]))
                ax.set_ylim(-ymax-0.2*ymax, ymax+0.2*ymax)
                
                num_color += 1
            else: # don't plot residuals
                # plot rvs
                ax = axes[0]
                if err_inflate: # inflate errors from jitter estimate
                    ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                                yerr=inflated_s_rv_np, fmt = 'o', color = colors[num_color],label=tel_name)
                else:
                    ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                                yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                
                # plot quadratic trend data
                ax = axes[1]
                if err_inflate: # inflate errors from jitter estimate
                    ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-D_s[num_color],
                                yerr=inflated_s_rv_np, fmt = 'o', color = colors[num_color],label=tel_name)
                else:
                    ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-D_s[num_color],
                                yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
                
                num_color += 1
        
        if residuals:# plot residuals
            ax = axes[0][0]
            if uncertanties:
                ax.plot(df_di_sorted['phase'], df_di_sorted['rvs'], color='#7829e6', linewidth=2,
                        label="e= %.4f $\pm$ %.4f, $\omega$= %.2f $\pm$ %.2f $^\circ$"%(e1, e_err,om1*180/np.pi, om_err*180/np.pi))
            else:
                ax.plot(df_di_sorted['phase'], df_di_sorted['rvs'], color='#7829e6', linewidth=2,
                        label="e= %.3f, $\omega$= %.3f $^\circ$"%(e1,om1*180/np.pi))
            
            ax.set_ylabel("RV (m/s)")
            ax.set_title(time_chunk+eval_method)
            
    
            # plot hat-p-2b residuals labels
            ax = axes[1][0]
            ax.set_ylabel("Residuals")
            ax.set_xlabel("Phase (days)")
            ax.plot([np.min(df_di_sorted['phase']), np.max(df_di_sorted['phase'])], [0,0], color='#7829e6', linewidth=2 )
    
            # plot hat-p-2c?
            ax = axes[0][1]
            ax.plot(df_ti_sorted2['phase'], df_ti_sorted2['y_preds'], color='#7829e6', linewidth=2)
            ax.set_title(time_chunk+" 2nd Planet")
            ax.legend(loc="upper right")
            legend = ax.legend(loc="upper right", prop={'size': legendfontsize}, facecolor='#24ab9b')
            plt.setp(legend.get_texts(), color='k')
    
            ax = axes[1][1]
            ax.plot([np.min(df_ti_sorted2['phase']), np.max(df_ti_sorted2['phase'])], [0,0], color='#7829e6', linewidth=2 )
            ax.set_xlabel("Phase (days)")
        else: # don't plot residuals
            ax = axes[0]
            if uncertanties:
                ax.plot(df_di_sorted['phase'], df_di_sorted['rvs'], color='#7829e6', linewidth=2,
                        label="e= %.4f $\pm$ %.4f, $\omega$= %.2f $\pm$ %.2f $^\circ$"%(e1, e_err,om1*180/np.pi, om_err*180/np.pi))
            else:
                ax.plot(df_di_sorted['phase'], df_di_sorted['rvs'], color='#7829e6', linewidth=2,
                        label="e= %.3f, $\omega$= %.3f $^\circ$"%(e1,om1*180/np.pi))            
            
            ax.set_ylabel("RV (m/s)")
            ax.set_title(time_chunk+eval_method)
            #ax.legend(loc="lower right")
            #legend = ax.legend(loc="lower right", prop={'size': legendfontsize}, facecolor='#24ab9b')
            #plt.setp(legend.get_texts(), color='k')
            ax.set_xlabel("Phase (days)")
    
            # plot hat-p-2c?
            ax = axes[1]
            ax.plot(df_ti_sorted2['phase'], df_ti_sorted2['y_preds'], color='#7829e6', linewidth=2)
            ax.set_title(time_chunk+" 2nd Planet")
            ax.set_xlabel("Phase (days)")
        
def fair_draws_plot_ext(time_np, rv_np, s_rv_np, out, num_draws, nburnin, telescopes, time_chunk, eval_method,
                    amp_mcmc=0, period_mcmc=0, tp_mcmc=0,D_mcmc=0, e1=0, om1=0, lin_mcmc=0, quad_mcmc=0,
                    D1_mcmc=0,D2_mcmc=0,D3_mcmc=0, extend_time=0, min_time=0, df_used=[], use_mass_parametrization=False):
    flat_samples = out.get_chains(nthin =5, nburnin = nburnin, flat=True)
    
    if use_mass_parametrization:
        Mp_mcmc, logperiod_mcmc, tp_mcmc, e_mcmc, de_dt_mcmc, om_mcmc,dom_dt_mcmc, de2_dt2_mcmc, dom2_dt2_mcmc, Mp2_mcmc, logperiod2_mcmc, tp2_mcmc, e02_mcmc, de2_dt_mcmc, om02_mcmc,dom2_dt_mcmc, D1_mcmc,D2_mcmc,D3_mcmc, jit1_mcmc, jit2_mcmc, jit3_mcmc,  lin_mcmc, quad_mcmc = np.percentile(flat_samples[:, :], [50], axis=0)[0]
        Mp_mcmc_l, logperiod_mcmc_l, tp_mcmc_l, e_mcmc_l, de_dt_mcmc_l, om_mcmc_l,dom_dt_mcmc_l, de2_dt2_mcmc_l, dom2_dt2_mcmc_l, Mp2_mcmc_l, logperiod2_mcmc_l, tp2_mcmc_l, e02_mcmc_l, de2_dt_mcmc_l, om02_mcmc_l,dom2_dt_mcmc_l, D1_mcmc_l,D2_mcmc_l,D3_mcmc_l, jit1_mcmc_l, jit2_mcmc_l, jit3_mcmc_l,  lin_mcmc_l, quad_mcmc_l = np.percentile(flat_samples[:, :], [16], axis=0)[0]
        Mp_mcmc_u, logperiod_mcmc_u, tp_mcmc_u, e_mcmc_u, de_dt_mcmc_u, om_mcmc_u,dom_dt_mcmc_u, de2_dt2_mcmc_u, dom2_dt2_mcmc_u, Mp2_mcmc_u, logperiod2_mcmc_u, tp2_mcmc_u, e02_mcmc_u, de2_dt_mcmc_u, om02_mcmc_u,dom2_dt_mcmc_u, D1_mcmc_u,D2_mcmc_u,D3_mcmc_u, jit1_mcmc_u, jit2_mcmc_u, jit3_mcmc_u,  lin_mcmc_u, quad_mcmc_u = np.percentile(flat_samples[:, :], [84], axis=0)[0]
        
        amp_mcmc, amp2_mcmc = K_from_Mp(Mp_mcmc, 10**logperiod_mcmc, tp_mcmc,  e_mcmc, om_mcmc,  Mp2=Mp2_mcmc, P2=10**logperiod2_mcmc, tp2=tp2_mcmc,  e02=e02_mcmc, om02=om02_mcmc, M1=1.34)
        amp_mcmc_l, amp2_mcmc_l = K_from_Mp(Mp_mcmc_l, 10**logperiod_mcmc_l, tp_mcmc_l,  e_mcmc_l, om_mcmc_l,  Mp2=Mp2_mcmc_l, P2=10**logperiod2_mcmc_l, tp2=tp2_mcmc_l,  e02=e02_mcmc_l, om02=om02_mcmc_l, M1=1.34)
        amp_mcmc_u, amp2_mcmc_u = K_from_Mp(Mp_mcmc_u, 10**logperiod_mcmc_u, tp_mcmc_u,  e_mcmc_u, om_mcmc_u,  Mp2=Mp2_mcmc_u, P2=10**logperiod2_mcmc_u, tp2=tp2_mcmc_u,  e02=e02_mcmc_u, om02=om02_mcmc_u, M1=1.34)
    else:
        amp_mcmc, logperiod_mcmc, tp_mcmc, e_mcmc, de_dt_mcmc, om_mcmc,dom_dt_mcmc, de2_dt2_mcmc, dom2_dt2_mcmc, amp2_mcmc, logperiod2_mcmc, tp2_mcmc, e02_mcmc, de2_dt_mcmc, om02_mcmc,dom2_dt_mcmc, D1_mcmc,D2_mcmc,D3_mcmc, jit1_mcmc, jit2_mcmc, jit3_mcmc,  lin_mcmc, quad_mcmc = np.percentile(flat_samples[:, :], [50], axis=0)[0]
        amp_mcmc_l, logperiod_mcmc_l, tp_mcmc_l, e_mcmc_l, de_dt_mcmc_l, om_mcmc_l,dom_dt_mcmc_l, de2_dt2_mcmc_l, dom2_dt2_mcmc_l, amp2_mcmc_l, logperiod2_mcmc_l, tp2_mcmc_l, e02_mcmc_l, de2_dt_mcmc_l, om02_mcmc_l,dom2_dt_mcmc_l, D1_mcmc_l,D2_mcmc_l,D3_mcmc_l, jit1_mcmc_l, jit2_mcmc_l, jit3_mcmc_l,  lin_mcmc_l, quad_mcmc_l = np.percentile(flat_samples[:, :], [16], axis=0)[0]
        amp_mcmc_u, logperiod_mcmc_u, tp_mcmc_u, e_mcmc_u, de_dt_mcmc_u, om_mcmc_u,dom_dt_mcmc_u, de2_dt2_mcmc_u, dom2_dt2_mcmc_u, amp2_mcmc_u, logperiod2_mcmc_u, tp2_mcmc_u, e02_mcmc_u, de2_dt_mcmc_u, om02_mcmc_u,dom2_dt_mcmc_u, D1_mcmc_u,D2_mcmc_u,D3_mcmc_u, jit1_mcmc_u, jit2_mcmc_u, jit3_mcmc_u,  lin_mcmc_u, quad_mcmc_u = np.percentile(flat_samples[:, :], [84], axis=0)[0]

    period_mcmc = 10**logperiod_mcmc
    period2_mcmc = 10**logperiod2_mcmc
    
    if e1==0:
        e1 = e_mcmc
        om1 = om_mcmc
    
    D_s = [D1_mcmc,D2_mcmc,D3_mcmc]
    
    if use_mass_parametrization:
        Mp_dr, logP_dr, tp_dr, e0_dr, de_dt_dr, om0_dr,dom_dt_dr,de2_dt2_dr, dom2_dt2_dr, Mp2_dr, logP2_dr, tp2_dr, e02_dr, de2_dt_dr, om02_dr,dom2_dt_dr, D1_dr,D2_dr,D3_dr, jit1_dr, jit2_dr, jit3_dr, lin_dr, quad_dr= np.array(sample(flat_samples.tolist(), num_draws)).T
        #K_dr, K2_dr = K_from_Mp(Mp_dr, 10**logP_dr, tp_dr,  e0_dr, om0_dr,  Mp2=Mp2_dr, P2=10**logP2_dr, tp2=tp2_dr,  e02=e02_dr, om02=om02_dr, M1=1.34)
        K2_dr = K_from_Mp(Mp_dr, 10**logP_dr, tp_dr,  e0_dr, om0_dr,  Mp2=Mp2_dr, P2=10**logP2_dr, tp2=tp2_dr,  e02=e02_dr, om02=om02_dr, M1=1.34)[1]
    else:
       K_dr, logP_dr, tp_dr, e0_dr, de_dt_dr, om0_dr,dom_dt_dr,de2_dt2_dr, dom2_dt2_dr, K2_dr, logP2_dr, tp2_dr, e02_dr, de2_dt_dr, om02_dr,dom2_dt_dr, D1_dr,D2_dr,D3_dr, jit1_dr, jit2_dr, jit3_dr, lin_dr, quad_dr= np.array(sample(flat_samples.tolist(), num_draws)).T
     
    
    P_dr = 10**logP_dr
    P2_dr = 10**logP2_dr

    fig, axes = plt.subplots(2,2, figsize=(20, 8),  gridspec_kw={'height_ratios': [3, 0.6]})
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    if min_time==0:
        ti_extend = np.arange(np.max(time_np), np.max(time_np)+extend_time, 0.1)
    else:
        min_time_minbjd = min_time - np.min(df_used["bjd_merged"])
        ti_extend = np.arange(min_time_minbjd, min_time_minbjd+extend_time, 0.1)

    if np.median(K2_dr)==0.0:
        for i in range(num_draws):
            if use_mass_parametrization:
                Mp_dr_d = Mp_dr[i]
            else:
                K_dr_d = K_dr[i]
            P_dr_d = P_dr[i]
            tp_dr_d = tp_dr[i]
            e0_dr_d = e0_dr[i]
            de_dt_dr_d = de_dt_dr[i]
            om0_dr_d = om0_dr[i]
            dom_dt_dr_d = dom_dt_dr[i]
            de2_dt2_dr_d = de2_dt2_dr[i]
            dom2_dt2_dr_d = dom2_dt2_dr[i]
            D1_dr_d = D1_dr[i]
            D2_dr_d = D2_dr[i]
            D3_dr_d = D3_dr[i]
            lin_dr_d = lin_dr[i]
            quad_dr_d = quad_dr[i]

            # compute median e and median om
            med_e_d = np.median(de_dt_dr_d*time_np+e0_dr_d+de2_dt2_dr_d*time_np**2)
            med_om_d = np.median(dom_dt_dr_d*time_np+om0_dr_d+dom2_dt2_dr_d*time_np**2)

            #print(avg_e_mcmc, avg_om_mcmc)
            
            if use_mass_parametrization:
                # Compute K_dr based on Mp, P, tp, and median e and median omega
                K_dr_d = K_from_Mp(Mp_dr_d, P_dr_d, tp_dr_d,  med_e_d, med_om_d, M1=1.34)[0]

            ti_np = np.arange(np.min(time_np), np.max(time_np), 0.1)
            planet_y_preds_ti_np = one_planet_model(ti_np, np.array([K_dr_d, P_dr_d, tp_dr_d,
                                                        med_e_d, med_om_d,0, 0,0]), telescopes)
            #planet_y_preds_ti_extend = one_planet_model(ti_extend, np.array([K_dr_d, P_dr_d, tp_dr_d,
            #                                            med_e_d, med_om_d,0, 0,0]), telescopes)
            planet_y_preds_time_np = one_planet_model(time_np, np.array([K_dr_d, P_dr_d, tp_dr_d,
                                                        med_e_d, med_om_d,0, 0,0]), telescopes)
            lin_quad_y_preds_time_np = lin_quad_model(time_np, np.array([lin_dr_d, quad_dr_d]))+D1_dr_d-D1_mcmc

            planet_data_ti = {'phase': ti_np%P_dr_d-0.5*P_dr_d,
                'rvs': planet_y_preds_ti_np}#+D2_dr_d+D3_dr_d}
            df_planet_ti = pd.DataFrame(data=planet_data_ti)
            df_planet_ti_sorted = df_planet_ti.sort_values(by=['phase'])
            #avg_phase, avg_rvs = average_y_preds(df_di_sorted)

            lin_quad_data_ti = {'phase': ti_np-0.5*P_dr_d,
                   'y_preds': lin_quad_model(ti_np, np.array([lin_dr_d, quad_dr_d]))}
            df_lin_quad_data_ti = pd.DataFrame(data=lin_quad_data_ti)
            df_lin_quad_data_ti_sorted = df_lin_quad_data_ti.sort_values(by=['phase'])

            # plot hat-p-2b
            ax = axes[0][0]
            ax.plot(df_planet_ti_sorted['phase'], df_planet_ti_sorted['rvs'], color='#cebce8', alpha=0.3)

            # plot quadratic trend
            ax = axes[0][1]
            ax.plot(df_lin_quad_data_ti_sorted['phase'], df_lin_quad_data_ti_sorted['y_preds'], color='#cebce8', alpha=0.3)
        
        # plot the average model
        print(e1, om1)
        y_preds = one_planet_model(ti_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds_ext = one_planet_model(ti_extend, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds_time_np = one_planet_model(time_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds_time_np2 = lin_quad_model(time_np, np.array([lin_mcmc, quad_mcmc]))
        data_ti = {'time': ti_np,
                'phase': ti_np%period_mcmc-0.5*period_mcmc,
                'rvs': y_preds}
        df_di = pd.DataFrame(data=data_ti)
        df_di_sorted = df_di.sort_values(by=['phase'])

        data2 = {'time': ti_np,
                   'phase': ti_np-0.5*period_mcmc,
                   'y_preds': lin_quad_model(ti_np, np.array([lin_mcmc, quad_mcmc]))}
        df_ti2 = pd.DataFrame(data=data2)
        df_ti_sorted2 = df_ti2.sort_values(by=['phase'])
        
        # plot extended data predictions
        data_ext_planet = {'time':ti_extend,
                   'phase':ti_extend%period_mcmc-0.5*period_mcmc,
                   'y_preds_ext': y_preds_ext}
        df_ext_planet_sorted = pd.DataFrame(data=data_ext_planet).sort_values(by=['phase'])
        
        data_ext_quad = {'time':ti_extend,
                   'phase':ti_extend-0.5*period_mcmc,
                   'y_quad_ext': lin_quad_model(ti_extend, np.array([lin_mcmc, quad_mcmc]))}
        df_ext_quad_sorted = pd.DataFrame(data=data_ext_quad).sort_values(by=['phase'])


        # plot hat-p-2b RVs
        
        colors = ['#24ab9b', 'r', 'k']
        num_color = 0
        # plot each of the rv datasets
        for tel_name in np.unique(telescopes):
            indexes = np.where(telescopes==tel_name)
            #print(i, indexes, rv_np[indexes])
            
            # plot rvs
            ax = axes[0][0]
            ax.errorbar(time_np[indexes]%period_mcmc-0.5*period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                        yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
            # plot residuals
            ax = axes[1][0]
            ax.errorbar(time_np[indexes]%period_mcmc-0.5*period_mcmc, 
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
                label="e="+str(round(e1,3))+", $\omega$="+str(round(om1*180/np.pi,3))+" degrees")
        ax.plot(df_ext_planet_sorted['phase'], df_ext_planet_sorted['y_preds_ext'], color='#559B47', linewidth=2,
               label="Forward Propogated Model")
        ax.set_ylabel("RV (m/s)")
        ax.set_title(time_chunk+eval_method)
        ax.legend(loc="lower right")
        legend = ax.legend(loc="lower right", prop={'size': 21}, facecolor='#24ab9b')
        plt.setp(legend.get_texts(), color='k')

        # plot hat-p-2b residuals labels
        ax = axes[1][0]
        ax.set_ylabel("Residuals")
        ax.set_xlabel("Phase (days)")

        # plot quadratic trend
        ax = axes[0][1]
        ax.plot(df_ti_sorted2['phase'], df_ti_sorted2['y_preds'], color='#7829e6', linewidth=2)
        ax.plot(df_ext_quad_sorted['phase'], df_ext_quad_sorted['y_quad_ext'], color='#559B47', linewidth=2,
                label="Forward Propogated Model")
        ax.set_title(time_chunk+" Quadratic background trend")

        ax = axes[1][1]
        ax.set_xlabel("Phase (days)")
        
        return df_di_sorted, df_ti_sorted2, df_ext_planet_sorted, df_ext_quad_sorted
    else: # plot two planet case
        #print("You should maybe write the code to plot two planets hehehe")
        for i in range(num_draws):
            if use_mass_parametrization:
                Mp_dr_d = Mp_dr[i]
                Mp2_dr_d = Mp2_dr[i]
            else:
                K_dr_d = K_dr[i]
                K2_dr_d = K2_dr[i]
            
            #K_dr_d = K_dr[i]
            P_dr_d = P_dr[i]
            tp_dr_d = tp_dr[i]
            e0_dr_d = e0_dr[i]
            de_dt_dr_d = de_dt_dr[i]
            om0_dr_d = om0_dr[i]
            dom_dt_dr_d = dom_dt_dr[i]
            de2_dt2_dr_d = de2_dt2_dr[i]
            dom2_dt2_dr_d = dom2_dt2_dr[i]
            
            P2_dr_d = P2_dr[i]
            tp2_dr_d = tp2_dr[i]
            e02_dr_d = e02_dr[i]
            de2_dt_dr_d = de2_dt_dr[i]
            om02_dr_d = om02_dr[i]
            dom2_dt_dr_d = dom2_dt_dr[i]
            
            
            D1_dr_d = D1_dr[i]
            D2_dr_d = D2_dr[i]
            D3_dr_d = D3_dr[i]
            lin_dr_d = lin_dr[i]
            quad_dr_d = quad_dr[i]

            # compute median e and median om
            med_e_d = np.median(de_dt_dr_d*time_np + e0_dr_d + de2_dt2_dr_d*time_np**2)
            med_om_d = np.median(dom_dt_dr_d*time_np + om0_dr_d + dom2_dt2_dr_d*time_np**2)
            # compute median e and median om
            med_e2_d = np.median(de2_dt_dr_d*time_np+e02_dr_d)
            med_om2_d = np.median(dom2_dt_dr_d*time_np+om02_dr_d)
            
            if use_mass_parametrization:
                # Compute K_dr based on Mp, P, tp, and median e and median omega
                K_dr_d = K_from_Mp(Mp_dr_d, P_dr_d, tp_dr_d,  med_e_d, med_om_d, M1=1.34)[0]
                K2_dr_d = K_from_Mp(Mp_dr_d, P_dr_d, tp_dr_d,  med_e_d, med_om_d, Mp2=Mp2_dr_d, P2=P2_dr_d, tp2=tp2_dr_d,  e02=med_e2_d, om02=med_om2_d, M1=1.34)[1]

            #print(avg_e_mcmc, avg_om_mcmc)


            ti_np = np.arange(np.min(time_np), np.max(time_np), 0.1)
            planet_y_preds_ti_np = one_planet_model(ti_np, np.array([K_dr_d, P_dr_d, tp_dr_d,
                                                        med_e_d, med_om_d,0, 0,0]), telescopes)
            planet_y_preds_time_np = one_planet_model(time_np, np.array([K_dr_d, P_dr_d, tp_dr_d,
                                                        med_e_d, med_om_d,0, 0,0]), telescopes)
            planet2_y_preds_ti_np = one_planet_model(ti_np, np.array([K2_dr_d, P2_dr_d, tp2_dr_d,
                                                        med_e2_d, med_om2_d,0, 0,0]), telescopes)
            planet2_y_preds_time_np = one_planet_model(time_np, np.array([K2_dr_d, P2_dr_d, tp2_dr_d,
                                                        med_e2_d, med_om2_d,0, 0,0]), telescopes)
            #lin_quad_y_preds_time_np = lin_quad_model(time_np, np.array([lin_dr_d, quad_dr_d]))+D1_dr_d-D1_mcmc

            planet_data_ti = {'phase': ti_np%P_dr_d-0.5*P_dr_d,
                'rvs': planet_y_preds_ti_np}#+D2_dr_d+D3_dr_d}
            df_planet_ti = pd.DataFrame(data=planet_data_ti)
            df_planet_ti_sorted = df_planet_ti.sort_values(by=['phase']).reset_index()
            
            planet2_data_ti = {'phase': ti_np%P2_dr_d-0.5*P_dr_d,
                'planet2_rvs': planet2_y_preds_ti_np}#+D2_dr_d+D3_dr_d}
            df_planet2_ti = pd.DataFrame(data=planet2_data_ti)
            df_planet2_ti_sorted = df_planet2_ti.sort_values(by=['phase']).reset_index()
            #avg_phase, avg_rvs = average_y_preds(df_di_sorted)

            #lin_quad_data_ti = {'phase': ti_np-0.5*P_dr_d,
            #       'y_preds': lin_quad_model(ti_np, np.array([lin_dr_d, quad_dr_d]))}
            #df_lin_quad_data_ti = pd.DataFrame(data=lin_quad_data_ti)
            #df_lin_quad_data_ti_sorted = df_lin_quad_data_ti.sort_values(by=['phase'])

            # plot hat-p-2b
            ax = axes[0][0]
            ax.plot(df_planet_ti_sorted['phase'], df_planet_ti_sorted['rvs'], color='#cebce8', alpha=0.3)

            # plot hat-p-2c
            ax = axes[0][1]
            ax.plot(df_planet2_ti_sorted['phase'], df_planet2_ti_sorted['planet2_rvs'], color='#cebce8', alpha=0.3)
        
        # plot the average model
        print(e1, om1)
        y_preds = one_planet_model(ti_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds_ext = one_planet_model(ti_extend, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds2 = one_planet_model(ti_np, np.array([amp2_mcmc, period2_mcmc, tp2_mcmc,
                                                        e02_mcmc, om02_mcmc,0, 0, 0]), telescopes)
        y_preds2_ext = one_planet_model(ti_extend, np.array([amp2_mcmc, period2_mcmc, tp2_mcmc,
                                                        e02_mcmc, om02_mcmc,0, 0, 0]), telescopes)
        y_preds_time_np = one_planet_model(time_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds_time_np2 = one_planet_model(time_np, np.array([amp2_mcmc, period2_mcmc, tp2_mcmc,
                                                        e02_mcmc, om02_mcmc,0, 0, 0]), telescopes)
        
        
        data_ti = {'time': ti_np,
                'phase': ti_np%period_mcmc-0.5*period_mcmc,
                'rvs': y_preds}
        df_di = pd.DataFrame(data=data_ti)
        df_di_sorted = df_di.sort_values(by=['phase']).reset_index()

        data2 = {'time': ti_np,
                'phase': ti_np%period2_mcmc-0.5*period_mcmc,
                'y_preds': y_preds2}
        df_ti2 = pd.DataFrame(data=data2)
        df_ti_sorted2 = df_ti2.sort_values(by=['phase']).reset_index()
        
        # plot extended data predictions
        data_ext_planet = {'time':ti_extend,
                   'phase':ti_extend%period_mcmc-0.5*period_mcmc,
                   'y_preds_ext': y_preds_ext}
        df_ext_planet_sorted = pd.DataFrame(data=data_ext_planet).sort_values(by=['phase'])
        
        data_ext_planet2 = {'time':ti_extend,
                   'phase':ti_extend%period2_mcmc-0.5*period_mcmc,
                   'y_preds_planet2_ext': y_preds2_ext}
        data_ext_planet2_sorted = pd.DataFrame(data=data_ext_planet2).sort_values(by=['phase'])


        # plot hat-p-2b RVs
        
        colors = ['#24ab9b', 'r', 'k']
        num_color = 0
        # plot each of the rv datasets
        for tel_name in np.unique(telescopes):
            indexes = np.where(telescopes==tel_name)
            #print(i, indexes, rv_np[indexes])
            
            # plot rvs
            ax = axes[0][0]
            ax.errorbar(time_np[indexes]%period_mcmc-0.5*period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                        yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
            # plot residuals
            ax = axes[1][0]
            ax.errorbar(time_np[indexes]%period_mcmc-0.5*period_mcmc, 
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
                label="e="+str(round(e1,3))+", $\omega$="+str(round(om1*180/np.pi,3))+" degrees")
        ax.plot(df_ext_planet_sorted['phase'], df_ext_planet_sorted['y_preds_ext'], color='#559B47', linewidth=2,
                label="Forward Propogated Model")
        ax.set_ylabel("RV (m/s)")
        ax.set_title(time_chunk+eval_method)
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
        ax.plot(data_ext_planet2_sorted['phase'], data_ext_planet2_sorted['y_preds_planet2_ext'], color='#559B47', linewidth=2,
               label="Forward Propogated Model")
        ax.set_title(time_chunk+" 2nd Planet")

        ax = axes[1][1]
        ax.set_xlabel("Phase (days)")
        
        return df_di_sorted, df_ti_sorted2, df_ext_planet_sorted, data_ext_planet2_sorted
    
def fair_draws_plot_chunk(time_np, rv_np, s_rv_np, out, num_draws, nburnin, telescopes, time_chunk, eval_method,
                    amp_mcmc=0, period_mcmc=0, tp_mcmc=0,D_mcmc=0, e1=0, om1=0, lin_mcmc=0, quad_mcmc=0,
                    D1_mcmc=0,D2_mcmc=0,D3_mcmc=0, phase_plot=0.5, legendfontsize=21, D_s =[], use_mass_parametrization=False):
    flat_samples = out.get_chains(nthin =5, nburnin = nburnin, flat=True)
    
    '''fix this code to include use_mass_parametrization = False posibility'''
    
    if use_mass_parametrization:
        Mp_mcmc, logperiod_mcmc, tp_mcmc, e_mcmc, de_dt_mcmc, om_mcmc,dom_dt_mcmc, de2_dt2_mcmc, dom2_dt2_mcmc, Mp2_mcmc, logperiod2_mcmc, tp2_mcmc, e02_mcmc, de2_dt_mcmc, om02_mcmc,dom2_dt_mcmc, D1_mcmc,D2_mcmc,D3_mcmc, jit1_mcmc, jit2_mcmc, jit3_mcmc,  lin_mcmc, quad_mcmc = np.percentile(flat_samples[:, :], [50], axis=0)[0]
        Mp_mcmc_l, logperiod_mcmc_l, tp_mcmc_l, e_mcmc_l, de_dt_mcmc_l, om_mcmc_l,dom_dt_mcmc_l, de2_dt2_mcmc_l, dom2_dt2_mcmc_l, Mp2_mcmc_l, logperiod2_mcmc_l, tp2_mcmc_l, e02_mcmc_l, de2_dt_mcmc_l, om02_mcmc_l,dom2_dt_mcmc_l, D1_mcmc_l,D2_mcmc_l,D3_mcmc_l, jit1_mcmc_l, jit2_mcmc_l, jit3_mcmc_l,  lin_mcmc_l, quad_mcmc_l = np.percentile(flat_samples[:, :], [16], axis=0)[0]
        Mp_mcmc_u, logperiod_mcmc_u, tp_mcmc_u, e_mcmc_u, de_dt_mcmc_u, om_mcmc_u,dom_dt_mcmc_u, de2_dt2_mcmc_u, dom2_dt2_mcmc_u, Mp2_mcmc_u, logperiod2_mcmc_u, tp2_mcmc_u, e02_mcmc_u, de2_dt_mcmc_u, om02_mcmc_u,dom2_dt_mcmc_u, D1_mcmc_u,D2_mcmc_u,D3_mcmc_u, jit1_mcmc_u, jit2_mcmc_u, jit3_mcmc_u,  lin_mcmc_u, quad_mcmc_u = np.percentile(flat_samples[:, :], [84], axis=0)[0]
        
        amp_mcmc, amp2_mcmc = K_from_Mp(Mp_mcmc, 10**logperiod_mcmc, tp_mcmc,  e_mcmc, om_mcmc,  Mp2=Mp2_mcmc, P2=10**logperiod2_mcmc, tp2=tp2_mcmc,  e02=e02_mcmc, om02=om02_mcmc, M1=1.34)
        amp_mcmc_l, amp2_mcmc_l = K_from_Mp(Mp_mcmc_l, 10**logperiod_mcmc_l, tp_mcmc_l,  e_mcmc_l, om_mcmc_l,  Mp2=Mp2_mcmc_l, P2=10**logperiod2_mcmc_l, tp2=tp2_mcmc_l,  e02=e02_mcmc_l, om02=om02_mcmc_l, M1=1.34)
        amp_mcmc_u, amp2_mcmc_u = K_from_Mp(Mp_mcmc_u, 10**logperiod_mcmc_u, tp_mcmc_u,  e_mcmc_u, om_mcmc_u,  Mp2=Mp2_mcmc_u, P2=10**logperiod2_mcmc_u, tp2=tp2_mcmc_u,  e02=e02_mcmc_u, om02=om02_mcmc_u, M1=1.34)
    else:
        amp_mcmc, logperiod_mcmc, tp_mcmc, e_mcmc, de_dt_mcmc, om_mcmc,dom_dt_mcmc, de2_dt2_mcmc, dom2_dt2_mcmc, amp2_mcmc, logperiod2_mcmc, tp2_mcmc, e02_mcmc, de2_dt_mcmc, om02_mcmc,dom2_dt_mcmc, D1_mcmc,D2_mcmc,D3_mcmc, jit1_mcmc, jit2_mcmc, jit3_mcmc,  lin_mcmc, quad_mcmc = np.percentile(flat_samples[:, :], [50], axis=0)[0]
        amp_mcmc_l, logperiod_mcmc_l, tp_mcmc_l, e_mcmc_l, de_dt_mcmc_l, om_mcmc_l,dom_dt_mcmc_l, de2_dt2_mcmc_l, dom2_dt2_mcmc_l, amp2_mcmc_l, logperiod2_mcmc_l, tp2_mcmc_l, e02_mcmc_l, de2_dt_mcmc_l, om02_mcmc_l,dom2_dt_mcmc_l, D1_mcmc_l,D2_mcmc_l,D3_mcmc_l, jit1_mcmc_l, jit2_mcmc_l, jit3_mcmc_l,  lin_mcmc_l, quad_mcmc_l = np.percentile(flat_samples[:, :], [16], axis=0)[0]
        amp_mcmc_u, logperiod_mcmc_u, tp_mcmc_u, e_mcmc_u, de_dt_mcmc_u, om_mcmc_u,dom_dt_mcmc_u, de2_dt2_mcmc_u, dom2_dt2_mcmc_u, amp2_mcmc_u, logperiod2_mcmc_u, tp2_mcmc_u, e02_mcmc_u, de2_dt_mcmc_u, om02_mcmc_u,dom2_dt_mcmc_u, D1_mcmc_u,D2_mcmc_u,D3_mcmc_u, jit1_mcmc_u, jit2_mcmc_u, jit3_mcmc_u,  lin_mcmc_u, quad_mcmc_u = np.percentile(flat_samples[:, :], [84], axis=0)[0]

    period_mcmc = 10**logperiod_mcmc
    period2_mcmc = 10**logperiod2_mcmc
    
    if e1==0:
        e1 = e_mcmc
        om1 = om_mcmc
    if D_s ==[]:
        D_s = [D1_mcmc,D2_mcmc,D3_mcmc]

    if use_mass_parametrization:
        Mp_dr, logP_dr, tp_dr, e0_dr, de_dt_dr, om0_dr,dom_dt_dr,de2_dt2_dr, dom2_dt2_dr, Mp2_dr, logP2_dr, tp2_dr, e02_dr, de2_dt_dr, om02_dr,dom2_dt_dr, D1_dr,D2_dr,D3_dr, jit1_dr, jit2_dr, jit3_dr, lin_dr, quad_dr= np.array(sample(flat_samples.tolist(), num_draws)).T
        #K_dr, K2_dr = K_from_Mp(Mp_dr, 10**logP_dr, tp_dr,  e0_dr, om0_dr,  Mp2=Mp2_dr, P2=10**logP2_dr, tp2=tp2_dr,  e02=e02_dr, om02=om02_dr, M1=1.34)
        K2_dr = K_from_Mp(Mp_dr, 10**logP_dr, tp_dr,  e0_dr, om0_dr,  Mp2=Mp2_dr, P2=10**logP2_dr, tp2=tp2_dr,  e02=e02_dr, om02=om02_dr, M1=1.34)[1]
    else:
       K_dr, logP_dr, tp_dr, e0_dr, de_dt_dr, om0_dr,dom_dt_dr,de2_dt2_dr, dom2_dt2_dr, K2_dr, logP2_dr, tp2_dr, e02_dr, de2_dt_dr, om02_dr,dom2_dt_dr, D1_dr,D2_dr,D3_dr, jit1_dr, jit2_dr, jit3_dr, lin_dr, quad_dr= np.array(sample(flat_samples.tolist(), num_draws)).T
     
    K_dr, logP_dr, tp_dr, e0_dr, de_dt_dr, om0_dr,dom_dt_dr,de2_dt2_dr, dom2_dt2_dr, K2_dr, logP2_dr, tp2_dr, e02_dr, de2_dt_dr, om02_dr,dom2_dt_dr, D1_dr,D2_dr,D3_dr, jit1_dr, jit2_dr, jit3_dr, lin_dr, quad_dr= np.array(sample(flat_samples.tolist(), num_draws)).T
    P_dr = 10**logP_dr
    P2_dr = 10**logP2_dr

    fig, axes = plt.subplots(2,2, figsize=(20, 8),  gridspec_kw={'height_ratios': [3, 0.6]})
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    if np.median(K2_dr)==0.0:
        for i in range(num_draws):
            if use_mass_parametrization:
                Mp_dr_d = Mp_dr[i]
            else:
                K_dr_d = K_dr[i]
            P_dr_d = P_dr[i]
            tp_dr_d = tp_dr[i]
            e0_dr_d = e0_dr[i]
            de_dt_dr_d = de_dt_dr[i]
            om0_dr_d = om0_dr[i]
            dom_dt_dr_d = dom_dt_dr[i]
            de2_dt2_dr_d = de2_dt2_dr[i]
            dom2_dt2_dr_d = dom2_dt2_dr[i]
            D1_dr_d = D1_dr[i]
            D2_dr_d = D2_dr[i]
            D3_dr_d = D3_dr[i]
            lin_dr_d = lin_dr[i]
            quad_dr_d = quad_dr[i]

            # compute median e and median om
            med_e_d = np.median(de_dt_dr_d*time_np+e0_dr_d+de2_dt2_dr_d*time_np**2)
            med_om_d = np.median(dom_dt_dr_d*time_np+om0_dr_d+dom2_dt2_dr_d*time_np**2)

            #print(avg_e_mcmc, avg_om_mcmc)
            if use_mass_parametrization:
                # Compute K_dr based on Mp, P, tp, and median e and median omega
                K_dr_d = K_from_Mp(Mp_dr_d, P_dr_d, tp_dr_d,  med_e_d, med_om_d, M1=1.34)[0]

            ti_np = np.arange(np.min(time_np), np.max(time_np), 0.1)
            planet_y_preds_ti_np = one_planet_model(ti_np, np.array([K_dr_d, P_dr_d, tp_dr_d,
                                                        med_e_d, med_om_d,0, 0,0]), telescopes)
            planet_y_preds_time_np = one_planet_model(time_np, np.array([K_dr_d, P_dr_d, tp_dr_d,
                                                        med_e_d, med_om_d,0, 0,0]), telescopes)
            lin_quad_y_preds_time_np = lin_quad_model(time_np, np.array([lin_dr_d, quad_dr_d]))+D1_dr_d-D1_mcmc

            planet_data_ti = {'phase': (ti_np-phase_plot*P_dr_d)%P_dr_d,
                'rvs': planet_y_preds_ti_np}#+D2_dr_d+D3_dr_d}
            df_planet_ti = pd.DataFrame(data=planet_data_ti)
            df_planet_ti_sorted = df_planet_ti.sort_values(by=['phase'])
            #avg_phase, avg_rvs = average_y_preds(df_di_sorted)

            lin_quad_data_ti = {'phase': ti_np,
                   'y_preds': lin_quad_model(ti_np, np.array([lin_dr_d, quad_dr_d]))+D1_dr_d-D1_mcmc}
            df_lin_quad_data_ti = pd.DataFrame(data=lin_quad_data_ti)
            df_lin_quad_data_ti_sorted = df_lin_quad_data_ti.sort_values(by=['phase'])

            # plot hat-p-2b
            ax = axes[0][0]
            ax.plot(df_planet_ti_sorted['phase'], df_planet_ti_sorted['rvs'], color='#cebce8', alpha=0.3)

            # plot quadratic trend
            ax = axes[0][1]
            ax.plot(df_lin_quad_data_ti_sorted['phase'], df_lin_quad_data_ti_sorted['y_preds'], color='#cebce8', alpha=0.3)
        
        # plot the average model
        print(e1, om1)
        y_preds = one_planet_model(ti_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds_time_np = one_planet_model(time_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds_time_np2 = lin_quad_model(time_np, np.array([lin_mcmc, quad_mcmc]))
        data_ti = {'phase': (ti_np-phase_plot*period_mcmc)%period_mcmc,
                'rvs': y_preds}
        df_di = pd.DataFrame(data=data_ti)
        df_di_sorted = df_di.sort_values(by=['phase'])

        data2 = {'phase': ti_np-phase_plot*period_mcmc,
                   'y_preds': lin_quad_model(ti_np, np.array([lin_mcmc, quad_mcmc]))}
        df_ti2 = pd.DataFrame(data=data2)
        df_ti_sorted2 = df_ti2.sort_values(by=['phase'])


        # plot hat-p-2b RVs
        
        colors = ['#24ab9b', 'r', 'k']
        num_color = 0
        # plot each of the rv datasets
        for tel_name in np.unique(telescopes):
            indexes = np.where(telescopes==tel_name)
            #print(i, indexes, rv_np[indexes])
            ymax = np.max(np.abs(rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color]))

            # plot rvs
            ax = axes[0][0]
            ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                        yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
            # plot residuals
            ax = axes[1][0]
            ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, 
                        rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],yerr=s_rv_np[indexes],
                        fmt = 'o', color = colors[num_color],label=tel_name)
            ax.set_ylim(-ymax-0.2*ymax, ymax+0.2*ymax)
            # plot quadratic trend data
            ax = axes[0][1]
            ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-D_s[num_color],
                        yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
            
            ax = axes[1][1]
            ax.errorbar(time_np[indexes], rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_color],
                        yerr=s_rv_np[indexes], fmt = 'o', color = colors[num_color],label=tel_name)
            ax.set_ylim(-ymax-0.2*ymax, ymax+0.2*ymax)
            num_color += 1
        
        ax = axes[0][0]
        ax.plot(df_di_sorted['phase'], df_di_sorted['rvs'], color='#7829e6', linewidth=2,
                label="e="+str(round(e1,3))+", $\omega$="+str(round(om1*180/np.pi,3))+" $^\circ$")
        ax.set_ylabel("RV (m/s)")
        ax.set_title(time_chunk+eval_method)
        ax.legend(loc="lower right")
        legend = ax.legend(loc="lower right", prop={'size': legendfontsize}, facecolor='#24ab9b')
        plt.setp(legend.get_texts(), color='k')

        # plot hat-p-2b residuals labels
        ax = axes[1][0]
        ax.set_ylabel("Residuals")
        ax.set_xlabel("Phase (days)")

        # plot quadratic trend
        ax = axes[0][1]
        ax.plot(df_ti_sorted2['phase'], df_ti_sorted2['y_preds'], color='#7829e6', linewidth=2)
        ax.set_title(time_chunk+" Quadratic background trend")

        ax = axes[1][1]
        ax.set_xlabel("Phase (days)")
        return D_s, period_mcmc,df_di_sorted, time_np, rv_np, y_preds_time_np, y_preds_time_np2, s_rv_np, telescopes
    else: # plot two planet case
        #print("You should maybe write the code to plot two planets hehehe")
        for i in range(num_draws):
            if use_mass_parametrization:
                Mp_dr_d = Mp_dr[i]
                Mp2_dr_d = Mp2_dr[i]
            else:
                K_dr_d = K_dr[i]
                K2_dr_d = K2_dr[i]
            
            P_dr_d = P_dr[i]
            tp_dr_d = tp_dr[i]
            e0_dr_d = e0_dr[i]
            de_dt_dr_d = de_dt_dr[i]
            om0_dr_d = om0_dr[i]
            dom_dt_dr_d = dom_dt_dr[i]
            de2_dt2_dr_d = de2_dt2_dr[i]
            dom2_dt2_dr_d = dom2_dt2_dr[i]
            
            P2_dr_d = P2_dr[i]
            tp2_dr_d = tp2_dr[i]
            e02_dr_d = e02_dr[i]
            de2_dt_dr_d = de2_dt_dr[i]
            om02_dr_d = om02_dr[i]
            dom2_dt_dr_d = dom2_dt_dr[i]
            
            
            D1_dr_d = D1_dr[i]
            D2_dr_d = D2_dr[i]
            D3_dr_d = D3_dr[i]
            lin_dr_d = lin_dr[i]
            quad_dr_d = quad_dr[i]

            # compute median e and median om
            med_e_d = np.median(de_dt_dr_d*time_np + e0_dr_d + de2_dt2_dr_d*time_np**2)
            med_om_d = np.median(dom_dt_dr_d*time_np + om0_dr_d + dom2_dt2_dr_d*time_np**2)
            # compute median e and median om
            med_e2_d = np.median(de2_dt_dr_d*time_np+e02_dr_d)
            med_om2_d = np.median(dom2_dt_dr_d*time_np+om02_dr_d)
            
            if use_mass_parametrization:
                # Compute K_dr based on Mp, P, tp, and median e and median omega
                K_dr_d = K_from_Mp(Mp_dr_d, P_dr_d, tp_dr_d,  med_e_d, med_om_d, M1=1.34)[0]
                K2_dr_d = K_from_Mp(Mp_dr_d, P_dr_d, tp_dr_d,  med_e_d, med_om_d, Mp2=Mp2_dr_d, P2=P2_dr_d, tp2=tp2_dr_d,  e02=med_e2_d, om02=med_om2_d, M1=1.34)[1]


            #print(avg_e_mcmc, avg_om_mcmc)


            ti_np = np.arange(np.min(time_np), np.max(time_np), 0.1)
            planet_y_preds_ti_np = one_planet_model(ti_np, np.array([K_dr_d, P_dr_d, tp_dr_d,
                                                        med_e_d, med_om_d,0, 0,0]), telescopes)
            planet_y_preds_time_np = one_planet_model(time_np, np.array([K_dr_d, P_dr_d, tp_dr_d,
                                                        med_e_d, med_om_d,0, 0,0]), telescopes)
            planet2_y_preds_ti_np = one_planet_model(ti_np, np.array([K2_dr_d, P2_dr_d, tp2_dr_d,
                                                        med_e2_d, med_om2_d,0, 0,0]), telescopes)
            planet2_y_preds_time_np = one_planet_model(time_np, np.array([K2_dr_d, P2_dr_d, tp2_dr_d,
                                                        med_e2_d, med_om2_d,0, 0,0]), telescopes)
            #lin_quad_y_preds_time_np = lin_quad_model(time_np, np.array([lin_dr_d, quad_dr_d]))+D1_dr_d-D1_mcmc

            planet_data_ti = {'phase': (ti_np-phase_plot*P_dr_d)%P_dr_d,
                'rvs': planet_y_preds_ti_np}#+D2_dr_d+D3_dr_d}
            df_planet_ti = pd.DataFrame(data=planet_data_ti)
            df_planet_ti_sorted = df_planet_ti.sort_values(by=['phase'])
            
            planet2_data_ti = {'phase': ti_np,
                'planet2_rvs': planet2_y_preds_ti_np}#+D2_dr_d+D3_dr_d}
            df_planet2_ti = pd.DataFrame(data=planet2_data_ti)
            df_planet2_ti_sorted = df_planet2_ti.sort_values(by=['phase'])
            #avg_phase, avg_rvs = average_y_preds(df_di_sorted)

            #lin_quad_data_ti = {'phase': ti_np-phase_plot*P_dr_d,
            #       'y_preds': lin_quad_model(ti_np, np.array([lin_dr_d, quad_dr_d]))}
            #df_lin_quad_data_ti = pd.DataFrame(data=lin_quad_data_ti)
            #df_lin_quad_data_ti_sorted = df_lin_quad_data_ti.sort_values(by=['phase'])

            # plot hat-p-2b
            ax = axes[0][0]
            ax.plot(df_planet_ti_sorted['phase'], df_planet_ti_sorted['rvs'], color='#cebce8', alpha=0.3)

            # plot hat-p-2c
            ax = axes[0][1]
            ax.plot(df_planet2_ti_sorted['phase'], df_planet2_ti_sorted['planet2_rvs'], color='#cebce8', alpha=0.3)
        
        # plot the average model
        print(e1, om1)
        y_preds = one_planet_model(ti_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds2 = one_planet_model(ti_np, np.array([amp2_mcmc, period2_mcmc, tp2_mcmc,
                                                        e02_mcmc, om02_mcmc,0, 0, 0]), telescopes)
        y_preds_time_np = one_planet_model(time_np, np.array([amp_mcmc, period_mcmc, tp_mcmc,
                                                        e1, om1,0, 0, 0]), telescopes)
        y_preds_time_np2 = one_planet_model(time_np, np.array([amp2_mcmc, period2_mcmc, tp2_mcmc,
                                                        e02_mcmc, om02_mcmc,0, 0, 0]), telescopes)
        
        
        data_ti = {'phase': (ti_np-phase_plot*period_mcmc)%period_mcmc,
                'rvs': y_preds}
        df_di = pd.DataFrame(data=data_ti)
        df_di_sorted = df_di.sort_values(by=['phase'])

        data2 = {'phase': ti_np,
                   'y_preds': y_preds2}
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
                label="e="+str(round(e1,3))+", $\omega$="+str(round(om1*180/np.pi,3))+" $^\circ$")
        ax.set_ylabel("RV (m/s)")
        ax.set_title(time_chunk+eval_method)
        ax.legend(loc="lower right")
        legend = ax.legend(loc="lower right", prop={'size': legendfontsize}, facecolor='#24ab9b')
        plt.setp(legend.get_texts(), color='k')

        # plot hat-p-2b residuals labels
        ax = axes[1][0]
        ax.set_ylabel("Residuals")
        ax.set_xlabel("Phase (days)")

        # plot hat-p-2c?
        ax = axes[0][1]
        ax.plot(df_ti_sorted2['phase'], df_ti_sorted2['y_preds'], color='#7829e6', linewidth=2)
        ax.set_title(time_chunk+" 2nd Planet")

        ax = axes[1][1]
        ax.set_xlabel("Phase (days)")
        
        return D_s, period_mcmc,df_di_sorted, time_np, rv_np, y_preds_time_np, y_preds_time_np2, s_rv_np, telescopes

    
def RV_plotter_without_offset_corr(telescopes, D_s, axes, colors, rv_np, s_rv_np, time_np, period_mcmc,
                                  y_preds_time_np, y_preds_time_np2, phase_plot=0.15):
    # plot each of the rv datasets
    num_tel = 0
    for tel_name in np.unique(telescopes):
        indexes = np.where(telescopes==tel_name)
        #print(i, indexes, rv_np[indexes])

        # plot rvs
        ax = axes[0]
        ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, rv_np[indexes]-y_preds_time_np2[indexes]-D_s[num_tel],
                    yerr=s_rv_np[indexes], fmt = 'o', color = colors)
        # plot residuals
        ax = axes[1]
        ax.errorbar((time_np[indexes]-phase_plot*period_mcmc)%period_mcmc, 
                    rv_np[indexes]-y_preds_time_np[indexes]-y_preds_time_np2[indexes]-D_s[num_tel],yerr=s_rv_np[indexes],
                    fmt = 'o', color = colors)
        
        num_tel +=0