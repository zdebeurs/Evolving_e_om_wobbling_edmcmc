#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 16:05:44 2022

@author: zdebeurs
"""

import numpy as np


def K_from_Mp(Mp, e, M1, P, Mp2=0, e2=0, P2=0):
    ''' Takes semi-amplitude (K) in ms^-1, eccentricity (e), mass of star (M1) in solar masses, Period (P) in years
    and return Minimum mass (Mp_sin_i) in Jupiter masses
    '''
    # Assume edge-on orbit
    i = 90
    
    if Mp2 ==0:
        return Mp * (28.4329/(np.sqrt(1-e**2))) * (np.sin(i)) * (M1)**(-1/2) * (P/365.25)**(-1/3)
    else: 
        return Mp * (28.4329/(np.sqrt(1-e**2))) * (np.sin(i)) * (M1)**(-1/2) * (P/365.25)**(-1/3), Mp2 * (28.4329/(np.sqrt(1-e2**2))) * (np.sin(i)) * (M1)**(-1/2) * P2**(-1/3)

def Mp_from_K(K, e, P, M1, K2=0, e2=0, P2=0):
    ''' Takes semi-amplitude (K) in ms^-1, eccentricity (e), mass of star (M1) in solar masses, Period (P) in years
    and return Minimum mass (Mp_sin_i) in Jupiter masses
    '''
    
    # asumme edge-on orbit. Thus the mass is just the minimum mass Mp*sin(i)
    i = 90
    if K2==0.0:
        return K*((np.sqrt(1-e**2))/28.4329) * (1/np.sin(i)) * (M1)**(1/2) * (P/365.25)**(1/3)
    else:
        return K*((np.sqrt(1-e**2))/28.4329) * (1/np.sin(i)) * (M1)**(1/2) * (P/365.25)**(1/3),  K2*((np.sqrt(1-e2**2))/28.4329) * (1/np.sin(i)) * (M1)**(1/2) * (P2/365.25)**(1/3)
