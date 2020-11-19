# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:39:13 2019

@author: situy
"""

# <codecell> import
import numpy as np
import matplotlib.pyplot as plt

# <codecell> constants
# 常数都放在C类中
class C:
    R =8.314                  #J/(K*mol)
    atm =1013.25e2            #Pa
    T = 298
    pi = 3.141592653589793
    avogadro = 6.022e23       #/mole
    boltzmann = 1.381e-23     #J/K
    planck = 6.626e-34        #Js
    c = 2.9979e8              #m/s  lightspeed
    
# <codecell> hw1
# 1.3.1
#PV = nRT, n = PV/RT
n = (3500*1e-6)/(C.R*220)
nMolec = n*C.avogadro
cMix = 5e12/nMolec
# 1.3.2
nsurfM= 100000*1e-6/(C.R*300)*C.avogadro
cMix2= 5e12/nsurfM


# <codecell> 