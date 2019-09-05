# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:10:41 2019

@author: situy
"""
import sys
sys.path.append(r"C:\Users\situy\Dropbox\Python_scripts")
import atmos_funcs_1 as a
# <codecell>
import numpy as np
import matplotlib.pyplot as plt

# <codecell> try integration
#def f1(z,h):
#    return np.exp(-1*z/h)
#f2 = lambda x : np.exp(-x/7.4)
#int1 = si.quad(f1,0,1000,args=7.4)
#int2 = si.quad(f2,0,1000)
# <codecell> try plotting
#x1 = np.linspace(0.1, 1000, 3000)
#y1 = f2(x1)
#fig = plt.figure()
#ax  =fig.gca()
#ax.plot(y1,x1)
# <codecell> exercise 1
#p0 = calc_p0(298)
#p1 = 0.9*p0
#c_water = p1/8.314/298
#ans1 = p1*100/8.314/298*6.022e23/1e6
#ans2 = p1/1013.25
#
#pn2o = 1013.25*100*311e-9
#pn2o*44/8.314/298
# 1013.25e2 * 44 /8.314/298 *0.311
#五十立方米的水，湿度从99%到50%，排出的水约有559g
#(7.556-3.816)*1e17/6.022e23*18*390*1e6

#(1e12/6.022e23)/(1013.25e2/8.314/298)

#4
#8.314*298*0.25/1013.25e2/62
#8.314*298*0.5/1013.25e2/62

#5
#500*3.7879*1000*0.85 / 114 *8*44/1000

# <codecell> exercise 3
#3.5
z = a.gas_molecule_speed(298, (67*17/(17+67))/a.C.avogadro/1000)
z*a.C.pi*(2.1e-8+2e-8)**2
#3.6
v1 = a.gas_molecule_speed(298, 108/a.C.avogadro/1000 )
ap = a.b_surface_area(0.2)
v1*100/4*5*1e-10

# 3.10
def tau2():
    part1 = 1-np.exp(-1/7.4)
    
# <codecell> exercise 4
np.log(1/7.4e5/1e-17/0.21/2.55e19)*-7.4
np.log(1/7.4e5/1e-23/0.21/2.55e19)*-7.4

m=1
i = np.arange(20,60,0.05)
r = []
for z in i:
    ex = np.exp(-z/7.4-m*1e-23*0.21*2.55e19*np.exp(-z/7.4))
    I = blackbody_radiation(6000, 200e-9)
    rate = 1e-23*1*I*ex*np.pi
    r.append(rate)
plt.plot(i,r)

# <codecell>

x = np.arange(200,500,1)
y = np.exp(70/x)
plt.plot(x,y)




