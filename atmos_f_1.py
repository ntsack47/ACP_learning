# -*- coding: utf-8 -*-

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
    
#n(air,0)=2.55e19    #molecules/cm3
#1 DU = 2.69e16 #molecules/cm2
# <codecell> functions
# calculate p0 at ground level>
def atmos_water_p0(t1,p_atm=1013.25):
    """
    temperature is in K
    vapor pressure output is in hPa (mbar)
    default p_atm is ground level
    """
    a = 1 - 373.15/t1
    p0 = p_atm*np.exp(a*13.3185 - a*a*1.97 - a**3*0.6445 - a**4*0.1299)
    return p0

def react_termolecule(M,T,k0,n,kinf,m,full=True):
    k = 0
    if True:
        part1 = k0*(T/300)**-n*M
        part2 = 1 + (part1/(kinf*(T/300)**-m))
        part3 = (1 + np.log10(part1/(kinf*(T/300)**-m))**2)**-1
        k = (part1/part2)*(0.6**part3)
    else:
        print('No preset function')
    return k

def gas_molecular_concentration(p,T,N=True):
    if N:
        return C.avogadro*p/(C.R*T)
    else:
        return p/(C.R*T)

def atmos_altitude_pressure(z,H=7.4):
    """altitude should be in km
        H by default is 7.4 km
    """
    return C.atm*np.e**(-z/H)

def gas_molecule_speed(T,ma):
    return np.sqrt(8*C.boltzmann*T/(C.pi*ma))

def b_surface_area(radi):
    return(4*C.pi*radi**2)
    
def blackbody_radiation(temp,wavelen):
    part1 = 2*C.pi*(C.c**2)*C.planck*(wavelen**-5)
    part2 = np.e**(C.c*C.planck/(C.boltzmann*wavelen*temp))-1
    return part1/part2

def blackbody_total_energy(temp):
    return 5.671e-8*temp**4

def cal_to_J(cal, order = True):
    if order:
        x = cal*4.184
    else:
        x = cal/4.184
    return x
