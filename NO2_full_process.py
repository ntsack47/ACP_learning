# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 14:23:03 2019

@author: situy
"""

# <codecell> import libs
import numpy as np
import netCDF4 as nc
#import datetime
#from datetime  import datetime as dtm
#import re
import os
import matplotlib.pyplot as plt

# <codecell> other libs
import pandas as pd
import cv2
#import scipy.ndimage as scn
a = cv2.imread("D:/downloads/2w.jpg")
basemap = -a[:,:,0]
# <codecell> basic parameters
class loc:
    xpix = 1440
    ypix = 720
    xlim = np.array([-179.875,179.875])
    ylim = np.array([-89.875,89.875])
    step = 0.25
    scystep = 0.125

class city:
    Mumbai = [72.88, 19.09]
    Seoul = [127,37.55]
    MexicoCity = [-99.13, 19.43]
    Rome = [12.5, 41.9]
    Houston = [-95.37, 29.75]
    Baghdad = [44.37, 33.3]
    Cairo = [31.24, 30.04]
    Karachi = [67.04, 24.87]
    Beijing = [116.4, 39.9]
    Nanjing = [118.8, 32]
# <codecell> functions

def conv_lonlat(xlim, ylim, step):
    """xind is lon(min,max), yind is lat(min,max)
    be sure to enter the right step (e.g. 0.25 deg)"""
    lon_array = np.arange(xlim[0],xlim[1]+step,step)
    lat_array = np.arange(ylim[0],ylim[1]+step,step)
    return lon_array,lat_array
#def read_lonlat(nc_file,var,lon,lat)

def conv_city_loc(loc):
    lon = loc[0]
    lat = loc[1]
    ind_x = int((lon-(-180))/0.25)
    ind_y = int((lat-(-90))/0.25)
    return ind_x,ind_y

def conv_line(line_str):
    a = line_str
    b = re.findall(r'.{4}',a)       #re函数
    c = []
    for i in b:
        i1 = i.split()            #split函数用法
        c.append(i1)
    d0 = np.array(c)
    d = [[int(y) for y in x] for x in d0]
    #d = map(int,d0)   不管用
    #d = d0[:,0]
    return d

def avg_13(array20):      # for SCY data
    count = np.arange(0,len(array20))
    newarray=[]
    #inds = [i-6,i-5,i-4,i-3,i-2,i-1,i,i+1,i+2,i+3,i+4,i+5,i+6]
    for i in count+7:
        if i>=len(array20):
            inds = np.arange(i-len(array20)-13
                             ,i-len(array20))
        else:
            inds = np.arange(i-13,i)
        newarray.append(np.nanmean(array20[inds]))
        #print(array20[inds])
    return newarray


def read_toms_file(fname,lonpix):
    dat = []
    with open(fname) as file1:
        content = file1.readlines()
        linesize = len(content)
        skipline = 4
        groupsize = int(lonpix/20)+1
        while skipline <= linesize-1:
            for i in range(skipline+1,skipline+groupsize):  # +1 to skip lat
                dat.append(conv_line(content[i]))
            skipline = skipline + groupsize
    return dat

def det_month(sat_type,month_ind):
    if month_ind>150:
        print("You probably used the wrong function")
    if sat_type=="GOME":
        init = 4
    elif sat_type=="SCY":
        init = 8
    elif sat_type=="GOME2A":
        init = 1
    elif sat_type=="GOME2B":
        init = 1
    t1 = month_ind%12
    month = t1+init if t1+init <= 12 else t1+init-12
    return month

def plot_city(cityname,gome, scy):
    x,y = conv_city_loc(cityname)
    plt.plot(gome[y,x,76:87],marker='o',linewidth = '1' \
             , label = "GOME", linestyle='-')
    plt.plot(scy[y,x,0:11],marker='o', label = "SCIAMACHY", linewidth = "1")
    plt.legend(loc='lower center')
    
def tick_generate(base_string, start, num):
    t = []
    for i in range(num):
        t.append(base_string+'%02d'%(start+i))
    return t

def write_to_nc(data,varname, file_name_path,
                lon1=-179.875,lon2=179.875,
                lat1=-89.875,lat2=89.875,
                pointnum1=loc.xpix,pointnum2=loc.ypix):    
    lons  = np.linspace(lon1, lon2, pointnum1)
    lats = np.linspace(lat1, lat2, pointnum2)
    da = nc.Dataset(file_name_path, 'w', format = 'NETCDF4')
    da.createDimension('lon', pointnum1)
    da.createDimension('lat', pointnum2)
    da.createVariable('lon', 'f', ('lon'))     #f 为数据类型，此处为浮点
    da.createVariable('lat', 'f', ('lat'))
    da.variables['lat'][:] = lats
    da.variables['lon'][:] = lons
    da.createVariable('datavar', 'f8', ('lat','lon'))
    da.variables[varname][:] = data
    
# <codecell> asc文件读取并按时间顺序拼接 

# GOME，GOME2的点数量是一样
# SCY 第一个版本和GOME读取方法一样

fdir = "D:/Projects/假期小任务/数据下载/GOME/unzip/"
fnlist = os.listdir(fdir)
fnlist.sort()
flist = []
for i in fnlist:
    flist.append(fdir+i)
x1, y1 = conv_lonlat(loc.xlim, loc.ylim, loc.step)
datall = np.array(np.zeros([51840,20,1]))
for fn in flist:
    t1 = np.array(read_toms_file(fn,loc.xpix)
    t2 = t1.reshape([720,1440,1])
    datall = np.append(datall, t2), axis=2)
    
# <codecell> asc文件读取并按时间顺序拼接
# SCY 第二个版本分辨率高一倍 
    
fdir = "D:/Projects/假期小任务/数据下载/SCY/TotalColumn(totno2)/qa4ecv/"
fnlist = os.listdir(fdir)
fnlist.sort()
flist = []
for i in fnlist:
    flist.append(fdir+i)

datall = np.array(np.zeros([loc.ypix,loc.xpix,1]))
t2 = None
for fn in flist:
    t1 = np.array(read_toms_file(fn,2880))
    t2 = t1.reshape([1440,2880])
    t3 = scn.zoom(t2, zoom = 0.5)
    t4 = np.where(t3<-500, np.nan, t3)
    t4 = t4[::-1,:,np.newaxis]
    datall = np.append(datall, t4, axis=2)

datall = datall[:,:,1:(1+len(fnlist))]
#dat = np.where(datall<-500, datall, np.nan)

np.save("D:/Projects/假期小任务/数据下载/SCY_new.npy", datall)
# <codecell> 从netcdf获取SCY数据
fname1 = 'D:/Projects/假期小任务/TroposNO2.nc'
nc1 = nc.Dataset(fname1)


plt.imshow(scy1[1,::-1,:], vmin = 0 , vmax = 20)

scy1 = np.array(nc1.variables['TroposNO2'][76:(76+116)])#.reshape([720,1440])
scy2 = np.where(scy1!=-999, scy1, np.nan)
scy3 = np.transpose(scy2,(1,2,0))

np.save("D:/Projects/假期小任务/数据下载/SCY_nc.npy", scy3)
# <codecell> 平滑SCY,乘以100得到GOME的1e13单位
scy0 = np.load("D:/Projects/假期小任务/数据下载/SCY_nc.npy")
scy = scy0*100
scy0 = None
scsm = np.zeros(scy.shape)
for j in range(116):
    for i in range(loc.ypix):
        scsm[i,:,j] = np.array(avg_13(scy[i,:,j]))

np.save("D:/Projects/假期小任务/数据下载/SCY_nc_scsm.npy", scsm)        
# <codecell> 计算CF1，重新读取scsm的话要乘以100
scsm0 = np.load("D:/Projects/假期小任务/数据下载/SCY_nc_scsm.npy")
scsm = scsm0*100
scsm0 = None
#plt.imshow(scsm[::-1,:,2])
m8 = np.array([0,12,24,36,48,60,72,84,96,108])
m9 = m8+1; m10 = m8+2; m11 = m8+3; m12 = m8+4
m1 = m8+5; m2 = m8+6; m3 = m8+7
m4 = (m8+8)[0:-1]
m5 = m4+1; m6 = m4+2; m7 = m4+3

# mask VCD-SCsm < 10e13 molecules/cm3
# 1,取绝对值<10的进行屏蔽，scy和scsm同时修改为1，相除得1
sc_m = np.array(np.ones([720,1440,1]))
scsm_m = np.array(np.ones([720,1440,1]))
for i in m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12:
    t1 = np.nanmean(scy[:,:,i], axis = 2)
    t2 = np.nanmean(scsm[:,:,i], axis = 2)
    # 屏蔽scy
    t3 = np.where(np.abs(t2)<10, 1, t1)
    # 屏蔽scsm
    t4 = np.where(np.abs(t2)<10, 1, t2)
    t5 = t3[:,:,np.newaxis]
    t6 = t4[:,:,np.newaxis]
    sc_m = np.append(sc_m, t5, axis = 2)
    scsm_m = np.append(scsm_m, t6, axis = 2)
sc_m = sc_m[:,:,1:13]
scsm_m = scsm_m[:,:,1:13]

CF1a = sc_m/scsm_m

np.save("D:/Projects/假期小任务/数据下载/CF1_abs_10.npy", arr= CF1a)

plt.imshow(CF1a[::-1,:,2], vmin = -1 , vmax = 3)
plt.hist(np.ravel(CF1a[:,:,5]), 200, range = (-5,5))
# 方法2，不考虑负值
# 暂不执行方法2
#sc_m = np.array(np.ones([720,1440,1]))
#scsm_m = np.array(np.ones([720,1440,1]))
#for i in m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12:
#    t1 = np.nanmean(scy[:,:,i], axis = 2)
#    t2 = np.nanmean(scsm[:,:,i], axis = 2)
#    t3 = t1[:,:,np.newaxis]
#    t4 = t2[:,:,np.newaxis]
#    sc_m = np.append(sc_m, t3, axis = 2)
#    scsm_m = np.append(scsm_m, t4, axis = 2)
#sc_m = sc_m[:,:,1:13]
#scsm_m = scsm_m[:,:,1:13]
#
#top2 = np.where(np.abs(SC_m)>10, SC_m, 1)
#bot2 = np.where(np.abs(SCsm_m)>10, SCsm_m, 1)
#CF1 = top2/bot2

# <codecell> GC1
# 屏蔽nan与否不影响的
gome = np.load("D:/Projects/假期小任务/数据下载/GOME1a.npy")
GOME = np.ma.masked_invalid(gome)
SCY = np.ma.masked_invalid(scy)
# use 1 to prevent missing data
GC1 = np.ones(GOME.shape)
for i in range(GC1.shape[2]):
    month_ind = det_month("GOME",i)-1
    GC1[:,:,i] = GOME[:,:,i] * CF1a[:,:,month_ind]

np.save(file = "D:/Projects/假期小任务/数据下载/GC1.npy", arr = GC1)
# <codecell> CF2 and GC2
# convert to 1e13 
overlap_gome = np.arange(76,87)
overlap_scy  = np.arange(0,11)

CF2 = (np.nanmean(SCY[:,:,overlap_scy],axis = 2) \
      -np.nanmean(GC1[:,:,overlap_gome],axis = 2))
  
#np.save(file = "D:/Projects/假期小任务/数据下载/CF2.npy", arr = CF2.data) 

#CF2 = np.load("D:/Projects/假期小任务/数据下载/CF2.npy")

CF2 = CF2[:,:,np.newaxis]
GC2 = GC1+ CF2
np.save(file = "D:/Projects/假期小任务/数据下载/GC2.npy", arr = GC2.data) 

# <codecell> CF3
# get monthly average （分子）
# n for GOME
n4 = np.array([0,12,24,36,48,60,72,84])
n5 = n4+1; n6 = n4+2
n7 = (n4+3)[0:-1]
n8 = n7+1; n9 = n7+2; n10 = n7+3; n11 = n7+4; n12 = n7+5
n1 = n7+6; n2 = n7+7; n3 = n7+8
# mask VCD-SCsm < 10e13 molecules/cm3
# 1,取绝对值<10的进行屏蔽.在第二版本中10而不是0.1，scy和scsm同时修改为1，相除得1
gc_m = np.array(np.ones([720,1440,1]))
for i in n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12:
    t1 = np.nanmean(GC2[:,:,i], axis = 2)
    # 屏蔽gc2<10的
    t2 = np.where(np.abs(t1)<10, 1, t1)
    t3 = t2[:,:,np.newaxis]
    gc_m = np.append(gc_m, t3, axis = 2)
gc_m = gc_m[:,:,1:13]

#mSC = np.zeros([720,1440,12])
#month_count = np.zeros(12)
#for i in range(SCY.shape[2]):
#    month_ind = det_month("SCY",i)-1
#    mSC[:,:,month_ind] = SCY[:,:,month_ind]+ mSC[:,:,month_ind]
#    month_count[month_ind] += 1
#for i in range(12):
#    mSC[:,:,i] = mSC[:,:,i]/month_count[i]

#plt.imshow(mSC[::-1,:,1])
#plt.imshow(sc_m[::-1,:,1])
plt.imshow(gc_m[::-1,:,3])
# here gc_m is GC2(x,y,m) and sc_m from CF1 is SC(x,y,m)

# 分母
ngc = np.nanmean(GC2, axis = 2)
nsc = np.nanmean(SCY, axis = 2)

mGC2 = np.nan_to_num(x = gc_m)
mGC2 = np.where(mGC2==0, 1, mGC2)
mSC = np.nan_to_num(x = sc_m)
mSC = np.where(mSC==0, 1, mSC)

nGC = np.where(~np.isnan(ngc), ngc, 1)
nSC = np.where(~np.isnan(nsc), nsc, 1)

bot = mGC2/nGC[:,:,np.newaxis]
top = mSC/nSC[:,:,np.newaxis]

CF3 = top/bot

np.save(file = "D:/Projects/假期小任务/数据下载/CF3.npy", arr = CF3) 


# <codecell> GC3
GC3 = np.ones(GC2.shape)
for i in range(GC2.shape[2]):
    month_ind = det_month("GOME",i)-1
    GC3[:,:,i] = GC2[:,:,i] * CF3[:,:,month_ind]

np.save(file = "D:/Projects/假期小任务/数据下载/GC3.npy", arr = GC3)

# <codecell> 调整图像方向
agc3 = GC3[::-1,:,:]
agome = GOME[::-1,:,:]
ascy = SCY[::-1,:,:]

#plt.hist(np.ravel(CF3[:,:,5]), 200, range = (-5,5))
#plt.imshow(CF3[:,:,1])
# <codecell> 对比GC3, netCDF, SCY
#fname1 = 'D:/Projects/假期小任务/TroposNO2.nc'
#nc1 = nc.Dataset(fname1)
#knmi = np.load("D:/Projects/假期小任务/数据下载/SCY1a.npy")[::-1,:,:]
#gome = np.load("D:/Projects/假期小任务/数据下载/GOME1a.npy")[::-1, :,:]
#gc3 = np.load("D:/Projects/假期小任务/数据下载/GC3.npy")[::-1, :,:]
# GC3 看起来很不错
#gc3 = np.load("D:/Projects/假期小任务/数据下载/GC3_all_CF1nan.npy")[::-1, :,:]
# GC3_all_CF1不太行
# <codecell> 画图1
x,y = conv_city_loc(city.Houston)
g1 = np.append(agome[y,x,:], np.zeros(105))
g3 = np.append(agc3[y,x,:], np.zeros(105))
s1 = np.append(np.zeros(76), ascy[y,x,:])
#G2 = np.load("D:/Projects/假期小任务/数据下载/GOME2m.npy")[::-1,:,0:63]
#G2 = np.append(np.zeros(129), G2[y,x,:])

s1 = np.where(s1==0, np.nan, s1)
g1 = np.where(g1==0, np.nan, g1)
g3 = np.where(g3==0, np.nan, g3)

nc_slice = np.array(nc1.variables['TroposNO2'])[0:192,::-1,x]
nc_slice = nc_slice[:,y]
n1 = np.where(nc_slice <=-10, np.nan, nc_slice)
n1 = n1*100

fig, ax = plt.subplots()
t = np.arange(0,129)
a, = ax.plot(t, g1[t], color = 'blue')
b, = ax.plot(t, g3[t], color = 'yellow')
c, = ax.plot(t, s1[t], color = 'orange')
#d, = ax.plot(t, s2, color = 'orange', linestyle = ':')
e, = ax.plot(t, n1[t], color = 'grey',linestyle = ':')

year_tick = np.arange(9,129,step = 12)
tick1= tick_generate("Jan 19", 96, 4)
tick2 = tick_generate("Jan 20", 0, 7)

ax.set_xticks(year_tick)
ax.set_xticklabels(np.append(tick1,tick2), rotation = 45)
ax.set_title('Houston')
ax.set_xlabel("Time")
ax.set_ylabel(r"NO$_2$, 1e13 molecules/cm$^2$")

ax.legend(handles = [a,b,c,e],labels \
          = ["GOME1","GC3","SCY","Combined_NetCDF"])
plt.show()

# <codecell> 拼接GOME2 数据
g2a = np.load(file = "D:/Projects/假期小任务/数据下载/GOME2A.npy")
g2b = np.load(file = "D:/Projects/假期小任务/数据下载/G2mean.npy")

dat1 = np.array(np.zeros([720,1440,1]))
for i in range(g2a.shape[2]):
    t1 = g2a[:,:,i]
    t2 = t1.reshape([720,1440,1], order = 'C')
    dat1 = np.append(dat1, t2, axis = 2)
dat2 = dat1[:,:,1:151]

a_time = np.arange(0,72)
g2m = np.append(dat2[:,:,a_time],g2b, axis = 2)
g2m = np.where(g2m==-999, np.nan, g2m)
np.save("D:/Projects/假期小任务/数据下载/GOME2m.npy", arr = g2m)

g2m = np.load("D:/Projects/假期小任务/数据下载/GOME2m.npy")

# <codecell> 只计算GOME和GOME2 的 CF2，CF3
scy0 = np.load("D:/Projects/假期小任务/数据下载/SCY_nc.npy")
scy = scy0*100
scy0 = None
gome2 = np.load("D:/Projects/假期小任务/数据下载/GOME2m.npy")
gome2 = np.ma.masked_invalid(gome2)

# <codecell> GOME2 CF2
overlap_g2 = np.arange(0,63)
overlap_sc  = np.arange(53,116)
t1 = (np.nanmean(scy[:,:,overlap_sc],axis = 2) \
      -np.nanmean(gome2[:,:,overlap_g2],axis = 2))

CF2 = t1[:,:,np.newaxis]
GC2 = gome2+ CF2

# <codecell> GOME3 CF3
m8 = np.array([0,12,24,36,48,60,72,84,96,108])
m9 = m8+1; m10 = m8+2; m11 = m8+3; m12 = m8+4
m1 = m8+5; m2 = m8+6; m3 = m8+7
m4 = (m8+8)[0:-1]
m5 = m4+1; m6 = m4+2; m7 = m4+3

# SC(x,y,m)
sc_m = np.array(np.ones([720,1440,1]))
for i in m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12:
    t1 = np.nanmean(scy[:,:,i], axis = 2)
    t2 = np.where(np.abs(t1)<10, 1, t1)
    t3 = t2[:,:,np.newaxis]
    sc_m = np.append(sc_m, t3, axis = 2)
sc_m = sc_m[:,:,1:13]


n4 = np.array([0,12,24,36,48,60,72,84])
n5 = n4+1; n6 = n4+2
n7 = (n4+3)[0:-1]
n8 = n7+1; n9 = n7+2; n10 = n7+3; n11 = n7+4; n12 = n7+5
n1 = n7+6; n2 = n7+7; n3 = n7+8
# mask VCD-SCsm < 10e13 molecules/cm3
# 1,取绝对值<10的进行屏蔽.在第二版本中10而不是0.1，scy和scsm同时修改为1，相除得1
gc_m = np.array(np.ones([720,1440,1]))
for i in n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12:
    t1 = np.nanmean(GC2[:,:,i], axis = 2)
    # 屏蔽gc2<10的
    t2 = np.where(np.abs(t1)<10, 1, t1)
    t3 = t2[:,:,np.newaxis]
    gc_m = np.append(gc_m, t3, axis = 2)
gc_m = gc_m[:,:,1:13]


#plt.imshow(mSC[::-1,:,1])
#plt.imshow(sc_m[::-1,:,1])
#plt.imshow(gc_m[::-1,:,3])
# here gc_m is GC2(x,y,m) and sc_m from CF1 is SC(x,y,m)

# 分母
ngc = np.nanmean(GC2, axis = 2)
nsc = np.nanmean(scy, axis = 2)

mGC2 = np.nan_to_num(x = gc_m)
mGC2 = np.where(mGC2==0, 1, mGC2)
mSC = np.nan_to_num(x = sc_m)
mSC = np.where(mSC==0, 1, mSC)

nGC = np.where(~np.isnan(ngc), ngc, 1)
nSC = np.where(~np.isnan(nsc), nsc, 1)

bot = mGC2/nGC[:,:,np.newaxis]
top = mSC/nSC[:,:,np.newaxis]

CF3 = top/bot

#np.save(file = "D:/Projects/假期小任务/数据下载/CF3.npy", arr = CF3) 
plt.hist(np.ravel(CF3[:,:,5]), 200, range = (-5,5))

GC3 = np.ones(GC2.shape)
for i in range(GC2.shape[2]):
    month_ind = det_month("GOME2A",i)-1
    GC3[:,:,i] = GC2[:,:,i] * CF3[:,:,month_ind]
    
np.save(file = "D:/Projects/假期小任务/数据下载/GC3_GOME2m.npy", arr = GC3) 

# <codecell> 画图2 准备数据
g13 = np.load("D:/Projects/假期小任务/数据下载/GC3.npy")[::-1, :,:]
g23 = np.load("D:/Projects/假期小任务/数据下载/GC3_GOME2m.npy")[::-1,:,:]
g20 = np.load("D:/Projects/假期小任务/数据下载/GOME2m.npy")[::-1,:,:]

scy0 = np.load("D:/Projects/假期小任务/数据下载/SCY_nc.npy")[::-1, :,:]
s1 = scy0*100
scy0 = None



# <codecell> 画图2
x,y = conv_city_loc(city.Houston)
g1c = np.append(g13[y,x,:], np.zeros(192))
g2a = np.append(np.zeros(129), g20[y,x,:])
g2c = np.append(np.zeros(129), g23[y,x,:])
s10 = np.append(np.zeros(76), s1[y,x,:])
s1a = np.append(s10, np.zeros(87))
#G2 = np.load("D:/Projects/假期小任务/数据下载/GOME2m.npy")[::-1,:,0:63]
#G2 = np.append(np.zeros(129), G2[y,x,:])

g1c = np.where(g1c==0, np.nan, g1c)
g2a = np.where(g2a==0, np.nan, g2a)
g2c = np.where(g2c==0, np.nan, g2c)
s1a = np.where(s1a==0, np.nan, s1a)

nc_slice = np.array(nc1.variables['TroposNO2'])[:,::-1,x]
nc_slice = nc_slice[:,y]
n1 = np.where(nc_slice <=-10, np.nan, nc_slice)
n1 = n1*100
n1 = np.append(n1,np.zeros(21))

g1c = np.where(g1c<=-100, np.nan, g1c)
g2a = np.where(g2a<=-100, np.nan, g2a)
g2c = np.where(g2c<=-100, np.nan, g2c)
s1a = np.where(s1a<=-100, np.nan, s1a)
n1 =  np.where(n1<=-100, np.nan, n1)

g1c = np.where(g1c>=200, np.nan, g1c)
g2a = np.where(g2a>=200, np.nan, g2a)
g2c = np.where(g2c>=200, np.nan, g2c)
s1a = np.where(s1a>=200, np.nan, s1a)
n1 =  np.where(n1>=200, np.nan, n1)

fig, ax = plt.subplots()
t = np.arange(0,279)
a, = ax.plot(t, g1c, color = 'green')
b, = ax.plot(t, g2a, color = 'yellow')
c, = ax.plot(t, g2c, color = 'orange')
d, = ax.plot(t, s1a, color = 'pink', linestyle = '--')
e, = ax.plot(t, n1, color = 'black',linestyle = ':')

year_tick = np.arange(9,279,step = 12)
tick1= tick_generate("Jan 19", 96, 4)
tick2 = tick_generate("Jan 20", 0, 20)

ax.set_xticks(year_tick)
ax.set_xticklabels(np.append(tick1,tick2), rotation = 45)
ax.set_title('Houston')
ax.set_xlabel("Time")
ax.set_ylabel(r"NO$_2$, 1e13 molecules/cm$^2$")

ax.legend(handles = [a,b,c,d,e],labels \
          = ["GOME_3","GOME2","GOME2_3","SCY","NetCDF"])
plt.show()