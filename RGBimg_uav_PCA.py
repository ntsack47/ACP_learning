# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:12:34 2019

@author: situy
"""

# <codecell>
import cv2
import matplotlib.pyplot as plt
import numpy as np
import xml.dom.minidom as xmd
import os

from sklearn.decomposition import PCA
import scipy.signal as sig
# <codecell>
def clip_rgb(img,coor):
    t4 = np.zeros(1)
    for i in range(coor.shape[0]):
        t1x = img[:,coor[i,0]:coor[i,2]]
        t2y = t1x[coor[i,1]:coor[i,3],:]
        t3 = np.ravel(t2y)
        t4 = np.append(t4,t3)
    #x = np.ravel(t4)
    return t4
# <codecell>
filename = "ym0012"
jpgn = "D:/downloads/uav/"+filename+".JPG"
xmdn = "D:/downloads/uav/"+filename+".xml"
o1 = cv2.imread(jpgn)
#plt.imshow(b1)
## <codecell>
r1 = o1[:,:,0]
g1 = o1[:,:,1]
b1 = o1[:,:,2]
base = np.where(2.*g1+r1*1.+b1*1.<=0, 1, 2*g1+r1+b1)
vdvi = (2.0*g1-r1*1.-b1*1.)/(base)

#rvi  = (r1*2.-g1*1.)/(r1*1.+g1*1.)     # not good
#rvi2 = (r1*2.-b1*1.)/(r1*1.+b1*1.)     #一般
#rvi3 = (b1*2.-g1*1.)/(g1*1.+b1*1.)
#rvi4 = (b1*2.-g1*1.-r1*0.5)/(g1*1.+r1*1.)  # ok
#rvi5 = (b1*2.-g1*2.+r1*1.)/(g1*1.+r1*1.)
rvi6 = (b1*2.-g1*1.+r1*0.5)/(g1*1.+r1*1.5)   # good
rvi7 = (b1*1.+r1*1.-g1*1.)/(r1*1.+b1*1.+g1*1.)

rvi8 = (b1*-0.57+r1*0.8-g1*0.14)/(b1*+0.57+r1*0.8+g1*0.14) #用的总PCA
rvi9 = (b1*0.753-r1*0.449-g1*0.481)/(b1*0.753+r1*0.449+g1*0.481)  #取了一部分出来的PCA
rvi9a = (255/np.max(rvi9))*rvi9

rvix = (g1*1.-b1*1.)/(g1*1.+b1*1.)

sxi = (g1*1.+b1*1.-r1*1.)/(g1*1.+b1*1.+r1*1.)
sxi_norm = (sxi+0.2)/1.2
## <codecell> parse xml file
xm1 = xmd.parse(xmdn)

a1 = xm1.documentElement
a2 = a1.getElementsByTagName("xmin")

coords = np.array([[0,0,0,0]])
for i in range(len(a2)):
    t1 = np.int(a1.getElementsByTagName("xmin")[i].childNodes[0].data)
    t2 = np.int(a1.getElementsByTagName("ymin")[i].childNodes[0].data)
    t3 = np.int(a1.getElementsByTagName("xmax")[i].childNodes[0].data)
    t4 = np.int(a1.getElementsByTagName("ymax")[i].childNodes[0].data)
    t5 = np.array([t1,t2,t3,t4])
    t6 = t5[np.newaxis,:]
    coords = np.append(coords,t6,axis = 0)
coords = coords[1:,:]

## <codecell>
#v2 = np.array(vdvi)
#plt.imshow(vdvi)
#ba

np.save("D:/Projects/松材线虫_无人机/rvi9a.npy", rvi9a)
# <codecell>
# start, end(xmin,ymin,xmax,ymax), color, thickness...
for i in range(len(a2)):
    cv2.rectangle(rvi9a,(coords[i,0],coords[i,1]),(coords[i,2],coords[i,3]),1,3)
cv2.namedWindow("vd1",0)
cv2.imshow("vd1",rvi9a)

# <codecell>
# start, end(xmin,ymin,xmax,ymax), color, thickness...
for i in range(len(a2)):
    cv2.rectangle(sxi,(coords[i,0],coords[i,1]),(coords[i,2],coords[i,3]),1,3)
cv2.namedWindow("vd",0)
cv2.imshow("vd",rvi9a*sxi_norm)

# <codecell>
# start, end(xmin,ymin,xmax,ymax), color, thickness...
cv2.rectangle(rvix, (soilc[0,0],soilc[0,1]),(soilc[0,2],soilc[0,3]),1,3)
cv2.namedWindow("vd",0)
cv2.imshow("vd",rvix)

# <codecell>

xap  = np.array([1600,900,2600,2100])
tc1 = xap[np.newaxis,:]
tc2 = np.array([2600,500,3100,1100])[np.newaxis,:]
soilc = np.array([4100,  600, 4325,  1045])[np.newaxis,:]
#coords = np.append(coords, xap[np.newaxis,:], axis = 0)
# <codecell> 通过coords来获取范围内RGB三个波段值的分布
cr = clip_rgb(r1,coords)
cg = clip_rgb(g1, coords)
cb = clip_rgb(b1, coords)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(cr, bins = 20, color = "red")
ax.hist(cg, bins = 20, color = "green")
ax.hist(cb, bins = 20, color = "blue")

ax.set_title("sick trees")

# <codecell> 用均值的方法求中心距离
cr1 = clip_rgb(r1, coords)
cg1 = clip_rgb(g1, coords)
cb1 = clip_rgb(b1, coords)

cr2 = clip_rgb(r1, soilc)
cg2 = clip_rgb(g1, soilc)
cb2 = clip_rgb(b1, soilc)

rx = np.mean(cr1)
bx = np.mean(cb1)
gx = np.mean(cg1)

ry = np.mean(cr2)
by = np.mean(cb2)
gy = np.mean(cg2)

dist = np.sqrt((rx-ry)**2+(gx-gy)**2+(bx-by)**2)

plt.scatter(cr1,cb1)
plt.scatter(cr2,cb2)

plt.scatter(cr1,cg1)
plt.scatter(cr2,cg2)
# <codecell>
cr = clip_rgb(r1,tc1)
cg = clip_rgb(g1, tc1)
cb = clip_rgb(b1, tc1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(cr, bins = 20, color = "red")
ax.hist(cg, bins = 20, color = "green")
ax.hist(cb, bins = 20, color = "blue")

ax.set_title("normal trees")


# <codecell> 

t1 = clip_rgb(rvi6,tc1)
t2 = clip_rgb(rvi6,coords)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#ax.hist(t2, bins = 20, color = "red")
ax.hist(t1, bins = 20, color = "green")


ax.set_title("comparison")

# <codecell> 
可以尝试一下用过滤的方法强化边缘，减小道路的影响。

# <codecell> PCA RVI9
def clip_mat(img,coor):
    t1x = img[coor[0,1]:coor[0,3],tc2[0,0]:coor[0,2]]
    t2x = np.ravel(t1x)
    return t2x

cr = clip_mat(r1,tc2)
cg = clip_mat(g1, tc2)
cb = clip_mat(b1, tc2)

red_x = np.ravel(cr)
green_x = np.ravel(cg)
blue_x = np.ravel(cb)
total_y = np.array([red_x,green_x, blue_x])
p1 = PCA(n_components=3)
p1.fit(total_y.T)
px1 = p1.transform(total_y.T)

p1.explained_variance_ratio_

p1.components_

# <codecell> PCA

red_x = np.ravel(r1)
green_x = np.ravel(g1)
blue_x = np.ravel(b1)
total_y = np.array([red_x,green_x, blue_x])
p1 = PCA(n_components=3)
p1.fit(total_y.T)
px1 = p1.transform(total_y.T)

p1.explained_variance_ratio_

p1.components_

# <codecell> PCA
def pca(X,k):#k is the components you want
  #mean of each feature
  n_samples, n_features = X.shape
  mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
  #normalization
  norm_X=X-mean
  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True)
  # select the top k eig_vec
  feature=np.array([ele[1] for ele in eig_pairs[:k]])
  #get new data
  data=np.dot(norm_X,np.transpose(feature))
  print([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],
        [eig_vec[:,1][0],0],[eig_vec[:,1][1],0])
  
  return data

#X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
X = np.array([x,y]).T
p2 = PCA(n_components = 1)
p2.fit(X)
cp = p2.components_
c1 = cp[0][0]
c2 = cp[0][1]
cmat = np.array([[c2,-1*c1],[c1,c2]])
pca(X,1)

plt.scatter(X[:,1],X[:,0])
plt.plot(cp)

plt.scatter(new_data[:,0],new_data[:,1])

new_data=np.transpose(np.dot(cmat,np.transpose(X)))

# <codecell>
dat0 = np.where(rvi9a<=0, 0, 1)
dat = sig.medfilt2d(dat0, (3,3))
sig.


cv2.imwrite("D:/Projects/松材线虫_无人机/r9a.jpg", rvi9a)















