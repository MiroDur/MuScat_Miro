# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:07:18 2020

@author: Miro
"""
import numpy as np
import matplotlib.pyplot as plt
import time

lambda0 = 0.650 # wavelenght in vacuum [in micrometers]
refrIndexM = 1.47 # refractive index of medium
lambdaM = lambda0/refrIndexM # wavelenght in medium
NAo = 0.5 # objective NA
NAc = 0.3 # condenser NA
gridSize = np.array([64, 64, 64]) # [z-layers, x-pixels, y-pixels]

dz, dx, dy = np.array([0.3, 0.3, 0.3]) # in micrometers

realSize = gridSize*[dz,dx,dy]


zzz, xxx, yyy = np.mgrid[-gridSize[0]/2:gridSize[0]/2-1:1j*gridSize[0],
                         -gridSize[1]/2:gridSize[1]/2-1:1j*gridSize[1],
                         -gridSize[2]/2:gridSize[2]/2-1:1j*gridSize[2]]
realxxx = xxx*dx
realyyy = yyy*dy
realzzz = zzz*dz

xx, yy = np.mgrid[-gridSize[1]/2:gridSize[1]/2-1:1j*gridSize[1],
                  -gridSize[2]/2:gridSize[2]/2-1:1j*gridSize[2]]
realxx = xx*dx
realyy = yy*dy

Kxx = xx/gridSize[1]/dx
Kyy = yy/gridSize[2]/dy

# Kz in medium
KzzM = np.sqrt((1/lambdaM**2 - Kxx**2 - Kyy**2)*((1/lambdaM**2 - Kxx**2 - Kyy**2)>0))

refrIndexO = 1.52

refrIndexDif = refrIndexO-refrIndexM

RIDistrib = np.zeros(gridSize) # refractive index difference distribution of the object

# create bead object in the center
RIDistrib = RIDistrib + np.float64(np.sqrt(realxxx**2 + \
                                           realyyy**2 + realzzz**2)<2) * refrIndexDif

condenserPupil = np.float64(lambda0 * np.sqrt(Kxx**2 + Kyy**2) < NAc)
objectivePupil = np.float64(lambda0 * np.sqrt(Kxx**2 + Kyy**2) < NAo)
# calculate spatial frequencies of illumination plane waves
KxxIllum = Kxx * condenserPupil
KyyIllum = Kyy * condenserPupil

KtIllum = np.column_stack((KxxIllum[np.nonzero(condenserPupil)],
                          KyyIllum[np.nonzero(condenserPupil)]))
Kzillum = KzzM[np.nonzero(condenserPupil)].reshape(-1, 1, 1)

# calculate illumination plane waves
planeWaves = np.exp(1j * 2 * np.pi * (KtIllum[:,0].reshape(-1,1,1) * realxx + \
                                     KtIllum[:,1].reshape(-1,1,1) * realyy))

propagator = np.fft.ifftshift(
    np.exp(-1j * 2 * np.pi * KzzM * dz)).reshape(1,gridSize[1],gridSize[2])


timeStart = time.time();
fields = planeWaves * np.exp(1j * 2 * np.pi / lambda0 * dz * \
                             RIDistrib[0,:,:]).reshape(1,gridSize[1],gridSize[2])
for layer in range(1,gridSize[0]):
    fields = np.fft.ifft2(np.fft.fft2(fields) * propagator)
    fields = fields * np.exp(1j * 2 * np.pi / lambda0 * dz * \
                             RIDistrib[layer,:,:]).reshape(1,gridSize[1],gridSize[2])
print(time.time()-timeStart)
# depending on the sample position, there is illumination defocus,
# and field has to be propagated to the focus of objective
fields = np.fft.ifft2(
    np.fft.fft2(fields) * np.fft.ifftshift(objectivePupil).reshape(1, gridSize[1], gridSize[2]))
referenceWaves = planeWaves
zStack = np.zeros(gridSize,dtype=np.complex128)

for layer in range(gridSize[0]):
    propagatedFields = fields * np.exp(1j * 2 * np.pi * Kzillum * layer * dz)
    propagatedFields = np.fft.ifft2(np.fft.fft2(propagatedFields) * np.fft.ifftshift(
        np.exp(1j * 2 * np.pi * KzzM * (gridSize[0] - layer - 1) * dz)).reshape(1,gridSize[1],gridSize[2]))
    
    zStack[layer,:,:] = np.sum(propagatedFields * np.conj(referenceWaves),0)
print(time.time()-timeStart)
    
    
plt.figure(1)
plt.subplot(121)
plt.imshow(np.angle(zStack[:,32,:],0))
# plt.subplot(122)
# plt.plot(np.angle(np.sum(interference,0))[:,32])

