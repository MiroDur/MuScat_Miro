# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:29:07 2020

@author: Miro
"""

from MuScatObject import MuScatObject
from MuScatField import MuScatField
from MuScatParameters import MuScatParameters
from MuScatMicroscopeSim import MuScatMicroscopeSim
import tensorflow as tf
import time
import numpy as np
from skimage.measure import block_reduce
from scipy.ndimage import median_filter
tf.config.set_visible_devices([], 'GPU')
start = time.time()

refX = tf.cast(tf.linspace(-5., 5., 11), tf.float32)
refY = tf.cast(tf.linspace(-0., 0., 1), tf.float32)

refXX, refYY = tf.meshgrid(refX, refY, indexing='ij')
refShifts = tf.stack([tf.reshape(refXX, [-1]), tf.reshape(refYY, [-1])],1)
#refShifts = tf.cast(tf.constant([[0., 0.]]), tf.float32)

#%% generate USAF
parametersUSAF = MuScatParameters(0.650,        # wavelength in vacuum
                              [1, 90, 90], # gridSize [z, x ,y]
                              0.3,          # dx
                              0.3,          # dy
                              0.25,          # dz
                              1.4881,          # refractive index in medium
                              0.3,          # NAc
                              0.5)          # NAo
USAFObject = MuScatObject(parametersUSAF)
usaf = np.loadtxt('USAF.csv', delimiter=',')
usafPart = usaf[-100:-10,-91:-1]
USAFObject.RIDistrib = tf.cast(tf.reshape(tf.constant(
    usafPart[::1, ::1] * (1 - USAFObject.refrIndexM)), USAFObject.gridSize), tf.float32)


#%% generate diffraction grating
parametersGrating = MuScatParameters(0.650,        # wavelength in vacuum
                              [1, 90, 90], # gridSize [z, x ,y]
                              0.3,          # dx
                              0.3,          # dy
                              0.666,          # dz
                              1.4881,          # refractive index in medium
                              0.3,          # NAc
                              0.5)          # NAo
GratingObject = MuScatObject(parametersGrating)
grating = np.zeros(GratingObject.gridSize)
grating[0,::30,::1] = 1
GratingObject.RIDistrib = tf.cast(tf.reshape(tf.constant(
    grating * (1 - GratingObject.refrIndexM)), GratingObject.gridSize), tf.float32)
#%% free space
parametersFreeSpace = MuScatParameters(0.650,        # wavelength in vacuum
                              [1, 90, 90], # gridSize [z, x ,y]
                              0.3,          # dx
                              0.3,          # dy
                              0.3,          # dz
                              1.521,           # refractive index in medium
                              0.3,          # NAc
                              0.5)          # NAo
freeSpaceObject = MuScatObject(parametersFreeSpace)
#%%

imagingSim = MuScatMicroscopeSim(parametersFreeSpace)

imagingSim.Illumination()
imagingSim.Detection()
zPositions = -(USAFObject.dz + 170. + GratingObject.gridSize[0] *  GratingObject.dz)

#%% MLB scattering on USAF
imagingSim.ComputeScatteredField(USAFObject, imagingSim.planeWaves)

#%% propagation through empty space
imagingSim.scatteredField = imagingSim.PropagateField(imagingSim.scatteredField,
                                                      170.,
                                                      freeSpaceObject)

#%% MLB scattering on grating
imagingSim.ComputeScatteredField(GratingObject, imagingSim.scatteredField)

#%%
MCF = imagingSim.CCHMImaging(imagingSim.scatteredField,
                                        zPositions, refShifts)
print(time.time()-start)