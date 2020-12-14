# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:38:06 2020

@author: Miro
"""
from MuScatObject import MuScatObject
from MuScatParameters import MuScatParameters
from MuScatMicroscopeSim import MuScatMicroscopeSim
import tensorflow as tf
import time
import numpy as np
from skimage.measure import block_reduce
from scipy.ndimage import median_filter
tf.config.set_visible_devices([], 'GPU')
start = time.time()

refX = tf.cast(tf.linspace(-2., 2., 5), tf.float32)
refY = tf.cast(tf.linspace(-2., 2., 5), tf.float32)

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
                              0.5,          # NAc
                              0.5)          # NAo
USAFObject = MuScatObject(parametersUSAF)
usaf = np.loadtxt('USAF.csv', delimiter=',')
usafPart = usaf[-100:-10,-91:-1]
USAFObject.RIDistrib = tf.cast(tf.reshape(tf.constant(
    usafPart[::1, ::1] * (1 - USAFObject.refrIndexM)), USAFObject.gridSize), tf.float32)


#%% generate random thick layer
parametersRandLay = MuScatParameters(0.650,        # wavelength in vacuum
                              [1, 90, 90], # gridSize [z, x ,y]
                              0.3,          # dx
                              0.3,          # dy
                              0.3,          # dz
                              1.5210,          # refractive index in medium
                              0.5,          # NAc
                              0.5)          # NAo
RandLayObject = MuScatObject(parametersRandLay)
RandLayObject.RIDistrib = tf.cast(tf.reshape(tf.constant(
     median_filter(np.random.randn(1,90,90), 1) * 10.5 * (1 - RandLayObject.refrIndexM)),
    RandLayObject.gridSize), tf.float32)
#%% free space
parametersFreeSpace = MuScatParameters(0.650,        # wavelength in vacuum
                              [1, 90, 90], # gridSize [z, x ,y]
                              0.3,          # dx
                              0.3,          # dy
                              0.3,          # dz
                              1.,           # refractive index in medium
                              0.5,          # NAc
                              0.5)          # NAo
freeSpaceObject = MuScatObject(parametersFreeSpace)
#%%

imagingSim = MuScatMicroscopeSim(parametersFreeSpace)

imagingSim.Illumination()
imagingSim.Detection()
zPositions = -(USAFObject.dz + 100. + RandLayObject.gridSize[0] *  RandLayObject.dz)

#%% MLB scattering on USAF
imagingSim.ComputeScatteredField(USAFObject, imagingSim.planeWaves)

#%% propagation through empty space
imagingSim.scatteredField = imagingSim.PropagateField(imagingSim.scatteredField,
                                                      100.,
                                                      freeSpaceObject)

#%% MLB scattering on random layer
imagingSim.ComputeScatteredField(RandLayObject, imagingSim.scatteredField)

#%%
MCF = imagingSim.CCHMImaging(imagingSim.scatteredField,
                                        zPositions, refShifts)