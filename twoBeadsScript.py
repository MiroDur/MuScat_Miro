# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:27:41 2020

@author: Miro
"""
from MuScatObject import MuScatObject
from MuScatParameters import MuScatParameters
from MuScatMicroscopeSim import MuScatMicroscopeSim
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU') # remove in Google Colab

start = time.time()
parameters = MuScatParameters(0.650,        # wavelength in vacuum
                              [50, 32, 32], # gridSize [z, x ,y]
                              0.3,          # dx
                              0.3,          # dy
                              0.3,          # dz
                              1.5,          # refractive index in medium
                              0.9,          # NAc
                              0.95)          # NAo
regLambdaL1 = 0.0001
regLambdaL2 = 0.001
learnRate = 0.01
imagedObject = MuScatObject(parameters)
imagedObject.GenerateTwoBeads(2., 6., 1.52)

imagingSim = MuScatMicroscopeSim(parameters)

imagingSim.Illumination()
imagingSim.Detection()
zPositions = imagedObject.realzzz[:,0,0] - imagedObject.realzzz[0,0,0]
zStackMeasured = imagingSim.CCHMImaging(imagedObject, zPositions)

print(time.time()-start)

optimizedObject = MuScatObject(parameters)
regularizer = tf.keras.regularizers.L1L2(regLambdaL1, regLambdaL2)
opt = tf.keras.optimizers.Adamax(learning_rate=learnRate)

@tf.function
def loss_fn(zStackSimulated, zStackMeasured):
    return tf.reduce_mean(tf.abs(zStackMeasured - zStackSimulated)**2) +\
        regularizer(optimizedObject.RIDistrib)
        
loss = lambda: loss_fn(imagingSim.CCHMImaging(optimizedObject, zPositions),
                      zStackMeasured)

for i in range(1):
    opt_op = opt.minimize(loss, var_list=[optimizedObject.RIDistrib])
    print(time.time()-start)