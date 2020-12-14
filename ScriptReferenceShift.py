# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:47:50 2020

@author: Miro
"""
from MuScatObject import MuScatObject
from MuScatParameters import MuScatParameters
from MuScatMicroscopeSim import MuScatMicroscopeSim
import tensorflow as tf
import time
import numpy as np

tf.config.set_visible_devices([], 'GPU')
start = time.time()
parameters = MuScatParameters(0.650,        # wavelength in vacuum
                              [32, 32, 32], # gridSize [z, x ,y]
                              0.3,          # dx
                              0.3,          # dy
                              0.3,          # dz
                              1.5,          # refractive index in medium
                              0.5,          # NAc
                              0.95)          # NAo
regLambdaL1 = 0.0001
regLambdaL2 = 0.001
learnRate = 0.01

refX = tf.cast(tf.linspace(-1., 1., 5), tf.float64)
refY = tf.cast(tf.linspace(-1., 1., 5), tf.float64)

refXX, refYY = tf.meshgrid(refX, refY, indexing='ij')
refShifts = tf.stack([tf.reshape(refXX, [-1]), tf.reshape(refYY, [-1])],1)

imagedObject = MuScatObject(parameters)
imagedObject.GenerateBead(2., 1.52)

imagingSim = MuScatMicroscopeSim(parameters)

imagingSim.Illumination()
imagingSim.Detection()
zPositions = imagedObject.realzzz[:,0,0] - imagedObject.realzzz[0,0,0]
zStackMeasured = imagingSim.CCHMImaging(imagedObject, zPositions, refShifts)
print(time.time()-start)

optimizedObject = MuScatObject(parameters)
regularizer = tf.keras.regularizers.L1L2(regLambdaL1, regLambdaL2)
opt = tf.keras.optimizers.Adamax(learning_rate=learnRate)

@tf.function
def loss_fn(zStackSimulated, zStackMeasured):
    return tf.reduce_mean(tf.abs(zStackMeasured - zStackSimulated)**2) +\
        regularizer(optimizedObject.RIDistrib)
        
loss = lambda: loss_fn(imagingSim.CCHMImaging(optimizedObject, zPositions, refShifts),
                      zStackMeasured)

for i in range(1):
    opt_op = opt.minimize(loss, var_list=[optimizedObject.RIDistrib])
    print(time.time()-start)