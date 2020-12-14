# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:54:25 2020

@author: Miro
"""
from MuScatObject import MuScatObject
from MuScatField import MuScatField
from MuScatParameters import MuScatParameters
from MuScatMicroscopeSim import MuScatMicroscopeSim
import tensorflow as tf
import tensorflow_addons as tfa
import time
import numpy as np
tf.config.set_visible_devices([], 'GPU')
#%%
start = time.time()
parameters = MuScatParameters(0.650,        # wavelength in vacuum
                              [50, 64, 64], # gridSize [z, x ,y]
                              0.45,          # dx
                              0.45,          # dy
                              0.3,          # dz
                              1.5,          # refractive index in medium
                              0.5,          # NAc
                              0.5)          # NAo
regLambdaL1 = 0.001
regLambdaL2 = 0.0000#1
learnRate = 0.01
refShifts = tf.cast(tf.constant([[0., 0.]]), tf.float32)
imagedObject = MuScatObject(parameters)
#imagedObject.GenerateBox(4., 4., 4., 1.52)
imagedObject.GenerateBead(8., 1.52)
#imagedObject.GenerateSheppLogan(1.55)

imagingSim = MuScatMicroscopeSim(parameters)

imagingSim.Illumination()
imagingSim.Detection()
zPositions = imagedObject.realzzz[::2, 0, 0] - imagedObject.realzzz[0, 0, 0]
fieldMeasured = MuScatField(imagingSim.planeWaves, parameters).ComputeMuScat(imagedObject, method='MLB')
#fieldMeasured.ComputeMuScat(imagedObject, method='MLB')
zStackMeasured = imagingSim.CCHMImaging(fieldMeasured, 
                                        zPositions, 
                                        refShifts)
print(time.time()-start)
#%%
optimizedObject = MuScatObject(parameters)
regularizer = tf.keras.regularizers.L1L2(regLambdaL1, regLambdaL2)
#opt = tf.keras.optimizers.Adamax(learning_rate=learnRate)
opt = tfa.optimizers.ProximalAdagrad(learning_rate=learnRate,
                                     l1_regularization_strength=regLambdaL1,
                                     l2_regularization_strength=regLambdaL2,)
def regTV(variable, parameter):
    diffZ = (2*variable - tf.roll(variable, 1, 0) - tf.roll(variable, -1, 0)) / 2
    diffX = (2*variable - tf.roll(variable, 1, 1) - tf.roll(variable, -1, 1)) / 2
    diffY = (2*variable - tf.roll(variable, 1, 2) - tf.roll(variable, -1, 2)) / 2
    return parameter * tf.reduce_sum(tf.abs(diffX) + tf.abs(diffY) +\
                                     tf.abs(diffZ))
    
@tf.function
def loss_fn(zStackSimulated, zStackMeasured):
    return tf.reduce_mean(tf.abs(zStackMeasured - zStackSimulated)**2) #+\
        #regularizer(optimizedObject.RIDistrib)
            
#scatteredField = MuScatField(imagingSim.planeWaves, parameters).ComputeMuScat(optimizedObject, method='MLB')        
loss = lambda: loss_fn(imagingSim.CCHMImaging(
    MuScatField(imagingSim.planeWaves, parameters).ComputeMuScat(optimizedObject, method='MLB'),
                                              zPositions,
                                              refShifts),
                      zStackMeasured)
lossF=[]
for i in range(0):
    opt_op = opt.minimize(loss, var_list=[optimizedObject.RIDistrib])
    #lossF.append(loss_fn(imagingSim.CCHMImaging(imagingSim.ComputeScatteredField(optimizedObject), zPositions, refShifts),
    #                  zStackMeasured).numpy)
    print(time.time()-start)
