# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:54:25 2020

@author: Miro
"""
from MuScatObject import MuScatObject
from MuScatParameters import MuScatParameters
from MuScatMicroscopeSim import MuScatMicroscopeSim
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import time
import numpy as np
import matplotlib.pyplot as plt
from pyunlocbox import functions

tf.config.set_visible_devices([], 'GPU')
start = time.time()
parameters = MuScatParameters(0.650,        # wavelength in vacuum
                              [50, 32, 32], # gridSize [z, x ,y]
                              0.3,          # dx
                              0.3,          # dy
                              0.3,          # dz
                              1.5,          # refractive index in medium
                              0.5,          # NAc
                              0.95)          # NAo
regLambdaL1 = 0.0001
regLambdaL2 = 0.0#00001
learnRate = 0.01
imagedObject = MuScatObject(parameters)
imagedObject.GenerateTwoBeads(2., 8., 1.54)

imagingSim = MuScatMicroscopeSim(parameters)
refShifts =tf.cast(tf.constant([[0.,0.]]), tf.float32)
imagingSim.Illumination()
imagingSim.Detection()
zPositions = imagedObject.realzzz[:,0,0] - imagedObject.realzzz[0,0,0]
zStackMeasured = imagingSim.CCHMImaging(imagedObject, zPositions, refShifts)

print(time.time()-start)

optimizedObject = MuScatObject(parameters)
regularizer = tf.keras.regularizers.L1L2(regLambdaL1, regLambdaL2)
#opt = tf.keras.optimizers.Adamax(learning_rate=learnRate)
opt = tfa.optimizers.Yogi(learning_rate=learnRate,
                                    l1_regularization_strength = regLambdaL1,
                                    l2_regularization_strength = regLambdaL2)

def regTV(variable, parameter):
    diffZ = (variable - tf.roll(variable,1,0))
    diffX = (variable - tf.roll(variable,1,1))
    diffY = (variable - tf.roll(variable,1,2))
    return parameter * tf.reduce_sum(tf.sqrt(diffZ**2 + diffX**2 + diffY**2 + \
                                             tf.keras.backend.epsilon()))
@tf.function
def loss_fn(zStackSimulated, zStackMeasured):
    return tf.reduce_mean(tf.abs(zStackMeasured - zStackSimulated)**2) #+\
        #regularizer(optimizedObject.RIDistrib) #+ regTV(optimizedObject.RIDistrib, 0.0005)
        
def loss_fn_withReg(zStackSimulated, zStackMeasured):
    return tf.reduce_mean(tf.abs(zStackMeasured - zStackSimulated)**2) +\
        regularizer(optimizedObject.RIDistrib)   
        
loss = lambda: loss_fn(imagingSim.CCHMImaging(optimizedObject, zPositions, refShifts),
                      zStackMeasured)
lossF=[]
param_t = tf.constant(1.)
TVnorm = functions.norm_tv(maxit=50, dim=3)
TVnorm.verbosity = 'NONE'
gradStep = 2
treshLambda = 0.000005
current_loss_withReg = loss_fn_withReg(imagingSim.CCHMImaging(optimizedObject,
                                                              zPositions, 
                                                              refShifts),
                                        zStackMeasured)
previous_RIDistrib = tf.convert_to_tensor(optimizedObject.RIDistrib)
for i in range(100):
    previous_param_t = param_t
    previous_loss_withReg = current_loss_withReg
    previous_RIDistrib = tf.convert_to_tensor(optimizedObject.RIDistrib)
    with tf.GradientTape() as t:
        current_loss = loss_fn(imagingSim.CCHMImaging(optimizedObject, zPositions, refShifts),
                          zStackMeasured)
    dRIDistrib = t.gradient(current_loss, optimizedObject.RIDistrib)
    proxStep0 = tfp.math.soft_threshold(
        optimizedObject.RIDistrib - (gradStep *dRIDistrib),
        treshLambda)
    #descentStep = optimizedObject.RIDistrib - (gradStep *dRIDistrib)
    proxStep = TVnorm.prox(proxStep0.numpy(), 0.00005)
    optimizedObject.RIDistrib.assign(proxStep)
    
    current_loss_withReg =  loss_fn_withReg(imagingSim.CCHMImaging(optimizedObject,
                                                                  zPositions, 
                                                                  refShifts),
                                            zStackMeasured)
    if current_loss_withReg > previous_loss_withReg:
        optimizedObject.RIDistrib.assign(previous_RIDistrib)
        
    param_t = (1. + tf.sqrt(1. + 4. * previous_param_t **2)) / 2
    optimizedObject.RIDistrib.assign(
        optimizedObject.RIDistrib + (previous_param_t / param_t) * \
            (proxStep - optimizedObject.RIDistrib) + (previous_param_t - 1.) / \
                param_t * (optimizedObject.RIDistrib - previous_RIDistrib)
        )
    if i%5==0:
        plt.figure(1)
        plt.subplot(121),plt.colorbar(plt.imshow(np.abs(optimizedObject.RIDistrib[:, :, 16]))),plt.clim(0,0.04)
        plt.subplot(122),plt.colorbar(plt.imshow(np.abs(imagedObject.RIDistrib[:, :, 16]))),plt.clim(0,0.04)
        plt.show()
print(time.time()-start)