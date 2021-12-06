# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:54:25 2020

@author: Miro
"""
from MuScatObject import MuScatObject
from MuScatField import MuScatField
from MuScatParameters import MuScatParameters
from MuScatMicroscopeSim import MuScatMicroscopeSim
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import time
import numpy as np
tf.config.set_visible_devices([], 'GPU')
# %%
start = time.time()
parameters = MuScatParameters(lambda0=0.650,         # wavelength in vacuum
                              gridSize=[40, 64, 64],  # gridSize [z, x ,y]
                              dx=0.2,          # dx
                              dy=0.2,          # dy
                              dz=0.3,           # dz
                              refrIndexM=1.5,           # refr index in medium
                              NAc=0.7,           # NAc
                              NAo=1.0)           # NAo
                                
centerInd = np.int32(parameters.gridSize[1] / 2)
regLambdaL1 = 0.0001
regLambdaL2 = 0.000
learnRate = 0.01
refShifts = tf.cast(tf.constant([[0.0, 0.0]]), tf.float32)
imagedObject = MuScatObject(parameters)
#imagedObject.GenerateBox(3., 3., 3., 1.505)
imagedObject.GenerateBead(1.5, 1.51)
# imagedObject.GenerateSheppLogan(1.55)

imagingSim = MuScatMicroscopeSim(parameters)

imagingSim.Illumination(HollowCone=0.0, sampling=1, shift=[0,1])
imagingSim.Detection()
zernCoef = tf.Variable([[0., 0., 0., 0., 0., 0., -0.0, -0.0, 0., 0., 0.]], tf.float32)
imagingSim.ZernikeCoefficients = zernCoef;
zPositions = imagedObject.realzzz[::1, 0, 0] - imagedObject.realzzz[0, 0, 0]
fieldMeasured = MuScatField(imagingSim.planeWaves, parameters).ComputeMuScat(
    imagedObject, method='MLB')
# fieldMeasured.ComputeMuScat(imagedObject, method='MLB')
zStackMeasured = imagingSim.CCHMImaging(fieldMeasured,
                                        zPositions,
                                        refShifts)

plt.figure(11)
plt.subplot(121),plt.colorbar(plt.imshow(
    np.angle(zStackMeasured[0, :, centerInd, :]),
    aspect=parameters.dz/parameters.dx))
plt.title('Measured phase')
plt.subplot(122),plt.colorbar(plt.imshow(
    np.abs(zStackMeasured[0, :, centerInd, :]),
    aspect=parameters.dz/parameters.dx))
plt.title('Measured amplitude')
plt.show()

imagingSim.Compute3DCTF(refShifts)
deconvPot = imagingSim.CCHMdeconvolution(zStackMeasured[0,:,:,:],1e-2)
plt.figure(13)
plt.subplot(121),plt.colorbar(plt.imshow(
    np.angle(deconvPot[:, centerInd, :]),
    aspect=parameters.dz/parameters.dx))
plt.title('Deconv Potential phase')
plt.subplot(122),plt.colorbar(plt.imshow(
    np.abs(deconvPot[ :, centerInd, :]),
    aspect=parameters.dz/parameters.dx))
plt.title('Deconv Potential amplitude')
plt.show()

plt.figure(12)
plt.figure(6),plt.colorbar(plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fftn(np.squeeze(zStackMeasured[0, :, centerInd, :])))))))
print(time.time()-start)
# %%
optimizedObject = MuScatObject(parameters)
regularizer = tf.keras.regularizers.L1L2(regLambdaL1, regLambdaL2)
# opt = tf.keras.optimizers.Adamax(learning_rate=learnRate)
opt = tfa.optimizers.Yogi(learning_rate=learnRate,
                          l1_regularization_strength=regLambdaL1,
                          l2_regularization_strength=regLambdaL2,)


def regTV(variable, parameter):
    diffZ = (2*variable - tf.roll(variable, 1, 0) - tf.roll(variable, -1, 0))/2
    diffX = (2*variable - tf.roll(variable, 1, 1) - tf.roll(variable, -1, 1))/2
    diffY = (2*variable - tf.roll(variable, 1, 2) - tf.roll(variable, -1, 2))/2
    return parameter * tf.reduce_sum(tf.abs(diffX) + tf.abs(diffY) +
                                     tf.abs(diffZ))


@tf.function
def loss_fn(zStackSimulated, zStackMeasured):
    return tf.reduce_mean(tf.abs(zStackMeasured - zStackSimulated)**2) #+\
        #regularizer(optimizedObject.RIDistrib)

# scatteredField = MuScatField(imagingSim.planeWaves, parameters).ComputeMuScat(optimizedObject, method='MLB')


loss = lambda: loss_fn(imagingSim.CCHMImaging(
    MuScatField(imagingSim.planeWaves, parameters).ComputeMuScat(
        optimizedObject, method='MLB'), zPositions, refShifts), zStackMeasured)
lossF = []

for i in range(0):
    opt_op = opt.minimize(loss, var_list=[optimizedObject.RIDistrib])
    plt.figure(1)
    plt.subplot(121), plt.colorbar(plt.imshow(
        optimizedObject.RIDistrib[:, centerInd, :],
        aspect=parameters.dz/parameters.dx))
    plt.subplot(122), plt.colorbar(plt.imshow(
        np.angle(zStackMeasured[0, :, centerInd, :]),
        aspect=parameters.dz/parameters.dx))
    plt.tight_layout(pad=3.0)
    plt.show()
    #lossF.append(loss_fn(imagingSim.CCHMImaging(imagingSim.ComputeScatteredField(optimizedObject), zPositions, refShifts),
    #                  zStackMeasured).numpy)
    print(time.time()-start)
