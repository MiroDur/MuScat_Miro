# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:19:28 2020

@author: Miro
"""

import tensorflow as tf
import numpy as np


class MuScatField(tf.keras.Model):
    def __init__(self, field, parameters, **kwargs):
        super().__init__(**kwargs)
        # initilize parameters from the object parameters
        self.lambda0 = parameters.lambda0
        self.gridSize = parameters.gridSize
        self.dx = parameters.dx
        self.dy = parameters.dy
        self.dz = parameters.dz
        self.ComputeGrids()
        self.field = field

    def __call__(self, *args, **kwargs):

        return

    def ComputeGrids(self):
        # x is in pixels
        # realx is in um (the units of lambda and dx)
        # Kx is in um^-1
        self.x = tf.cast(tf.linspace(
            -self.gridSize[1]/2, self.gridSize[1]/2-1, self.gridSize[1]),
            tf.float32)
        self.y = tf.cast(tf.linspace(
            -self.gridSize[2]/2, self.gridSize[2]/2-1, self.gridSize[2]),
            tf.float32)
        self.z = tf.cast(tf.linspace(
            -self.gridSize[0]/2, self.gridSize[0]/2-1, self.gridSize[0]),
            tf.float32)

        self.zzz, self.xxx, self.yyy = tf.meshgrid(self.z, self.x, self.y,
                                                   indexing='ij')

        self.realxxx = self.xxx * self.dx
        self.realyyy = self.yyy * self.dy
        self.realzzz = self.zzz * self.dz

        self.xx, self.yy = tf.meshgrid(self.x, self.y, indexing='ij')
        self.realxx = self.xx * self.dx
        self.realyy = self.yy * self.dy

        self.Kxx = self.xx / self.gridSize[1] / self.dx
        self.Kyy = self.yy / self.gridSize[2] / self.dy

    def Propagate(self, MuScatObject, distance):
        propagator = tf.reshape(tf.signal.ifftshift(tf.exp(
            tf.complex(tf.cast(0., tf.float32),
                       2 * np.pi * MuScatObject.KzzM * distance))),
            [1, self.gridSize[1], self.gridSize[2]])
        self.field = tf.signal.ifft2d(tf.signal.fft2d(self.field) * propagator)
        return self.field

    def Refract(self, RIDistribLayer, MuScatObject):
        self.field = self.field * tf.reshape(tf.exp(tf.complex(
            tf.cast(0., tf.float32),
            2 * np.pi / self.lambda0 * MuScatObject.dz * RIDistribLayer)),
            [1, self.gridSize[1], self.gridSize[2]])
        return self.field

    def ConvolveWithGreen(self, RIDistribLayer, MuScatObject):
        scatteringPotential = tf.reshape(
            (2 * np.pi / self.lambda0)**2 * (MuScatObject.refrIndexM**2 -
                                             (MuScatObject.refrIndexM +
                                              RIDistribLayer)**2),
            [1, self.gridSize[1], self.gridSize[2]])

        return tf.signal.ifft2d(self.GreensFuncFFT * tf.signal.fft2d(
            self.field * tf.complex(scatteringPotential *
                                    MuScatObject.dz, 0.)))

    def MultipleScatteringMS(self, MuScatObject):
        for layer in range(MuScatObject.gridSize[0]):
            self.Refract(MuScatObject.RIDistrib[layer, :, :], MuScatObject)
            self.Propagate(MuScatObject, MuScatObject.dz)
        return self.field

    def MultipleScatteringMLB(self, MuScatObject):
        self.GreensFuncFFT = tf.reshape(
            tf.signal.ifftshift((-1j * tf.exp(tf.complex(
                tf.cast(0., tf.float32),
                2 * np.pi * MuScatObject.KzzM * MuScatObject.dz)) /
                tf.complex((4 * np.pi * MuScatObject.KzzM + 1e-07 ) + MuScatObject.mask,
                           0.)) * tf.complex(1 - MuScatObject.mask, 0.)),
            [1, self.gridSize[1], self.gridSize[2]])

        for layer in range(MuScatObject.gridSize[0]):
            self.field = self.ConvolveWithGreen(
                MuScatObject.RIDistrib[layer, :, :],
                MuScatObject) + self.Propagate(MuScatObject, MuScatObject.dz)
        return self.field
    
    def PinholeScattering(self, MuScatObject):
        
        pinhole = np.ones((self.gridSize[1],self.gridSize[2])) * 0.
        pinhole[np.int32(self.gridSize[1]/2), np.int32(self.gridSize[2]/2)] = 1.
        pinhole = tf.cast(pinhole, tf.complex64)
        
        for layer in range(MuScatObject.gridSize[0]):
            if layer == np.int32(self.gridSize[0]/2):
                self.field = self.field * tf.reshape(
                    pinhole, [1, self.gridSize[1], self.gridSize[2]])    
                print(np.abs(pinhole[16:18,16:18]))
            self.Propagate(MuScatObject, MuScatObject.dz)
        return self.field
    
    def SingleScatteringBorn(self, MuScatObject):
        scatteringPotential = tf.complex64((2 * np.pi / self.lambda0)**2 * (
            MuScatObject.refrIndexM**2 - (MuScatObject.refrIndexM + MuScatObject.RIDistrib)**2))
        scatteringFunction = tf.signal.fftshift(
            tf.signal.fft3d(scatteringPotential))
        incidentField = self.field
        scatteredField = tf.constant(tf.zeros(tf.shape(self.field),
                                              tf.complex64))
        for layer in range(MuScatObject.gridSize[0]):
            scatteredField = scatteredField + self.ConvolveWithGreen(
                MuScatObject.RIDistrib[layer, :, :],
                MuScatObject)
            
        self.field = incidentField    
        for layer in range(MuScatObject.gridSize[0]):
            incidentField = self.Propagate(MuScatObject, MuScatObject.dz)
            
        return incidentField + scatteredField

    def ComputeMuScat(self, MuScatObject, method='MLB'):
        if method == 'MLB':
            return self.MultipleScatteringMLB(MuScatObject)
        elif method == 'MS':
            return self.MultipleScatteringMS(MuScatObject)
        elif method == 'SingleBorn':
            return self.SingleScatteringBorn(MuScatObject)
        elif method == 'Pinhole':
            return self.PinholeScattering(MuScatObject)
