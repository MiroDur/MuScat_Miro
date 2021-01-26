# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:42:53 2020

@author: Miro
"""
import tensorflow as tf
import numpy as np


class MuScatMicroscopeSim(tf.keras.Model):

    def __init__(self, parameters, **kwargs):
        super().__init__(**kwargs)
        # initilize parameters from the object parameters
        self.lambda0 = parameters.lambda0
        self.NAo = parameters.NAo
        if parameters.NAc < parameters.NAo:
            self.NAc = parameters.NAc
        else:
            self.NAc = parameters.NAo
        self.gridSize = parameters.gridSize
        self.dx = parameters.dx
        self.dy = parameters.dy
        self.dz = parameters.dz
        self.refrIndexM = parameters.refrIndexM
        self.lambdaM = self.lambda0 / self.refrIndexM
        self.ComputeGrids()

    def __call__(self, MuScatObject, *args, **kwargs):

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

        # Kz in medium
        self.KzzSq = tf.cast(1 / tf.pow(self.lambdaM, 2), tf.float32) - \
            tf.pow(self.Kxx, 2) - tf.pow(self.Kyy, 2)
        self.KzzM = tf.sqrt(self.KzzSq * tf.cast(self.KzzSq >= 0, tf.float32))

    def Illumination(self):
        self.condenserPupil = tf.cast(
            (self.lambda0 * tf.sqrt(self.Kxx**2 + self.Kyy**2)) < self.NAc,
            tf.float32)

        # calculate spatial frequencies of illumination plane waves
        self.KxxIllum = self.Kxx * self.condenserPupil
        self.KyyIllum = self.Kyy * self.condenserPupil

        self.KtIllum = tf.stack(
            [self.KxxIllum[tf.not_equal(self.condenserPupil, 0)],
             self.KyyIllum[tf.not_equal(self.condenserPupil, 0)]], 1)
        self.Kzillum = tf.reshape(
            self.KzzM[tf.not_equal(self.condenserPupil, 0)], [-1, 1, 1])

        # calculate illumination plane waves
        self.planeWaves = tf.exp(
            tf.complex(tf.cast(0., tf.float32), 2 * np.pi * (
                tf.reshape(self.KtIllum[:, 0], [-1, 1, 1]) * self.realxx +
                tf.reshape(self.KtIllum[:, 1], [-1, 1, 1]) * self.realyy)))

        self.planeWavesNum = self.planeWaves.get_shape().as_list()[0]

    def Detection(self):
        # create objective pupil function
        self.objectivePupil = tf.cast(
            (self.lambda0 * tf.sqrt(self.Kxx**2 + self.Kyy**2)) <
            self.NAo, tf.float32)

    def FiltByObjectivePupil(self, field):
        return tf.signal.ifft2d(tf.signal.fft2d(field) * tf.reshape(
            tf.signal.ifftshift(tf.cast(self.objectivePupil, tf.complex64)),
            [1, self.gridSize[1], self.gridSize[2]]))

    def CCHMImaging(self, ScatteredField, zPositions, refShifts):
        """
        Generate images at defined Z positions for each reference obj. shift.

        Indexing order: [illum plane wave, refShift, zPos, xCoor, yCoor]

        Parameters
        ----------
        ScatteredField : Complex64 3D-Tensor
            Output of light propagation simulation
            Indexeing order: [illum plane wave, xCoor, yCoor]
        zPositions : Float32 1D-Tensor
            Focus positions relative to the top (illum side) of the sim. grid
        refShifts : Float32 2D-Tensor
            reference objective positions
            Standard imaging [refX, refY] = [0., 0.]

        Returns
        -------
        zStack : Complex64 4D-Tensor
            [refShift, zPos, xCoor, yCoor]

        """
        # filter scattered field by objective pupil
        FiltScatteredField = tf.reshape(
            self.FiltByObjectivePupil(ScatteredField),
            [-1, 1, 1, self.gridSize[1], self.gridSize[2]])

        # reference waves are illumination plane waves at z=0
        # and transversally shifted in relation to illumination plane waves
        referenceWaves = tf.exp(
            tf.complex(tf.cast(0., tf.float32), 2 * np.pi *
                       (tf.reshape(self.KtIllum[:, 0], [-1, 1, 1, 1, 1]) *
                        (tf.reshape(self.realxx, [1, 1, 1, self.gridSize[1],
                                                  self.gridSize[2]]) -
                         tf.reshape(refShifts[:, 0], [1, -1, 1, 1, 1])) +
                        tf.reshape(self.KtIllum[:, 1], [-1, 1, 1, 1, 1]) *
                        (tf.reshape(self.realyy, [1, 1, 1, self.gridSize[1],
                                                  self.gridSize[2]]) -
                         tf.reshape(refShifts[:, 1], [1, -1, 1, 1, 1])))))

        # apply defocus related to the illumination of volumetric sample
        propagatedIllum = FiltScatteredField * tf.exp(tf.complex(
            tf.cast(0., tf.float32),
            -2 * np.pi * tf.reshape(self.Kzillum, [-1, 1, 1, 1, 1]) *
            tf.reshape(zPositions, [1, 1, -1, 1, 1])))

        # propagate all fields for each illumination plane wave
        # to z-stack positions
        propagator = tf.signal.ifftshift(tf.exp(tf.complex(
            tf.cast(0., tf.float32), 2 * np.pi * self.KzzM * (
                -((self.gridSize[0]) * self.dz -
                  tf.reshape(zPositions, [1, 1, -1, 1, 1]))))), (3, 4))

        propagatedFields = tf.signal.ifft2d(
            tf.signal.fft2d(propagatedIllum) * propagator)

        # returns summed up interference of scattered fields and
        # reference waves

        zStack = tf.reduce_sum(propagatedFields * tf.math.conj(referenceWaves),
                               0) / self.planeWavesNum

        return zStack

    def TomoImaging(self, ScatteredField):
        """
        Generate images at defined Z positions for each reference obj. shift.

        Indexing order: [illum plane wave, xCoor, yCoor]

        Parameters
        ----------
        ScatteredField : Complex64 3D-Tensor
            Output of light propagation simulation
            Indexeing order: [illum plane wave, xCoor, yCoor]

        Returns
        -------
        zStack : Complex64 4D-Tensor
            [illum plane wave, xCoor, yCoor]

        """
        # filter scattered field by objective pupil
        FiltScatteredField = tf.reshape(
            self.FiltByObjectivePupil(ScatteredField),
            [-1, self.gridSize[1], self.gridSize[2]])

        # reference waves are illumination plane waves at z=0
        # and transversally shifted in relation to illumination plane waves
        referenceWaves = self.planeWaves

        # apply defocus related to the illumination of volumetric sample
        propagatedIllum = FiltScatteredField * tf.exp(tf.complex(
            tf.cast(0., tf.float32),
            -2 * np.pi * tf.reshape(self.Kzillum, [-1, 1, 1]) *
            tf.reshape(self.realzzz[0, 0, 0], [-1, 1, 1])))

        # propagate all fields for each illumination plane wave
        # to z-stack positions
        propagator = tf.signal.ifftshift(tf.exp(tf.complex(
            tf.cast(0., tf.float32), 2 * np.pi * self.KzzM * (
                -((self.gridSize[0]) * self.dz -
                  tf.reshape(0., [-1, 1, 1]))))), (1, 2))

        propagatedFields = tf.signal.ifft2d(
            tf.signal.fft2d(propagatedIllum) * propagator)

        # returns summed up interference of scattered fields and
        # reference waves

        TomoImages = propagatedFields * tf.math.conj(referenceWaves)

        return TomoImages
