# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:42:53 2020

@author: Miro
"""
import tensorflow as tf
import numpy as np
from MuScatFieldLOOP import MuScatField
class MuScatMicroscopeSim(tf.keras.Model):
    def __init__(self, parameters, **kwargs):
        super().__init__(**kwargs)
        # initilize parameters from the object parameters
        self.lambda0 = parameters.lambda0
        self.NAo = parameters.NAo
        if parameters.NAc < parameters.NAo: self.NAc = parameters.NAc
        else: self.NAc = parameters.NAo
        self.gridSize = parameters.gridSize
        self.dx = parameters.dx
        self.dy = parameters.dy
        self.dz = parameters.dz
        self.refrIndexM = parameters.refrIndexM
        self.lambdaM = self.lambda0 / self.refrIndexM
        self.ComputeGrids()
        self.parameters = parameters
        

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
        self.KzzSq = tf.cast(1 / tf.pow(self.lambdaM, 2), tf.float32) \
                             - tf.pow(self.Kxx, 2) - tf.pow(self.Kyy, 2)
        self.KzzM = tf.sqrt(self.KzzSq * tf.cast(self.KzzSq >= 0, tf.float32)) 
        
        
    def Illumination(self):
        self.condenserPupil = tf.cast(
            (self.lambda0 * tf.sqrt(self.Kxx**2 + self.Kyy**2)) \
                < self.NAc, tf.float32)
            
        # calculate spatial frequencies of illumination plane waves
        self.KxxIllum = self.Kxx * self.condenserPupil
        self.KyyIllum = self.Kyy * self.condenserPupil
        
        self.KIllum = tf.stack(
            [self.KzzM[tf.not_equal(self.condenserPupil, 0)],
             self.KxxIllum[tf.not_equal(self.condenserPupil, 0)],
             self.KyyIllum[tf.not_equal(self.condenserPupil, 0)]], 1)
        #self.Kzillum = tf.reshape(
        #    self.KzzM[tf.not_equal(self.condenserPupil, 0)], [-1, 1, 1])
        
        self.planeWavesNum = self.KIllum.get_shape().as_list()[0]
        
    def Detection(self):
        # create objective pupil function
        self.objectivePupil = tf.cast(
            (self.lambda0 * tf.sqrt(self.Kxx**2 + self.Kyy**2)) \
                < self.NAo, tf.float32) 
            

    
    def FiltByObjectivePupil(self, field):
        return tf.signal.ifft2d(tf.signal.fft2d(field) * \
            tf.signal.ifftshift(tf.cast(self.objectivePupil, tf.complex64)) 
            )
    
    def CCHMImaging(self, MuScatObject, zPositions, refShifts):
        
        # [illum plane wave, refShift, zPos, xCoor, yCoor]
        # first compute BPM propagation by MultipleScattering func
        # second filter scattered field by objective pupil
        fields = tf.TensorArray(tf.complex64, size=self.planeWavesNum)
        fieldObj = MuScatField(tf.zeros(self.gridSize), self.parameters)
        for refShift in refShifts:
            zStack = tf.TensorArray(tf.complex64, size=zPositions.get_shape().as_list()[0])
            j = 0
            for zPos in zPositions:
                fields = tf.TensorArray(tf.complex64, size=self.planeWavesNum)
                for i in range(self.planeWavesNum):
                    fieldObj.field = tf.exp(tf.complex(
                        0.,
                        2 * np.pi * (self.KIllum[i, 1] * self.realxx + \
                                     self.KIllum[i, 2] * self.realyy - \
                                         self.KIllum[i, 0] * zPos)))
                    fieldObj.ComputeMuScat(MuScatObject, 'MLB')
                    
                    
                    # reference waves are illumination plane waves at z=0
                    
                       
                    fieldObj.Propagate(MuScatObject, -(self.gridSize[0] * self.dz - zPos))   
                     
                    FiltScatteredField = self.FiltByObjectivePupil(fieldObj.field)
                    # returns summed up interference of scattered fields and 
                    # reference waves
                    referenceWave = tf.exp(tf.complex(
                        0.,
                        -2 * np.pi * (self.KIllum[i, 1] * (self.realxx + refShift[0]) + \
                            self.KIllum[i, 2] * (self.realyy + refShift[1]))))
                    fields = fields.write(i, FiltScatteredField * referenceWave)
                                         
                    #0) / self.planeWavesNum        
                zStack = zStack.write(j, tf.reduce_sum(fields.stack(), 0) / self.planeWavesNum)
                j += 1         
        return zStack.stack()
    
        # def PropagateField(self, field, distance, MuScatObject):
    #     propagator = tf.reshape(tf.signal.ifftshift(tf.exp(tf.complex(
    #         tf.cast(0., tf.float32), 2 * np.pi * MuScatObject.KzzM * distance))),
    #         [1, self.gridSize[1], self.gridSize[2]])
    #     return tf.signal.ifft2d(tf.signal.fft2d(field) * propagator)
    
    # def RefractField(self, field, RIDistribLayer, MuScatObject):
    #     return field * tf.reshape(tf.exp(tf.complex(tf.cast(0., tf.float32),
    #                                      2 * np.pi / self.lambda0 * MuScatObject.dz * \
    #                                          RIDistribLayer)), 
    #                               [1, self.gridSize[1], self.gridSize[2]])
            
    # def ConvolveWithGreen(self, field, RIDistribLayer, MuScatObject):
    #     GreensFuncFFT = tf.reshape(tf.signal.ifftshift(-1j*tf.exp(tf.complex(
    #         tf.cast(0., tf.float32), 2 * np.pi * MuScatObject.KzzM * MuScatObject.dz) + \
    #             tf.cast(MuScatObject.KzzSq < 0, tf.complex64)*1e-07) / \
    #             tf.complex((4 * np.pi * MuScatObject.KzzM) + tf.cast(MuScatObject.KzzSq < 0,
    #                                                          tf.float32), 0.)),
    #             [1, self.gridSize[1], self.gridSize[2]])
            
    #     scatteringPotential =  tf.reshape((2 * np.pi / self.lambda0)**2 * \
    #         (MuScatObject.refrIndexM**2 - (MuScatObject.refrIndexM + RIDistribLayer)**2), 
    #         [1, self.gridSize[1], self.gridSize[2]])
                
    #     return tf.signal.ifft2d(GreensFuncFFT * tf.signal.fft2d(
    #         field * tf.complex(scatteringPotential * MuScatObject.dz, 0.)))
        
    # def MultipleScatteringMS(self, MuScatObject, illumination):
    #     scatteredField = illumination
    #     for layer in range(MuScatObject.gridSize[0]):
    #         scatteredField = self.RefractField(scatteredField,
    #                                            MuScatObject.RIDistrib[layer, :, :],
    #                                            MuScatObject)
    #         scatteredField = self.PropagateField(scatteredField,
    #                                              MuScatObject.dz,
    #                                              MuScatObject)
    #     return scatteredField
    
    # def MultipleScatteringMLB(self, MuScatObject, illumination):
    #     scatteredField = illumination
    #     for layer in range(MuScatObject.gridSize[0]):
    #         scatteredField = self.PropagateField(scatteredField,
    #                                              MuScatObject.dz,
    #                                              MuScatObject) + \
    #             self.ConvolveWithGreen(scatteredField,
    #                                    MuScatObject.RIDistrib[layer, :, :],
    #                                    MuScatObject)
    #     return scatteredField
        
    # def ComputeScatteredField(self, MuScatObject, illumination):
    #     self.scatteredField = self.MultipleScatteringMLB(MuScatObject,
    #                                                      illumination)
    #     return self.scatteredField