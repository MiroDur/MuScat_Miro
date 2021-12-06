# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:42:53 2020

@author: Miro
"""
import tensorflow as tf
import numpy as np
import zernike1 as zern

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
        self.ComputeZernike()
        self.ZernikeCoefficients = parameters.ZernikeCoefficients
        
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

    def Illumination(self, HollowCone=0., sampling=1, shift=[0,0]):
        self.illumSampling = sampling
        self.condenserPupil = tf.cast(
            ((self.lambda0 * tf.sqrt(self.Kxx**2 + self.Kyy**2)) < self.NAc) &
            ((self.lambda0 * tf.sqrt(self.Kxx**2 + self.Kyy**2)) >= HollowCone),
            tf.float32)
        self.pupilShift = shift
        self.condenserPupilShifted = tf.roll(self.condenserPupil,self.pupilShift,[0,1])
        self.ZernikePolynomial = tf.roll(self.ZernikePolynomial,self.pupilShift,[1,2])
        # calculate spatial frequencies of illumination plane waves
        self.KxxIllum = self.Kxx[::sampling, ::sampling] * \
            self.condenserPupil[::sampling, ::sampling]
        self.KyyIllum = self.Kyy[::sampling, ::sampling] * \
            self.condenserPupil[::sampling, ::sampling]

        self.KtIllum = tf.stack(
            [self.KxxIllum[tf.not_equal(
                self.condenserPupil[::sampling, ::sampling], 0.)],
             self.KyyIllum[tf.not_equal(
                 self.condenserPupil[::sampling, ::sampling], 0.)]], 1)
        KzzMsampled = self.KzzM[::sampling, ::sampling]
        self.Kzillum = tf.reshape(KzzMsampled[tf.not_equal(
            self.condenserPupil[::sampling, ::sampling], 0.)], [-1, 1, 1])
        
        # Apodization - Debye approximation + sine condition
        
        self.apodization2D = tf.sqrt(self.lambdaM * KzzMsampled + 1e-7)
        
        self.apodizationNotNull = self.apodization2D[tf.not_equal(
            self.condenserPupil[:: self.illumSampling,
                                :: self.illumSampling], 0.)]
        self.apodization2D = tf.roll(self.apodization2D,self.pupilShift,[0,1])

        # calculate illumination plane waves
        self.planeWaves = tf.exp(
            tf.complex(tf.cast(0., tf.float32), 2 * np.pi * (
                tf.reshape(self.KtIllum[:, 0], [-1, 1, 1]) * self.realxx +
                tf.reshape(self.KtIllum[:, 1], [-1, 1, 1]) * self.realyy)))

        self.planeWavesNum = self.planeWaves.get_shape().as_list()[0]
        
        self.illumIntensity = tf.cast(tf.reduce_sum(tf.math.abs(
            self.apodizationNotNull)**4, axis=None), tf.complex64)
        
    def Detection(self):
        # create objective pupil function
        self.objectivePupil = tf.cast(
            (self.lambda0 * tf.sqrt(self.Kxx**2 + self.Kyy**2)) <
            self.NAo, tf.float32)
        # Apodization - Debye approximation + sine condition
        
        self.apodizationObj2D = tf.sqrt(self.lambdaM * self.KzzM + 1e-7)
        self.apodizationObjNotNull = self.apodizationObj2D[tf.not_equal(
                self.objectivePupil, 0.)]
        
        self.objectivePupilApod = self.objectivePupil * self.apodizationObj2D
        
    def ComputeZernike(self):
        self.xxPupil = self.lambda0 * self.Kxx / self.NAc
        self.yyPupil = self.lambda0 * self.Kyy / self.NAc
        r, theta = zern.cart2pol(self.yyPupil, self.xxPupil)
        self.ZernikePolynomial = []     
        for i in range(1,12):
            self.ZernikePolynomial.append(zern.zernike(r, theta, i))
        self.ZernikePolynomial = tf.constant(self.ZernikePolynomial,
                                             tf.float32)

        
    def FiltByObjectivePupil(self, field):
        return tf.signal.ifft2d(tf.signal.fft2d(field) * tf.reshape(
            tf.signal.ifftshift(tf.cast(self.objectivePupilApod, tf.complex64)),
            [1, self.gridSize[1], self.gridSize[2]]))

    def Compute3DCTF(self, refShifts):
        
        # CCHM apodization due to HIGH NA (Debye approx + sine condition)
        self.apodizationCCHM = tf.complex(self.apodization2D**3 ,
            tf.cast(0., tf.float32)) * tf.reduce_prod(tf.exp(tf.complex(tf.cast(0., tf.float32), self.ZernikePolynomial * tf.reshape(self.ZernikeCoefficients, [-1, 1 ,1]))), 0) * \
            tf.complex(self.condenserPupilShifted, tf.cast(0., tf.float32))
        self.apodizationCCHMNotNull = self.apodizationCCHM[tf.not_equal(
            self.condenserPupilShifted[:: self.illumSampling,
                                :: self.illumSampling], 0.)]
        self.intensity = tf.reduce_sum(tf.squeeze(self.apodizationCCHMNotNull) * tf.cast(self.apodization2D[tf.not_equal(
            self.condenserPupilShifted, 0.)], tf.complex64))
        self.SPupil = tf.reverse(tf.math.conj(
                tf.reshape(self.apodizationCCHM, [1, self.gridSize[1], self.gridSize[2]]) * tf.exp(tf.complex(tf.cast(0., tf.float32), 2 * np.pi * tf.reshape(-tf.reshape(self.Kxx * refShifts[0, 0] + self.Kyy * refShifts[0,1], [1, self.gridSize[1], self.gridSize[2]]) + self.KzzM, [1, self.gridSize[1], self.gridSize[2]]) * tf.reshape(self.realzzz[:,0,0], [-1, 1, 1])))),[1,2])
        self.PPupil = tf.math.conj(tf.complex(tf.reshape(self.objectivePupil, [1, self.gridSize[1], self.gridSize[2]]), tf.cast(0., tf.float32)) * tf.exp(tf.complex(tf.cast(0., tf.float32), -2 * np.pi * tf.reshape(self.KzzM, [1, self.gridSize[1], self.gridSize[2]]) * tf.reshape(self.realzzz[:,0,0], [-1, 1, 1]))))
        self.Sfunc = tf.signal.fft2d(tf.signal.fftshift(self.SPupil))#/ tf.cast(tf.reduce_sum(self.apodizationNotNull), tf.complex64)#/(self.gridSize[1]*self.gridSize[2])
        self.Pfunc = tf.signal.fft2d(tf.signal.fftshift(self.PPupil))#/ tf.cast(tf.reduce_sum(self.apodizationObjNotNull), tf.complex64)#/(self.gridSize[1]*self.gridSize[2])
        self.CTF_2D = tf.signal.ifftshift(tf.signal.ifft2d(self.Sfunc * self.Pfunc)) /self.intensity#* (self.gridSize[1]*self.gridSize[2])
        self.invKzzM = tf.complex(tf.reshape(1/(self.KzzM+1e-7)* tf.cast(self.KzzSq >= 0, tf.float32), [1, self.gridSize[1], self.gridSize[2]]),tf.cast(0., tf.float32))
        self.CTF_3D = 1j*self.invKzzM*tf.transpose(tf.signal.fftshift(tf.signal.fft(tf.signal.fftshift(tf.transpose(self.CTF_2D, perm=[1,2,0]), axes=2)), axes=2),[2,0,1]) /self.gridSize[0] #/self.intensity #/self.gridSize[0] #
    
    def CCHMconvolution(self, MuScatObject):
        """
        Generate images at defined Z positions for each reference obj. shift.

        Indexing order: [refShift, zPos, xCoor, yCoor]

        Parameters
        ----------
        MuScatObject : Contains 3D RI distribution

        Returns
        -------
        zStack : Complex64 4D-Tensor
            [refShift, zPos, xCoor, yCoor]

        """
        #1/ self.lambda0 * MuScatObject.refrIndexM
        #self.deltaFuncz = np.zeros(self.gridSize)
        #self.deltaFuncz[np.int32(self.gridSize[0]/2),:,:] = 1.
        #self.deltaFuncz = tf.constant(self.deltaFuncz, tf.complex64)/self.dz
        
        
        self.scatteringPotential = tf.complex(np.pi*(1/ self.lambda0)**2 * ( 
            -self.refrIndexM**2 + (self.refrIndexM + MuScatObject.RIDistrib)**2),tf.cast(0., tf.float32))
        self.scatteringFunction = tf.signal.fftshift(
            tf.signal.fft3d(tf.signal.fftshift(self.scatteringPotential)))*self.gridSize[0]*self.dz #/(self.gridSize[0]*self.gridSize[1]*self.gridSize[2])*(self.dx*self.dy*self.dz) 
        #self.scatteredFieldBorn =  tf.signal.ifftshift(tf.signal.ifft3d(tf.signal.ifftshift(self.scatteringFunction)))
        #self.scatteringFunctionCCHM = tf.signal.fftshift(
        #    tf.signal.fft3d(tf.signal.fftshift(self.scatteredFieldBorn)))
        
        #tf.transpose(tf.signal.fftshift(tf.signal.fft(tf.signal.fftshift(tf.transpose(self.CTF_2D, perm=[1,2,0]), axes=2)), axes=2),[2,0,1])
        self.convPotential = tf.signal.fftshift(tf.signal.ifft3d(tf.signal.ifftshift(self.CTF_3D * self.scatteringFunction)))
        #self.deconv = tf.signal.fftshift(tf.signal.ifft3d(tf.signal.fft3d(tf.signal.ifftshift(self.convPotential))/tf.signal.ifftshift(((self.lambdaM)**2 *self.CTF_3D+1e-4))))
        return tf.complex(tf.cast(1.,tf.float32), tf.cast(0.,tf.float32))+self.convPotential#*(self.gridSize[0]*self.gridSize[1]*self.gridSize[2])
    
    def CCHMdeconvolution(self, zStack, regParam=1e-3):
        """
        Compute deconvolution of zStack with microscope 3D psf

        Parameters
        ----------
        zStack : Complex64 3D-Tensor
            Measurement or output of simulation

        Returns
        -------
        scattteringPotential: Complex64 3D-Tensor

        """
        # remove background from zStack
        self.scatteredField = zStack - tf.complex(tf.cast(1.,tf.float32), tf.cast(0.,tf.float32))
        
        # tikhonov regularization
        self.deconvPotential = tf.signal.fftshift(tf.signal.ifft3d(tf.math.conj(tf.signal.ifftshift(self.CTF_3D))*(tf.signal.fft3d(tf.signal.ifftshift(self.scatteredField))/self.gridSize[0]/self.dz)/(tf.signal.ifftshift(self.CTF_3D)*tf.math.conj(tf.signal.ifftshift(self.CTF_3D))+regParam)))
        
        return RIDiff
    
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
        
        # CCHM apodization due to HIGH NA (Debye approx + sine condition)
        self.apodizationCCHM = tf.complex(self.apodization2D**2 * self.apodizationObj2D,
            tf.cast(0., tf.float32)) * tf.reduce_prod(tf.exp(tf.complex(tf.cast(0., tf.float32), self.ZernikePolynomial * tf.reshape(self.ZernikeCoefficients, [-1, 1 ,1]))), 0)
        self.apodizationCCHMNotNull = self.apodizationCCHM[tf.not_equal(
            self.condenserPupil[:: self.illumSampling,
                                :: self.illumSampling], 0.)]
        self.apodizationCCHMNotNull = tf.reshape(
            self.apodizationCCHMNotNull, [-1, 1, 1, 1, 1])
        self.intensity = tf.reduce_sum(tf.squeeze(self.apodizationCCHMNotNull) * tf.cast(self.apodizationObj2D[tf.not_equal(
            self.condenserPupil, 0.)], tf.complex64))
        
        # reference waves are illumination plane waves at z=0
        # and transversally shifted in relation to illumination plane waves
        referenceWaves = tf.exp(tf.complex(tf.cast(0., tf.float32), 2 * np.pi *
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

        zStack = tf.reduce_sum(self.apodizationCCHMNotNull * \
                               propagatedFields * tf.math.conj(referenceWaves),
                               0) / self.intensity

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
        referenceWaves = self.planeWaves
        
         # CCHM apodization due to HIGH NA (Debye approx + sine condition)
        self.apodizationCCHM = self.apodization2D**2 * self.apodizationObj2D
        self.apodizationCCHMNotNull = self.apodizationCCHM[tf.not_equal(
            self.condenserPupil[:: self.illumSampling,
                                :: self.illumSampling], 0.)]
        self.apodizationCCHMNotNull = tf.complex(tf.reshape(
            self.apodizationCCHMNotNull, [-1, 1, 1]),
            tf.cast(0., tf.float32))
        # apply defocus related to the illumination of volumetric sample
        propagatedIllum = FiltScatteredField * tf.exp(tf.complex(
            tf.cast(0., tf.float32),
            -2 * np.pi * tf.reshape(self.Kzillum, [-1, 1, 1]) *
            tf.reshape(self.realzzz[-1,0,0], [-1, 1, 1])))

        # propagate all fields for each illumination plane wave
        # to z-stack positions
        propagator = tf.signal.ifftshift(tf.exp(tf.complex(
            tf.cast(0., tf.float32), 2 * np.pi * self.KzzM * (tf.reshape(-((self.gridSize[0]) * self.dz - self.realzzz[-1, 0, 0]), [-1, 1, 1])))), (1, 2))

        propagatedFields = tf.signal.ifft2d(
            tf.signal.fft2d(propagatedIllum) * propagator)

        # returns summed up interference of scattered fields and
        # reference waves

        TomoImages = self.apodizationCCHMNotNull * propagatedFields * tf.math.conj(referenceWaves)

        return TomoImages
