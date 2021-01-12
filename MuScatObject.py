# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:57:18 2020

@author: Miro
"""
import tensorflow as tf
# from phantominator import shepp_logan


class MuScatObject(tf.keras.Model):
    def __init__(self, parameters, **kwargs):
        super().__init__(**kwargs)
        # initilize parameters from the object parameters
        self.lambda0 = parameters.lambda0
        self.gridSize = parameters.gridSize
        self.dx = parameters.dx
        self.dy = parameters.dy
        self.dz = parameters.dz
        self.refrIndexM = parameters.refrIndexM
        self.lambdaM = self.lambda0 / self.refrIndexM
        self.ComputeGrids()

    def __call__(self, *args, **kwargs):

        return self

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
        self.KzzM = tf.sqrt(self.KzzSq * tf.cast(self.KzzSq >= 0., tf.float32))

        # mask used in MultiLayerBorn to not divide by 0
        self.mask = tf.cast(self.KzzM * self.dz <= 0., tf.float32)

        self.RIDistrib = tf.Variable(
            tf.zeros(self.gridSize, tf.float32),
            constraint=tf.keras.constraints.MinMaxNorm(min_value=-0.08,
                                                       max_value=0.08,
                                                       rate=1.0,
                                                       axis=0))

    def GenerateBead(self, radius, refrIndex):
        self.RIDistrib = tf.cast(tf.sqrt(
            tf.pow(self.realxxx, 2) + tf.pow(self.realyyy, 2) +
            tf.pow(self.realzzz, 2)) < radius, tf.float32) * \
            (refrIndex - self.refrIndexM)

    def GenerateBox(self, a, b, c, refrIndex):
        self.RIDistrib = (tf.cast(tf.abs(self.realxxx) < a/2, tf.float32) *
                          tf.cast(tf.abs(self.realyyy) < b/2, tf.float32) *
                          tf.cast(tf.abs(self.realzzz) < c/2, tf.float32) *
                          (refrIndex - self.refrIndexM))

    def GenerateCylinder(self, radius, height, refrIndex):
        self.RIDistrib = (tf.cast(
            tf.sqrt(self.realxxx**2 + self.realyyy**2) < radius,
            tf.float32) * tf.cast(tf.abs(self.realzzz) < height/2, tf.float32)
            * (refrIndex - self.refrIndexM))

    def GenerateTwoBeads(self, radius, separation, refrIndex):
        self.RIDistrib = (tf.cast(tf.sqrt(
            tf.pow(self.realxxx, 2) + tf.pow(self.realyyy, 2) +
            tf.pow(self.realzzz + separation / 2, 2)) < radius, tf.float32) +
            tf.cast(tf.sqrt(tf.pow(self.realxxx, 2) + tf.pow(self.realyyy, 2) +
                            tf.pow(self.realzzz - separation / 2, 2)) < radius,
                    tf.float32)) * (refrIndex - self.refrIndexM)

    # def GenerateSheppLogan(self, refrIndex):
    #     self.RIDistrib = tf.cast(tf.constant(shepp_logan(self.gridSize) * \
    #                                  (refrIndex - self.refrIndexM)), tf.float32)
