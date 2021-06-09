# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:36:57 2020

@author: Miro
"""


class MuScatParameters:
    def __init__(self, lambda0=0.650, gridSize=[64, 64, 64], dx=0.3, dy=0.3,
                 dz=0.3, refrIndexM=1.5, NAc=0.3, NAo=0.5, 
                 ZernikeCoefficients=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]):
        self.lambda0 = lambda0
        self.gridSize = gridSize
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.refrIndexM = refrIndexM
        self.NAc = NAc
        self.NAo = NAo
        self.ZernikeCoefficients = ZernikeCoefficients
