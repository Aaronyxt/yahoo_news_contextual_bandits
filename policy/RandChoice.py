# -*- coding: utf-8 -*-

import numpy as np
import random as rand
from policy.IndexPolicy import IndexPolicy

class RandChoice(IndexPolicy):

    def __init__(self, nbArms, nbFeatures):
        self.nbArms = nbArms
        self.nbFeatures = nbFeatures        

    def startGame(self, rounds, reductionDim):
        self.t = 0
        self.newFeatures = reductionDim
        
    def getReward(self, armid, arm, reward):   
        pass
    
    def computeIndex(self, armid, arm):
        result = rand.random()
        return result