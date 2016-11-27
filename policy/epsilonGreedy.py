# -*- coding: utf-8 -*-

import numpy as np
import random as rand
from policy.IndexPolicy import IndexPolicy

class epsilonGreedy(IndexPolicy):

    def __init__(self, nbArms, nbFeatures):
        self.nbArms = nbArms
        self.nbFeatures = nbFeatures 
        self.epsilon = 0.2
        self.arms_set = {}

    def startGame(self, rounds, reductionDim):
        self.t = 0
        self.newFeatures = reductionDim
        self.randomValue = rand.random()
        
        
    def getReward(self, armid, arm, reward):
        if armid not in self.arms_set:
            self.arms_set.update({armid:[reward, 1.0, 0.0]})
        else:
            self.arms_set[armid][0] = self.arms_set[armid][0] + reward
            self.arms_set[armid][1] = self.arms_set[armid][1] + 1.0
        self.arms_set[armid][2] = self.arms_set[armid][0]/self.arms_set[armid][1]
        self.randomValue = rand.random()
        
    def computeIndex(self, armid, arm):
        if not bool(self.arms_set):
            if armid == 0:
                result = 1.0
            else:
                result = 0.0
            return result
        if len(self.arms_set) < self.nbArms:
            if armid == (len(self.arms_set)):
                result = 1.0
            else:
                result = 0.0
            return result
            
        maxIndex = max(self.arms_set.values())
                
        if self.randomValue <= self.epsilon:
            if self.arms_set[armid] == maxIndex:
                result = -0.1                
            else:
                result = rand.random()
        else:
            if self.arms_set[armid] == maxIndex:
                result = 1.0
            else:
                result = 0.0
        return result