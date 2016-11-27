# -*- coding: utf-8 -*-


class load_arms:
    def __init__(self, datapath):
        self.datapath = datapath
        self.arm2features=None
        self.user2arms={}
        self.nbfeatures = 6
        self.nbusers = 1
        self.nbarms = 20
        with open(self.datapath, 'r') as f:
            for line in f:
                lstr = line.strip().split()
                self.user2arms.update({int(lstr[0]):int(lstr[1])})