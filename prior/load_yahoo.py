# -*- coding: utf-8 -*-

import numpy as np
import os
import shutil

class load_yahoo:
    def __init__(self, datapath):
        self.datapath = datapath
        self.path_origin = datapath.split("/") 
        self.writepath = self.path_origin[0]+'/data'
        self.user_id = 0
        self.path_id = 0
        self.users = {}
        self.path = {} # self.user2arms={}
        
        
        self.nusers = 0
        self.nbfeatures = 6
        try:
            shutil.rmtree(self.writepath)
            os.makedirs(self.writepath)
        except:
            os.makedirs(self.writepath)
        
        with open(self.datapath, 'r') as f:
            for line in f:
                line = line.strip('\n')
                data = line.split("|")
                if data[1] not in self.users:
                    self.users.update({data[1]:self.user_id})
                    os.makedirs(self.writepath+'/'+str(self.user_id))
                    self.user_id = self.user_id + 1
                    
                self.path_id = self.users[data[1]]
                if self.path_id not in self.path:
                    self.path.update({self.path_id:0})   
                    
                recommend_info = data[0].split(" ")
                article_id = recommend_info[1]
                reward = recommend_info[2]
                data[0] = article_id
                data[1] = reward
                
                with open(self.writepath+'/'+str(self.path_id)+'/'+str(self.path[self.path_id]), 'w') as outfile: 
                    outfile.writelines('\n'.join(data))
                outfile.close()
                self.path[self.path_id] = self.path[self.path_id] + 1
                if self.path_id > 20000:
                    break
        f.close()
        
        with open(self.path_origin[0]+'/'+'users','w') as outfile:
            for key, value in self.users.items():
                outfile.write('%s\t%s\n' %(value, key))
        outfile.close()
        
        with open(self.path_origin[0]+'/'+'arms','w') as outfile:
            for key, value in self.path.items():                
                outfile.write('%s %s\n' %(key, value+1))
        outfile.close()
        
        self.nusers = len(self.path)