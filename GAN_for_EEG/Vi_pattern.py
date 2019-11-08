# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 02:59:00 2019

@author: Nightmare
"""

import numpy as np
from PIL import Image
from numpy import newaxis as na

def vi_pattern():
      zhongyang=np.array([
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,1,1,1,1,0,0,0],
                  [0,0,0,1,1,1,1,0,0,0],
                  [0,0,0,1,1,1,1,0,0,0],
                  [0,0,0,1,1,1,1,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  ])
      sizhou=np.array([
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,1,1,1,1,1,1,1,1,0],
                  [0,1,1,1,1,1,1,1,1,0],
                  [0,1,1,0,0,0,0,1,1,0],
                  [0,1,1,0,0,0,0,1,1,0],
                  [0,1,1,0,0,0,0,1,1,0],
                  [0,1,1,0,0,0,0,1,1,0],
                  [0,1,1,1,1,1,1,1,1,0],
                  [0,1,1,1,1,1,1,1,1,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  ])
      shizi=np.array([
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  [1,1,1,1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1,1,1,1],
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  ])
      xiegang=np.array([
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,1,1,0,0,0,0,0,0,0],
                  [0,1,1,1,0,0,0,0,0,0],
                  [0,0,1,1,1,0,0,0,0,0],
                  [0,0,0,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,0,0,0],
                  [0,0,0,0,0,1,1,1,0,0],
                  [0,0,0,0,0,0,1,1,1,0],
                  [0,0,0,0,0,0,0,1,1,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  ])
      sidian=np.array([
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,1,1,0,0,0,0,1,1,0],
                  [0,1,1,0,0,0,0,1,1,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,1,1,0,0,0,0,1,1,0],
                  [0,1,1,0,0,0,0,1,1,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  ])
      shangxia=np.array([
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,1,1,0,0,0,0,1,1,0],
                  [0,1,1,0,0,0,0,1,1,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  ])
      space=np.array([
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  ])
    
      hunhe12=zhongyang | sizhou
      hunhe13=zhongyang | shizi
      hunhe14=zhongyang | xiegang
      hunhe23=sizhou | shizi
      hunhe24=sizhou | xiegang
      hunhe34=shizi | xiegang

      fanxie=xiegang[::-1]
      image               = np.array([zhongyang,sizhou,shizi,xiegang,sidian,shangxia,fanxie,space,hunhe12,hunhe13,hunhe14,hunhe23,hunhe24,hunhe34])
      image=image[:,:,:,na]
      label = np.array([0,1,2,3,4,5,6,7])
      return image,label
  
if __name__ == '__main__':
    x,y=vi_pattern() 
#    seq=list(range(0,64,1))
#    seq=np.random.randint(0,high=7,size=64)
#    ff=x[seq].reshape([64,10,10,1])
##    ff=np.random.randint(0,high=300,size=(64,10,10,1))
##    image=np.rollaxis(ff.reshape([8,8,10,10]),2,start=1).reshape([80,80])
#    image=ff.reshape(640,10)
#    
###    image=np.append(image[0:80],[image[i*80:(i+1)+80] for i in range(1,8)],axis=0)
##    image=x[1]
##    image=np.random.randint(0,high=300,size=(64,10,10,1))
#    gg=np.array([[i]*10 for i in range(10)]).reshape(-1)
#    ggg=np.array([[i]*10 for i in range(640)]).reshape(-1)
#    image=image[ggg,:]
#    image=image[:,gg]
#    img = Image.fromarray(image*255).convert('L')
#    img.show()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            