import numpy as np
import cv2
import glob
import random

class BatchGenerator:
    def __init__(self, img_size, datadir):
        self.folderPath = datadir
        self.imagePath = glob.glob(self.folderPath+"/*")
        #self.orgSize = (218,173)
        self.imgSize = (img_size,img_size)
        assert self.imgSize[0]==self.imgSize[1]


    def add_mosaic(self,img,ocp):
        #flg = True
        occupancy = ocp#np.random.uniform(min_occupancy, max_occupancy)
        mosaic =img
        h, w, _ = img.shape
        img_for_cnt = np.zeros((h, w), np.uint8)

        while True:
            ms_size = np.random.randint(8,16)
            scale = np.random.randint(4,8)
            x = random.randint(0, max(0, w - 1 - ms_size))
            y = random.randint(0, h - 1 - ms_size)
            area = img[y:y+ms_size,x:x+ms_size]
            area = cv2.resize(area,(ms_size//scale,ms_size//scale), interpolation=cv2.INTER_NEAREST)
            area = cv2.resize(area,(ms_size,ms_size), interpolation=cv2.INTER_NEAREST)
            mosaic[y:y+ms_size,x:x+ms_size] = area
            area = np.where(area>0, 255, area)
            img_for_cnt[y:y+ms_size,x:x+ms_size] = cv2.cvtColor(area, cv2.COLOR_RGB2GRAY)
            #print((img_for_cnt > 0).sum())
            if (img_for_cnt > 0).sum() > h * w * occupancy:

                break
        return mosaic

    def add_impulse(self, img, ocp):
        rate = np.random.rand() * ocp/2 + ocp/2
        mask = np.random.binomial(size=img.shape, n=1, p=rate)
        noise = np.random.randint(256, size=img.shape)
        img = img * (1 - mask) + noise * mask
        return img


    def add_noise(self, img, ocp):
        #noise = (np.random.rand(self.imgSize[0],self.imgSize[1],3) -0.5) * 512
        rate = np.random.rand() * ocp/2 + ocp/2
        noise = np.random.normal(0,100,(self.imgSize[0],self.imgSize[1],3))
        occupy = np.random.binomial(n=1,p=rate,size=[self.imgSize[0],self.imgSize[1],3])
        noise = noise * occupy
        img = img+noise
        img = np.clip(img,0,255)
        return img

    def strongNoise(self, img, ntype="gauss"):
        height = img.shape[0]
        width = img.shape[1]
        #print(height,width)
        noise = np.zeros_like(img)

        if ntype == "random":
            noise = (np.random.rand(height,width,3) - 0.5) * 255

        elif ntype == "gauss":
            noise = (np.random.randn(height,width,3) - 0.5) * 255

        elif ntype == "sandp":
            noise = (np.random.rand(height,width) - 0.5) * 255
            noise = np.where( noise > 255/4, 255, 0)
            noise2 =(np.random.rand(height,width) - 0.5) * 255
            noise2 = np.where( noise2 > 255/4, 0, -255)
            noise += noise2
            noise = np.tile(noise,3).reshape(height,width,3)

        new = img + noise
        return new

    def getBatch(self, nBatch, id, ocp=0.5):
        x   = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],3), dtype=np.float32)
        for i,j in enumerate(id):

            img = cv2.imread(self.imagePath[j])
            dmin = min(img.shape[0],img.shape[1])
            img = img[int(0.5*(img.shape[0]-dmin)):int(0.5*(img.shape[0]+dmin)),int(0.5*(img.shape[1]-dmin)):int(0.5*(img.shape[1]+dmin)),:]
            img = cv2.resize(img,self.imgSize)

            img = self.add_noise(img,ocp)
            #img = self.add_mosaic(img,ocp)
            #img = self.add_impulse(img,ocp)


            x[i,:,:,:] = (img - 127.5) / 127.5 

        return x
