import os
import cv2
import numpy as np
import sys

args = sys.argv

folder = "set14"
output = "test"
ntype = "weak"

try:
    folder = args[1]
    output = args[2]
    ntype = args[3]
except:
    print("receiving augment is failed")

if not os.path.exists(output):
    os.mkdir(output)
files =os.listdir(folder)

for i, j in enumerate(files):

    path ="{}/{}".format(folder,j)

    img = cv2.imread(path)

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

    elif ntype == "weak":
        ocp = 0.25
        rate = ocp
        noise = np.random.normal(0,100,(height,width,3))
        occupy = np.random.binomial(n=1,p=rate,size=[height,width,3])
        noise = noise * occupy



    new = img + noise
    new = np.clip(new, 0 ,255)
    cv2.imwrite("{}/n_{}".format(output,j),new)
