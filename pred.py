import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from main import *
from model2 import *
DATASET_DIR = "data"
VAL_DIR ="val"
SAVE_DIR = "model"
OUT_DIR = "ouput"

def main(folder="test"):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    folder=folder
    files = os.listdir(folder)
    img_size = 576
    bs =8
    val_size =2

    start = time.time()

    x = tf.placeholder(tf.float32, [1, None, None, 3])
    y =buildGenerator(x,nBatch=bs)

    print("%.4e sec took building model"%(time.time()-start))

    sess = tf.Session()
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
    if ckpt: # checkpointがある場合
        #last_model = ckpt.all_model_checkpoint_paths[1]
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print ("load " + last_model)
        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    print("%.4e sec took initializing"%(time.time()-start))

    start = time.time()

    files = os.listdir(folder)
    for i in range(len(files)):


        print(files[i])
        img = cv2.imread("{}/{}".format(folder,files[i]))
        img = (img-127.5)/127.5
        h, w = img.shape[:2]
        print(y)

        img = img.reshape(1,h,w,3)
        out = sess.run(y,feed_dict={x:img})

        img = img.reshape(h,w,3)
        out = np.squeeze(out)

        X_ = (img + 1)*127.5
        Y_ = (out + 1)*127.5
        cv2.imwrite("{}/{}_xval.png".format(OUT_DIR, i), X_)
        cv2.imwrite("{}/{}_yval.png".format(OUT_DIR, i), Y_)
        #Z_ = np.concatenate((X_,Y_), axis=1)

        #cv2.imwrite("{}/{}_val.png".format(OUT_DIR,i), Z_)

    print("%.4e sec took for predicting" %(time.time()-start))

if __name__ == '__main__':
    folder = "test"
    try:
        folder = sys.argv[1]
    except:
        pass
    main(folder)
