import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from btgen import BatchGenerator
from model2 import *

DATASET_DIR = "data"
VAL_DIR ="val"
SAVE_DIR = "model"
SVIM_DIR = "samples"


def loss_g(y, t):
    #mse = tf.reduce_mean(tf.square(y - t))
    loss = tf.nn.l2_loss(y-t) + tf.reduce_sum(tf.abs(y-t))
    return loss


def training(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01,
                                       beta1=0.9,
                                       beta2=0.999)
    train_step = optimizer.minimize(loss)
    return train_step

def tileImage(imgs):
    d = int(math.sqrt(imgs.shape[0]-1))+1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.zeros((h*d,w*d,3),dtype=np.float32)
    for idx,img in enumerate(imgs):
        idx_y = int(idx/d)
        idx_x = idx-idx_y*d
        r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
    return r

def foloderLength(folder):
    dir = folder
    paths = os.listdir(dir)
    return len(paths)

def printParam(scope):
    total_parameters = 0
    for variable in tf.trainable_variables(scope=scope):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("{} has {} parameters".format(scope, total_parameters))

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(SVIM_DIR):
        os.makedirs(SVIM_DIR)

    img_size = 256
    bs = 16

    dir = DATASET_DIR
    val = VAL_DIR
    datalen = foloderLength(DATASET_DIR)
    vallen = foloderLength(VAL_DIR)

    # loading images on training
    batch = BatchGenerator(img_size=img_size,datadir=dir)
    val = BatchGenerator(img_size=img_size,datadir=val)

    id = np.random.choice(range(datalen),bs)
    IN_ = tileImage(batch.getBatch(bs,id)[:4])

    IN_ = (IN_ + 1)*127.5
    cv2.imwrite("input.png",IN_)


    start = time.time()

    x = tf.placeholder(tf.float32, [bs, img_size, img_size, 3])
    t = tf.placeholder(tf.float32, [bs, img_size, img_size, 3])

    y =buildGenerator(x,nBatch=bs)

    loss = loss_g(y, t)
    printParam(scope="generator")

    train_step = training(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state(SAVE_DIR)

    if ckpt: # checkpointがある場合
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print ("load " + last_model)
        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")
    else:
        print("models were not found")
        init = tf.global_variables_initializer()
        sess.run(init)

    print("%.4e sec took initializing"%(time.time()-start))


    hist =[]


    start = time.time()
    for i in range(100000):
        # loading images on training
        id = np.random.choice(range(datalen),bs)
        batch_images_x = batch.getBatch(bs,id,ocp=0.5)
        batch_images_t = batch.getBatch(bs,id,ocp=0.5)

        tmp, yloss = sess.run([train_step,loss], feed_dict={
            x: batch_images_x,
            t: batch_images_t
        })

        print("in step %s loss = %.4e" %(i,yloss))
        hist.append(yloss)

        if i %100 ==0:
            id = np.random.choice(range(vallen),bs)
            batch_images_x = val.getBatch(bs,id,ocp=0.5)
            out = sess.run(y,feed_dict={
                x:batch_images_x})
            X_ = tileImage(batch_images_x[:4])
            Y_ = tileImage(out[:4])

            X_ = (X_ + 1)*127.5
            Y_ = (Y_ + 1)*127.5
            Z_ = np.concatenate((X_,Y_), axis=1)
            #print(np.max(X_))
            cv2.imwrite("{}/{}.png".format(SVIM_DIR,i),Z_)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.title("Loss")
            plt.grid(which="both")
            plt.yscale("log")
            ax.plot(hist,label="test", linewidth = 0.5)
            plt.savefig("hist.png")
            plt.close()

            print("%.4e sec took per 100steps" %(time.time()-start))
            start = time.time()

        if i%1000==0 :
            if i>1900:
                loss_1k_old = np.mean(hist[-2000:-1000])
                loss_1k_new = np.mean(hist[-1000:])
                print("old loss=%.4e , new loss=%.4e"%(loss_1k_old,loss_1k_new))
                if loss_1k_old*2 < loss_1k_new:
                    break

            saver.save(sess,os.path.join(SAVE_DIR,"model.ckpt"),i)

if __name__ == '__main__':
    main()
