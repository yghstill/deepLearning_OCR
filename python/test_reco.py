# -*- coding: utf-8 -*-
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import caffe
import json
import numpy as np
import os
import cv2
import shutil
import copy




if __name__ == "__main__":
    #base_dir = "/home/user/Projects/data/caffe_dataset_cn_sim"
    base_dir = "/home/user/Projects/data/caffe_dataset_id_num"
    #base_dir = "/home/user/Projects/deep_ocr_workspace/data/chongdata_train_ualpha_digits_64_64"

    model_def = os.path.join(base_dir, "deploy_lenet_train_test.prototxt")
    model_weights = os.path.join(base_dir, "lenet_iter_50000.caffemodel")
    y_tag_json_path = os.path.join(base_dir, "y_tag.json")

    net = caffe.Net(model_def, model_weights, caffe.TEST)
    cv2_color_img = cv2.imread('/home/user/Projects/data/2.jpg')
    cv2_img = cv2.cvtColor(cv2_color_img, cv2.COLOR_RGB2GRAY)
    cv2_img = cv2_img.reshape((1, 1, 64, 64))
    print(cv2_img.shape)
    #np_img = np.asarray(cv2_img)
    
    #print(net.blobs['data'].data.shape)
    
    #transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    #transformer.set_transpose('data', (2, 0, 1))
    #transformer.set_raw_scale('data', 255)                                # 缩放到【0，255】之间
    #transformer.set_channel_swap('data', (2, 1, 0))
    #net.blobs['data'].reshape(1, 1, 64, 64)
    #im=caffe.io.load_image('/home/user/Projects/data/0.jpg')

    #net.blobs['data'].data[...] = transformer.preprocess('data', cv2_img)
    net.blobs['data'].data[...] = cv2_img
    out = net.forward()
    #print(out)

    pridects=out['prob']
    print(max(pridects))

    #print([(k, v.data.shape) for k, v in net.blobs.items()])
    #print(net.params['conv11'][0].data)
    test=net.params['conv11'][0].data
    #print(net.blobs['data'].data)
