#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from argparse import RawTextHelpFormatter
import fnmatch
import os
import cv2
import json
import random
import numpy as np
import shutil
from deep_ocr.lang_aux import LangCharsGenerate
from deep_ocr.lang_aux import FontCheck
from deep_ocr.lang_aux import Font2Image
from deep_ocr.lang_aux import DataAugmentation

if __name__ == "__main__":

    out_caffe_dir = os.path.expanduser("/home/user/Projects/data/caffe_dataset_id_num/")
    #font_dir = os.path.expanduser("/home/user/Projects/deepLearning_OCR/chinese_fonts/")
    font_dir = os.path.expanduser("/home/user/Projects/deepLearning_OCR/id_num_fonts/")
    test_ratio = float(0.3)
    width = int(64)
    height = int(64)
    need_crop = False
    margin = int(4)
    langs = "id_num"
    rotate = 10
    rotate_step = 1

    image_dir_name = "images"

    images_dir = os.path.join(out_caffe_dir, image_dir_name)
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir)

    lang_chars_gen = LangCharsGenerate(langs)
    lang_chars = lang_chars_gen.do()
    font_check = FontCheck(lang_chars)

    y_to_tag = {}
    y_tag_json_file = os.path.join(out_caffe_dir, "y_tag.json")
    y_tag_text_file = os.path.join(out_caffe_dir, "y_tag.txt")
    path_train = os.path.join(out_caffe_dir, "train.txt")
    path_test = os.path.join(out_caffe_dir, "test.txt")

    ## rotate
    if rotate < 0:
        rotate = - rotate

    if rotate > 0 and rotate <= 45:
        all_rotate_angles = []
        for i in range(0, rotate + 1, rotate_step):
            all_rotate_angles.append(i)
        for i in range(-rotate, 0, rotate_step):
            all_rotate_angles.append(i)
        #print(all_rotate_angles)

    verified_font_paths = []
    ## search for file fonts
    for font_name in os.listdir(font_dir):
        path_font_file = os.path.join(font_dir, font_name)
        if font_check.do(path_font_file):
            verified_font_paths.append(path_font_file)

    train_list = []
    test_list = []
    max_train_i = int(len(verified_font_paths) * (1.0 - test_ratio))

    font2image = Font2Image(width, height, need_crop, margin)

    # loop font
    for i, verified_font_path in enumerate(verified_font_paths):
        is_train = True
        if i >= max_train_i:
            is_train = False
        # loop char
        for j, char in enumerate(lang_chars):
            if j not in y_to_tag:
                y_to_tag[j] = char
            char_dir = os.path.join(images_dir, "%d" % j)
            if not os.path.isdir(char_dir):
                os.makedirs(char_dir)
            if rotate == 0:
                relative_path_image = os.path.join(image_dir_name, "%d" % j, "%d_%s.jpg" % (i, os.path.basename(verified_font_path)))
                path_image = os.path.join(char_dir, "%d_%s.jpg" % (i, os.path.basename(verified_font_path)))
                font2image.do(verified_font_path, path_image, char)
                if is_train:
                    train_list.append((relative_path_image, j))
                else:
                    test_list.append((relative_path_image, j))
            else:
                for k in all_rotate_angles:
                    relative_path_image = os.path.join(image_dir_name, "%d" % j, "%d_%s_%d.jpg" % (i, os.path.basename(verified_font_path), k))
                    path_image = os.path.join(char_dir, "%d_%s_%d.jpg" % (i, os.path.basename(verified_font_path), k))
                    font2image.do(verified_font_path, char, path_image, rotate=k)
                    #font2image.do(verified_font_path, char, path_image)
                    if is_train:
                        train_list.append((relative_path_image, j))
                    else:
                        test_list.append((relative_path_image, j))




    h_y_tag_json_file = open(y_tag_json_file, "w+")
    json.dump(y_to_tag, h_y_tag_json_file)
    h_y_tag_json_file.close()

    h_y_tag_text_file = open(y_tag_text_file, "w+")
    for key in y_to_tag:
        h_y_tag_text_file.write("%d %s\n" % (key, y_to_tag[key].encode("utf-8")))
    h_y_tag_text_file.close()

    fout = open(path_train, "w+")
    for item in train_list:
        fout.write("%s %d\n" % (item[0], item[1]))
    fout.close()

    fout = open(path_test, "w+")
    for item in test_list:
        fout.write("%s %d\n" % (item[0], item[1]))
    fout.close()
