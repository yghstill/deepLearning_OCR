# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from argparse import RawTextHelpFormatter
import os
import shutil
import cv2
from deep_ocr.caffe_clf import CaffeClsBuilder
from deep_ocr.cv2_img_proc import PreprocessResizeKeepRatio
from deep_ocr.cv2_img_proc import PreprocessBackgroundMask
from deep_ocr.id_cards.segmentation import Segmentation
from deep_ocr.id_cards.char_set import CharSet
from deep_ocr.reco_text_line import RecoTextLine
from deep_ocr.reco_text_line import RectImageClassifier


if __name__ == "__main__":

    path_img = os.path.expanduser("/home/user/Projects/data/test_id_card/hehe4.jpg")
    debug_path = os.path.expanduser("/home/user/Projects/data/debug")
    if debug_path is not None:
        if os.path.isdir(debug_path):
            shutil.rmtree(debug_path)
        os.makedirs(debug_path)

    cls_dir_ua = os.path.expanduser("/home/user/Projects/data/caffe_dataset_id_num")
    caffe_cls_builder = CaffeClsBuilder()
    cls_ua = caffe_cls_builder.build(cls_dir=cls_dir_ua, )

    seg_norm_width = 600
    seg_norm_height = 600
    preprocess_resize = PreprocessResizeKeepRatio(seg_norm_width, seg_norm_height)
    id_card_img = cv2.imread(path_img)
    id_card_img = preprocess_resize.do(id_card_img)
    segmentation = Segmentation(debug_path)
    key_to_segmentation = segmentation.do(id_card_img)

    boundaries = [
        ((0, 0, 0), (100, 100, 100)),
    ]
    boundary2binimgs = []
    for boundary in boundaries:
        preprocess_bg_mask = PreprocessBackgroundMask(boundary)
        id_card_img_mask = preprocess_bg_mask.do(id_card_img)
        boundary2binimgs.append((boundary, id_card_img_mask))

    char_set = CharSet()
    char_set_data = char_set.get()

    rect_img_clf = RectImageClassifier(None, None, char_set, caffe_cls_width=64, caffe_cls_height=64)

    reco_text_line = RecoTextLine(rect_img_clf)


    ## just test id card num
    for i, segment in enumerate(key_to_segmentation["id"]):
        if debug_path is not None:
            line_debug_path = "key_%s_%i" % ("id", i)
            line_debug_path = os.path.join(debug_path, line_debug_path)
            reco_text_line.debug_path = line_debug_path
        reco_text_line.char_set = char_set_data["id"]
        caffe_cls = cls_ua
        ocr_res = reco_text_line.do(boundary2binimgs, segment, caffe_cls)
        print("=" * 64)
        print(ocr_res)


    if debug_path is not None:
        path_debug_image_mask = os.path.join(debug_path, "reco_debug_01_image_mask.jpg")
        cv2.imwrite(path_debug_image_mask, id_card_img_mask)

