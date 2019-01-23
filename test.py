#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

warnings.filterwarnings('ignore')


def main(yolo):

    ####################################
    # images from Camera
    ####################################
    '''
    video_capture = cv2.VideoCapture(0)
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break;
        t1 = time.time()
        image = Image.fromarray(frame)
        det_img = yolo.detect_image(image)
        resultimg = np.array(det_img)
        cv2.imshow('', resultimg)
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    '''


    ####################################
    # images from Image_Path_TXT
    ####################################
    data_src_path = '/media/yl/Elements/data/FPointnet_kitti/training/image_2'
    data_detection_txt = '/media/yl/Elements/data/FPointnet_kitti/training/train.txt'
    fps = 0.0

    with open(data_detection_txt, 'r') as f:
        for line in f.readlines():
            # src_img_path = line.rstrip()
            src_img_path = data_src_path + '/' + line.rstrip() + '.png'
            frame = cv2.imread(src_img_path)
            if frame is None:
                print('Canot open image:', src_img_path)
                continue
            t1 = time.time()
            image = Image.fromarray(frame)
            det_img = yolo.detect_image(image)
            resultimg = np.array(det_img)
            cv2.imshow('', resultimg)
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %f" % (fps))
            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
