# -*- coding: utf-8 -*-

import time
import logging
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import glob
import os
import sys

from abc import ABCMeta, abstractmethod

from collections import OrderedDict
from tqdm import tqdm


#------------------------------------------------------------------------------#
#                                  LOGGING                                     #
#------------------------------------------------------------------------------#
logging.basicConfig(level=logging.INFO,
                    format=' %(levelname)s: %(name)s%(message)s')
logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------#
#                                 PARAMETERS                                   #
#------------------------------------------------------------------------------#

timestr = time.strftime('%Y%b%d_%H%M%S%Z')   # timestamp string
# CONSECUTIVE_WHITE_PIXEL_CUTOFF = 0.1 # 10%
# WHITE_PIXEL_CUTOFF = 0.5 #50%
# INTERVAL = 75 # Min interval between two row

BAND_WIDTH = 0.03
IMG_RATIO_DICT = [
    (2/3.-BAND_WIDTH, 2/3.+BAND_WIDTH),
    (3/4.-BAND_WIDTH, 3/4.+BAND_WIDTH),
    (4/5.-BAND_WIDTH, 4/5.+BAND_WIDTH),
    (9/16.-BAND_WIDTH, 9/16.+BAND_WIDTH),
    (1-BAND_WIDTH, 1+BAND_WIDTH),
    (10/16.-BAND_WIDTH, 10/16.+BAND_WIDTH),
]

class Image(object):

    def __init__(self):
        pass
    @abstractmethod
    def from_path(self):
        raise NotImplementedError

    @abstractmethod
    def split(self):
        raise NotImplementedError       

    @abstractmethod
    def detect(self):
        raise NotImplementedError       

    @abstractmethod
    def transform(self):
        raise NotImplementedError    

    @abstractmethod
    def resize(self):
        raise NotImplementedError

class HaoShiYouImage(Image):

    diff_order = 4
    cutoff = 99.5
    long_img_ratio = 2.5
    img_ratio = IMG_RATIO_DICT
    output_size = (256,256) 

    # def __init__(self):
    #     super(HaoShiYouImage, self).__init__()
    #     self.diff_order = 4
    #     self.cutoff = 99.5
    #     self.long_img_ratio = 2.5

    @staticmethod
    def transform(img, tfms):
        pass

    @staticmethod
    def detect(img, tfms):
        pass

    @staticmethod
    def save(img, path):
        pass

    @classmethod
    def split(cls, img, save=False):
        """
        HAOSHIYOU: Split long Ins/Wechat/FB images based on 
        1D difference.

        Arguments:
        ---------
            img(np.array): 
                3D image array
            save(boolen):
                Save split results if True

        Returns
        -------    
            img_list(list):
                image split results

        Note:
        -----
        Use common image ratio to filter out split results

        """

        if isinstance(img,str):
            img_name = img.split('/')[-1]
            img = cv2.imread(img)
        else:
            img_name = 'img.jpg'

        height, width, channels = img.shape

        ## Check long image ratio
        ratio = height/float(width)        
        if ratio <= cls.long_img_ratio: return img

        ## Calculate 1D difference
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_diff = (gray_img[1:, :]-gray_img[:-1, :])**cls.diff_order
        img_diff = np.sum(img_diff, axis=1)

        cut_off = np.percentile(img_diff, cls.cutoff)

        ## Split image based on cutoff value
        result = cls._split(img, img_diff, cut_off)

        if save:
            for ind, _img in enumerate(result):                
                _img_name = img_name.split(
                    '.')[0]+'_'+str(ind)+img_name.split('.')[-1]
                cls.save(_img, _img_name)
        return result

    @classmethod
    def _split(cls, img, img_diff, cut_off):
        """Split long image"""

        height, width, channels = img.shape
        result = []
        ind1 = 0

        for ind2, pixel_value in enumerate(img_diff):
            if pixel_value > cut_off:
                if (ind2 - ind1) < width:
                    # Use common picture ratio for postprocessing
                    for low, high in cls.img_ratio:
                        if low <= (ind2 - ind1)/float(width) <= high:
                            result.append(img[ind1:ind2,:,:])
                            ind1 = ind2
                else:
                    result.append(img[ind1:ind2, :, :])
                    ind1 = ind2

        return result


