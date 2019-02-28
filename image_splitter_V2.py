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

from collections import OrderedDict

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
CONSECUTIVE_WHITE_PIXEL_CUTOFF = 0.1 # 10%
WHITE_PIXEL_CUTOFF = 0.5 #50%
INTERVAL = 75 # Min interval between two row
 
def process(input_path):
    """Split images based on second order gradient"""

    print('Input:', input_path)
    output_path = input_path.replace('input', 'dw_output2')
    mid_path = input_path.replace('input', 'dw_mid2')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    if not os.path.exists(os.path.dirname(mid_path)):
        os.makedirs(os.path.dirname(mid_path))

    input_img = cv2.imread(input_path)
    output_img = input_img.copy()


    height, width, channels = input_img.shape
    
    # # Blur image
    # kernel = np.ones((5, 5), np.float32) / 25
    # blur_img = cv2.filter2D(input_img, -1, kernel)

    # # RGB -> gray
    # gray_img = cv2.cvtColor(blur_img , cv2.COLOR_BGR2GRAY)

    # Get first order gradient
    gradient_img = cv2.Sobel(input_img,cv2.CV_64F,0,1,ksize=1)
    gradient_img = cv2.convertScaleAbs(gradient_img)

    # Get second order gradient
    gradient_img= cv2.Sobel(gradient_img,cv2.CV_64F,0,1,ksize=1)
    gradient_img = cv2.convertScaleAbs(gradient_img)

    # BGR -> Gray
    gradient_gray_img = cv2.cvtColor(gradient_img,cv2.COLOR_BGR2GRAY)


    kernel_size = 7
    kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size**2
    gradient_gray_img = cv2.filter2D(gradient_gray_img, -1, kernel)

    linear_size = min(width, height)

    edges = cv2.Canny(gradient_gray_img, 0, 3 * linear_size, apertureSize=5)
    mid_img= cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        for line in lines:
            rho, theta  = line[0,0], line[0,1]

            if (int(theta / np.pi * 1000.0) + 2) % 500 <= 2 and theta > 0:
            #if theta <0.001:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 10000 * (-b))
                y1 = int(y0 + 10000 * (a))
                x2 = int(x0 - 10000 * (-b))
                y2 = int(y0 - 10000 * (a))
                
                #cv2.line(output_img, (0,x1), (width,x2), (0, 0, 255), 2)
                cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 10)

    gradient_img = cv2.cvtColor(gradient_gray_img, cv2.COLOR_GRAY2BGR)
    full_img = np.concatenate([input_img, gradient_img, mid_img, output_img], axis=1)

    cv2.imwrite(mid_path, full_img)

    return 

def _process(gradient_img):
    """ Split images into multiple images based on gradient.
    
    Arguments
    ---------
        gradient_img(/np.array): 
            Object Image 

    Returns
    -------
        result(list):
            List of result row

    """

    result = {}
    height, width = gradient_img.shape

    # Find row based on consecutive white pixel numer 
    # and nonzero pixel num
    for ind in np.arange(0,height,1):
        row = gradient_img[ind,:]

        consecutive_white_list = get_consecutive_nonzero_num(row, CONSECUTIVE_WHITE_PIXEL_CUTOFF, 253)
        num_zero = len(row[row==0])
        
        if (num_zero==0 and len(consecutive_white_list)>0) or len(consecutive_white_list)>1 :
            result[ind] = consecutive_white_list[0]
        else:
            result[ind] = 0

    ordered_result = OrderedDict()
    for k,v in result.items():
        if v>0:
            ordered_result[k] = v

    previous = -1000
    filter_result = OrderedDict()

    for k,v in ordered_result.items():
        if v>0:
            if k-1 in ordered_result and k+1 in ordered_result or v >width*WHITE_PIXEL_CUTOFF:            
                if k >previous+INTERVAL:               
                    previous = k
                    filter_result[k] = v

    return filter_result.keys()

def get_consecutive_nonzero_num(array, ratio=0.6, pixel_cutoff = 0):
    
    width = len(array)
    consecutive_cutoff = width * ratio
    
    result = []
    count = 1
    previous = 0
    for num in array:
        if previous>pixel_cutoff and num>pixel_cutoff:
            count+=1       
        if previous>pixel_cutoff and num<=pixel_cutoff:
            if count>=consecutive_cutoff:
                result.append(count)               
            count = 1
            
        previous = num
        
    if count>=consecutive_cutoff and count not in result:
        result.append(count) 
                   
    result = sorted(result,reverse=True)
    return result

def main(argv):
    input_path = argv[1] if len(argv) >= 2 else '../haoshiyou-testdata/images/input/'
    # input_path = '../haoshiyou-testdata/images/input/tricky/tricky-7.jpeg'
    print('inputpath=',input_path)
    if os.path.isdir(input_path):
        print('process directory:', input_path)
        for input_file_path in glob.glob(input_path + '**/*.jpeg'):
            process(input_file_path)
    elif os.path.isfile(input_path):
        print('process single image:', input_path)
        process(input_path)
    else:
        print('path does\'t exist')

if __name__ == "__main__":
    main(sys.argv)
