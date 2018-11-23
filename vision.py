#!/usr/bin/python3

import cv2
import numpy as np
import glob
import os
import sys
# from matplotlib import pyplot as plt

def process(input_path):

    print('Input:', input_path)
    output_path = input_path.replace('input', 'output')
    mid_path = input_path.replace('input', 'mid')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(mid_path), exist_ok=True)
    input_img = cv2.imread(input_path)
    output_img = input_img.copy()
    height, width, channels = input_img.shape
    linear_size = min(width, height)

    kernel = np.ones((5, 5), np.float32) / 25
    blur_img = cv2.filter2D(input_img, -1, kernel)
    laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)
    laplacian_img = cv2.convertScaleAbs(laplacian)

    edges = cv2.Canny(laplacian_img, 0, 3 * linear_size, apertureSize=5)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:

        for line in lines:
            for rho, theta in line:
                if (int(theta / np.pi * 1000.0) + 2) % 500 <= 2:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 10000 * (-b))
                    y1 = int(y0 + 10000 * (a))
                    x2 = int(x0 - 10000 * (-b))
                    y2 = int(y0 - 10000 * (a))

                    cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(output_path, output_img)
    print('Output:', output_path)

    mid_img= cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    full_img = np.concatenate([input_img, laplacian_img, mid_img, output_img], axis=1)
    cv2.imwrite(mid_path, full_img)
    print('Mid:', mid_path)

def main(argv):
    input_path = argv[1] if len(argv) >= 2 else '../haoshiyou-testdata/images/input/'
    # input_path = '../haoshiyou-testdata/images/input/tricky/tricky-7.jpeg'
    print('inputpath=',input_path)
    if os.path.isdir(input_path):
        print('process directory:', input_path)
        for input_file_path in glob.iglob(input_path + '**/*.jpeg', recursive=True):
            process(input_file_path)
    elif os.path.isfile(input_path):
        print('process single image:', input_path)
        process(input_path)
    else:
        print('path does\'t exist')

if __name__ == "__main__":
    main(sys.argv)