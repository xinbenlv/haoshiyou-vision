#!/usr/bin/python3

import cv2
import numpy as np
import glob

def process(input_path, output_path):
    img = cv2.imread(input_path)
    height, width, channels = img.shape
    print(height, width)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 0, width, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        for line in lines:
            for rho, theta in line:

                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                if 88 < abs(angle) or abs(angle) < 2:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    print('angle', angle)
                else:
                    print('===angle', angle)
    cv2.imwrite(output_path, img)
    print('Wrote to ', output_path, 'after process')

root_dir = '../haoshiyou-testdata/images/input/tricky'

for input_file_path in glob.iglob(root_dir + '**/*.jpeg', recursive=True):
    print('Input:', input_file_path)
    output_file_path = input_file_path.replace('input', 'output')
    process(input_file_path, output_file_path)
    print('Output', output_file_path)

# process('../haoshiyou-testdata/images/input/tricky/tricky-8.jpeg', '../haoshiyou-testdata/images/output/tricky/tricky-8.jpeg')