import cv2
import numpy as np


def process(input_path, output_path):
    img = cv2.imread(input_path)
    height, width, channels = img.shape
    print(height, width)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 0, width, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
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
            if angle > 0 and angle < 10:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(output_path, img)

process('input/normal/normal-1.jpg', 'houghlines3.jpg')
