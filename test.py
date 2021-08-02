#!/usr/bin/env python
import cv2
import numpy as np

prevx = []

ploty = np.linspace(0, 9, 10)

print("\n\n ploty: ", ploty)

left_plotx = 3 * ploty

print("\n\n left_plotx: ", left_plotx)


for i in range(3):
    prevx.append(left_plotx)


print("\n\n prevx: ", prevx)


lines = np.squeeze(prevx)

print("\n\n lines: ", lines)

k = len(prevx)
n = len(left_plotx)
m = len(lines)

print("\n\n len(prevx): ", k)
print("\n len(left_plotx): ", n)
print("\n len(lines): ", m)
