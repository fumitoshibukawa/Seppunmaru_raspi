# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 05:50:57 2021

@author: Takumi.H

This module is used for debugging.
"""

import cv2
import numpy as np

class DebugImage():
    def __init__(self, image, name):
        self._image = image.copy()
        self._name = name

    def add_contour_line(self, contours, rbg, width):
        for contour in contours:
            if len(contour) > 0:
                rect = contour
                #滑らかな境界線を描画
                cv2.polylines(self._image, contour, True, rbg, width)

    def add_circle_line(self, circle, rbg, width):
        if not circle is None:
            cv2.circle(self._image, (circle[0], circle[1]), circle[2], rbg, width)

    def show(self):
        cv2.imshow(self._name, self._image)
        #0キーを押して表示を終了
        cv2.waitKey(0)
        cv2.destroyAllWindows()
