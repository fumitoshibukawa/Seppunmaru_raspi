# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 05:50:57 2021

@author: Takumi.H

This module is a set of functions used for debugging.
"""

import cv2
import numpy as np

def show_detected_Item( image, contours ):
    """
    認識した対象物の境界線を元画像データの上に描画し、別ウィンドウに表示する関数。
    元画像データには変更を加えない。
    引数 image:元の画像データ contours:輪郭座標のリスト
    """

    image_debug = image.copy()
    for contour in contours:
        if len(contour) > 0:
            rect = contour
            x, y, w, h = cv2.boundingRect(rect)
            cv2.rectangle(image_debug, (x, y), (x + w, y + h), (0, 255, 0), 10)

    #画像を表示
    cv2.imshow('image', image_debug)
    #0キーを押して表示を終了
    cv2.waitKey(0)
    cv2.destroyAllWindows()
