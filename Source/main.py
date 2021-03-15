# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:07:37 2018

@author: S.M
"""

# 距離を比較して近い距離とその形状RXに返す
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import math
import serial
from debug import DebugImage
ser = serial.Serial('/dev/ttyUSB0', 115200)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

#picameraシャッタースピード設定
camera.shutter_speed = 30000 #周期　×　[マイクロ秒]

# allow the camera to warmup
time.sleep(0.1)

#グローバル変数宣言
image = 0
image_hsv = 0
back_image = 0
back_image_hsv = 0

#赤ボール用閾値設定(HSV値, 画像領域)
lower_darkred_hsv = np.array([0, 100, 50])
upper_darkred_hsv = np.array([30, 255, 255])
lower_lightred_hsv = np.array([160, 100, 50])
upper_lightred_hsv = np.array([180, 255, 255])

lower_red_hsv = np.array([160, 40, 100])
upper_red_hsv = np.array([180, 255, 255])
lower_red_area = 7000
upper_red_area = 10000

#青ボール用HSV値
lower_blue_hsv = np.array([80, 60, 20])
upper_blue_hsv = np.array([110, 255, 255])
lower_blue_area = 7000
upper_blue_area = 12000

#贅沢微糖ボール用HSV値
#lower_cun_hsv = np.array([18, 50, 80])
#upper_cun_hsv = np.array([30, 255, 255])

lower_cun_hsv = np.array([18, 30, 80])
upper_cun_hsv = np.array([30, 255, 255])
lower_cun_area = 6000
upper_cun_area = 10000

#赤ボールピラミッド用HSV値
lower_darkred_hsv = np.array([0, 100, 50])
upper_darkred_hsv = np.array([30, 255, 255])
lower_lightred_hsv = np.array([160, 100, 50])
upper_lightred_hsv = np.array([180, 255, 255])
#lower_pyramid_area = 20000
#upper_pyramid_area = 40000

lower_pyramid_area = 20000
upper_pyramid_area = 40000

#マックスコーヒー用HSV値
#lower_maxyellow_hsv = np.array([0, 20, 28])
#upper_maxyellow_hsv = np.array([40, 255, 255])

lower_maxyellow_hsv = np.array([0, 50, 0])
upper_maxyellow_hsv = np.array([30, 255, 255])

# lower_maxbrown = np.array([122, 25, 18])
# upper_maxbrown = np.array([179, 255, 255])
lower_maxcoffee_area = 15000
#upper_maxcoffee_area = 45000

#image = cv2.imread("./picture/ballpyramid.jpg")
#front_image = cv2.imread("./picture/redball.jpg")
back_image = cv2.imread("./robocon_picture/back_image1.jpg")
#back_image = cv2.imread("./robocon_picture/back_image1.jpg")
#maxcoffee_image = cv2.imread("./picture/maxcoffee.jpg")
#cv2.line(back_image, (0,0), (0, 480), (0,0,0), 50)  #トリミング処理
#cv2.line(back_image, (640,0), (640, 480), (0,0,0), 50)

def redball_detection():

    global image
    global back_image
    global lower_darkred_hsv
    global upper_darkred_hsv
    global lower_lightred_hsv
    global upper_lightred_hsv
    global lower_red_hsv
    global upper_red_hsv
    global lower_red_area
    global upper_red_area

    c_area = []
    area = []
    count = 0
    number = 0

    #DEBUG:
    di = DebugImage(image, 'RedBall Detection')

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #back_image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 背景差分処理
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    fgmask_redball = fgbg.apply(back_image)
    fgmask_redball = fgbg.apply(image)

    and_mask_redball = cv2.bitwise_and(image, image, mask = fgmask_redball)
    and_mask_redball = cv2.cvtColor(and_mask_redball, cv2.COLOR_BGR2HSV)
    # 指定した色のマスク画像の生成
    #range_mask_darkred = cv2.inRange(and_mask_redball, lower_darkred_hsv, upper_darkred_hsv)
    #range_mask_lightred = cv2.inRange(and_mask_redball, lower_lightred_hsv, upper_lightred_hsv)

    range_mask_redball = cv2.inRange(and_mask_redball, lower_red_hsv, upper_red_hsv)
    #range_mask_redball = cv2.bitwise_or(range_mask_darkred, range_mask_lightred)
    cnt1,cnt2,cnt3 = cv2.findContours(range_mask_redball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnt2:
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if(radius > 40 and radius < 80):
            c_area.append(c)
            area.append(cv2.contourArea(c))

    if len(area) == 0:
        area.append(0)

    print("redball_area"+str(area))
    for i in area:
        count += 1
        if i > lower_red_area and i < upper_red_area:
            number = count -1
            c_para = c_area[number]
            max_area = area[number]
            break
        else:
            c_para = 0
            max_area = 0

    #DEBUG:
    di.add_contour_line(c_area, (0, 255, 0), 5)
    di.add_contour_line([c_para], (128, 0, 255), 5)
    di.show()

    #print(area)
    print(max_area)
    #cv2.imshow("redball_image", back_image_hsv)
    #cv2.imshow("image", image_hsv)
    #cv2.imshow("red", range_mask_redball)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #rawCapture.truncate(0)

    return c_para, max_area

def blueball_detection():

    global image
    global back_image
    global lower_blue_hsv
    global upper_blue_hsv
    global lower_blue_area
    global upper_blue_area

    c_area = []
    area = []
    count = 0
    number = 0

    #DEBUG:
    di = DebugImage(image, 'BlueBall Detection')

    #image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 背景差分処理
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    fgmask_blueball = fgbg.apply(back_image)
    fgmask_blueball = fgbg.apply(image)

    and_mask_blueball = cv2.bitwise_and(image, image, mask = fgmask_blueball)
    #cv2.imshow("and_mask_blueball", and_mask_blueball)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #rawCapture.truncate(0)
    and_mask_blueball = cv2.cvtColor(and_mask_blueball, cv2.COLOR_BGR2HSV)
    # 指定した色のマスク画像の生成
    range_mask_blueball = cv2.inRange(and_mask_blueball, lower_blue_hsv, upper_blue_hsv)
    cnt1,cnt2,cnt3 = cv2.findContours(range_mask_blueball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnt2:
        c_area.append(c)
        area.append(cv2.contourArea(c))

    if len(area) == 0:
        area.append(0)

    print("blueball_area"+str(area))
    for i in area:
        count += 1
        if i > lower_blue_area and i < upper_blue_area:
            number = count -1
            c_para = c_area[number]
            max_area = area[number]
            #print("ari")
            break
        else:
            c_para = 0
            max_area = 0

    #DEBUG:
    di.add_contour_line(c_area, (0, 255, 0), 5)
    di.add_contour_line([c_para], (128, 0, 255), 5)
    di.show()

    #print(area)

    #cv2.imshow("redball_image", back_image_hsv)
    #cv2.imshow("image", image_hsv)
    print(max_area)
    #cv2.imshow("blue", range_mask_blueball)
    #cv2.imshow("back_image", back_image)
    #cv2.imshow("image", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #rawCapture.truncate(0)

    return c_para, max_area


def cun_detection():

    global image
    global back_image
    global lower_red_hsv
    global upper_red_hsv
    global lower_blue_hsv
    global upper_blue_hsv


    c_area = []
    area = []
    count = 0
    number = 0

    #DEBUG:
    di = DebugImage(image, 'Cun Detection')

    #画像縮小膨張のためのカーネル
    #縮小
    kernel_1 = np.ones((2, 2), np.uint8)
    #膨張
    kernel_2 = np.ones((4, 4), np.uint8)


    image_blur = cv2.medianBlur(image, 3)

    # 背景差分処理
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    fgmask = fgbg.apply(back_image)
    fgmask = fgbg.apply(image)
    fgmask_and = cv2.bitwise_and(image, image, mask = fgmask)

    fgmask_image_hsv = cv2.cvtColor(fgmask_and, cv2.COLOR_BGR2HSV)

    # 赤ボール用閾値処理
    range_mask_red = cv2.inRange(fgmask_image_hsv, lower_red_hsv, upper_red_hsv)
    dilate_mask_red = cv2.dilate(range_mask_red, kernel_2, iterations = 1)
    # 青ボール用閾値処理
    range_mask_blue = cv2.inRange(fgmask_image_hsv, lower_blue_hsv, upper_blue_hsv)
    dilate_mask_blue = cv2.dilate(range_mask_blue, kernel_2, iterations = 1)

    dilate_mask_and = cv2.bitwise_or(dilate_mask_red, dilate_mask_blue)
    image_impaint = cv2.inpaint(image_blur, dilate_mask_and, 10, cv2.INPAINT_NS)

    impaint_blur = cv2.medianBlur(image_impaint, 3)
    image_gray = cv2.cvtColor(impaint_blur, cv2.COLOR_BGR2GRAY)
    image_canny = cv2.Canny(image_gray, 0, 60)

    # ハフ変換処理
    circles = cv2.HoughCircles(image_canny, cv2.HOUGH_GRADIENT, 1,
                              minDist = 100, param1 = 50, param2 = 30,
                              minRadius = 40, maxRadius = 60)

    #circles = np.uint16(np.around(circles))
    if not circles is None:
        circles_para = circles[0][0]

    else:
        circles_para = None

    #DEBUG:
    di.add_circle_line(circles_para, (128, 0, 255), 5)
    di.show()

    rawCapture.truncate(0)
    return circles_para

def ballpyramid_detection():

    global image_hsv
    global back_image_hsv
    global lower_darkred_hsv
    global upper_darkred_hsv
    global lower_lightred_hsv
    global upper_lightred_hsv
    global lower_pyramid_area
    global upper_pyramid_area

    c_area = []
    area = []
    count = 0
    number = 0

    #DEBUG:
    di = DebugImage(image, 'BallPyramid Detection')

    # 背景差分処理
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    fgmask_redball = fgbg.apply(back_image)
    fgmask_redball = fgbg.apply(image)

    and_mask_redball = cv2.bitwise_and(image, image, mask = fgmask_redball)
    and_mask_redball = cv2.cvtColor(and_mask_redball, cv2.COLOR_BGR2HSV)
    # 指定した色のマスク画像の生成
    range_mask_darkred = cv2.inRange(and_mask_redball, lower_darkred_hsv, upper_darkred_hsv)
    range_mask_lightred = cv2.inRange(and_mask_redball, lower_lightred_hsv, upper_lightred_hsv)

    range_mask_pyramid = cv2.bitwise_or(range_mask_darkred, range_mask_lightred)
    kernel = np.ones((2, 2), np.uint8)
    #morpho_mask_maxcoffee = cv2.morphologyEx(range_mask_maxcoffee, cv2.MORPH_OPEN, kernel)
    dilate_mask_pyramid = cv2.dilate(range_mask_pyramid, kernel, iterations = 1)

    cnt1,cnt2,cnt3 = cv2.findContours(dilate_mask_pyramid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #print("cnt2=" + str(cnt2))

    for c in cnt2:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if(radius > 100):
            c_area.append(c)
            area.append(cv2.contourArea(c))

    if len(area) == 0:
        area.append(0)

    print("pyramid_area"+str(area))

    for i in area:
        count += 1
        print("i="+str(i))
        if i > lower_pyramid_area and i < upper_pyramid_area:
            number = count -1
            c_para = c_area[number]
            max_area = area[number]
            print("ari")
            break
        else:
            c_para = 0
            max_area = 0

    #DEBUG:
    di.add_contour_line(c_area, (0, 255, 0), 5)
    di.add_contour_line([c_para], (128, 0, 255), 5)
    di.show()

    print(max_area)
    #cv2.imshow("pyramid", range_mask_pyramid)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #cv2.imshow("pyramid_d", dilate_mask_pyramid)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #rawCapture.truncate(0)
    #rawCapture.truncate(0)
    return c_para, max_area


def maxcoffee_detection():

    global image
    global back_image
    global lower_maxyellow_hsv
    global upper_maxyellow_hsv
    global lower_maxcoffee_area
    global upper_maxcoffee_area

    pre_area =[]
    c_area = []
    area = []
    count = 0
    number = 0

    #DEBUG:
    di = DebugImage(image, 'MaxCoffee Detection')

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    back_image_hsv = cv2.cvtColor(back_image, cv2.COLOR_BGR2HSV)

    # 背景差分処理
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    fgmask_maxcoffee = fgbg.apply(back_image)
    fgmask_maxcoffee = fgbg.apply(image)

    and_mask_maxcoffee = cv2.bitwise_and(image, image, mask = fgmask_maxcoffee)
    and_mask_maxcoffee = cv2.cvtColor(and_mask_maxcoffee, cv2.COLOR_BGR2HSV)

    #画像処理1
    range_mask_maxcoffee = cv2.inRange(and_mask_maxcoffee, lower_maxyellow_hsv, upper_maxyellow_hsv)
    #range_mask_cun_front = cv2.inRange(image_hsv, lower_maxyellow_hsv, upper_maxyellow_hsv)

    kernel = np.ones((6, 6), np.uint8)
    #morpho_mask_maxcoffee = cv2.morphologyEx(fgmask_maxcoffee, cv2.MORPH_OPEN, kernel)
    dilate_mask_maxcoffee = cv2.dilate(range_mask_maxcoffee, kernel, iterations = 1)

    cnt1,cnt2,cnt3 = cv2.findContours(dilate_mask_maxcoffee, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnt2:
        #c_area.append(c)
        pre_area.append(cv2.contourArea(c))
        ((x,y),(width,height),angle) = cv2.minAreaRect(c)
        #area.sort()
        menseki = width * height
        if menseki > 13000 and menseki < 30000:
            c_area.append(c)
            area.append(cv2.contourArea(c))

    if len(area) == 0:
        area.append(0)

    print("maxcoffee"+str(pre_area))

    for i in area:
        count += 1
        if i > lower_maxcoffee_area:
            number = count -1
            c_para = c_area[number]
            max_area = area[number]

            break
        else:
            c_para = 0
            max_area = 0

    #DEBUG:
    di.add_contour_line(c_area, (0, 255, 0), 5)
    di.add_contour_line([c_para], (128, 0, 255), 5)
    di.show()

    print(max_area)
    #cv2.imshow("mask", dilate_mask_maxcoffee)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    rawCapture.truncate(0)
    return c_para, max_area

#画像処理して得た結果を基にそれぞれの形状について，距離，角度を算出
def calaculate_distance( c_para, max_area, shape ):

    global image
    global front_image

    real_distance = 0
    real_angle = 0
    radius = 0
    ENABLE_COMAR = "ban"

    if max_area > 0:
        #c = max(c_para, key = cv2.contourArea)
        print("計算開始")

        #マックス缶の処理
        if shape == 'm':
            rect = cv2.minAreaRect(c_para)
            ((x,y),(width,height),angle) = cv2.minAreaRect(c_para)
            rect = cv2.minAreaRect(c_para)
            M = cv2.moments(c_para)

            if width > 0:

                box = cv2.boxPoints(rect)
                #box = cv2.boxPoints((x,y),(width,height),angle)
                box = np.int0(box)
                #front_image_copy = cv2.drawContours(front_image,[box],0,(0,0,255),2)
                image = cv2.drawContours(image,[box],0,(0,0,255),2)
                print("height"+str(height))
                print("width"+str(width))
                h_rag = math.radians(31.1)   #水平画角　Horizontal field of view
                v_rag = math.radians(24.4)   #垂直画角　Vertical field of view

                camera_height = 357   #床からカメラまでの高さ

                real_view_x = camera_height * math.tan(h_rag) * 2
                real_view_y = camera_height * math.tan(v_rag) * 2

                miri_per_pixel_x = real_view_x / 640
                miri_per_pixel_y = real_view_y / 480

                pixel_x = x - 320     #(320,385)を原点とした
                pixel_y = y - 380

                real_x = pixel_x * miri_per_pixel_x
                real_y = pixel_y * miri_per_pixel_y

                real_distance = math.sqrt(real_x ** 2 + real_y ** 2)
                #要検証
                #real_y = real_y + 185
                rad = math.atan(real_x / real_y)
                ang = math.degrees(rad)     #ライントレース線から左が正zeitakubito_detection(),右が負

                real_distance = real_distance / 10

                real_distance = round(real_distance, 2)* 0.8  #小数第2位で四捨五入
                real_angle = round(ang, 2)                      #小数第2位で四捨五入
                ENABLE_COMAR = "let"

                #shape = 'm'

            else:
                print('マックス缶なし')
                real_distance = 0
                real_angle = 0
                ENABLE_COMAR = "ban"

        #赤ボールかボールピラミッドの処理
        else:
            ((x, y), radius) = cv2.minEnclosingCircle(c_para)
            #front_image_copy = front_image
            #cv2.circle(front_image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            M = cv2.moments(c_para)

            print( "radius = " + str(radius) )

            if radius > 20:

                imagesensor_size_x = 3.674  #イメージセンサの水平サイズ
                imggesensor_size_y = 2.760  #イメージセンサの水平サイズ

                height = 357   #床からカメラまでの高さ
                focal_length = 3.04  #焦点距離

                real_view_x = imagesensor_size_x * height / focal_length
                real_view_y = imagesensor_size_x * height / focal_length

                miri_per_pixel_x = real_view_x / 640
                miri_per_pixel_y = real_view_y / 480

                pixel_x = x - 320     #(320,385)を原点とした
                pixel_y = y - 380

                real_x = pixel_x * miri_per_pixel_x
                real_y = pixel_y * miri_per_pixel_y

                real_distance = math.sqrt(real_x ** 2 + real_y ** 2)
                #要検証
                #real_y = real_y + 185
                rad = math.atan(real_x / real_y)
                ang = math.degrees(rad)     #ライントレース線から左が正zeitakubito_detection(),右が負

                real_distance = real_distance / 10

                real_distance = round(real_distance, 2) * 0.8  #小数第2位で四捨五入
                real_angle = round(ang, 2)                      #小数第2位で四捨五入
                ENABLE_COMAR = "let"

                #print( " real_distance = " + str( real_distance ) )
                #print( " ang = "+  str( ang ) )

                #if radius > 45 and radius < 55:
                 #   shape ='p'
                  #  print("Pのはず")

                #elif radius > 20 and radius < 25:
                 #   shape = 'r'
                  #  print("rのはず")

                print(shape)

    else:
        print("赤ボールもボールピラミッドも何もなし")
        #ser.write(b"t,10,0,3\n")
        real_distance = 0
        real_angle = 0
        time.sleep(0.1)
        ENABLE_COMAR = "ban"


    object_taple = ( real_distance, real_angle, shape, ENABLE_COMAR )

    return object_taple

#画像処理して得た結果を基にそれぞれの形状について，距離，角度を算出
def hough_calaculate_distance( circles_para, shape ):

    global image
    global front_image

    real_distance = 0
    real_angle = 0
    radius = 0
    ENABLE_COMAR = "ban"


    if not circles_para is None:
        #c = max(c_para, key = cv2.contourArea)
        print("計算開始")
        (x, y, radius) = (circles_para[0], circles_para[1], circles_para[2])
        # 贅沢微糖の処理
        if radius > 45.0:
            shape ='z'

            cv2.circle(image, (x, y), radius, (255, 0, 0), 2)
            cv2.circle(image, (x, y), 1, (0, 255, 0), 3)

            h_rag = math.radians(31.1)   #水平画角　Horizontal field of view
            v_rag = math.radians(24.4)   #垂直画角　Vertical field of view

            imagesensor_size_x = 3.674  #イメージセンサの水平サイズ
            imggesensor_size_y = 2.760  #イメージセンサの水平サイズ

            camera_height = 357   #床からカメラまでの高さ
            focal_length = 3.04  #焦点距離

            real_view_x = imagesensor_size_x * camera_height / focal_length
            real_view_y = imagesensor_size_x * camera_height / focal_length

            miri_per_pixel_x = real_view_x / 640
            miri_per_pixel_y = real_view_y / 480

            pixel_x = x - 320     #(320,370)を原点とした
            pixel_y = y - 380

            real_x = pixel_x * miri_per_pixel_x
            real_y = pixel_y * miri_per_pixel_y

            real_distance = math.sqrt(real_x ** 2 + real_y ** 2)
            #要検証
            #real_y = real_y + 185
            rad = math.atan(real_x / real_y)
            ang = math.degrees(rad)     #ライントレース線から左が正zeitakubito_detection(),右が負

            real_distance = real_distance / 10

            real_distance = round(real_distance, 2)  #小数第2位で四捨五入
            real_angle = round(ang, 2)                     #小数第2位で四捨五入
            ENABLE_COMAR = "let"

    else:
        print('贅沢微糖なし')
        real_distance = 0
        real_angle = 0
        ENABLE_COMAR = "ban"

    object_taple = ( real_distance, real_angle, shape, ENABLE_COMAR )

    return object_taple


#それぞれの形状の距離，角度のデータからRXに送るデータを選択し，RXに送る
def compar_distance_and_send2RX( data1, data2, data3, data4, data5 ):
    if data1[3] == "ban" and data2[3] == "ban" and data3[3] == "ban" and data4[3] == "ban" and data5[3] == "ban":
        print("何もないため，前進")
        ser.write(b"n\n")

    else:
        sum_taple = ( data1 , data2, data3, data4, data5 )
        #result = min( sum_taple,  key = lambda x: x[0] )
        result = min(filter(lambda x: x[3] == 'let', sum_taple), key = lambda x:x[0])
        print( "result = " + str(result))
        comand = bytes("t,{0},{1},{2},{3}\n".format(result[0],result[1],3,result[2]),"ascii")
        ser.write(comand)

    #return result

#3つの結果からそれぞれの形状の距離，角度のデータからRXに送るデータを選択し，RXに送る
def compar_distance_and_send2RX_a( data1, data2, data3 ):
    if data1[3] == "ban" and data2[3] == "ban" and data3[3] == "ban":
        print("何もないため，前進")
        ser.write(b"n\n")

    else:
        sum_taple = ( data1 , data2, data3 )
        #result = min( sum_taple,  key = lambda x: x[0] )
        result = min(filter(lambda x: x[3] == 'let', sum_taple), key = lambda x:x[0])
        print( "result = " + str(result))
        comand = bytes("t,{0},{1},{2},{3}\n".format(result[0],result[1],3,result[2]),"ascii")
        ser.write(comand)

    #return result

#2つの結果からそれぞれの形状の距離，角度のデータからRXに送るデータを選択し，RXに送る
def compar_distance_and_send2RX_d( data1, data2 ):
    if data1[3] == "ban" and data2[3] == "ban":
        print("何もないため，前進")
        ser.write(b"n\n")

    else:
        sum_taple = ( data1 , data2 )
        #result = min( sum_taple,  key = lambda x: x[0] )
        result = min(filter(lambda x: x[3] == 'let', sum_taple), key = lambda x:x[0])
        print( "result = " + str(result))
        comand = bytes("t,{0},{1},{2},{3}\n".format(result[0],result[1],3,result[2]),"ascii")
        ser.write(comand)

    #return result

def main():

    global image #画像処理する画像
    global back_image #背景画像


    #RXからsが送られてくるまで待機
    while True:

        available = ser.in_waiting
        print("waiting", str(available))
        time.sleep(0.1)

        if available > 0:
            values = ser.readline()
            print("Receive",values)
            time.sleep(0.2)

            if values == b"c\r\n":
                # 背景画像を撮る
                #print("背景画面を撮る")
                #camera.capture(rawCapture, format="bgr")
                #back_image = rawCapture.array
                #back_image_hsv = cv2.cvtColor(back_image, cv2.COLOR_BGR2HSV)
                #cv2.imshow("back_image", back_image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                #rawCapture.truncate(0)
                pass

            if values == b"s\r\n":
                print("RXに送ります")

                #カメラで写真を撮る
                camera.capture(rawCapture, format="bgr")
                image = rawCapture.array
                #image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                #cv2.imshow("image", image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                #撮った画像をそれぞれの形状について画像処理をする
                c_para_r, max_area_r = redball_detection()
                real_distance_r, real_angle_r, shape_r, ENABLE_COMAR_r = calaculate_distance(c_para_r, max_area_r, "r")

                c_para_m, max_area_m = maxcoffee_detection()
                real_distance_m, real_angle_m, shape_m, ENABLE_COMAR_m = calaculate_distance(c_para_m, max_area_m, "m")
                #cv2.imshow("image_max", image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                c_para_b, max_area_b = blueball_detection()
                real_distance_b, real_angle_b, shape_b, ENABLE_COMAR_b = calaculate_distance(c_para_b, max_area_b, "b")

                c_para_p, max_area_p = ballpyramid_detection()
                real_distance_p, real_angle_p, shape_p, ENABLE_COMAR_p = calaculate_distance(c_para_p, max_area_p, "p")

                circles_para = cun_detection()
                real_distance_z, real_angle_z, shape_z, ENABLE_COMAR_z = hough_calaculate_distance(circles_para, "z")
                #print(max_area_z)

                #それぞれの結果をタプルに代入
                data_r = ( real_distance_r, real_angle_r, shape_r, ENABLE_COMAR_r )
                data_m = ( real_distance_m, real_angle_m, shape_m, ENABLE_COMAR_m )
                data_b = ( real_distance_b, real_angle_b, shape_b, ENABLE_COMAR_b )
                data_p = ( real_distance_p, real_angle_p, shape_p, ENABLE_COMAR_p )
                data_z = ( real_distance_z, real_angle_z, shape_z, ENABLE_COMAR_z )
                print(" data_r = " + str(data_r) )
                print(" data_m = " + str(data_m) )
                print(" data_b = " + str(data_b) )
                print(" data_p = " + str(data_p) )
                print(" data_z = " + str(data_z) )

                compar_distance_and_send2RX( data_r, data_m, data_b, data_p, data_z )

            if values == b"a\r\n":
                print("RXに送ります")
                print("1，2，4枠の処理")

                #カメラで写真を撮る
                camera.capture(rawCapture, format="bgr")
                image = rawCapture.array

                cv2.line(image, (0,0), (0, 480), (0,0,0), 50)  #トリミング処理
                cv2.line(image, (640,0), (640, 480), (0,0,0), 50)

                #image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                #cv2.imshow("image", image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                #撮った画像をそれぞれの形状について画像処理をする
                c_para_r, max_area_r = redball_detection()
                real_distance_r, real_angle_r, shape_r, ENABLE_COMAR_r = calaculate_distance(c_para_r, max_area_r, "r")

                c_para_b, max_area_b = blueball_detection()
                real_distance_b, real_angle_b, shape_b, ENABLE_COMAR_b = calaculate_distance(c_para_b, max_area_b, "b")

                circles_para = cun_detection()
                real_distance_z, real_angle_z, shape_z, ENABLE_COMAR_z = hough_calaculate_distance(circles_para, "z")
                #print(max_area_z)

                #それぞれの結果をタプルに代入
                data_r = ( real_distance_r, real_angle_r, shape_r, ENABLE_COMAR_r )
                data_b = ( real_distance_b, real_angle_b, shape_b, ENABLE_COMAR_b )
                data_z = ( real_distance_z, real_angle_z, shape_z, ENABLE_COMAR_z )
                print(" data_r = " + str(data_r) )
                print(" data_b = " + str(data_b) )
                print(" data_z = " + str(data_z) )

                compar_distance_and_send2RX_a( data_r, data_b, data_z )

            if values == b"d\r\n":
                print("3枠の処理")
                print("RXに送ります")

                #カメラで写真を撮る
                camera.capture(rawCapture, format="bgr")
                image = rawCapture.array
                #image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                #cv2.imshow("image", image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                #撮った画像をそれぞれの形状について画像処理をする
                c_para_m, max_area_m = maxcoffee_detection()
                real_distance_m, real_angle_m, shape_m, ENABLE_COMAR_m = calaculate_distance(c_para_m, max_area_m, "m")
                #cv2.imshow("image_max", image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                c_para_p, max_area_p = ballpyramid_detection()
                real_distance_p, real_angle_p, shape_p, ENABLE_COMAR_p = calaculate_distance(c_para_p, max_area_p, "p")

                #それぞれの結果をタプルに代入
                data_m = ( real_distance_m, real_angle_m, shape_m, ENABLE_COMAR_m )
                data_p = ( real_distance_p, real_angle_p, shape_p, ENABLE_COMAR_p )
                print(" data_m = " + str(data_m) )
                print(" data_p = " + str(data_p) )

                compar_distance_and_send2RX_d( data_m, data_p )

            else:
                print("信号が違うため何もしません")

        else:
            print("RXからの信号を待機中")

if __name__ == '__main__':
    main()
