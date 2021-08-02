#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from ros_img_process import line, birdeye_view, img_process, draw_lane


import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

FPS = 40
wait = 1000 / FPS


# 화면크기 설정 (기본값)
width = 640
height = 480

roi_Area = 200  # 관심영역 크기

# 차체를 가리기위한 상수
correct_car = 10

roi_Area_correct = roi_Area - correct_car


# 버드아이뷰 설정

src = np.float32(
    [(0, roi_Area), (0, 0), (0.9 * width, roi_Area), (0.95 * width, 0)])
dst = np.float32([(240, roi_Area), (0, 0), (407, roi_Area), (width, 0)])


Line = line()


# 트랙바

# def nothing(x):
#     pass


# def track_bar():
#     cv2.namedWindow("ColorTest")
#     cv2.createTrackbar('low', 'ColorTest', 0, 255, nothing)
#     cv2.createTrackbar('mid', 'ColorTest', 0, 255, nothing)
#     cv2.createTrackbar('high', 'ColorTest', 0, 255, nothing)

#     cv2.namedWindow("Warp_Img")
#     cv2.createTrackbar('width_Top', 'Warp_Img', 0, width // 2, nothing)
#     cv2.createTrackbar('width_Bottom', 'Warp_Img', 0, height, nothing)
#     cv2.createTrackbar('height_Top', 'Warp_Img', 0, width // 2, nothing)
#     cv2.createTrackbar('height_Bottom', 'Warp_Img', 0, height, nothing)


class AD():
    def __init__(self):
        self._sub = rospy.Subscriber(
            '/raspicam_node/image/compressed', CompressedImage, self.callback, queue_size=1)
        self.bridge = CvBridge()

    def callback(self, image_msg):

        # track_bar()

        # converting compressed image to opencv image
        np_arr = np.fromstring(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        img_main = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        img_main = cv2.flip(img_main, -1)

        rows, cols = img_main.shape[:2]
        height = rows
        width = cols
        # 관심영역 설정
        # -60은 차체를 가리기위한 상수
        img_roi = img_main[(height - roi_Area):(height - correct_car), 0:width]

        # 트랙바
        # low = cv2.getTrackbarPos('low', 'ColorTest')
        # mid = cv2.getTrackbarPos('mid', 'ColorTest')
        # high = cv2.getTrackbarPos('high', 'ColorTest')

        # width_Top = cv2.getTrackbarPos('width_Top', 'Warp_Img')
        # height_Top = cv2.getTrackbarPos('height_Top', 'Warp_Img')
        # Width_Bottom = cv2.getTrackbarPos('Width_Bottom', 'Warp_Img')
        # height_Bottom = cv2.getTrackbarPos('height_Bottom', 'Warp_Img')

        # dst = np.float32([(width_Top, height_Top), (width - width_Top, height_Top),
        #                   (Width_Bottom, height_Bottom), (width - Width_Bottom, height_Bottom)])

        # 버드아이뷰
        img_bird, Perspect_back = birdeye_view(
            img_roi, src, dst, (width, roi_Area_correct))

        #cv2.imshow("bird", img_bird)

        # 영상처리
        window = img_process(img_bird, roi_Area_correct, Line)
        #cv2.imshow('test', window)

        # 선 긋기
        img_draw = draw_lane(window, Line, roi_Area_correct)

        #cv2.imshow('Test', img_draw)

        # 버드아이뷰 해제
        img_bird_back = cv2.warpPerspective(
            img_draw, Perspect_back, (width, height))

        # 밑에는 검정이니깐 윗부분 크기만큼만 따서
        result = img_bird_back[:roi_Area, :]

        # 원본이미지에 갖다붙이기
        result = np.vstack((img_main[:height - roi_Area, :], result))

        cv2.imshow("Warp_Img", result)

        # FPS만큼 대기
        # if cv2.waitKey(int(wait)) & 0xFF == 27:
        #    break
        cv2.waitKey(int(wait))

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('AD')

    node = AD()
    node.main()
