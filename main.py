#!/usr/bin/env python
# <Copyright 2020. ech97. All rights reserved.>

import numpy as np
import cv2
from img_process import line, birdeye_view, img_process, draw_lane
'''
도전과제
1. 리스트에 이전 x좌표들 넣을 때, 비울 방법 생각 

2. 파라미터를 나중에 일괄 관리하며, 차선에 맞춰 Threshold 조정할수있게 제작

★★ 아마 밑에가 휘는건 그래프를 그릴때 쓸데없는 y좌표까지 들어가서인듯함 ★★★
'''


'''
윈도우 관련 상수
: img_process의 line class 참조

슬라이딩 윈도우 확인
: img_process에서 cv2.imshow("test", window)

픽셀검출 확인
: img_process에서 '검출된 픽셀에 색칠하기' 찾기

버드아이뷰
: 나중에 트랙바 함수만 따로 빼서 사용

'''


# 동영상 이름 설정
video_name = "project_video.mp4"

# 프레임 설정
FPS = 40
wait = 1000 / FPS


# 화면크기 설정 (기본값)
width = 1280
height = 720

roi_Area = 250  # 관심영역 크기

# 차체를 가리기위한 상수
correct_car = 20

roi_Area_correct = roi_Area - correct_car


# 버드아이뷰 설정

src = np.float32(
    [(0, roi_Area), (0, 0), (0.9 * width, roi_Area), (0.95 * width, 0)])
dst = np.float32([(480, roi_Area), (0, 0), (815, roi_Area), (width, 0)])


Line = line()

'''
# 트랙바
def nothing(x):
    pass

def track_bar(width, height):

    
    cv2.namedWindow("ColorTest")
    cv2.createTrackbar('low', 'ColorTest', 0, 255, nothing)
    cv2.createTrackbar('mid', 'ColorTest', 0, 255, nothing)
    cv2.createTrackbar('high', 'ColorTest', 0, 255, nothing)


    cv2.namedWindow("Warp_Img")
    cv2.createTrackbar('width_Top', 'Warp_Img', 0, width//2, nothing)
    cv2.createTrackbar('width_Bottom', 'Warp_Img', 0, height, nothing)
    cv2.createTrackbar('height_Top', 'Warp_Img', 0, width//2, nothing)
    cv2.createTrackbar('height_Bottom', 'Warp_Img', 0, height, nothing)
'''


if __name__ == '__main__':

    #track_bar(width, roi_Area)

    cap = cv2.VideoCapture(video_name)

    while (cap.isOpened()):
        _, frame = cap.read()

        img_main = frame

        # 화면 크기 조정
        rows, cols = img_main.shape[:2]
        height = rows
        width = cols

        # 관심영역 설정
        # -60은 차체를 가리기위한 상수
        img_roi = img_main[(height - roi_Area):(height - correct_car), 0:width]

        # 트랙바
        '''
        low = cv2.getTrackbarPos('low', 'ColorTest')
        mid = cv2.getTrackbarPos('mid', 'ColorTest')
        high = cv2.getTrackbarPos('high', 'ColorTest')
        '''
        
        '''
        width_Top = cv2.getTrackbarPos('width_Top', 'Warp_Img')
        height_Top = cv2.getTrackbarPos('height_Top', 'Warp_Img')
        Width_Bottom = cv2.getTrackbarPos('Width_Bottom', 'Warp_Img')
        height_Bottom = cv2.getTrackbarPos('height_Bottom', 'Warp_Img')
        dst = np.float32([(width_Top, height_Top), (width-width_Top, height_Top), (Width_Bottom, height_Bottom), (width-Width_Bottom, height_Bottom)])
        '''

        # 버드아이뷰
        img_bird, Perspect_back = birdeye_view(
            img_roi, src, dst, (width, roi_Area_correct))

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
        if cv2.waitKey(int(wait)) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
