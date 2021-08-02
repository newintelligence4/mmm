#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2


class line:
    def __init__(self):
        # 차선검출의 최소 픽셀 수
        self.min_pixel = 50

        self.left_pre_x = []
        self.right_pre_x = []

        self.left_fit = [np.array([False])]
        self.right_fit = [np.array([False])]

        # 데이터를 10번 받았는지 안받았는지 췤
        self.left_fit_flag = True
        self.right_fit_flag = True

        self.plot_left_x = None
        self.plot_right_x = None

        self.plot_y = None

        # 윈도우로갈지 어쩔지??????????????
        self.detected = False

        '''
        # 안정화
        self.stab_flag = False
        self.stab_right_x = 0
        '''

        # 라인 몇개를 평균으로 잡을지
        self.avg_line_num = 10
        #self.avg_num_margin = 15

        self.window_num = 10
        self.window_margin = 40  # 15배수로 설정 권장

        # 흰 픽셀 색칠
        self.left_pixel_color = [10, 0, 100]
        self.right_pixel_color = [100, 0, 10]

        # 라인 색칠
        self.left_line_color = (128, 0, 105)
        self.right_line_color = (128, 0, 105)

        # 면 색칠
        self.lane_color = (224, 145, 206)

        # 슬라이딩 윈도우 색상
        self.window_color = (224, 145, 145)


def birdeye_view(img_roi, src, dst, size):
    Perspect = cv2.getPerspectiveTransform(src, dst)
    Perspect_back = cv2.getPerspectiveTransform(dst, src)
    img_bird = cv2.warpPerspective(
        img_roi, Perspect, size, flags=cv2.INTER_LINEAR)

    return img_bird, Perspect_back


def draw_lane(window, Line, roi_Area_correct):
    # 일단 포인트 설정부터 해야겠지

    #window_zero = np.zeros_like(window[:roi_Area_correct, :])
    #window_zero_plus = np.zeros_like(window[roi_Area_correct +1:, :])

    window_zero = np.zeros_like(window)

    #window_zero = cv2.cvtColor(window_zero, cv2.COLOR_BGR2RGB)
    #window_zero_plus = cv2.cvtColor(window_zero_plus, cv2.COLOR_BGR2RGB)

    # 점 찍을 x, y 좌표 설정
    left_ptr_a = np.array([np.transpose(
        np.vstack([Line.plot_left_x - Line.window_margin / 5, Line.plot_y]))])
    left_ptr_b = np.array([np.flipud(np.transpose(
        np.vstack([Line.plot_left_x + Line.window_margin / 5, Line.plot_y])))])

    left_ptr = np.hstack((left_ptr_a, left_ptr_b))

    right_ptr_a = np.array([np.transpose(
        np.vstack([Line.plot_right_x - Line.window_margin / 5, Line.plot_y]))])
    right_ptr_b = np.array([np.flipud(np.transpose(
        np.vstack([Line.plot_right_x + Line.window_margin / 5, Line.plot_y])))])

    right_ptr = np.hstack((right_ptr_a, right_ptr_b))
    '''
    ptr_a: 행벡터인 x좌표 집합과, y좌표 집합을 세로로 붙여(vstack) 전치시킴(transpose)
    ptr_b: a와 같은 작업 수행 후, 위아래로 뒤집음(filpud)
    ptr: ptr_a와 ptr_b를 가로로 합쳐(hstack) 추후 fillpoly에 활용
    '''

    # 라인테두리 색칠
    cv2.fillPoly(window_zero, np.int_([left_ptr]), Line.left_line_color)
    cv2.fillPoly(window_zero, np.int_([right_ptr]), Line.right_line_color)

    # 이제 라인 사이 색칠
    fill_a = np.array([np.transpose(
        np.vstack([Line.plot_left_x + Line.window_margin / 5, Line.plot_y]))])
    fill_b = np.array([np.flipud(np.transpose(
        np.vstack([Line.plot_right_x - Line.window_margin / 5, Line.plot_y])))])
    fill_ptr = np.hstack((fill_a, fill_b))

    cv2.fillPoly(window_zero, np.int_([fill_ptr]), Line.lane_color)

    #window_zero = np.vstack((window_zero, window_zero_plus))

    #cv2.imshow("asdf", window)
    #cv2.imshow("test", window_zero_a)

    img_draw = cv2.addWeighted(window, 1, window_zero, 0.3, 0)
    #cv2.imshow("asdf", img_draw)

    img_draw = img_draw[:roi_Area_correct, :]

    return img_draw
    #cv2.imshow('agas', window)


# 축적된 x값집합의 집합들을 이용해 x좌표의 평균 집합 제작
def average_line(pre_x, avg_line_num, roi_Area_correct):

    # array 합치기
    pre_x = np.squeeze(pre_x)

    '''
    이전에 append하면서 생성된
    [array([237., 239., 245., ...]), array([....]), array([....])] 를
    [ [237. 239. 245. ...] [....] [.....] ..] 로 합쳐줌
    '''

    # y에 대응하는 2차방정식의 x값이기 때문에 y보다 많이 나올수가 없음
    left_x_avg = np.zeros((roi_Area_correct))

    for i, line in enumerate(reversed(pre_x)):
        if i == avg_line_num:
            break
        left_x_avg += line
    left_x_avg = left_x_avg / avg_line_num

    return left_x_avg


def binary_image(img_bird, roi_Area_correct, Line):

    # 색공간 설정
    img_bird_gray = cv2.cvtColor(img_bird, cv2.COLOR_BGR2GRAY)
    img_bird_hls = cv2.cvtColor(img_bird, cv2.COLOR_BGR2HLS)
    img_bird_hls = cv2.inRange(img_bird_hls, (6, 120, 65), (100, 255, 255))

    #cv2.imshow("ColorTest", img_bird_hls)

    # 영상처리
    kernel = np.ones((3, 3), np.uint8)
    img_bird_hls = cv2.erode(img_bird_hls, kernel)
    img_bird_hls = cv2.dilate(img_bird_hls, kernel)

    img_canny = cv2.Canny(img_bird_gray, 200, 255)

    img_bird_result = img_canny | img_bird_hls

    return img_bird_result


def line_detect(img_bird, img_bird_result, roi_Area_correct, Line):

    # 슬라이딩 윈도우 옵션 (높이설정)
    window_height = int(roi_Area_correct / Line.window_num)

    #window = np.dstack((img_bird_result, img_bird_result, img_bird_result)) * 255
    window = img_bird
    #window = cv2.cvtColor(img_bird_result, cv2.COLOR_GRAY2BGR)

    # zero가 아닌애들의 인덱스 반환
    nonzero = img_bird_result.nonzero()
    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])

    # print(img_bird_result.nonzero()[0])

    # 히스토그램을 통한 라인 구별
    histogram = np.sum(img_bird_result, axis=0)

    midpoint = np.int(img_bird.shape[1] / 2)
    left_x = np.argmax(histogram[:midpoint])
    right_x = np.argmax(histogram[midpoint:]) + midpoint

    '''
    # 안정화 코드
    if Line.stab_flag == True:
        right_x = Line.stab_right_x
    '''

    window_left_lane = []
    window_right_lane = []

    # 첫 박스의 갑작스런 탈출을 막기위한 예외처리 (구현예정)
    '''
    append 된 것 중에서 뒤에서 두번째 박스의 값을 다음프레임 첫번째 박스의 값으로 설정할 예정
    '''

    # 윈도우 생성
    for i in range(Line.window_num):

        win_low = roi_Area_correct - i * window_height
        win_high = roi_Area_correct - (i + 1) * window_height
        win_lt_low = left_x - Line.window_margin
        win_lt_high = left_x + Line.window_margin
        win_rt_low = right_x - Line.window_margin
        win_rt_high = right_x + Line.window_margin

        # 윈도우 색칠
        cv2.rectangle(window, (win_lt_low, win_low),
                      (win_lt_high, win_high), Line.window_color, 2)
        cv2.rectangle(window, (win_rt_low, win_low),
                      (win_rt_high, win_high), Line.window_color, 2)

        # 박스안에 있는 nonzero 좌표 추출
        win_lt_ind = ((nonzero_y >= win_high) & (nonzero_y <= win_low) & (nonzero_x >= win_lt_low) & (
            nonzero_x <= win_lt_high)).nonzero()[0]
        win_rt_ind = ((nonzero_y >= win_high) & (nonzero_y <= win_low) & (nonzero_x >= win_rt_low) & (
            nonzero_x <= win_rt_high)).nonzero()[0]
        '''
        nonzero_y, nonzero_x(행 벡터)의 관계 및 논리 연산 후, 생성된 행벡터*에 nonzero()[0] 함수를 사용하여 참인 애들의 좌표값이있는 곳의 index를 받음
        (추후에 받은 좌표값으로 다양한 연산 진행) 

        * 이때 행벡터는 좌표가 True 또는 False로 변환되어있는 상태
            ex) [1 1 0 1]  -> 관계식 및 논리식 -> [F F T F]
                이때, nonzero()[0]을 수행하면 [2]가 나오고 이 [2]를 가지고 각각 nonzero_y와 nonzero_x에 접근하면 (y, x) 좌표값 도출가능


        일단은 1차로 nonzero를 통해 사진에서 0이 아닌곳의 좌표를 추려내고
        2차로 nonzero의 x와 y좌표를 분리
        이후 각각의 조건식과 논리식을 통해 T와 F의 상태로 만들어놓고, T의 인덱스를 저장

        나중에 인덱스를 이용해 nonzero또는 nonzero_x, nonzero_y에 각각 접근해 데이터 뽑아먹기 가능
        '''

        # 추출된 x좌표들 리스트에 삽입 //for문 도는만큼 리스트에 쌓임
        window_left_lane.append(win_lt_ind)
        window_right_lane.append(win_rt_ind)

        # 만약, 최소 픽셀 수 이상이라면, x좌표들의 평균을 다음 박스로
        # but 거리차이가 window 크기 이상이라면 그냥 이전값 유지 (구현예정)
        if len(win_lt_ind) > Line.min_pixel:
            left_x = np.int(np.mean(nonzero_x[win_lt_ind]))

            '''
            이전에 구했던 조건이 참인 인덱스를 nonzero_x에 넣어,
            사진에서의 x좌표값을 받아냄.

            이때의 x좌표는 사진에서 0이 아닌곳의 위치를 뜻함!
            '''
        if len(win_rt_ind) > Line.min_pixel:
            right_x = np.int(np.mean(nonzero_x[win_rt_ind]))
            '''
            # 안정화 코드
            Line.stab_flag = True
            '''

    '''
    # 안정화 코드
    if Line.stab_flag == True:
        wwindow = np.squeeze(window_right_lane)
        for i, x in enumerate(reversed(wwindow)):
            # 2번째 박스를 픽!
            if i == 2 & len(x) > 1:
                print(x)
                Line.stab_right_x = np.int(np.mean(nonzero_x[x]))
                break
    '''

    # 리스트가 for문 만큼 생기기 때문에 합쳐줘야함
    window_left_lane = np.concatenate(window_left_lane)
    window_right_lane = np.concatenate(window_right_lane)
    '''
    즉, 영상에서 값이 0이 아니고, 박스안에있는 픽셀들의 좌표가 저장된 nonzero의 인덱스가 저장되는 셈
    '''
    # print(window_left_lane)

    # 인덱스를 이용해 좌표 빼내기
    line_lt_x = nonzero_x[window_left_lane]
    line_lt_y = nonzero_y[window_left_lane]

    line_rt_x = nonzero_x[window_right_lane]
    line_rt_y = nonzero_y[window_right_lane]
    '''
    마찬가지로 이전에 구했던 조건들에 부합한 인덱스를 통해
    좌표값들을 get
    '''

    # 검출된 픽셀에 색칠하기
    #window[line_lt_y, line_lt_x] = Line.left_pixel_color
    #window[line_rt_y, line_rt_x] = Line.right_pixel_color

    '''
    2차함수 그래프를 그릴걸 생각하면 x와 y의 위치를 바꾸는게 마즘

    이제부터 축 바꿔서 생각해! (역함수 처럼)
    '''

    # 이후 계산한 방정식에 대입할 y값 (plot 생각해)
    plot_y = np.linspace(0, img_bird.shape[0] - 1, img_bird.shape[0])
    '''
    linspace 형식 [1. 2. 3. 4. 5. ...]
    '''

    Line.plot_y = plot_y

    # if Line.left_fit_flag | Line.right_fit_flag :
    '''
    이걸로 데이터가 충분히 쌓인상태이면 두번 좌표 딸 일 없게하려했는데
    오히려 나중엔 평균값만돌아서 고정됨.
    '''

    # y, x값과 차수를 넣으면 해당 방정식의 계수를 찾아줌
    left_fit = np.polyfit(line_lt_y, line_lt_x, 2)
    right_fit = np.polyfit(line_rt_y, line_rt_x, 2)

    # y값에 따른 x값 계산하는거죵
    left_plot_x = left_fit[0] * plot_y ** 2 + \
        left_fit[1] * plot_y + left_fit[2]
    right_plot_x = right_fit[0] * plot_y ** 2 + \
        right_fit[1] * plot_y + right_fit[2]
    '''
    마찬가지로 이때 left_plot_x 형식 [237. 239. 245. ...]
    '''

    # y값에 따른 x값 저장
    Line.left_pre_x.append(left_plot_x)
    Line.right_pre_x.append(right_plot_x)
    '''
    리스트형식에 append를 하게되면 [array([237., 239., 245., ....])]

    근데 이제 main의 while문이 돌며 append가 계속 되니깐
    [array([237., 239., 245., ...]), array([....]), array([....]), ...반복]
    
    이때 left_pre_x의 len은 while문이 반복된 만큼임 (안에있는 원소의 총 개수가 아님!)
    '''

    # 누적된 x값의 집합들이 n개 이상 나왔다면
    # if len(Line.left_pre_x) > Line.avg_line_num:
    #    left_avg = average_line()

    if len(Line.left_pre_x) > Line.avg_line_num:

        # 이전 x집합들의 평균값 집합
        #left_x_avg = average_line(Line.left_pre_x, Line.avg_line_num, height)

        left_plot_x = average_line(
            Line.left_pre_x, Line.avg_line_num, roi_Area_correct)

        #left_plot_x = left_x_avg
        '''
        어차피 평균값으로 한번 더 구할거면 1Line.avg_line_num(현재값 10), 만큼 데이터가 쌓이고 난 다음부턴
        굳이 위에서 left_plot_x 계산해 줄 필요 없음 (일단 보류)
        
        --> if문 생성완료
        '''

        # 좌표 저장 (y는 이미 저장함)
        Line.plot_left_x = left_plot_x

        # 계수 한번더 얻어내기
        #left_fit_avg = np.polyfit(plot_y, left_x_avg, 2)
        # 방정식으로 x값 한번더 도출
        #left_plot_x = left_fit_avg[0] * plot_y ** 2 + left_fit_avg[1] * plot_y + left_fit_avg[2]
        # 계수 저장 (이후 라인검출 함수에서 쓰임))
        #Line.left_fit = left_fit_avg

        '''
        if len(Line.left_pre_x) >= Line.avg_num_margin:
            Line.left_fit_flag = False
        '''

    else:
        # 좌표 저장 (드로잉 시 필요)
        Line.plot_left_x = left_plot_x

        # 계수 저장 (의미없음)
        Line.left_fit = left_fit

        # 예외처리
        #Line.left_fit_flag = True

    if len(Line.right_pre_x) > Line.avg_line_num:

        right_plot_x = average_line(
            Line.right_pre_x, Line.avg_line_num, roi_Area_correct)

        # 좌표 저장
        Line.plot_right_x = right_plot_x

        # 계수 저장
        #Line.right_fit = right_fit_avg
        # 평균값으로 방정식 한번 더 도출 및 계수 저장(필요없지요?)
        #right_fit_avg = np.polyfit(plot_y, right_x_avg, 2)

        #right_plot_x = right_fit_avg[0] * plot_y ** 2 + right_fit_avg[1] * plot_y + right_fit_avg[2]

    else:
        # 계수 저장
        Line.right_fit = right_fit
        Line.plot_right_x = right_plot_x

    #left_x_avg = np.average(left_plot_x)
    #right_x_avg = np.average(right_plot_x)
    '''
    나중에 얘네들로 윈도우 보정할 것
    '''

    #cv2.imshow("Test", window)
    Line.detected = True

    return window
    # 윈도우 안에있는 nonzero 픽셀들 확인
    # 몇개 이상있는지에 따라서 결정

    #cv2.imshow("ColorTest", img_bird_result)


def post_line_detect(img_bird, img_bird_result, roi_Area_correct, Line):

    # 값 가져오기
    window_margin = Line.window_margin

    # 슬라이딩 윈도우 옵션 (높이설정)
    window_height = int(roi_Area_correct / Line.window_num)

    # 사진 가져오기
    window = img_bird

    # zero가 아닌애들의 인덱스 반환
    nonzero = img_bird_result.nonzero()
    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])

    # line_detect()에서 저장한 계수들을 통해 y값에 따른 x값 계산
    # 다만 다른건 y값을 plot_y가 아닌 그냥 nonzero_y를 사용한다는 점?
    leftx_min = Line.left_fit[0] * nonzero_y ** 2 + \
        Line.left_fit[1] * nonzero_y + Line.left_fit[2] - window_margin
    leftx_max = Line.left_fit[0] * nonzero_y ** 2 + \
        Line.left_fit[1] * nonzero_y + Line.left_fit[2] + window_margin
    rightx_min = Line.right_fit[0] * nonzero_y ** 2 + \
        Line.right_fit[1] * nonzero_y + Line.right_fit[2] - window_margin
    rightx_max = Line.right_fit[0] * nonzero_y ** 2 + \
        Line.right_fit[1] * nonzero_y + Line.right_fit[2] + window_margin

    # 이전엔 window안에 있는 흰 픽셀 좌표의 인덱스를 뺐다면
    # 지금은 방금 계산한 x값 범위안에 있는 흰 픽셀의 인덱스만 뺌
    win_lt_ind = ((nonzero_x >= leftx_min) & (
        nonzero_x <= leftx_max)).nonzero()[0]
    win_rt_ind = ((nonzero_x >= rightx_min) & (
        nonzero_x <= rightx_max)).nonzero()[0]

    # 인덱스를 이용해 좌표 빼내기
    line_lt_x = nonzero_x[win_lt_ind]
    line_lt_y = nonzero_y[win_lt_ind]

    line_rt_x = nonzero_x[win_rt_ind]
    line_rt_y = nonzero_y[win_rt_ind]

    # 색칠띠
    window[line_lt_y, line_lt_x] = Line.left_pixel_color
    window[line_rt_y, line_rt_x] = Line.right_pixel_color

    # 값 불러오기
    plot_y = Line.plot_y

    # 검출한 애들로 방정식 도출
    left_fit = np.polyfit(line_lt_y, line_lt_x, 2)
    right_fit = np.polyfit(line_rt_y, line_rt_x, 2)

    # y값에 따른 x값 계산하는거죵
    left_plot_x = left_fit[0] * plot_y ** 2 + \
        left_fit[1] * plot_y + left_fit[2]
    right_plot_x = right_fit[0] * plot_y ** 2 + \
        right_fit[1] * plot_y + right_fit[2]

    # y값에 따른 x값 저장
    Line.left_pre_x.append(left_plot_x)
    Line.right_pre_x.append(right_plot_x)

    if len(Line.left_pre_x) > Line.avg_line_num:

        # 이전 x집합들의 평균값 집합
        left_plot_x = average_line(
            Line.left_pre_x, Line.avg_line_num, roi_Area_correct)

        # 좌표 저장 (y는 이미 저장함)
        Line.plot_left_x = left_plot_x

    else:
        # 좌표 저장 (드로잉 시 필요)
        Line.plot_left_x = left_plot_x

        # 계수 저장 (의미없음)
        Line.left_fit = left_fit

    if len(Line.right_pre_x) > Line.avg_line_num:

        right_plot_x = average_line(
            Line.right_pre_x, Line.avg_line_num, roi_Area_correct)

        # 좌표 저장
        Line.plot_right_x = right_plot_x

    else:
        # 좌표 저장
        Line.plot_right_x = right_plot_x

        # 계수 저장
        Line.right_fit = right_fit

    # 급작스런 변화에 대응키위해 다시 원래 line_detect로
    # 이거는 간소화된 버전이라서, 급작스러울땐 슬라이딩윈도우 사용해야해
    standard = np.std(right_plot_x - left_plot_x)

    if (standard > 60):
        Line.detected = False

    return window


def img_process(img_bird, roi_Area_correct, Line):

    # 2진화 후 색공간 추출
    img_bird_result = binary_image(img_bird, roi_Area_correct, Line)

    # 라인 검출 안되면, 혹은 변화가 심하면 슬라이딩
    if Line.detected == False:
        return line_detect(img_bird, img_bird_result, roi_Area_correct, Line)
    # 검출되면 그냥 post루다가
    else:
        return post_line_detect(img_bird, img_bird_result, roi_Area_correct, Line)
