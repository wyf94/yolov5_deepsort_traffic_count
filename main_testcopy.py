# -*- coding: utf-8 -*-
import numpy as np

import tracker
from detector import Detector
import cv2

"""
#生成一个尺寸为size的图片mask，包含1个polygon，（值范围 0、1、2），供撞线计算使用
#list_point：点数组
#color_value: polygon填充的值
#size：图片尺寸
"""
def image_mask(list_point, color_value, size):
    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros(size, dtype=np.uint8)

    # 初始化撞线polygon
    ndarray_pts = np.array(list_point, np.int32)
    polygon_color_value = cv2.fillPoly(mask_image_temp, [ndarray_pts], color=color_value)
    polygon_color_value = polygon_color_value[:, :, np.newaxis]

    return polygon_color_value
    
def traffic_count(image, frame_count, list_bboxs, polygon_mask_first_and_second, first_list, second_list,  up_count, down_count):
    first_num = 0
    second_num = 0
    point_radius = 3

    if len(list_bboxs) > 0:
        for item_bbox in list_bboxs:
            x1, y1, x2, y2, cls_id, conf = item_bbox
            
            # 撞线的点(中心点)
            x = int(x1 + ((x2 - x1) * 0.5))
            y = int(y1 + ((y2 - y1) * 0.5))

            if polygon_mask_first_and_second[y, x] == 1 or polygon_mask_first_and_second[y, x]  ==3:
                first_num += 1
            elif polygon_mask_first_and_second[y, x] == 2:
                second_num += 1

            #画出中心list_bboxs的中心点
            list_pts = []
            list_pts.append([x-point_radius, y-point_radius])
            list_pts.append([x-point_radius, y+point_radius])
            list_pts.append([x+point_radius, y+point_radius])
            list_pts.append([x+point_radius, y-point_radius])
            ndarray_pts = np.array(list_pts, np.int32)
            image = cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))           

        if frame_count > 2:
            second_list.pop(0)
            first_list.pop(0)

        first_list.append(first_num)
        second_list.append(second_num)
        # print("first_num", first_num)
        # print("second_num", second_num)

        if frame_count > 2 and first_list[0] > first_list[1]:
            first_diff = first_list[0] - first_list[1]
            second_diff =  second_list[1] - second_list[0]
            if first_diff == second_diff:
                up_count += first_diff
                print('up count:', up_count)
        elif frame_count >2 and second_list[0] > second_list[1]:
            second_diff =  second_list[0] - second_list[1]
            first_diff = first_list[1] - first_list[0]
            if first_diff == second_diff:
                down_count += first_diff  
                print('down count:', down_count)

    return up_count, down_count             

def polygon_mask(point_list_first, point_list_second, size):
    polygon_value_first = image_mask(point_list_first, 1, size)
    polygon_value_second = image_mask(point_list_second, 2, size)
    
    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_first_and_second = polygon_value_first + polygon_value_second

    # set the first  polygon to blue
    blue_color_plate = [255, 0, 0]
    blue_image = np.array(polygon_value_first * blue_color_plate, np.uint8)
    # set the first  polygon to yelllow
    yellow_color_plate = [0, 255, 255]
    yellow_image = np.array(polygon_value_second * yellow_color_plate, np.uint8)
    # 彩色图片（值范围 0-255）用于图片显示
    polygon_color_image = blue_image + yellow_image

    # polygon_mask_first_and_second = cv2.resize(polygon_mask_first_and_second, size)
    # polygon_color_image = cv2.resize(polygon_color_image, size)

    return polygon_mask_first_and_second,  polygon_color_image

if __name__ == '__main__':
    #多边形数组
    # list_pts_blue_1  =  [[300*2, 380*2],[800*2, 380*2], [800*2, 385*2],[300*2, 385*2]]
    # list_pts_yellow_2=  [[300*2, 375*2],[800*2, 375*2], [800*2, 380*2],[300*2, 380*2]]

    # polygon_blue_value_1 = image_mask(list_pts_blue_1, 1, (1080, 1920))
    # polygon_yellow_value_2 = image_mask(list_pts_yellow_2, 2, (1080, 1920))
    

    # # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    # polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2
    # # 缩小尺寸，1920x1080->960x540
    # polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))

    # # 蓝 色盘 b,g,r
    # blue_color_plate = [255, 0, 0]
    # # 蓝 polygon图片
    # blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)
    # # 黄 色盘
    # yellow_color_plate = [0, 255, 255]
    # # 黄 polygon图片
    # yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)
    # # 彩色图片（值范围 0-255）
    # polygon_color_image = blue_image + yellow_image
    # # 缩小尺寸，1920x1080->960x540
    # polygon_color_image = cv2.resize(polygon_color_image, (960, 540))

    point_blue = [[300 , 390 ],[800 , 390 ], [800 , 400],[300 , 400 ]]
    point_yellow =  [[300 , 380],[800 , 380 ], [800 , 390 ],[300 , 390 ]]
    # point_blue =  [[300*2, 380*2],[800*2, 380*2], [800*2, 390*2],[300*2, 390*2]]
    # point_yellow =   [[300*2, 370*2],[800*2, 370*2], [800*2, 380*2],[300*2, 380*2]]

    # polygon_mask_blue_and_yellow,  polygon_color_image = polygon_mask(point_blue, point_yellow,(1080, 1920))
    polygon_mask_blue_and_yellow,  polygon_color_image = polygon_mask(point_blue, point_yellow,(540, 960))
    # polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))
    # polygon_color_image = cv2.resize(polygon_color_image, (960, 540))
    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []
    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 进入数量
    down_count = 0
    # 离开数量
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))
    # draw_text_postion = (int(1920 * 0.01), int(1080 * 0.05))

    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture('./video/1_1080.mp4')
    # capture = cv2.VideoCapture('/mnt/datasets/datasets/towncentre/TownCentreXVID.avi')

    blue_list = []
    yellow_list = []

    cur_frame = 0

    while True:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break

        cur_frame += 1

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))
        # cv2.imshow("img",im)

        list_bboxs = []
        bboxes = detector.detect(im)

        for bbox in bboxes:
            if bbox[4] in [ 'car',  'bus', 'truck']:
                list_bboxs.append(bbox)

        # for bbox in bboxes:
        #     list_bboxs.append(bbox)
        # print(list_bboxs)
        # print(len(list_bboxs))

        up_count, down_count = traffic_count(im, cur_frame,  list_bboxs, polygon_mask_blue_and_yellow, blue_list, yellow_list,  up_count, down_count)


        output_image_frame = cv2.add(im, polygon_color_image)

        text_draw = 'DOWN: ' + str(down_count) + ' , UP: ' + str(up_count)
        
        # cv2.imshow('demo', output_image_frame)
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(0, 0, 255), thickness=2)
        cv2.imshow('demo3', output_image_frame)
        cv2.waitKey(1)
            

    capture.release()
    cv2.destroyAllWindows()


