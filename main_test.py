# -*- coding: utf-8 -*-
import numpy as np

import tracker
from detector import Detector
import cv2


def traffic_count(list_bboxs, polygon_mask_up_and_down, list_overlapping_first_polygon, list_overlapping_second_polygon, forward_count, backward_count): 
    if len(list_bboxs) > 0:
        # ----------------------判断撞线----------------------
        for item_bbox in list_bboxs:
            x1, y1, x2, y2, _, track_id = item_bbox

            # 撞线的点(中心点)
            x = int(x1 + ((x2 - x1) * 0.5))
            y = int(y1 + ((y2 - y1) * 0.5))

            if polygon_mask_up_and_down[y, x] == 1:
                # 如果撞 first polygon
                if track_id not in list_overlapping_second_polygon:
                    list_overlapping_second_polygon.append(track_id)
                pass

                # 判断 second polygon list 里是否有此 track_id
                # 有此 track_id，则 认为是second polygon 到  first polygon方向
                if track_id in list_overlapping_first_polygon:
                    # 外出+1
                    backward_count += 1

                    print('backward count:', backward_count, ', backward id:', list_overlapping_first_polygon)

                    # 删除 second polygon  list 中的此id
                    list_overlapping_first_polygon.remove(track_id)

                    pass
                else:
                    # 无此 track_id，不做其他操作
                    pass

            elif polygon_mask_up_and_down[y, x] == 2:
                # 如果撞 second polygon 
                if track_id not in list_overlapping_first_polygon:
                    list_overlapping_first_polygon.append(track_id)
                pass

                # 判断 first polygon list 里是否有此 track_id
                # 有此 track_id，则 认为是 进入方向
                if track_id in list_overlapping_second_polygon:
                    # 进入+1
                    forward_count += 1

                    print('forward count:', forward_count, ', forward id:', list_overlapping_second_polygon)

                    # 删除 first polygon list 中的此id
                    list_overlapping_second_polygon.remove(track_id)

                    pass
                else:
                    # 无此 track_id，不做其他操作
                    pass
                pass
            else:
                pass
            pass

        pass

        # ----------------------清除无用id----------------------
        list_overlapping_all = list_overlapping_first_polygon + list_overlapping_second_polygon
        for id1 in list_overlapping_all:
            is_found = False
            for _, _, _, _, _, bbox_id in list_bboxs:
                if bbox_id == id1:
                    is_found = True
                    break
                pass
            pass

            if not is_found:
                # 如果没找到，删除id
                if id1 in list_overlapping_first_polygon:
                    list_overlapping_first_polygon.remove(id1)
                pass
                if id1 in list_overlapping_second_polygon:
                    list_overlapping_second_polygon.remove(id1)
                pass
            pass
        list_overlapping_all.clear()
        pass

        # 清空list
        list_bboxs.clear()

        pass
    else:
        # 如果图像中没有任何的bbox，则清空list
        list_overlapping_second_polygon.clear()
        list_overlapping_first_polygon.clear()
        pass
    pass

    return forward_count, backward_count

#生成一个尺寸为size的图片mask，包含1个polygon，（值范围 0、1、2），供撞线计算使用
#list_point：点数组
#color_value: polygon填充的值
#size：图片尺寸
def image_mask(list_point, color_value, size):
    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros(size, dtype=np.uint8)

    # 初始化撞线polygon
    ndarray_pts = np.array(list_point, np.int32)
    polygon_color_value = cv2.fillPoly(mask_image_temp, [ndarray_pts], color=color_value)
    polygon_color_value = polygon_color_value[:, :, np.newaxis]

    return polygon_color_value
    

if __name__ == '__main__':
    #多边形数组
    list_pts_blue_1  =  [[300*2, 380*2],[800*2, 380*2], [800*2, 390*2],[300*2, 390*2]]
    list_pts_yellow_2=  [[300*2, 370*2],[800*2, 370*2], [800*2, 380*2],[300*2, 380*2]]

    polygon_blue_value_1 = image_mask(list_pts_blue_1, 1, (1080, 1920))
    polygon_yellow_value_2 = image_mask(list_pts_yellow_2, 2, (1080, 1920))
    

    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2
    # 缩小尺寸，1920x1080->960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)
    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)
    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (960, 540))

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

    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture('./video/1_1080.mp4')
    # capture = cv2.VideoCapture('/mnt/datasets/datasets/towncentre/TownCentreXVID.avi')

    while True:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))
        # cv2.imshow("img",im)
        # im = cv2.resize(im, (1920, 1080))

        output_image = cv2.add(im, color_polygons_image)
        # cv2.imshow('demo0', output_image)


        list_bboxs = []
        bboxs_car = []
        bboxes = detector.detect(im)

        for bbox in bboxes:
            if bbox[4] in [ 'car',  'bus', 'truck']:
                bboxs_car.append(bbox)
        if len(bboxs_car) > 0:
            list_bboxs = tracker.update(bboxs_car, im)

        # 如果画面中 有bbox
        # if len(bboxes) > 0:
        #     list_bboxs = tracker.update(bboxes, im)
            # print(list_bboxs)

            # 画框
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
            # cv2.imshow('demo1', output_image_frame)
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass

        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)
        
        # cv2.imshow('demo2', output_image_frame)

        # ----------------------判断撞线----------------------
        up_count, down_count = traffic_count(list_bboxs, polygon_mask_blue_and_yellow, list_overlapping_yellow_polygon, list_overlapping_blue_polygon, up_count, down_count)

        text_draw = 'DOWN: ' + str(down_count) + ' , UP: ' + str(up_count)
        
        # cv2.imshow('demo', output_image_frame)
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(0, 0, 255), thickness=2)

        cv2.imshow('demo3', output_image_frame)
        cv2.waitKey(1)

        pass
    pass

    capture.release()
    cv2.destroyAllWindows()


