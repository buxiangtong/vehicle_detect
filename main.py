from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon

import sys
import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2

import tracker
from detector import Detector

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import colors, plot_one_box, plot_one_box_PIL
from utils.plots import Annotator, colors, save_one_box

from utils.torch_utils import select_device
from utils.capnums import Camera
from dialog.rtsp_win import Window

class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(list)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = '0'
        self.barricade = []
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False                   # jump out of the loop
        self.is_continue = True                 # continue/pause
        self.percent_length = 1000              # progress bar
        self.rate_check = True                  # Whether to enable delay
        self.rate = 100
        self.save_fold = './result'
        self.left_count = 0
        self.right_count = 0
        self.up_count = 0 
        self.down_count = 0
        self.left_count1 = 0
        self.right_count1 = 0
        self.left_count2 = 0
        self.right_count2 = 0
        self.left_count3 = 0
        self.right_count3 = 0
        self.left_count4 = 0
        self.right_count4 = 0 
        self.count=0
        self.left_id = []
        self.right_id = []
        self.up_id = [] 
        self.down_id = []
        self.barricade_set = 0
        self.direction = 0
        self.left_id1 = []
        self.right_id1 = []
        self.left_id2 = []
        self.right_id2 = []
        self.left_id3 = []
        self.right_id3 = []
        self.left_id4 = []
        self.right_id4 = []
        self.output = ""
        self.turn = 0
    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):
    
        if self.barricade_set == 0:
            if self.direction == 0:
                list_pts_blue = [[960,1],[960,1078],[980,1078],[980,1]]
                list_pts_yellow = [[940,1],[940,1078],[960,1078],[960,1]]
            else:
                list_pts_blue = [[1,520],[1,540],[1919,540],[1919,520]]
                list_pts_yellow = [[1,540],[1,560],[1919,560],[1919,540]]
            mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)       
            ndarray_pts_blue = np.array(list_pts_blue, np.int32)
            polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
            polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]        
            mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)        
            ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
            polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
            polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]
            polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2
            polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))
            blue_color_plate = [255, 0, 0]
            blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)
            yellow_color_plate = [0, 255, 255]
            yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)
            color_polygons_image = blue_image + yellow_image
            color_polygons_image = cv2.resize(color_polygons_image, (960, 540))
            list_overlapping_blue_polygon = []
            list_overlapping_yellow_polygon = []
        else:
            # x111=760
            # y111=500
            # x222=1300
            # y222=1000
            # print(self.barricade)
            [x111,y111,x222,y222] = self.barricade     
        #左边竖线
            mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
            list_pts_blue1 = [[x111,y111],[x111,y222],[x111+20,y222],[x111+20,y111]]
            ndarray_pts_blue1 = np.array(list_pts_blue1, np.int32)
            polygon_blue_value_11 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue1], color=1)
            polygon_blue_value_11 = polygon_blue_value_11[:, :, np.newaxis]
            
            mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
            list_pts_yellow1 = [[x111-20,y111],[x111-20,y222],[x111,y222],[x111,y111]]
            ndarray_pts_yellow1 = np.array(list_pts_yellow1, np.int32)
            polygon_yellow_value_12 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow1], color=2)
            polygon_yellow_value_12 = polygon_yellow_value_12[:, :, np.newaxis]

            polygon_mask_blue_and_yellow1 = polygon_blue_value_11 + polygon_yellow_value_12
            polygon_mask_blue_and_yellow1 = cv2.resize(polygon_mask_blue_and_yellow1, (960, 540))
            blue_color_plate = [71,99,255]# blue_color_plate = [255, 0, 0]
            blue_image1 = np.array(polygon_blue_value_11 * blue_color_plate, np.uint8)
            yellow_color_plate = [192, 192, 192]# yellow_color_plate = [0, 255, 255]
            yellow_image1 = np.array(polygon_yellow_value_12 * yellow_color_plate, np.uint8)
            color_polygons_image1 = blue_image1 + yellow_image1
            color_polygons_image1 = cv2.resize(color_polygons_image1, (960, 540))
            list_overlapping_blue1_polygon = []
            list_overlapping_yellow1_polygon = []
            #右边竖线
            mask_image_temp1 = np.zeros((1080, 1920), dtype=np.uint8)
            list_pts_red2 = [[x222,y111],[x222,y222],[x222+20,y222],[x222+20,y111]]
            ndarray_pts_red2 = np.array(list_pts_red2, np.int32)
            polygon_red_value_21 = cv2.fillPoly(mask_image_temp1, [ndarray_pts_red2], color=1)
            polygon_red_value_21 = polygon_red_value_21[:, :, np.newaxis]
        
            mask_image_temp1 = np.zeros((1080, 1920), dtype=np.uint8)
            list_pts_green2 = [[x222-20,y111],[x222-20,y222],[x222,y222],[x222,y111]]
            ndarray_pts_green2 = np.array(list_pts_green2, np.int32)
            polygon_green_value_22 = cv2.fillPoly(mask_image_temp1, [ndarray_pts_green2], color=2)
            polygon_green_value_22 = polygon_green_value_22[:, :, np.newaxis]

            polygon_mask_red_and_green2 = polygon_red_value_21 + polygon_green_value_22
            polygon_mask_red_and_green2 = cv2.resize(polygon_mask_red_and_green2, (960, 540))
            red_color_plate = [192,192,192]#red_color_plate = [0, 0, 255]
            red_image2 = np.array(polygon_red_value_21 * red_color_plate, np.uint8)
            green_color_plate = [71, 99, 255]#green_color_plate = [0, 255, 0]
            green_image2 = np.array(polygon_green_value_22 * green_color_plate, np.uint8)
            color_polygons_image2 = red_image2 + green_image2
            color_polygons_image2 = cv2.resize(color_polygons_image2, (960, 540))
            list_overlapping_red_polygon = []
            list_overlapping_green_polygon = []
            #上线
            mask_image_temp2 = np.zeros((1080, 1920), dtype=np.uint8)
            list_pts_a13 = [[x111+20,y111],[x111+20,y111+20],[x222,y111+20],[x222,y111]]
            ndarray_pts_a13 = np.array(list_pts_a13, np.int32)
            polygon_a1_value_31 = cv2.fillPoly(mask_image_temp2, [ndarray_pts_a13], color=1)
            polygon_a1_value_31 = polygon_a1_value_31[:, :, np.newaxis]
        
            mask_image_temp2 = np.zeros((1080, 1920), dtype=np.uint8)
            list_pts_a23 = [[x111+20,y111+20],[x111+20,y111+40],[x222,y111+40],[x222,y111+20]]
            ndarray_pts_a23 = np.array(list_pts_a23, np.int32)
            polygon_a2_value_32 = cv2.fillPoly(mask_image_temp2, [ndarray_pts_a23], color=2)
            polygon_a2_value_32 = polygon_a2_value_32[:, :, np.newaxis]

            polygon_mask_a1_and_a23 = polygon_a1_value_31 + polygon_a2_value_32
            polygon_mask_a1_and_a23 = cv2.resize(polygon_mask_a1_and_a23, (960, 540))
            a1_color_plate = [140, 199, 0]#a1_color_plate = [192, 192, 192]
            a1_image3 = np.array(polygon_a1_value_31 * a1_color_plate, np.uint8)
            a2_color_plate = [201, 161, 51]#a2_color_plate = [71,99,255]
            a2_image3 = np.array(polygon_a2_value_32 * a2_color_plate, np.uint8)
            color_polygons_image3 = a1_image3 + a2_image3
            color_polygons_image3 = cv2.resize(color_polygons_image3, (960, 540))
            list_overlapping_a1_polygon = []
            list_overlapping_a2_polygon = []
            #下线
            mask_image_temp3 = np.zeros((1080, 1920), dtype=np.uint8)
            list_pts_a34 = [[x111+20,y222-40],[x111+20,y222-20],[x222,y222-20],[x222,y222-40]]
            ndarray_pts_a34 = np.array(list_pts_a34, np.int32)
            polygon_a3_value_41 = cv2.fillPoly(mask_image_temp3, [ndarray_pts_a34], color=1)
            polygon_a3_value_41 = polygon_a3_value_41[:, :, np.newaxis]
        
            mask_image_temp3 = np.zeros((1080, 1920), dtype=np.uint8)
            list_pts_a44 = [[x111+20,y222-20],[x111+20,y222],[x222,y222],[x222,y222-20]]
            ndarray_pts_a44 = np.array(list_pts_a44, np.int32)
            polygon_a4_value_42 = cv2.fillPoly(mask_image_temp3, [ndarray_pts_a44], color=2)
            polygon_a4_value_42 = polygon_a4_value_42[:, :, np.newaxis]

            polygon_mask_a3_and_a44 = polygon_a3_value_41 + polygon_a4_value_42
            polygon_mask_a3_and_a44 = cv2.resize(polygon_mask_a3_and_a44, (960, 540))
            a3_color_plate = [0, 69, 255]#a3_color_plate = [71, 99, 255]
            a3_image4 = np.array(polygon_a3_value_41 * a3_color_plate, np.uint8)
            a4_color_plate = [71, 99, 255]#a4_color_plate = [192,192,192]
            a4_image4 = np.array(polygon_a4_value_42 * a4_color_plate, np.uint8)
            color_polygons_image4 = a3_image4 + a4_image4
            color_polygons_image4 = cv2.resize(color_polygons_image4, (960, 540))
            list_overlapping_a3_polygon = []
            list_overlapping_a4_polygon = []
            #填充区
            mask_image = np.zeros((1080, 1920), dtype=np.uint8)
            list_pts = [[x111,y111],[x111,y222],[x222,y222],[x222,y111]]
            ndarray_pts = np.array(list_pts, np.int32)
            polygon = cv2.fillPoly(mask_image, [ndarray_pts],color=1)
            polygon = polygon[:, :, np.newaxis]
            color_plate = [255,0,0]
            color_area = np.array(polygon * color_plate, np.uint8)
            area = cv2.resize(color_area, (960, 540))
            
        # Initialize
        try:
            device = select_device(device)
            half &= device.type != 'cpu'  # half precision only supported on CUDA
            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if half:
                model.half()  # to FP16

            # Dataloader
            # if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
            #     view_img = check_imshow()
            #     cudnn.benchmark = True  # set True to speed up constant image size inference
            #     dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
            #     # bs = len(dataset)     # batch_size
            # else:
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            count = 0   
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset) 

            while True:
                if self.jump_out:   # 开始是false
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('Stop')  #send_msg 发送信号 stop
                    self.left_count = 0
                    self.right_count = 0
                    self.up_count = 0 
                    self.down_count = 0 
                    self.left_id = []
                    self.right_id = []
                    self.up_id = [] 
                    self.down_id = []
                    self.left_id1 = []
                    self.right_id1 = []
                    self.left_id2 = []
                    self.right_id2 = []
                    self.left_id3 = []
                    self.right_id3 = []
                    self.left_id4 = []
                    self.right_id4 = []
                    self.barricade_set = 0
                    self.turn = 0
                    if hasattr(self, 'out'):    #hasattr() 函数用于判断对象是否包含对应的属性。
                        self.out.release()
                    break

                # change model
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                    if half:
                        model.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    self.current_weight = self.weights

                if self.is_continue:    #一开始为true
                    path, img, im0s, self.vid_cap = next(dataset)
                    # print(self.barricade)
                    count += 1
                    if count % 30 == 0 and count >= 30:
                        fps = int(30/(time.time()-start_time))
                        self.send_fps.emit('fps:'+str(fps))    #发送帧率
                        start_time = time.time()                #刷新开始时间
                    if self.vid_cap:
                        percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length) #进度条显示
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length
                    list_bboxs = []
                    if self.barricade_set == 0:
                        if self.direction == 0: 
                            statistic_dic = [('LEFT',self.left_count),
                                            ('RIGHT',self.right_count),
                                            ('left_id',self.left_id),
                                            ('right_id',self.right_id)]
                        else :
                            statistic_dic = [('UP',self.up_count),
                                            ('DOWN',self.down_count),
                                            ('up_id',self.up_id),
                                            ('down_id',self.down_id)]
                    else:
                        if self.direction == 0: 
                            statistic_dic = [('Illegal Region',self.barricade),
                                            ('illegal vehicle',self.right_id1+self.left_id2+self.left_id3+self.left_id4),
                                            ('out illegal vehicle',self.left_id1+self.right_id2+self.right_id3+self.right_id4)]
                    self.output = ""
                    if self.right_id1+self.left_id2+self.left_id3+self.left_id4 != 0:
                        self.turn +=1
                    for i in statistic_dic: 
                        self.output += ' '+str(i[0]) + ':' + str(i[1]) +'\n'
                    # print(self.output)               
                    # statistic_dic.update({name: 0 for name in names})
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img, augment=augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)
                   
                    bboxes = []
                    im0s = cv2.resize(im0s,(960, 540))
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                if names[c] not in ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']:
                                    continue
                                # statistic_dic[names[c]] += 1
                                x1, y1 = int(xyxy[0]), int(xyxy[1])
                                x2, y2 = int(xyxy[2]), int(xyxy[3])
                                bboxes.append((x1, y1, x2, y2,names[c],conf))
                    if len(bboxes) > 0:     #bbox 为预测框 x1, y1, x2, y2, lbl, conf im为一帧一帧的图片
                        list_bboxs = tracker.update(bboxes, im0)      #对一帧的检测框进行处理                             
                    output_image_frame = tracker.draw_bboxes(im0, list_bboxs, line_thickness=None)
                    pass
                    if self.barricade_set == 0:
                        if self.rate_check:
                            time.sleep(1/self.rate)
                        output_image_frame_color = cv2.add(output_image_frame, color_polygons_image)
                    else:
                        if self.rate_check:
                            time.sleep(1/self.rate)
                        output_image_frame_color = cv2.add(output_image_frame, area)                       
                        # output_image_frame = cv2.add(output_image_frame, color_polygons_image1)
                        # output_image_frame = cv2.add(output_image_frame, color_polygons_image2)
                        # output_image_frame = cv2.add(output_image_frame, color_polygons_image3)
                        # output_image_frame = cv2.add(output_image_frame, color_polygons_image4)
                        # 函数 cv2.add() 对两张相同大小和类型的图像进行加法运算，或对一张图像与一个标量进行加法运算。
                        # 对一张图像与一个标量相加时，则将图像所有像素的各通道值分别与标量的各通道值相加。
                    if self.barricade_set == 0:
                        if len(list_bboxs) > 0: #(x1, y1, x2, y2, label, track_id)
                            # ----------------------判断撞线----------------------
                            for item_bbox in list_bboxs:
                                x1, y1, x2, y2, label, track_id = item_bbox

                                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                                # y1_offset = int(y1 + ((y2 - y1) * 0.6))

                                # 撞线的点
                                y = int((y1+y2)//2)
                                x = int((x1+x2)//2)
                                if polygon_mask_blue_and_yellow[y, x] == 1:
                                    # 如果撞 蓝polygon
                                    if track_id not in list_overlapping_blue_polygon:       #如果船不在蓝矩形内
                                        list_overlapping_blue_polygon.append(track_id)      #登记
                                    pass

                                    # 判断 黄polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 外出方向
                                    if track_id in list_overlapping_yellow_polygon:
                                        # 外出+1
                                        if self.direction == 0:
                                            self.right_count += 1
                                            self.right_id.append(track_id)
                                            # 删除 黄polygon list 中的此id
                                            list_overlapping_yellow_polygon.remove(track_id)
                                        elif self.direction == 1:
                                            self.up_count += 1
                                            self.up_id.append(track_id)
                                            # 删除 黄polygon list 中的此id
                                            list_overlapping_yellow_polygon.remove(track_id)
                                        pass
                                    else:
                                        # 无此 track_id，不做其他操作
                                        pass

                                elif polygon_mask_blue_and_yellow[y, x] == 2:
                                    # 如果撞 黄polygon
                                    if track_id not in list_overlapping_yellow_polygon:     
                                        list_overlapping_yellow_polygon.append(track_id)
                                    pass

                                    # 判断 蓝polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 进入方向
                                    if track_id in list_overlapping_blue_polygon:
                                        # 进入+1
                                        if self.direction == 0:
                                            self.left_count += 1
                                            self.left_id.append(track_id)
                                            # 删除 蓝polygon list 中的此id
                                            list_overlapping_blue_polygon.remove(track_id)
                                        elif self.direction == 1:
                                            self.down_count += 1
                                            self.down_id.append(track_id)
                                            # 删除 蓝polygon list 中的此id
                                            list_overlapping_blue_polygon.remove(track_id)
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
                            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
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
                                    if id1 in list_overlapping_yellow_polygon:
                                        list_overlapping_yellow_polygon.remove(id1)
                                    pass
                                    if id1 in list_overlapping_blue_polygon:
                                        list_overlapping_blue_polygon.remove(id1)
                                    pass
                                pass
                            list_overlapping_all.clear()
                            pass
                            # 清空list
                            list_bboxs.clear()
                            pass
                        else:
                            # 如果图像中没有任何的bbox，则清空list
                            list_overlapping_blue_polygon.clear()
                            list_overlapping_yellow_polygon.clear()
                            pass
                        pass
                    else:
                        if len(list_bboxs) > 0: #(x1, y1, x2, y2, label, track_id)
                        # ----------------------判断撞线----------------------
                            for item_bbox in list_bboxs:
                                x1, y1, x2, y2, label, track_id = item_bbox

                                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                                # y1_offset = int(y1 + ((y2 - y1) * 0.6))

                                # 撞线的点
                                y = int((y1+y2)//2)
                                x = int((x1+x2)//2)
                    #此处判断左侧撞线
                                if polygon_mask_blue_and_yellow1[y, x] == 1:
                                    # 如果撞 蓝polygon
                                    if track_id not in list_overlapping_blue1_polygon:       #如果船不在蓝矩形内
                                        list_overlapping_blue1_polygon.append(track_id)      #登记
                                    pass

                                    # 判断 黄polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 外出方向
                                    if track_id in list_overlapping_yellow1_polygon:
                                        # 外出+1
                                        self.right_count1 += 1
                                        self.count+=1
                                        self.right_id1.append(track_id)
                                        # 删除 黄polygon list 中的此id
                                        list_overlapping_yellow1_polygon.remove(track_id)

                                        pass
                                    else:
                                        # 无此 track_id，不做其他操作
                                        pass

                                elif polygon_mask_blue_and_yellow1[y, x] == 2:
                                    # 如果撞 黄polygon
                                    if track_id not in list_overlapping_yellow1_polygon:     
                                        list_overlapping_yellow1_polygon.append(track_id)
                                    pass

                                    # 判断 蓝polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 进入方向
                                    if track_id in list_overlapping_blue1_polygon:
                                        # 进入+1
                                        self.left_count1 += 1
                                        self.count-=1
                                        self.left_id1.append(track_id)
                                        # 删除 蓝polygon list 中的此id
                                        list_overlapping_blue1_polygon.remove(track_id)
                                        pass
                                    else:
                                        # 无此 track_id，不做其他操作
                                        pass
                                    pass
                                else:
                                    pass
                                pass
                        #此处判断右侧撞线
                                if polygon_mask_red_and_green2[y, x] == 1:
                                    # 如果撞 蓝polygon
                                    if track_id not in list_overlapping_red_polygon:       #如果船不在蓝矩形内
                                        list_overlapping_red_polygon.append(track_id)      #登记
                                    pass

                                    # 判断 黄polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 外出方向
                                    if track_id in list_overlapping_green_polygon:
                                        # 外出+1
                                        self.right_count2 += 1
                                        self.count-=1
                                        self.right_id2.append(track_id)
                                        # 删除 黄polygon list 中的此id
                                        list_overlapping_green_polygon.remove(track_id)

                                        pass
                                    else:
                                        # 无此 track_id，不做其他操作
                                        pass

                                elif polygon_mask_red_and_green2[y, x] == 2:
                                    # 如果撞 黄polygon
                                    if track_id not in list_overlapping_green_polygon:     
                                        list_overlapping_green_polygon.append(track_id)
                                    pass

                                    # 判断 蓝polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 进入方向
                                    if track_id in list_overlapping_red_polygon:
                                        # 进入+1
                                        self.left_count2 += 1
                                        self.count+=1
                                        self.left_id2.append(track_id)
                                        # 删除 蓝polygon list 中的此id
                                        list_overlapping_red_polygon.remove(track_id)
                                        pass
                                    else:
                                        # 无此 track_id，不做其他操作
                                        pass
                                    pass
                                else:
                                    pass
                                pass
                        #此处判断上侧撞线
                                if polygon_mask_a1_and_a23[y, x] == 1:
                                    # 如果撞 蓝polygon
                                    if track_id not in list_overlapping_a1_polygon:       #如果船不在蓝矩形内
                                        list_overlapping_a1_polygon.append(track_id)      #登记
                                    pass

                                    # 判断 黄polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 外出方向
                                    if track_id in list_overlapping_a2_polygon:
                                        # 外出+1
                                        self.right_count3 += 1
                                        self.count-=1
                                        self.right_id3.append(track_id)
                                        # 删除 黄polygon list 中的此id
                                        list_overlapping_a2_polygon.remove(track_id)

                                        pass
                                    else:
                                        # 无此 track_id，不做其他操作
                                        pass

                                elif polygon_mask_a1_and_a23[y, x] == 2:
                                    # 如果撞 黄polygon
                                    if track_id not in list_overlapping_a2_polygon:     
                                        list_overlapping_a2_polygon.append(track_id)
                                    pass

                                    # 判断 蓝polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 进入方向
                                    if track_id in list_overlapping_a1_polygon:
                                        # 进入+1
                                        self.left_count3 += 1
                                        self.count+=1
                                        self.left_id3.append(track_id)
                                        # 删除 蓝polygon list 中的此id
                                        list_overlapping_a1_polygon.remove(track_id)
                                        pass
                                    else:
                                        # 无此 track_id，不做其他操作
                                        pass
                                    pass
                                else:
                                    pass
                                pass
                        #此处判断下侧撞线
                                if polygon_mask_a3_and_a44[y, x] == 1:
                                    # 如果撞 蓝polygon
                                    if track_id not in list_overlapping_a3_polygon:       #如果船不在蓝矩形内
                                        list_overlapping_a3_polygon.append(track_id)      #登记
                                    pass

                                    # 判断 黄polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 外出方向
                                    if track_id in list_overlapping_a4_polygon:
                                        # 外出+1
                                        self.right_count4 += 1
                                        self.count-=1
                                        self.right_id4.append(track_id)
                                        # 删除 黄polygon list 中的此id
                                        list_overlapping_a4_polygon.remove(track_id)

                                        pass
                                    else:
                                        # 无此 track_id，不做其他操作
                                        pass

                                elif polygon_mask_a3_and_a44[y, x] == 2:
                                    # 如果撞 黄polygon
                                    if track_id not in list_overlapping_a4_polygon:     
                                        list_overlapping_a4_polygon.append(track_id)
                                    pass

                                    # 判断 蓝polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 进入方向
                                    if track_id in list_overlapping_a3_polygon:
                                        # 进入+1
                                        self.left_count4 += 1
                                        self.count+=1
                                        self.left_id4.append(track_id)
                                        # 删除 蓝polygon list 中的此id
                                        list_overlapping_a3_polygon.remove(track_id)
                                        pass
                                    else:
                                        # 无此 track_id，不做其他操作
                                        pass
                                    pass
                                else:
                                    pass
                                pass
                            # self.count = (self.left_count2+self.right_count1-self.right_count2-self.left_count1)
                            pass
                            # ----------------------清除无用id----------------------
                            list_overlapping_all = list_overlapping_yellow1_polygon + list_overlapping_blue1_polygon +list_overlapping_red_polygon + list_overlapping_green_polygon + list_overlapping_a1_polygon + list_overlapping_a2_polygon + list_overlapping_a3_polygon + list_overlapping_a4_polygon
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
                                    if id1 in list_overlapping_yellow1_polygon:
                                        list_overlapping_yellow1_polygon.remove(id1)
                                    pass
                                    if id1 in list_overlapping_blue1_polygon:
                                        list_overlapping_blue1_polygon.remove(id1)
                                    pass
                                    if id1 in list_overlapping_red_polygon:
                                        list_overlapping_red_polygon.remove(id1)
                                    pass
                                    if id1 in list_overlapping_green_polygon:
                                        list_overlapping_green_polygon.remove(id1)
                                    pass
                                    if id1 in list_overlapping_a1_polygon:
                                        list_overlapping_a1_polygon.remove(id1)
                                    pass
                                    if id1 in list_overlapping_a2_polygon:
                                        list_overlapping_a2_polygon.remove(id1)
                                    pass
                                    if id1 in list_overlapping_a3_polygon:
                                        list_overlapping_a3_polygon.remove(id1)
                                    pass
                                    if id1 in list_overlapping_a4_polygon:
                                        list_overlapping_a4_polygon.remove(id1)
                                    pass
                                pass
                            list_overlapping_all.clear()
                            pass
                            # 清空list
                            list_bboxs.clear()
                            pass
                        else:
                            # 如果图像中没有任何的bbox，则清空list
                            list_overlapping_blue1_polygon.clear()
                            list_overlapping_yellow1_polygon.clear()
                            list_overlapping_red_polygon.clear()
                            list_overlapping_green_polygon.clear()
                            list_overlapping_a1_polygon.clear()
                            list_overlapping_a2_polygon.clear()
                            list_overlapping_a3_polygon.clear()
                            list_overlapping_a4_polygon.clear()
                            pass
                        pass
                    im0 = cv2.resize(output_image_frame_color,(im0s.shape[1],im0s.shape[0]))
                    if self.barricade_set == 1:
                        if self.save_fold:
                            os.makedirs(self.save_fold, exist_ok=True)
                            os.makedirs(self.save_fold+'/crops', exist_ok=True)
                            [scx1,scy1,scx2,scy2]=[x111//2,y111//2,x222//2,y222//2]
                            if self.turn:
                                screenshot = output_image_frame[scx1:scx2,scy1:scy2,:]
                                size = (4*(x222-x111),4*(y222-y111)) if 4*(x222-x111)<1920 and 4*(y222-y111)<1080 else (960,540)
                                screenshot = cv2.resize(screenshot,size)
                                save_path = os.path.join(self.save_fold+'/crops' ,'{}'.format(self.turn) + '.jpg')#'1'
                                cv2.imwrite(save_path,screenshot)
                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_statistic.emit(statistic_dic)
                    if self.save_fold:
                        os.makedirs(self.save_fold, exist_ok=True)
                        if self.vid_cap is None:
                            save_path = os.path.join(self.save_fold,
                                                     time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                   time.localtime()) + '.jpg')
                            cv2.imwrite(save_path, im0)
                        else:
                            if count == 1:
                                ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                                if ori_fps == 0:
                                    ori_fps = 25
                                width, height = im0.shape[1], im0.shape[0]
                                save_path = os.path.join(self.save_fold, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.mp4')
                                self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                           (width, height))
                            self.out.write(im0)
                    if percent == self.percent_length:
                        # print(count)
                        if self.save_fold:
                            save_path = os.path.join(self.save_fold, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.txt')
                            file = open(save_path,'w')
                            file.write(self.output)
                            file.close()
                        self.send_percent.emit(0)
                        self.send_msg.emit('finished')
                        if hasattr(self, 'out'):
                            self.out.release()
                        break

        except Exception as e:
            self.send_msg.emit('%s' % e)


class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False

        # style 1: window can be stretched
        # self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)

        # style 2: window can not be stretched
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)#设置窗口函数
        # self.setWindowOpacity(0.85)  # Transparency of window

        self.minButton.clicked.connect(self.showMinimized)  #半屏
        self.maxButton.clicked.connect(self.max_or_restore) #全屏
        # show Maximized window
        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close) #关闭界面

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # search models automatically
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5 thread
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type   #选择权重的位置
        self.det_thread.source = '0'    #摄像头
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_direction)
        self.rtspButton.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.saveCheckBox.clicked.connect(self.is_save)
        self.load_setting()

    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def is_save(self):
        if self.saveCheckBox.isChecked():
            self.det_thread.save_fold = './result'
        else:
            self.det_thread.save_fold = None
    def checkrate(self):
        if self.checkBox.isChecked():
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False
    def chose_rtsp(self):#——————————————————————————————————————————————————————————————————————————————————————
        self.det_thread.barricade_set = 1  #执行危险区域
        MessageBox(
            self.closeButton, title='mark', text='请标注出违规区的左上，右下坐标，例：x0 y0 x1 y1', time=1000, auto=True).exec_()
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "0 0 1920 1080"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        # try:
        self.stop()
        self.det_thread.barricade = [int(x) for x in ip.split(' ')] 
        new_config = {"ip": ip}
        new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
        with open('config/ip.json', 'w', encoding='utf-8') as f:
            f.write(new_json)
        # self.statistic_msg('Loading rtsp：{}'.format(ip))
        self.rtsp_window.close()
        # except Exception as e:
        #     self.statistic_msg('%s' % e)
    def chose_direction(self):
            # if self.cameraButton.isChecked():
            if self.det_thread.direction == 0:
                MessageBox(self.cameraButton, title='切换模式', text='切换到横向计数', time=1000, auto=True).exec_()
                self.det_thread.direction = 1
            else:
                MessageBox(self.cameraButton, title='切换模式', text='切换到竖向计数', time=1000, auto=True).exec_()
                self.det_thread.direction = 0
            #     self.saveCheckBox.setEnabled(False)
            #     self.det_thread.is_continue = True
            # if not self.det_thread.isRunning():
            #     self.det_thread.start()
            # source = os.path.basename(self.det_thread.source)
            # source = 'camera' if source.isnumeric() else source
            # self.statistic_msg('Detecting >> model：{}，file：{}'.
            #                    format(os.path.basename(self.det_thread.weights),
            #                           source))
        
            
            # self.stop()
            # MessageBox(
            #     self.closeButton, title='Tips', text='Loading camera', time=2000, auto=True).exec_()
            # # get the number of local cameras
            # _, cams = Camera().get_cam_num()
            # popMenu = QMenu()
            # popMenu.setFixedWidth(self.cameraButton.width())
            # popMenu.setStyleSheet('''
            #                                 QMenu {
            #                                 font-size: 16px;
            #                                 font-family: "Microsoft YaHei UI";
            #                                 font-weight: light;
            #                                 color:white;
            #                                 padding-left: 5px;
            #                                 padding-right: 5px;
            #                                 padding-top: 4px;
            #                                 padding-bottom: 4px;
            #                                 border-style: solid;
            #                                 border-width: 0px;
            #                                 border-color: rgba(255, 255, 255, 255);
            #                                 border-radius: 3px;
            #                                 background-color: rgba(200, 200, 200,50);}
            #                                 ''')

            # for cam in cams:
            #     exec("action_%s = QAction('%s')" % (cam, cam))
            #     exec("popMenu.addAction(action_%s)" % cam)

            # x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            # y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            # y = y + self.cameraButton.frameGeometry().height()
            # pos = QPoint(x, y)
            # action = popMenu.exec_(pos)
            # if action:
            #     self.det_thread.source = action.text()
            #     self.statistic_msg('Loading camera：{}'.format(action.text()))
            # except Exception as e:
            #     self.statistic_msg('%s' % e)

    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "check": check,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                check = 0
                savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                check = config['check']
                savecheck = config['savecheck']
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save()

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
            self.det_thread.iou_thres = x/100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        output = self.det_thread.output
        MessageBox(
        self.closeButton, title='output', text=output, mod=1,time=3000, auto=True).exec_()
        if msg == "Finished":
            self.saveCheckBox.setEnabled(True)
            
    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Change model to %s' % x)

    def open_file(self):

        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('Detecting >> model：{}，file：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('Pause')

    def stop(self):
        self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True)

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            # statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            # statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + ':' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        self.det_thread.jump_out = True
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config['savecheck'] = self.saveCheckBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        # MessageBox(
        #     self.closeButton, title='Tips', text='Closing the program', time=2000, auto=True).exec_()
        sys.exit(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    # myWin.showMaximized()
    sys.exit(app.exec_())


