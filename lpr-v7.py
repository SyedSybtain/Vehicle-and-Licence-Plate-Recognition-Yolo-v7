import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import pandas as pd
import time
import bluetooth
import re

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    # video_capture = cv2.VideoCapture(0)  # Open the default camera
    # # Set the desired resolution (720p)
    # width = 1280
    # height = 720
    # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # # Define the codec and create a VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_file = cv2.VideoWriter('captured_video.mp4', fourcc, 20.0, (width, height))
    # duration = 10  # Duration in seconds
    # start_time = cv2.getTickCount()
    # while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < duration:
    #     ret, frame = video_capture.read()  # Capture frame-by-frame
    #     output_file.write(frame)  # Write the captured frame to the video file
    #     cv2.imshow('Video Capture', frame)  # Display the resulting frame
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # # Release everything when done
    # video_capture.release()
    # output_file.release()
    # cv2.destroyAllWindows()

    weights = 'yolov7type.pt'
    weights2 = 'yolov7lp.pt'
    weights3 = 'yolov7char.pt'
    lplate = None
    lp_num = None
    characters = None
    source, view_img, save_txt, imgsz, trace = 'sample.mp4', opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    # save_img = not opt.nosave and not isinstance(source, str) and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Make crop Folders
    if not os.path.exists("crop_vehicle"):
        os.mkdir("crop_vehicle")
    crp_cnt1=0

    if not os.path.exists("crop_LP_Num"):
        os.mkdir("crop_LP_Num")
    crp_cnt2=0
    dict1={}
    dict2={}



    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA



################################################## Vehicle Type ##########################################################


    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    model2 = attempt_load(weights2, map_location=device)  # load FP32 model
    stride2 = int(model2.stride.max())  # model stride
    imgsz2 = check_img_size(imgsz, s=stride2)  # check img_size
    model3 = attempt_load(weights3, map_location=device)  # load FP32 model
    stride3 = int(model3.stride.max())  # model stride
    imgsz3 = check_img_size(imgsz, s=stride3)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)
        model2 = TracedModel(model2, device, opt.img_size)
        model3 = TracedModel(model3, device, opt.img_size)

    if half:
        model.half()  # to FP16
        model2.half()  # to FP16
        model3.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im1s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im1s)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, vehicle, im1, frame = path[i], '%g: ' % i, im1s[i].copy(), dataset.count
            else:
                p, vehicle, im1, frame = path, '', im1s, getattr(dataset, 'frame',0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im1.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im1.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    # n = (det[:, -1] == c).sum()  # detections per class
                    vehicle += f"{names[int(c)]}{''}"  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #crop an image based on coordinates
                    object_coordinates = [int (xyxy [0]), int (xyxy [1]), int(xyxy [2]), int(xyxy[3])]
                    cropobj = im1[int (xyxy [1]): int(xyxy [3]), int (xyxy[0]): int(xyxy [2])]
                    #save crop part
                    crop_file_path = os.path.join("crop_vehicle", str(crp_cnt1)+".jpg")
                    cv2.imwrite(crop_file_path,cropobj)
                    crp_cnt1 = crp_cnt1+0


################################################## Licence Plate Number ##########################################################



                    names2 = model2.module.names2 if hasattr(model2, 'module') else model2.names
                    colors2 = [[random.randint(0, 255) for _ in range(3)] for _ in names2]
                    # if trace:
                    #     model2 = TracedModel(model2, device, opt.img_size)
                    # if half:
                    #     model2.half()  # to FP16
                    # Second-stage classifier
                    classify = False
                    if classify:
                        modelc2 = load_classifier(name='resnet101', n=2)  # initialize
                        modelc2.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model2']).to(device).eval()
                    if device.type != 'cpu':
                        model2(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model2.parameters())))  # run once

                    t1 = time_synchronized()
                    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                        pred2 = model2(img, augment=opt.augment)[0]

                    t2 = time_synchronized()
                    # Apply NMS
                    pred2 = non_max_suppression(pred2, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                    t3 = time_synchronized()

                    # Apply Classifier
                    if classify:
                        pred2 = apply_classifier(pred2, modelc2, img, im1s)
                    # Process detections
                    for i, det2 in enumerate(pred2):  # detections per image
                        if webcam:  # batch_size >= 1
                            p, lplate, im1, frame = path[i], '%g: ' % i, im1s[i].copy(), dataset.count
                        else:
                            p, lplate, im1, frame = path, '', im1s, getattr(dataset, 'frame', 0)
                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # img.jpg
                        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                        gn = torch.tensor(im1.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        if len(det2):
                           # Rescale boxes from img_size to im0 size
                            det2[:, :4] = scale_coords(img.shape[2:], det2[:, :4], im1.shape).round()
                        # Print results
                        for c in det2[:, -1].unique():
                            # n = (det[:, -1] == c).sum()  # detections per class
                            lplate += f"{names2[int(c)]}{''}"  # add to string
                        # Write results
                        for *xyxy, conf, cls in reversed(det2):
                            #crop an image based on coordinates
                            object_coordinates = [int (xyxy [0]), int (xyxy [1]), int(xyxy [2]), int(xyxy[3])]
                            cropobj2 = im1[int (xyxy [1]): int(xyxy [3]), int (xyxy[0]): int(xyxy [2])]
                            #save crop part
                            crop_file_path = os.path.join("crop_LP_Num", str(crp_cnt2)+".jpg")
                            cv2.imwrite(crop_file_path,cropobj2)
                            crp_cnt2 = crp_cnt2+0



################################################## Characters ##########################################################

#uptil now 

                            imgchar = 'crop_LP_Num/0.jpg'
                            imgsz3 = opt.img_size
                            # if trace:
                            #     model3 = TracedModel(model3, device, opt.img_size)
                            # if half:
                            #     model3.half()  # to FP16
                            # Second-stage classifier
                            classify = False
                            if classify:
                                modelc3 = load_classifier(name='resnet101', n=2)  # initialize
                                modelc3.load_state_dict(torch.load('weights3/resnet101.pt', map_location=device)['model3']).to(device).eval()
                            # Set Dataloader
                            vid_path, vid_writer = None, None
                            # if webcam:
                            #     view_img = check_imshow()
                            #     cudnn.benchmark = True  # set True to speed up constant image size inference
                            #     dataset3 = LoadStreams(imgchar, img_size=imgsz3, stride=stride3)
                            # else:
                            dataset3 = LoadImages(imgchar, img_size=imgsz3, stride=stride3)
                            names3 = model3.module.names3 if hasattr(model3, 'module') else model3.names
                            colors3 = [[random.randint(0, 255) for _ in range(3)] for _ in names3]
                            if device.type != 'cpu':
                                model3(torch.zeros(1, 3, imgsz3, imgsz3).to(device).type_as(next(model3.parameters())))  # run once
                            t0 = time.time()
                            for path, imgchar, im0s, vid_cap in dataset3:
                                imgchar = torch.from_numpy(imgchar).to(device)
                                imgchar = imgchar.half() if half else imgchar.float()  # uint8 to fp16/32
                                imgchar /= 255.0  # 0 - 255 to 0.0 - 1.0
                                if imgchar.ndimension() == 3:
                                    imgchar = imgchar.unsqueeze(0)
                                t1 = time_synchronized()
                                with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                                    pred3 = model3(imgchar, augment=opt.augment)[0]
                                t2 = time_synchronized()
                                # Apply NMS
                                pred3 = non_max_suppression(pred3, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                                t3 = time_synchronized()
                                # Apply Classifier
                                if classify:
                                    pred3 = apply_classifier(pred3, modelc3, imgchar, im0s)
                                # Process detections
                                for i, det3 in enumerate(pred3):  # detections per image
                                    if webcam:  # batch_size >= 1
                                        p, sc, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset3.count
                                    else:
                                        p, sc, im0, frame = path, '', im0s, getattr(dataset3, 'frame', 0)
                                    p = Path(p)  # to Path
                                    save_path = str(save_dir / p.name)  # img.jpg
                                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset3.mode == 'image' else f'_{frame}')  # img.txt
                                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                                    if len(det3):
                                    # Rescale boxes from img_size to im0 size
                                        det3[:, :4] = scale_coords(imgchar.shape[2:], det3[:, :4], im0.shape).round()
                                    # Print results

                                    for *xyxy,c3 in det3:
                                        n = (det3[:, -1] == c3).sum()  # detections per class
                                        sc = f"{names3[int(c3)]}{'sc' * (n > 1)}"  # add to string
                                        object_coordinates = [int (xyxy [0]), int (xyxy [1]), int(xyxy [2]), int(xyxy[3])]
                                        dict1[sum(object_coordinates)] = sc
                                    dict2=dict1
                                    sorted_data = dict(sorted(dict2.items()))
                                    characters = ''.join(sorted_data.values())
                                    char = re.findall(r'[A-Z]',characters)
                                    numbers = re.findall(r'[\d+]',characters)
                                    lastcharacters = char + numbers 
                                    characters = ''.join(lastcharacters)
                                    dict1.clear()
                                    # print(characters)




##############################################################################################################
# uptil Now








                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names2[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im1s, label=label, color=colors[int(cls)], line_thickness=1)
                        # labelc = f'{names[int(cls)]} {conf:.2f}'
                        # plot_one_box(xyxy, im0, labelc=labelc, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print('\n')
            print(f'{vehicle}  --- Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            if 'lplate' not in locals():
                lplate = "Not Found"
            else:
                lplate = lplate
            if 'lp_num' not in locals():
                lp_num = "Not Found"
            else:
                lp_num = lp_num
            print(f'{lplate}  --- Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            print(f'{characters}  ---Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            t=time.localtime()
            logg_time = time.strftime("%H:%M:%S %p",t)

            log_file = "Logging_File.csv"
            with open(log_file, 'r') as f:
                dataf= pd.DataFrame({"Vehicle_Type":[vehicle],"Licence_Number":[characters],"Logging_Time":logg_time})
                dataf.to_csv(log_file,index=False,mode='a',header=False)


            # Stream results
            if view_img:
                cv2.imshow(str(p), im1s)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im1s)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im1s.shape[1], im1s.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im1s)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        print("Vehicle Type is : ",vehicle)
        print("Licence Plate Number is : ",lplate)
        

    print(f'-------------------------------------------------------------------------. ({time.time() - t0:.3f}s)')







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=['yolov7type.pt','yolov7lp.pt','yolov7char.pt'], help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    # count = 1
    # while count != 0:
    #     with torch.no_grad():
    #         if opt.update:  # update all models (to fix SourceChangeWarning)
    #             for opt.weights in ['best.pt','yolov7lp','yolov7char']:
    #                 print("-----------time---------------")
    #                 detect()
    #                 strip_optimizer(opt.weights)
    #         else:
    #             print("-----------time---------------")
    #             detect()
    # # 
    # # 
    count = 1
    while count != 0:
        with torch.no_grad():
            devicename = "HC-05"
            nearby_devices = bluetooth.discover_devices()
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov7type.pt','yolov7lp.pt','yolov7char.pt']:
                    for addr in nearby_devices:
                        if bluetooth.lookup_name(addr) == devicename:
                            detect()
                            strip_optimizer(opt.weights)
                        else:
                            print("\r---------------------------------------------\r")
            else:
                for addr in nearby_devices:
                    if bluetooth.lookup_name(addr) == devicename:
                        detect()
                    else:
                        print("\r---------------------------------------------\r")
            print("\r----------------------No Device Found--------------------------\r")
                # nearby_devices = bluetooth.discover_devices()
                # for addr in nearby_devices:
                #     if bluetooth.lookup_name(addr) == devicename:
                #         detect()
    
    
    
