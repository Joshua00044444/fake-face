# -*- coding: utf-8 -*-

# 代码是一个基于PyTorch的目标检测模型的推理部分，主要完成了以下功能：
import argparse
import random

import cv2  # cv2 是 OpenCV 库的 Python 接口，用于图像处理。
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, check_img_size, scale_coords
from utils.torch_utils import time_synchronized, select_device

# 加载模型权重和相关参数。
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='./weights/liveness-best.pt',
                    help='model.pt path(s)')  # 模型路径仅支持.pt文件
parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')  # 检测图像大小，仅支持480
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')  # 置信度阈值
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')  # 来设置 NMS（非极大值抑制）的 IOU 阈值
# 选中运行机器的GPU或者cpu，有GPU则GPU，没有则cpu，若想仅使用cpu，可以填cpu即可
parser.add_argument('--device', default='0',
                    help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--save-dir', type=str, default='inference', help='directory to save results')  # 文件保存路径
parser.add_argument('--classes', nargs='+', type=int,
                    help='filter by class: --class 0, or --class 0 2 3')  # 分开类别
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')  # 使用NMS
opt = parser.parse_args()  # opt局部变量，重要
out, weight, imgsz = opt.save_dir, opt.weights, opt.img_size  # 得到文件保存路径，文件权重路径，图像尺寸
device = select_device(opt.device)  # 检验计算单元,gpu还是cpu
half = device.type != 'cpu'  # 如果使用gpu则进行半精度推理

model = attempt_load(weight, map_location=device)  # 读取模型
imgsz = check_img_size(imgsz, s=model.stride.max())  # 检查图像尺寸
if half:  # 如果是半精度推理
    model.half()  # 转换模型的格式
names = model.module.names if hasattr(model, 'module') else model.names  # 得到模型训练的类别名

colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]  # 给每个类别一个颜色
img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # 创建一个图像进行预推理
_ = model(img.half() if half else img) if device.type != 'cpu' else None  # 预推理


# 在主程序中，定义了 predict 函数用于进行图像推理，返回预测结果和推理时间；
# cv_imread 函数用于读取图片文件；
# plot_one_box 函数用于在图像上绘制边界框。


# 进行图像推理，获取预测结果。
# 使用模型进行推理，最后进行非极大值抑制（NMS）处理并返回预测结果和推理时间
def predict(img):
    img = torch.from_numpy(img).to(device)  # 使用模型进行推理，最后进行非极大值抑制（NMS）处理并返回预测结果和推理时间
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # 如果图像是单张图像，则在第 0 维上增加一个维度，用于模型推理

    t1 = time_synchronized()
    pred = model(img, augment=False)[0]  # augment=False 表示不进行数据增强
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                               agnostic=opt.agnostic_nms)  # 对预测结果进行非极大值抑制处理，根据置信度阈值、IOU 阈值和类别信息进行筛选。
    t2 = time_synchronized()
    InferNms = round((t2 - t1), 2)  # 返回我们整个推理的时间的大小

    return pred, InferNms


def cv_imread(filePath):
    # 读取图片
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)

    if len(cv_img.shape) > 2:
        if cv_img.shape[2] > 3:
            cv_img = cv_img[:, :, :3]
    return cv_img


# 对预测结果进行处理，包括筛选、缩放、绘制边界框等操作。
def plot_one_box(img, x, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# 展示检测结果。
# 通过 OpenCV 显示检测结果图像，并等待按键输入。
if __name__ == '__main__':
    img_path = "./UI_rec/test_/548622_png.rf.9981b4857d51eaefade4893f6bf8780d.jpg"
    image = cv_imread(img_path)
    image = cv2.resize(image, (850, 500))
    img0 = image.copy()
    img = letterbox(img0, new_shape=imgsz)[0]  # 使用 letterbox 函数将图像 img0 调整到指定的尺寸 new_shape=imgsz，并取返回结果的第一个元素
    img = np.stack(img, 0)  # 沿着新轴将数组堆叠起来，将 img 转换为 NumPy 数组。
    img = img[:, :, ::-1].transpose(2, 0, 1)  # 将图像通道顺序从 BGR 转换为 RGB，并将图像的维度顺序调整为 3x416x416
    img = np.ascontiguousarray(img)  # 创建一个连续的（contiguous）数组，以便在后续处理中能够更高效地访问图像数据。

    pred, useTime = predict(img)  # 进行预测推理的过程

    det = pred[0]
    p, s, im0 = None, '', img0
    if det is not None and len(det):  # 如果有检测信息则进入
        det[:, :4] = scale_coords(img.shape[1:], det[:, :4], im0.shape).round()  # 将检测框的坐标缩放至原始图像 im0 的尺寸，对坐标进行四舍五入处理。
        number_i = 0  # 类别预编号
        detInfo = []  # 存储检测的信息
        for *xyxy, conf, cls in reversed(det):  # 遍历检测信息
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))  # 将检测框的坐标转换为左上角和右下角的坐标形式。
            # 将检测信息添加到字典中，包括类别名称、坐标信息和置信度。
            detInfo.append([names[int(cls)], [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf])
            number_i += 1  # 编号数+1

            label = '%s %.2f' % (names[int(cls)], conf)

            # 画出检测到的目标物
            plot_one_box(image, xyxy, label=label, color=colors[int(cls)])
    # 实时显示检测画面
    cv2.imshow('检测画面Stream', image)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    # break
    c = cv2.waitKey(0) & 0xff
