import cv2
import numpy as np
import matplotlib.pyplot as plt
import setting
from PIL import Image


def get_obj_img_by_bbox(frame, bbox):
    """[Given a image and a bounding box it will return cropped image]

    Args:
        frame ([3d numpy array]): [It's an image]
        bbox ([list]): [bounding box with x, y, width, height]

    Returns:
        [np.array]: [cropped image]
    """
    img = Image.fromarray(frame)
    img = img.crop(bbox)
    img = np.asarray(img)
    return img

def get_yolo_v3_net():
    """[Create yolo v3 object detection net object]

    Returns:
        [type]: [description]
    """
    net = cv2.dnn.readNet(setting.YOLO_V3_WEIGHTS_PATH, setting.YOLO_V3_CONFIG_PATH)
    return net

def get_object_detection_result(net_obj, frame):
    with open(setting.COCO_NAMES, "r") as f:
        classes = [line.strip() for line in f.readlines()]
        layer_names = net_obj.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net_obj.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net_obj.setInput(blob)
    outs = net_obj.forward(output_layers)
    result = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                dct = {}
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                dct['box'] = [x, y, w, h]
                dct['confidence'] = float(confidence)
                dct['class_ids'] = class_id
                result.append(dct)
    return result