from ultralytics import YOLO
import cv2
import numpy as np

'''
    Use YOLOv8
    min-z = 175    

'''

class YOLODetector:

    def __init__(self, model_path):
        '''
        params:
        model_path: path to the model
        device: device to run the model
        '''
        self.model = YOLO(model_path)

    def detect(self, frame):
        '''
        params:
        frame: input frame
        return:
        boxes: detected boxes
        prob: detected probabilities
        '''
        results = self.model(frame)
        boxes = []
        conf = []
        for r in results:
            boxes.append(r.boxes.xywh)
            conf.append(r.boxes.conf)
        return boxes, conf
        

if __name__ == '__main__':
    yolo = YOLODetector('yolov8s.pt')
    frame = cv2.imread('test.jpg')
    boxes, conf = yolo.detect(frame)
    print(boxes)
    print(conf)

    for i in range(3) :
        print(boxes[0][i])
        print(conf[0][i])