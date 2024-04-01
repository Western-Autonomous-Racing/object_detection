import numpy as np
import torch

'''
    Use YOLOv8 with int8 quantization
    min-z = 175    

'''

class YOLO_Detector:

    def __init__(self, model_path, device):
        '''
        params:
        model_path: path to the model
        device: device to run the model
        '''
        self.model = torch.load(model_path)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    