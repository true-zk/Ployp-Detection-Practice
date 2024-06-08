"""
@Description: adapted from https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard
@Author: Ken Zh0ng
@date: 2024-06-08
"""
import numpy as np
from tensorboardX import SummaryWriter
from PIL import Image

class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)
    
    def log_training(self, dice_loss, iou, epoch):
        self.add_scalar('training/dice_loss', dice_loss, epoch)
        self.add_scalar('training/miou', iou, epoch)
    
    def log_validation(self, dice_loss, iou, epoch):
        self.add_scalar('validation/dice_loss', dice_loss, epoch)
        self.add_scalar('validation/miou', iou, epoch)
        
    def log_images(self, map, target, prediction, step):
        if len(map.shape) > 3:
            map = map.squeeze(0)
        if len(target.shape) > 2:
            target = target.squeeze()
        if len(prediction.shape) > 2:
            prediction = prediction.squeeze()
        self.add_image("map", map, step)
        self.add_image("mask", target.unsqueeze(0), step)
        self.add_image("prediction", prediction.unsqueeze(0), step)