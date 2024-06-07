"""
@Description: Image data augmentation utils; Modified based on Ref
@Ref: https://github.com/DebeshJha/ResUNetPlusPlus/blob/master/process_image.py
@Author: Ken Zh0ng
@date: 2024-06-06
"""

import os
import shutil
import random
from typing import List

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Useful funcs

def read_img(file, color=cv2.IMREAD_COLOR):
    """
    Read image from file
    
    Args:
        file: str, path of image file
        color: int, color mode, cv2.IMREAD_COLOR or cv2.IMREAD_GRAYSCALE
        
    Returns:
        img: np.ndarray, image data
    """
    img = cv2.imread(file, color)
    return img  


def read_bbox(file) -> List[List]:
    """
    Read bounding boxes from a CSV file
    
    Args:
        file: str, path of CSV file
    
    Returns:
        bboxes: list, bounding boxes
    """
    bboxes = []
    with open(file, "r") as f:
        for line in f.readlines()[1:]:
            bbox = line.strip().split(",")
            bbox[1:] = [float(x) for x in bbox[1:]]
            bboxes.append(bbox)
    return bboxes


def save_data(data:tuple, path:tuple, binary=True):
    """
    Save img, mask, bboxes to pathes
    
    Args:
        data: tuple, (img, mask, bboxes)
        path: tuple, (img_path, mask_path, bbox_path)
        binary: bool, whether to save mask as binary image
    """
    def round_int(x):
        return int(round(x))
    
    if binary:
        mask = (data[1] > 0).astype(np.uint8) * 255
    cv2.imwrite(path[0], data[0])
    cv2.imwrite(path[1], mask)
    with open(path[2], "w") as f:
        for b in data[2]:
            f.write(f"{b[0]} {round_int(b[1])} {round_int(b[2])} {round_int(b[3])} {round_int(b[4])}\n")


def resize_data(data:tuple, size=(640, 640)):
    """
    Resize image, mask and bbox
    
    Args:
        data: tuple, (img, mask, bboxes)
        size: tuple, (width, height)
        
    Returns:
        data: tuple, (img, mask, bbox)
    """
    img, mask, bboxes = data
    h, w = img.shape[:2]
    scale = (size[0]/w, size[1]/h)
    img = cv2.resize(img, size)
    mask = cv2.resize(mask, size)
    bboxes = [[bbox[0], bbox[1]*scale[0], bbox[2]*scale[1], bbox[3]*scale[0], bbox[4]*scale[1]] for bbox in bboxes]
    return (img, mask, bboxes)


def path_check(paths:list):
    """
    Check if paths(a list) exist
    """
    for p in paths:
        assert os.path.exists(p), f"Path {p} does not exist!"


def clean(path):
    shutil.rmtree(path, ignore_errors=True)


# Data Augmentatior Class

class DataAugmentor(object):
    def __init__(self, data:tuple, bbox_class=None, debug=False) -> None:
        self.img = data[0]
        self.mask = data[1]
        self.bboxes = data[2]
        self.h, self.w = self.img.shape[:2]
        self._bbox_class = bbox_class if bbox_class else "polyp"
        self.DEBUG = debug
    
    @staticmethod
    def help():
        print("DataAugmentor is a class for performing data augmentation on images, masks, and bounding boxes.")
        print("Available methods:")
        print("- random_crop(crop_rate): Randomly crops the image, mask, and bounding boxes.")
        print("- random_rotate(angle_range): Randomly rotates the image, mask, and bounding boxes.")
        print("- flip(flip_code): Randomly flips the image, mask, and bounding boxes.")
        print("- random_linear_trans(alpha_range, beta_range): Randomly applies a linear transformation to the image.")
        print("- random_brightness(factor): Randomly adjusts the brightness of the image.")
        print("- gray_scale(): Converts the image to grayscale.")
        print("- cutout(ratio, fill_value): Randomly cuts out a rectangle from the image.")
        print("- random_scale(scale_ratio_range, crop_ratio): Randomly scales the image, mask, and bounding boxes.")
    
    # private methods
    def _bbox_coords(self):
        return self.bboxes[1:]
    
    def _check_size(self, size, FLAG=None):
        assert len(size) == 2 and type(size[0]) == int, f"Invalid size: {size}"
        if FLAG:
            if FLAG == "crop":
                assert size[0] <= self.w and size[1] <= self.h, f"Invalid crop size: {size}"
            if FLAG == "cutout":
                assert size[0] <= self.w and size[1] <= self.h, f"Invalid cutout size: {size}"
    
    def _crop_bbox(self, lims):
        """
        Crop bbox
        
        Args:
            bboxes: list, [(class, x1, y1, x2, y2),...]
            lims: tuple, (x1, y1, x2-1, y2-1)
        """
        new_bboxes = []
        for bbox in self.bboxes:
            x1  = max(lims[0], bbox[1])
            y1  = max(lims[1], bbox[2])
            x2  = min(lims[2], bbox[3])
            y2  = min(lims[3], bbox[4])
            if x1 >= x2 or y1 >= y2:
                continue
            else:
                if self.DEBUG:
                    print("=> original bbox: ", bbox)
                    print("=> crop lims: ", lims)
                    print("=> cropped bbox: ", [bbox[0], x1, y1, x2, y2])
                new_bboxes.append([bbox[0], x1, y1, x2, y2])
        return new_bboxes
    
    def _re_extract_bbox(self, mask, class_=None):
        """
        Extract bounding box from mask image. bboxes might be None!
        
        Args:
            mask: np.ndarray, mask image
        
        Returns:
            bboxes: list, [(class, x1, y1, x2, y2),...]
        """
        class_ = self._bbox_class
        
        num_labels, labeled_mask, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        # num_labels: int, number of labels, contines background, so num_labels-1 is the number of objects
        # labeled_mask: np.ndarray, labeled mask, same size as mask
        index = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
        index = index[:len(self.bboxes)]
        index = index + 1
        
        if self.DEBUG:
            print("cc index: ", index)
            
        bboxes = []
        for i in range(1, num_labels):
            """
            BUG: there are very small ones, so we need to filter them by setting a threshold
            """
            if stats[i, cv2.CC_STAT_AREA] < 20:
                continue
            
            bianry_mask = np.where(labeled_mask == i, 1, 0) # bianry mask only contines pixes with value i
            points = np.argwhere(bianry_mask > 0) # get all points with value i, [(y1, x1), (y2, x2),...]
            points = np.fliplr(points) # flip at aix=1, (y, x) -> (x, y)
            x1, y1 = np.min(points, axis=0)
            x2, y2 = np.max(points, axis=0)
            
            bboxes.append([class_, x1, y1, x2, y2])
        return bboxes
    
    # data augmentation methods
    def random_crop(self, crop_rate=0.6):
        """
        Random crop image, mask and bboxes
        
        Args:
            crop_size: tuple, (width, height)
        """
        assert 0 < crop_rate < 1, f"Invalid crop_rate: {crop_rate}"
        crop_size = (int(self.w*crop_rate), int(self.h*crop_rate))
        self._check_size(crop_size, "crop")
        
        x1 = random.randint(0, self.w - crop_size[0])
        y1 = random.randint(0, self.h - crop_size[1])
        x2 = x1 + crop_size[0]
        y2 = y1 + crop_size[1]
        
        crop_img = self.img[y1:y2, x1:x2]
        crop_mask = self.mask[y1:y2, x1:x2]
        # crop_bboxes = self._crop_bbox(lims=(x1, y1, x2-1, y2-1))
        crop_bboxes = self._re_extract_bbox(crop_mask)
        if crop_bboxes:
            return (crop_img, crop_mask, crop_bboxes)
        else:
            return None
        
    def random_rotate(self, angle_range=30):
        """
        Random rotate image, mask and bboxes
        
        Args:
            angle: int, rotate angle range
        """
        assert 0 < angle_range < 180, f"Invalid angle_range: {angle_range}"
        angle = random.randint(-angle_range, angle_range)
        rotate_M = cv2.getRotationMatrix2D((self.w/2, self.h/2), angle, 1) # center, angle, scale
        rotated_img = cv2.warpAffine(self.img, rotate_M, (self.w, self.h))
        rotated_mask = cv2.warpAffine(self.mask, rotate_M, (self.w, self.h))
        rotated_bboxes = self._re_extract_bbox(rotated_mask)
        if rotated_bboxes:
            return (rotated_img, rotated_mask, rotated_bboxes)
        else:
            return None
    
    def flip(self, flip_code=1):
        """
        Random flip image, mask and bboxes
        
        Args:
            flip_code: int, flip code, 0: vertical flip, 1: horizontal flip, -1: both
        """
        assert flip_code in [0, 1, -1], f"Invalid flip_code: {flip_code}"
        flipped_img = cv2.flip(self.img, flip_code)
        flipped_mask = cv2.flip(self.mask, flip_code)
        if flip_code == 0:
            flipped_bboxes = [[bbox[0], bbox[1], self.h - bbox[4], bbox[3], self.h - bbox[2]] for bbox in self.bboxes]
        elif flip_code == 1:
            flipped_bboxes = [[bbox[0], self.w - bbox[3], bbox[2], self.w - bbox[1], bbox[4]] for bbox in self.bboxes]
        elif flip_code == -1:
            flipped_bboxes = [[bbox[0], self.w - bbox[3], self.h - bbox[4], self.w - bbox[1], self.h - bbox[2],] for bbox in self.bboxes]
        if self.DEBUG:
            print("=> original bboxes[0]: ", self.bboxes[0])
            print("=> flipped bboxes[0]: ", flipped_bboxes[0])
        return (flipped_img, flipped_mask, flipped_bboxes)
    
    def random_linear_trans(self, alpha_range=0.8, beta_range=3):
        """
        Random pix-wise liner transform; img = alpha * src + beta
        
        Args:
            alpha: float, contrast ratio, alpha > 1: higher contrast, alpha < 1: lower contrast
            beta: int, brightness shift, beta > 0: brighter, beta < 0: darker
        """
        assert 0 < alpha_range < 1, f"Invalid alpha_range: {alpha_range}"
        assert 0 < beta_range <100, f"Invalid beta_range: {beta_range}"
        alpha = np.random.uniform(alpha_range, 1)
        alpha = alpha if np.random.randint(0, 2) == 0 else 1/alpha
        beta = np.random.uniform(-beta_range, beta_range)
        
        img = cv2.convertScaleAbs(self.img, alpha, beta)
        return (img, self.mask, self.bboxes)

    def random_brightness(self, factor=0.5):
        """
        Random adjust brightness
        
        Args:
            factor: float, brightness scale
        """
        assert 0 < factor < 2, f"Invalid factor: {factor}"
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform())
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        brg = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
        return (brg, self.mask, self.bboxes)
    
    def gray_scale(self):
        """
        Convert image to gray scale
        """
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return (gray, self.mask, self.bboxes)
    
    def cutout(self, ratio=(0.3, 0.2), fill_value='mean'):
        """
        Randomly cutout a rectangle from image
        
        Args:
            ratio: tuple, (width_ratio, height_ratio)
            fill_value: str, fill value, "mean": mean value, "random": random value
        """
        assert 0 < ratio[0] < 1 and 0 < ratio[1] < 1, f"Invalid ratio: {ratio}"
        size = (int(self.w*ratio[0]), int(self.h*ratio[1]))
        self._check_size(size, "cutout")
        
        img = np.copy(self.img)
        mask = np.copy(self.mask)
        
        if fill_value == "mean":
            fill_value = np.mean(img)
        elif fill_value == "random":
            fill_value = np.random.randint(0, 256)
        
        top = np.random.randint(0, self.h - size[1])
        bottom = top + size[1]
        left = np.random.randint(0, self.w - size[0])
        right = left + size[0]
        
        """
        "img[top:bottom, left:right, :] = cut_value[top:bottom, left:right, :]"
        assignment is worse than .fill()
        """
        img[top:bottom, left:right, :].fill(fill_value)
        """
        BUG[fixed]:    mask[top:bottom, left:right, :].fill(0)
            IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
        """
        mask[top:bottom, left:right].fill(0)
        bboxes = self._re_extract_bbox(mask)
        if bboxes:
            return (img, mask, bboxes)
        else:
            return None
    
    def random_scale(self, scale_ratio_range=(0.2, 0.3), crop_ratio=0.6):
        """
        Random scale image, mask and bboxes
        
        Args:
            scale_ratio_range: tuple, (width_ratio, height_ratio); 0<x<0.5
        """
        assert 0 < scale_ratio_range[0] < 0.5 and 0 < scale_ratio_range[1] < 0.5, f"Invalid scale_ratio_range: {scale_ratio_range}"
        scale_size = (int(np.random.uniform(0.5 - scale_ratio_range[0], 0.5 + scale_ratio_range[0]) * self.w),
                      int(np.random.uniform(0.5 - scale_ratio_range[1], 0.5 + scale_ratio_range[1]) * self.h))
        img, mask, bboxes = resize_data((self.img, self.mask, self.bboxes), scale_size)
        crop_data = DataAugmentor((img, mask, bboxes)).random_crop(crop_ratio)
        if crop_data:
            return crop_data
        else:
            return None
        
        
# Main Function

def main(path, new_path="new_dataset", resize=(640, 640), if_augment=False):
    print("cwd: ", os.getcwd())
    clean(new_path)
    
    # Dataset Pathes
    img_path = os.path.join(path, "images")
    mask_path = os.path.join(path, "masks")
    bbox_path = os.path.join(path, "bbox")
    path_check([img_path, mask_path, bbox_path])
    
    # Create new directory
    train_path = os.path.join(new_path, "train")
    val_path = os.path.join(new_path, "val")
    test_path = os.path.join(new_path, "test")
    
    for set_ in [train_path, val_path, test_path]:
        os.makedirs(os.path.join(set_, "images"), exist_ok=True)
        os.makedirs(os.path.join(set_, "masks"), exist_ok=True)
        os.makedirs(os.path.join(set_, "bbox"), exist_ok=True)
    path_check([train_path, val_path, test_path])
    
    # Split dataset    
    train_index = os.path.join(path, "train.txt")
    test_index = os.path.join(path, "test.txt")
    path_check([train_index, test_index])
    
    with open(test_index, "r") as f:
        bar = tqdm(f.readlines())
        for test_img in bar:
            test_img = test_img.strip()
            data = (read_img(os.path.join(img_path, test_img), cv2.IMREAD_COLOR),       # read image
                    read_img(os.path.join(mask_path, test_img), cv2.IMREAD_GRAYSCALE),  # read mask
                    read_bbox(os.path.join(bbox_path, test_img.replace(".jpg", ".csv")))) # read bboxes, bboxes=[(class, x1, y1, x2, y2),...]
            
            data = resize_data(data, resize) # resize img, mask, bboxes
            save_pathes = [os.path.join(test_path, "images", test_img),
                           os.path.join(test_path, "masks", test_img),
                           os.path.join(test_path, "bbox", test_img.replace(".jpg", ".txt"))]
            save_data(data, save_pathes)
    
    with open(train_index, "r") as f:
        train_set = f.readlines()
        random.shuffle(train_set)
        val_set = train_set[:int(len(train_set)*0.1)]
        train_set = train_set[int(len(train_set)*0.1):]
        
        bar = tqdm(val_set)
        for val_img in bar:
            val_img = val_img.strip()
            data = (read_img(os.path.join(img_path, val_img), cv2.IMREAD_COLOR),       # read image
                    read_img(os.path.join(mask_path, val_img), cv2.IMREAD_GRAYSCALE),  # read mask
                    read_bbox(os.path.join(bbox_path, val_img.replace(".jpg", ".csv")))) # read bboxes, bboxes=[(class, x1, y1, x2, y2),...]
            
            data = resize_data(data, resize) # resize img, mask, bboxes
            save_pathes = [os.path.join(val_path, "images", val_img),
                           os.path.join(val_path, "masks", val_img),
                           os.path.join(val_path, "bbox", val_img.replace(".jpg", ".txt"))]
            save_data(data, save_pathes)
        
        bar = tqdm(train_set)
        for train_img in bar:
            train_img = train_img.strip()
            data = (read_img(os.path.join(img_path, train_img), cv2.IMREAD_COLOR),       # read image
                    read_img(os.path.join(mask_path, train_img), cv2.IMREAD_GRAYSCALE),  # read mask
                    read_bbox(os.path.join(bbox_path, train_img.replace(".jpg", ".csv")))) # read bboxes, bboxes=[(class, x1, y1, x2, y2),...]

            if if_augment:
                # Augment data
                data_augmentor = DataAugmentor(data)
        

# Test Functions

def blended_mask_bbox_img(img, mask, bboxes):
    """
    Blend mask and image, draw bboxes on image
    
    Args:
        img: 3d np.ndarray, image data
        mask: 2d np.ndarray, mask data
        bboxes: list, [(class, x1, y1, x2, y2),...]
        
    Returns:
        blended: np.ndarray, blended image
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    # blend mask and image
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # convert mask to 3 channels
    mask[np.where((mask == [255, 255, 255]).all(axis=2))] = [255, 0, 0] # change mask color to BLUE
    alpha = 1
    blended = cv2.addWeighted(img, alpha=alpha, src2=mask, beta=0.5, gamma=0)
    
    # draw bboxes
    for bbox in bboxes:
        assert len(bbox) == 5, f"Invalid bbox: {bbox}"
        cv2.rectangle(blended, (int(bbox[1]), int(bbox[2])), (int(bbox[3]), int(bbox[4])), (0, 255, 0), 2)
        cv2.putText(blended, bbox[0], (int(bbox[1]), int(bbox[2])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return blended


def see_data(path, name, resize=(640, 640)):
    """
    Visualize image, mask and bboxes on a single image
    """
    img_path = os.path.join(path, "images")
    mask_path = os.path.join(path, "masks")
    bbox_path = os.path.join(path, "bbox")
    
    data = (read_img(os.path.join(img_path, name), cv2.IMREAD_COLOR),       # read image
            read_img(os.path.join(mask_path, name), cv2.IMREAD_GRAYSCALE),  # read mask
            read_bbox(os.path.join(bbox_path, name.replace(".jpg", ".csv")))) # read bboxes, bboxes=[(class, x1, y1, x2, y2),...]
    
    # test resize img, mask and bboxes
    data = resize_data(data, resize)      
    
    # blend mask and image
    mask = cv2.cvtColor(data[1], cv2.COLOR_GRAY2BGR) # convert mask to 3 channels
    mask[np.where((mask == [255, 255, 255]).all(axis=2))] = [255, 0, 0] # change mask color to BLUE
    alpha = 0.5
    blended = cv2.addWeighted(data[0], alpha=alpha, src2=mask, beta=1-alpha, gamma=0)
    
    # draw bboxes
    for bbox in data[2]:
        assert len(bbox) == 5, f"Invalid bbox: {bbox}"
        cv2.rectangle(blended, (int(bbox[1]), int(bbox[2])), (int(bbox[3]), int(bbox[4])), (0, 255, 0), 2)
        cv2.putText(blended, bbox[0], (int(bbox[1]), int(bbox[2])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imwrite(f"see_raw_{name}", data[0])
    cv2.imwrite(f"see_blended_{name}", blended)


def see_augmentor(dataset_path, name):
    """
    Visualize image, mask and bboxes after data augmentation
    """
    shutil.rmtree("see_data", ignore_errors=True)
    os.mkdir("see_data")
    
    img_path = os.path.join(dataset_path, "images")
    mask_path = os.path.join(dataset_path, "masks")
    bbox_path = os.path.join(dataset_path, "bbox")
    
    data = (read_img(os.path.join(img_path, name), cv2.IMREAD_COLOR),       # read image, BGR
            read_img(os.path.join(mask_path, name), cv2.IMREAD_GRAYSCALE),  # read mask
            read_bbox(os.path.join(bbox_path, name.replace(".jpg", ".csv")))) # read bboxes, bboxes=[(class, x1, y1, x2, y2),...] 
    
    # Augment data
    DataAugmentor.help()
    data_augmentor = DataAugmentor(data, debug=True)
    random_crop_data = data_augmentor.random_crop()
    random_rotate_data = data_augmentor.random_rotate(120)
    flip_data = data_augmentor.flip(flip_code=-1)
    random_brightness_data = data_augmentor.random_brightness()
    random_lineartrans_data = data_augmentor.random_linear_trans()
    random_scale_data = data_augmentor.random_scale()
    cutout_data = data_augmentor.cutout()
    gray_scale_data = data_augmentor.gray_scale()
    
    # shape check
    print("shape check")
    if random_crop_data:
        print(random_crop_data[0].shape, random_crop_data[1].shape)
    if random_rotate_data:
        print(random_rotate_data[0].shape, random_rotate_data[1].shape)
    print(flip_data[0].shape, flip_data[1].shape)
    print(random_brightness_data[0].shape, random_brightness_data[1].shape)
    print(random_lineartrans_data[0].shape, random_lineartrans_data[1].shape)
    if random_scale_data:
        print(random_scale_data[0].shape, random_scale_data[1].shape)
    if cutout_data:
        print(cutout_data[0].shape, cutout_data[1].shape)
    print(gray_scale_data[0].shape, gray_scale_data[1].shape)
    
    # blended images
    blended_random_crop = blended_mask_bbox_img(*random_crop_data) if random_crop_data else None
    blended_random_rotate = blended_mask_bbox_img(*random_rotate_data) if random_rotate_data else None
    blended_flip = blended_mask_bbox_img(*flip_data)
    blended_random_brightness = blended_mask_bbox_img(*random_brightness_data)
    blended_random_lineartrans = blended_mask_bbox_img(*random_lineartrans_data)
    blended_random_scale = blended_mask_bbox_img(*random_scale_data) if random_scale_data else None
    blended_cutout = blended_mask_bbox_img(*cutout_data) if cutout_data else None
    blended_gray_scale = blended_mask_bbox_img(*gray_scale_data)
    
    # resize
    
    imgs = [data[0],
            blended_random_crop,
            blended_random_rotate,
            blended_flip,
            blended_random_brightness,
            blended_random_lineartrans,
            blended_random_scale,
            blended_cutout,
            blended_gray_scale]
    
    imgs = [cv2.resize(img, (640, 640)) if img is not None
            else np.zeros((640, 640, 3), dtype=np.uint8)
            for img in imgs]
    
    rows = [np.hstack(imgs[i:i+3]) for i in range(0, 9, 3)]
    see_img = np.vstack(rows)
    cv2.imwrite(f"see_data/see_raw_{name}", data[0])
    cv2.imwrite(f"see_data/see_augmentor_{name}", see_img)
    
                
if __name__ == "__main__" :
    # main("mini_dataset", "new_mini_dataset")
    # print(read_bbox("mini_dataset/bbox/0.csv"))
    # see_data("mini_dataset", "101.jpg")
    see_augmentor("mini_dataset", "101.jpg")
    
    