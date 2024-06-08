"""
@Description: Ployp Dataloader
@Ref: https://github.com/rishikksh20/ResUnet?tab=readme-ov-file
@Author: Ken Zh0ng
@date: 2024-06-08
"""
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# TODO: bboxes num not equal, dataloader error...


class PloypDataset(Dataset):
    """
    Ployp Dataset
    The dataset has 3 subdirs: images, masks, bbox
    """
    def __init__(self, data_path, transform=None, train=True, bbox=False):
        # Initialize the dataset
        self.train = train
        self.dataset_path = data_path
        self.bbox_path = os.path.join(data_path, 'bbox')
        self.img_path = os.path.join(data_path, 'images')
        self.mask_path = os.path.join(data_path, 'masks')
        self.imgs = os.listdir(self.img_path)
        self.transform = transform
        
        self.bbox = bbox # only for detection task; segmentation, bbox = False

    def load_bboxes(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            bboxes = [[bbox[0], int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4])] for bbox in map(lambda x: x.strip().split(' ') ,lines)]
            return bboxes
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # return a data = (image, mask, bboxes)
        img = cv2.imread(os.path.join(self.img_path, self.imgs[idx]))
        mask = cv2.imread(os.path.join(self.mask_path, self.imgs[idx]), cv2.IMREAD_GRAYSCALE)
        if self.bbox:
            bboxes = self.load_bboxes(os.path.join(self.bbox_path, self.imgs[idx].replace('.jpg', '.txt')))
            data = (img, mask, bboxes)
            if self.transform:
                return self.transform(data)
            return data
        else:
            data = (img, mask)
            if self.transform:
                return self.transform(data)
            return data


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    numpy image: H x W x C
    torch image: C x H x W
    """
    def __call__(self, data):
        image = data[0]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.functional.to_tensor(image)
        if len(data) == 2:
            return (image, torch.from_numpy(data[1]).unsqueeze(0).float().div(255))
        elif len(data) == 3:
            return (image, torch.from_numpy(data[1]).unsqueeze(0).float().div(255), data[2])
        else:
            raise ValueError('Data length not equal to 2 or 3')
        

class NormalizeTarget(transforms.Normalize):
    """Normalize a tensor and also return the target"""

    def __call__(self, img, mask, _):
        return transforms.functional.normalize(img, self.mean, self.std), mask
    

class UnNormalize(object):
    """
    Convert normalized image to original image
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    

def check_size_equal(dataset):
    check_set = PloypDataset(dataset, transform=transforms.Compose([ToTensor()]), train=True, bbox=False)
    # for i in check_set:
    # # print(i[0].shape, i[1].shape)
    # assert i[0].shape[2] == i[1].shape[2] == 640, f'img size: {i[0].shape[2]}, mask size: {i[1].shape[2]}'
    # assert i[0].shape[1] == i[1].shape[1] == 640, f'img size: {i[0].shape[1]}, mask size: {i[1].shape[1]}'
    check_set = DataLoader(check_set, batch_size=2, shuffle=True)
    for idx, data in enumerate(check_set):
        assert data[0].shape[2] == data[1].shape[2], 'Image and mask size not equal'


if __name__ == "__main__" :
    # mydataset = PloypDataset(data_path='new_dataset/train', transform=transforms.Compose([ToTensor()]), train=True)
    # print(len(mydataset))
    # print(mydataset[0])
    check_size_equal('new_dataset/train')
    