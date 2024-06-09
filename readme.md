# Ployp detection via YOLOv8 & ResUnet++
---
## Dataset

### Dataset
A ployp dataset with 1000 imgs/masks/bboxes. 800 for train and 200 for test.

While there's no Val dataset, split 10% from the training set as the verification set.
| Set       | Num     |
|---------- |---------|
|   Train   |   720   |
|   Test    |   200   |
|   Val     |   80    |

An example:

<img src="./see_data/see_raw_101.jpg" width="300" height="250">


### Data augmentation

See `data_augmentation.py` for details.

Offer a `see_augmentor()` function to check the augmentation. Result saved in `./see_data` dir.
```python
see_augmentor("mini_dataset", "101.jpg")
```
An example:

<img src="./see_data/see_augmentor_101.jpg" width="500" height="500">

---
## Models

ResUnet++, for details, see `model.py`, and modules in `modules.py`.

---
## File Structure
```
.
├── config.yml  # model config
├── data_augmentation.py    # data augmentation class & utils
├── dataloader.py           # data set & data loader
├── dataset                 # original dataset dir
│   ├── bbox                    # .csv bbox
│   ├── images                  
│   ├── masks
│   ├── test.txt
│   └── train.txt
├── logger.py               # tensorboardx utils
├── metrics.py              # loss functions
├── mini_dataset
│   ├── bbox
│   ├── images
│   ├── masks
│   ├── test.txt
│   └── train.txt
├── model.py                # model
├── modules.py              # model modules
├── pyutils.py              # Hyper parameter utils
├── see_data                # visualization data augmentation
│   ├── see_augmentor_101.jpg
│   └── see_raw_101.jpg
├── train.py                # train 
└── yolov8                  # yolov8 scripts
    ├── contours_detect_sample  # sample of my contours detect res
    ├── dataset             # dataset dir
    ├── dataset_utils.py    # trans original dataset to yolov8 format
    ├── result.jpg          # a predict res
    ├── runs                # yolov8 runs results, contains ckpts
    ├── yolo8.ipynb         # object detection script
    ├── yolo8_pretrained    # pretrained yolov8 models
    └── yolo8_seg.ipynb     # segmentation script
```

---
## Cites
It cannot be completed without these :zap: awesome :zap: works:

- [ResUnet++ Torch Implementation](https://github.com/rishikksh20/ResUnet)

- [ResUnet++](https://github.com/DebeshJha/ResUNetPlusPlus-with-CRF-and-TTA?tab=readme-ov-file)

- [Ultralytic YOLOv8](https://www.ultralytics.com/)
