"""
@Description: split the dataset into training and validation set
@Author: Ken Zh0ng
@date: 2024-06-03
"""
def yolo8(DEBUG=False):
    import os
    import shutil
    from tqdm import tqdm
    from PIL import Image
    raw_set = "dataset/raw/"
    
    # clear
    shutil.rmtree("dataset/yolo8", ignore_errors=True)
    print("clear dataset/yolo8 ok!")
    
    # make directories
    os.makedirs("dataset/yolo8", exist_ok=True)
    set = ["train", "val", "test"]
    for set_ in set:
        os.makedirs(f"dataset/yolo8/{set_}", exist_ok=True)
        os.makedirs(f"dataset/yolo8/{set_}/images", exist_ok=True)
        os.makedirs(f"dataset/yolo8/{set_}/bbox", exist_ok=True)
        os.makedirs(f"dataset/yolo8/{set_}/labels", exist_ok=True)
    print("make directories ok!")
    
    # split the dataset (train -> train val, test -> test)
    train_set = []
    val_set = []
    test_set = []
    with open(os.path.join(raw_set, "train.txt"), "r") as f:
        lines = f.readlines()
        train_size = int(len(lines) * 0.9)
        train_set = lines[:train_size]
        val_set = lines[train_size:]
    with open(os.path.join(raw_set, "test.txt"), "r") as f:
        test_set = f.readlines()
    print("split the dataset ok!")
    
    # cp images and labels to yolo8
    for set_, data in zip(set, [train_set, val_set, test_set]):
        for line in data:
            image_path = os.path.join(raw_set, "images", line.strip())
            label_path = os.path.join(raw_set, "bbox", line.strip().replace(".jpg", ".csv"))
            os.system(f"cp {image_path} dataset/yolo8/{set_}/images/")
            os.system(f"cp {label_path} dataset/yolo8/{set_}/bbox/")
    print("cp images and labels to yolo8 ok!")
    
    # convert bbox to yolo8 format
    if DEBUG:
        check_cnt = 10
        
    for set_ in set:
        bbox_dir = f"dataset/yolo8/{set_}/bbox"
        label_dir = f"dataset/yolo8/{set_}/labels"
        image_dir = f"dataset/yolo8/{set_}/images"
        print(f"Processing {set_} set...")
        bar = tqdm(os.listdir(bbox_dir))
        for bbox in bar:
            with open(os.path.join(bbox_dir, bbox), "r") as f:
                lines = f.readlines()
                
                image = Image.open(os.path.join(image_dir, bbox.strip().replace(".csv", ".jpg")))
                width, height = image.size
                
                with open(os.path.join(label_dir, bbox.strip().replace(".csv", ".txt")), "w") as f:
                    for line in lines[1:]:
                        line = line.split(",")[1:]
                        assert len(line) == 4
                        x_min, y_min, x_max, y_max = map(float, line)
                        # x, y, w, h
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        w = x_max - x_min
                        h = y_max - y_min
                        # uniform to [0, 1]
                        x_center = round(x_center/width, 6)
                        y_center = round(y_center/height, 6)
                        w = round(w/width, 6)
                        h = round(h/height, 6)
                        f.write(f"0 {x_center} {y_center} {w} {h}\n")
                        if DEBUG and check_cnt > 0:
                            check_cnt -= 1
                            print(f"0 {x_center} {y_center} {w} {h}")
    
    # rm bbox
    for set_ in set:
        shutil.rmtree(os.path.join("dataset/yolo8", set_, "bbox"))
    print("rm bbox ok!")
    
    # show info
    print(f"train set: {len(os.listdir('dataset/yolo8/train/images'))}")
    print(f"val set: {len(os.listdir('dataset/yolo8/val/images'))}")
    print(f"test set: {len(os.listdir('dataset/yolo8/test/images'))}")
    print("Done!")
           

def yolo8_yaml():
    import os
    import yaml
    # gen yaml data
    # suppose we are in rootdir 
    rootdir = os.getcwd()

    data = dict(
        path = os.path.join(rootdir, "dataset/yolo8"),
        train = "train/images",
        val = "val/images",
        test = "test/images",
        names = {
            "0": "polyp"
            },
    )
    
    # write to yaml
    with open(f"{rootdir}/dataset/yolo8/data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)       


def see_data():
    from pathlib import Path
    from PIL import Image
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    p = Path("dataset/raw/images")
    w_l = []
    h_l = []
    for img in p.glob("*.jpg"):
        image = Image.open(img)
        w, h = image.size
        w_l.append(w)
        h_l.append(h)
    
    center_x = np.mean(w_l)
    center_y = np.mean(h_l)
    
    plt.figure()
    
    # 绘制宽度和高度的二维分布图
    sns.jointplot(x=w_l, y=h_l, kind="kde")
    
    # mark center dot
    plt.plot(center_x, center_y, 'ro')
    plt.text(center_x, center_y, f'{center_x},{center_y}', color='red')
    
    # sacatter plot
    plt.scatter(w_l, h_l, color='green',alpha=0.5)
    
    plt.title('Width and Height Distribution')
    plt.xlabel('Width')
    plt.ylabel('Height')

    # 显示图形
    plt.tight_layout()
    plt.show()
    plt.savefig("dataset/raw/size.png")
    

def yolo8_seg(DEBUG=False):
    import os
    import shutil
    from tqdm import tqdm
    from PIL import Image
    import cv2
    import numpy as np
    raw_set = "dataset/raw/"
    
    # clear
    shutil.rmtree("dataset/yolo8_seg", ignore_errors=True)
    print("clear dataset/yolo8_seg ok!")
    
    # make directories
    os.makedirs("dataset/yolo8_seg")
    set = ["train", "val", "test"]
    for set_ in set:
        os.makedirs(f"dataset/yolo8_seg/{set_}")
        os.makedirs(f"dataset/yolo8_seg/{set_}/images")
        os.makedirs(f"dataset/yolo8_seg/{set_}/masks")
        os.makedirs(f"dataset/yolo8_seg/{set_}/labels")
        
    os.makedirs(f"dataset/yolo8_seg/test/raw_labels")
    
    print("make directories ok!")
    
    # split the dataset (train -> train val, test -> test)
    train_set = []
    val_set = []
    test_set = []
    with open(os.path.join(raw_set, "train.txt"), "r") as f:
        lines = f.readlines()
        train_size = int(len(lines) * 0.9)
        train_set = lines[:train_size]
        val_set = lines[train_size:]
    with open(os.path.join(raw_set, "test.txt"), "r") as f:
        test_set = f.readlines()
    print("split the dataset ok!")
    
    # cp images and masks to yolo8_seg
    for set_, data in zip(set, [train_set, val_set, test_set]):
        for line in data:
            image_path = os.path.join(raw_set, "images", line.strip())
            mask_path = os.path.join(raw_set, "masks", line.strip())
            os.system(f"cp {image_path} dataset/yolo8_seg/{set_}/images/")
            os.system(f"cp {mask_path} dataset/yolo8_seg/{set_}/masks/")
    print("cp images and labels to yolo8_seg ok!")
    
    # convert mask to yolo8_seg format
    if DEBUG:
        check_cnt = 10
        
    for set_ in set:
        mask_dir = f"dataset/yolo8_seg/{set_}/masks"
        label_dir = f"dataset/yolo8_seg/{set_}/labels"
        print(f"Processing {set_} set...")
        bar = tqdm(os.listdir(mask_dir))
        for file_name in bar:
            mask = cv2.imread(os.path.join(mask_dir, file_name), cv2.IMREAD_GRAYSCALE)
            h, w = mask.shape
            if DEBUG and check_cnt>1:
                check_cnt -= 1
                print("img h:", h, "img w:", w)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            index, labels = cv2.connectedComponents(mask)
            lines = []
            for i in range(1, index):
                contours, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_str = ''
                for contour in contours:
                    # uniform
                    if DEBUG and check_cnt>1:
                        print("contours:", len(contour))
                    """
                    BUG: some masks has a white point at the left-top!
                        this will cause that the result txt file has an extra wrong line:
                            1 0 0
                        assume that each connectedComponents has multi but not 1 contours.
                    """
                    if len(contour) <= 1:
                        continue
                    
                    contours_str += " ".join(['{} {}'.format(round(point[0][0]/w, 4), round(point[0][1]/h, 4)) for point in contour]) + ' '
                if contours_str != '':
                    lines.append(contours_str.strip())
            
            with open(os.path.join(label_dir, file_name.replace(".jpg", ".txt")), 'w') as f:
                if DEBUG and check_cnt>1:
                    print("lines", len(lines))
                for index, i in enumerate(lines):
                    if index == len(lines) - 1:
                        f.write("0 " + i)
                    else:
                        f.write("0 " + i + '\n')
                        
    bar = tqdm(os.listdir("dataset/yolo8_seg/test/masks"))                 
    for file_name in bar:
        mask = cv2.imread(os.path.join("dataset/yolo8_seg/test/masks", file_name), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        index, labels = cv2.connectedComponents(mask)
        lines = []
        for i in range(1, index):
            contours, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_str = ''
            for contour in contours:
                """
                BUG: some masks has a white point at the left-top!
                    this will cause that the result txt file has an extra wrong line:
                        1 0 0
                    assume that each connectedComponents has multi but not 1 contours.
                """
                if len(contour) <= 1:
                    continue
                contours_str += " ".join(['{} {}'.format(point[0][0], point[0][1]) for point in contour]) + ' '
            if contours_str != '':
                lines.append(contours_str.strip())
        
        with open(os.path.join("dataset/yolo8_seg/test/raw_labels", file_name.replace(".jpg", ".txt")), 'w') as f:
            for index, i in enumerate(lines):
                if index == len(lines) - 1:
                    f.write(i)
                else:
                    f.write(i + '\n')
    
    # show info
    print(f"train set: {len(os.listdir('dataset/yolo8_seg/train/images'))}")
    print(f"val set: {len(os.listdir('dataset/yolo8_seg/val/images'))}")
    print(f"test set: {len(os.listdir('dataset/yolo8_seg/test/images'))}")
    print("Done!")


def yolo8_seg_yaml():
    import os
    import yaml
    # gen yaml data
    # suppose we are in rootdir 
    rootdir = os.getcwd()

    data = dict(
        path = os.path.join(rootdir, "dataset/yolo8_seg"),
        train = "train/images",
        val = "val/images",
        test = "test/images",
        names = {
            "0": "polyp"
            },
    )
    
    # write to yaml
    with open(f"{rootdir}/dataset/yolo8_seg/data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)  


def test_mask_to_seg(mask_path="0.jpg"):
    import os
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape
    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    ret, labels = cv2.connectedComponents(mask)
    
    lines = []
    for i in range(1, ret):
        contours, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_str = ''
        for contour in contours:
            # 将每个边缘坐标转换为字符串，并用逗号分隔
            edge_str += ' '.join([f'{point[0][0]} {point[0][1]}' for point in contour]) + ' '
        lines.append(edge_str.strip())
        
    # 创建一个文件来保存边缘坐标
    with open('edges.txt', 'w') as f:
        print("lines", len(lines))
        for index, i in enumerate(lines):
            if index == len(lines) - 1:
                f.write("class " + i)
            else:
                f.write("class " + i + '\n')
    
    # redraw the coutour
    y, x = [], []
    with open('edges.txt', 'r') as f:
        for line in f.readlines():
            coords = line.split(' ')[1:]
            print(len(coords))
            x_coords = [int(x) for x in coords[::2]]
            y_coords = [int(y) for y in coords[1::2]]
            x.extend(x_coords)
            y.extend(y_coords)
            
    remask = np.zeros((h, w))
    print(h, w)
    remask[y, x] = 255
    plt.imshow(remask, cmap='gray')
    plt.xlim([0, w])  # 设置x轴的范围为[0, w]
    plt.ylim([h, 0])  # 设置y轴的范围为[0, h]
    plt.savefig("remask.png")
                   
              
if __name__ == "__main__" :
    # yolo8(DEBUG=True)      
    # yolo8_yaml()
    # see_data()
    # test_mask_to_seg("946.jpg")
    yolo8_seg(DEBUG=False)
    yolo8_seg_yaml()