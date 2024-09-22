import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import os
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F

class Custom_dataset(Dataset):
    def __init__(self, folder, class_names, shuffle=True,train= True,percent=0.8):
        super(Custom_dataset, self)
        self.folder = folder
        self.class_names = class_names
        self.sub_folders = [os.path.join(os.getcwd(),f.path) for f in os.scandir(self.folder) if f.is_dir()]
        self.total_images = []

        for dirname in list(self.sub_folders):
            images = [os.path.join(dirname,f) for f in os.listdir(dirname) if f.endswith(".png") or f.endswith(".jpeg")]
            self.total_images.extend(images)
        print(f"Total images: {len(self.total_images)}")
        if train:
            self.final_images = self.total_images[:int(len(self.total_images)*percent)]
            print(f"Trainset: {len(self.final_images)}")
        else:
            self.final_images = self.total_images[int(len(self.total_images)*percent):]
            print(f"Testset: {len(self.final_images)}")
        if shuffle:
            random.shuffle(self.final_images)

    def __len__(self):
        return len(self.final_images)
    

    def __getitem__(self, idx):
        # E:\\projects\\VisionTransformers\\gestures\\z\\hand5_z_dif_seg_3_cropped.jpeg
        img_path = self.final_images[idx]
        img = np.array(Image.open(img_path))
        img_norm = img / img.max()
        class_name = os.path.split(img_path)[0][-1]
        class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        if isinstance(class_name, str):
             class_idx = class_to_idx[class_name]
        # Convert index to one-hot encoded tensor
        label_one_hot = F.one_hot(torch.tensor(class_idx), num_classes=len(self.class_names)).float()
        sample = {"label":label_one_hot,"class_idx": class_idx,"image":img_norm , "class_name": class_name}
        return sample
    