import os
import random
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import copy

import torch
from torch.utils.data import Dataset
from albumentations import augmentations
from torchvision import transforms


class DATASET(Dataset):
    def __init__(self, dataframe, mode, val_fold, test_fold, nonzero_filter=100, imgaug_augment=None,
                 transforms_normalize=None, geo_augment=None, equal_sample=False, segment=False
    ):
        super().__init__()
        self.dataframe = dataframe
        self.mode = mode
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.resize = 256
        self.imgaug_augment = imgaug_augment
        self.geo_augment = geo_augment
        self.transforms_normalize = transforms_normalize
        self.label_smoothing = 0.1
        self.root_folder = "Image_Manipulation_Dataset"

        

        if self.mode == "train":
            rows = self.dataframe[~self.dataframe["fold"].isin([self.val_fold, self.test_fold])]
        elif self.mode == "val":
            rows = self.dataframe[self.dataframe["fold"] == self.val_fold]
        else:
            rows = self.dataframe[self.dataframe["fold"] == self.test_fold]

        #---- For checking. Get all rows -------#
        # rows = self.dataframe 

        if equal_sample:
            rows = self._equalize(rows)
        if segment:
            rows = self._segment(rows)

        print("real:{}, fakes:{}, mode = {}".format(
                len(rows[rows["label"] == 0]), len(rows[rows["label"] == 1]), self.mode
            )
        )

        self.data = rows.values
        np.random.shuffle(self.data)
        
        if nonzero_filter > 0: # filter images that do not have at least 'nonzero_filter' fake pixels
            self.data = self._filter_mask(self.data, nonzero_filter)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        while(True):
            image_name, image_patch, mask_patch, label, _, ela, root_dir = self.data[index]
            image_path = os.path.join(self.root_folder, root_dir, image_name, image_patch)
            ela_path = os.path.join(self.root_folder, root_dir, image_name, ela)

            if self.label_smoothing:
                label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)
                
                
            if(not os.path.exists(ela_path)): print(f"ELA Not Found : {ela_path}")
            if(not os.path.exists(image_path)): print(f"Image Not Found : {image_path}")

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            ela_image = cv2.imread(ela_path, cv2.IMREAD_COLOR)
            ela_image = cv2.cvtColor(ela_image, cv2.COLOR_BGR2RGB)
            

            if mask_patch == '':
                mask_image = np.zeros((image.shape[0], image.shape[1])).astype('uint8')
                mask_path = 'REAL'
            else:
                mask_path = os.path.join(self.root_folder, root_dir, image_name, mask_patch)

                if(not os.path.exists(mask_path)): print(f"Mask Not Found : {mask_path}")
                
                mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # if('NIST16' in root_dir):
                #     mask_image = 255 - mask_image
    

            if self.imgaug_augment:
                try :
                    image = self.imgaug_augment.augment_image(image)
                except Exception as e:
                    print(image_path, e) 
        
            # if('extend' not in root_dir):
            if self.geo_augment:
                data = self.geo_augment(image=image, mask=mask_image, ela=ela_image)
                image = data["image"]
                mask_image = data["mask"]
                ela_image = data["ela"]
            

            image = augmentations.geometric.functional.resize(image, self.resize, self.resize, cv2.INTER_AREA)
            ela_image = augmentations.geometric.functional.resize(ela_image, self.resize, self.resize, cv2.INTER_AREA)
            mask_image = augmentations.geometric.functional.resize(mask_image, self.resize, self.resize, cv2.INTER_AREA)


            ##########------Normalize-----##########
            image_normalize = {
                "mean": [0.4535408213875562, 0.42862278450748387, 0.41780105499276865],
                "std": [0.2672804038612597, 0.2550410416463668, 0.29475415579144293],
            }
            transNormalize = transforms.Normalize(mean=image_normalize['mean'], std=image_normalize['std'])
            transTensor = transforms.ToTensor()

            tensor_image = transTensor(image)
            tensor_ela = transTensor(ela_image)
            tensor_mask = transTensor(mask_image)

            tensor_image = transNormalize(tensor_image)
            tensor_ela = transNormalize(tensor_ela)

            ########################################

            if label == 1:
                if(np.count_nonzero(tensor_mask.numpy().ravel() >= 0.5) < 100):
                    index = random.randint(0, len(self.data) - 1)
                    continue

                
            return {
                "image": tensor_image,
                "image_path" : image_path,
                "mask_path" : mask_path, 
                "label": label, 
                "mask": tensor_mask,
                "ela" : tensor_ela,
            }


    def _equalize(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
            Equalizes count of fake and real samples
        """
        real = rows[rows["label"] == 0]
        fakes = rows[rows["label"] == 1]

        num_fake = fakes["label"].count()
        num_real = real["label"].count()

        if num_fake < num_real:
            real = real.sample(n=num_fake, replace=False)
        else:
            fakes = fakes.sample(n=num_real, replace=False)
        return pd.concat([real, fakes])


    def _segment(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
            Returns only fake rows for segmentation
        """
        return rows[rows["label"] == 1]
    
    
    def _filter_mask(self, data, count):
        temp_data = []
        
        removed_count = 0
        if os.path.exists("filtermask100.txt"):
            with open("filtermask100.txt", "r") as fp: lines = fp.read().splitlines()
        
        pbar = tqdm(data, desc="Filtering empty mask")
        for row in pbar:
            image_name, _, mask_patch, _, _, _, root_dir = row

            if mask_patch != '':
                mask_path = os.path.join(self.root_folder, root_dir, image_name, mask_patch)
                if(not os.path.exists(mask_path)): print(f"Mask Not Found : {mask_path}")
                
                flag = 0
                if len(lines) > 0:
                    if mask_path in lines: flag = 1
                else:
                    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if('NIST' in root_dir): mask_image = 255 - mask_image

                    mask_image = augmentations.geometric.functional.resize(mask_image, 256, 256, cv2.INTER_CUBIC)
                    if(np.count_nonzero(mask_image) < count): flag = 1
                
                if flag:
                    removed_count += 1
                    pbar.set_postfix({'removed': removed_count})
                    continue
                
            temp_data.append(row)
                
        return temp_data