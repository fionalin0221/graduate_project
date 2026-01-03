import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import re
import cv2
import random
import yaml
import time
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries
from skimage.measure import find_contours

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.combine_csv import merge_labels, merge_TI

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch_ema import ExponentialMovingAverage as ema
from torch.amp import autocast, GradScaler
import timm

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

torch.manual_seed(0)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = "cuda" if torch.cuda.is_available() else "cpu"

class Worker():
    def __init__(self, config):
        current_computer = config['current_computer']

        self.wsi_type = config['type']
        self.state = config['state']
        self.gen_type = config['gen_type']
        self.generation = config['generation']
        self.patch_size = config['patch_size']

        self.file_paths = config['computers'][current_computer]['file_paths']
        class_list = config["class_list"]
        self.classes = [class_list[i] for i in self.file_paths['classes']]
        self.class_num = len(self.classes)
        self.num_trial = self.file_paths['num_trial'] 
        
        # Model parameters
        self.backbone = self.file_paths['backbone']
        self.pretrain = self.file_paths['pretrain']
        self.batch_size = self.file_paths['batch_size']
        self.base_lr = float(self.file_paths['base_lr'])
        self.model_save_freq = self.file_paths['model_save_freq']
        self.valid_percentage = self.file_paths['valid_percentage']

        # Data parameters
        self.data_num = self.file_paths['data_num']
        self.data_trial = self.file_paths['data_trial'] 
        self.num_wsi = self.file_paths['num_wsi']
        self.load_dataset = self.file_paths['load_dataset']
        self.replay_data_num = self.file_paths['replay_data_num']
        self.other_validation = self.file_paths['other_validation']
        self.other_valid_size = self.file_paths['valid_size']

        # Test parameters
        self.test_state = self.file_paths['test_state']
        self.test_type = self.file_paths['test_type']

        self.test_model = self.file_paths['test_model']
        self.test_model_trial = self.file_paths['test_model_trial']
        self.test_model_wsis = self.file_paths['test_model_wsis']
        self.test_model_state = self.file_paths['test_model_state']
        self.test_model_type = self.file_paths['test_model_type']

        if self.gen_type:
            self.save_dir = self.file_paths[f'{self.wsi_type}_generation_save_path']
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = self.file_paths[f'{self.wsi_type}_WTC_result_save_path']
            self.save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}"
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.save_path, exist_ok=True)

        # wsi lists
        self.hcc_old_wsis = self.file_paths['HCC_old_wsis']
        self.hcc_wsis = self.file_paths['HCC_wsis']
        self.cc_wsis = self.file_paths['CC_wsis']

        self.replay_hcc_old_wsis = self.file_paths['replay_HCC_old_wsis']
        self.replay_hcc_wsis = self.file_paths['replay_HCC_wsis']
        self.replay_cc_wsis = self.file_paths['replay_CC_wsis']

        self.valid_hcc_old_wsis = self.file_paths['valid_HCC_old_wsis']
        self.valid_hcc_wsis = self.file_paths['valid_HCC_wsis']
        self.valid_cc_wsis = self.file_paths['valid_CC_wsis']

        # data paths
        self.hcc_data_dir = self.file_paths['HCC_new_patches_save_path']
        self.hcc_old_data_dir = self.file_paths['HCC_old_patches_save_path']
        self.cc_data_dir = self.file_paths['CC_patches_save_path']
        self.hcc_csv_dir = self.file_paths['HCC_csv_dir']
        self.cc_csv_dir = self.file_paths['CC_csv_dir']

        # transforms
        self.train_tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=(0,360), expand=False),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
        ])

        self.test_tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    class TrainDataset(Dataset):
        def __init__(self, data_dict, img_dir, classes, transform, state):
            self.data_dict = data_dict
            self.img_dir = img_dir
            self.classes = classes
            self.transform = transform
            self.state = state

            # mask = data_dict["label"] != 'N'
            # self.data_dict = data_dict[mask].reset_index(drop=True)

        def __getitem__(self, id):
            index = random.randint(0, len(self.data_dict["file_name"]) - 1)

            label_text = self.data_dict["label"][index]
            label = self.classes.index(label_text)
            img_name = self.data_dict["file_name"][index]

            if self.state == "old":
                if label_text == "H":
                    img_path = os.path.join(self.img_dir, "HCC", img_name)
                if label_text == "N":
                    img_path = os.path.join(self.img_dir, "Normal", img_name)
            elif self.state == "new":
                img_path = os.path.join(self.img_dir, img_name)
            
            image = Image.open(img_path)
            image = self.transform(image)
            if self.data_dict["augment"][index]:
                image = transforms.RandomRotation(degrees=(0, 360))(image)

            return image, label, img_path

        def __len__(self):
            # return len(self.data_dict["file_name"])
            return 800

    class ValidDataset(Dataset):
        def __init__(self, data_dict, img_dir, classes, transform, state):
            self.data_dict = data_dict
            self.img_dir = img_dir
            self.classes = classes
            self.transform = transform
            self.state = state

        def __getitem__(self, index):
            label_text = self.data_dict["label"][index]
            label = self.classes.index(label_text)
            img_name = self.data_dict["file_name"][index]

            if self.state == "old":
                if label_text == "H":
                    img_path = os.path.join(self.img_dir, "HCC", img_name)
                if label_text == "N":
                    img_path = os.path.join(self.img_dir, "Normal", img_name)
            elif self.state == "new":
                img_path = os.path.join(self.img_dir, img_name)
            
            image = Image.open(img_path)
            image = self.transform(image)
            if self.data_dict["augment"][index]:
                image = transforms.RandomRotation(degrees=(0, 360))(image)

            return image, label, img_path

        def __len__(self):
            return len(self.data_dict["file_name"])
    
    class TestDataset(Dataset):
        def __init__(self, data_dict, img_dir, classes, transform, state, label_exist=True):
            
            self.data_dict = data_dict
            self.img_dir = img_dir
            self.classes = classes
            self.transform = transform
            self.label_exist = label_exist
            self.state = state
            
        def __getitem__(self, index):
            img_name = self.data_dict["file_name"][index]

            if "label" in self.data_dict:
                label_text = self.data_dict["label"][index]
                label = self.classes.index(label_text)

            if self.state == 'old':
                if label_text == 'H':
                    image_path = os.path.join(self.img_dir, 'HCC', img_name)
                if label_text == 'N':
                    image_path = os.path.join(self.img_dir, 'Normal', img_name)

            if self.state == 'new':
                image_path = os.path.join(self.img_dir, img_name)

            image = Image.open(image_path)
            image = self.transform(image)

            if self.label_exist:
                return image, label, img_name
            else:
                return image, img_name
        
        def __len__(self):
            return len(self.data_dict["file_name"]) if "file_name" in self.data_dict else 0
        
    def check_overlap(self, *lists):
        sets = [set(lst) for lst in lists]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                if sets[i] & sets[j]:
                    return True
        return False
    
    def split_datas(self, selected_data, data_num, tp_data=None, fp_data=None, valid_percentage=None, test_percentage=None):
        if valid_percentage is None:
            valid_percentage = self.valid_percentage
        if test_percentage is None:
            test_percentage = 0.0

        file_names = selected_data['file_name'].to_numpy()
        labels = selected_data['label'].to_numpy()        
        datas = []

        class_file_names = []
        for cl in self.classes:
            class_file_names.append(list(file_names[labels == cl]))

        if self.gen_type:
            if tp_data is not None:
                tp_file_names = tp_data['file_name'].to_numpy()
                tp_labels = tp_data['label'].to_numpy()
            else:
                tp_file_names, tp_labels = np.array([]), np.array([])

            if fp_data is not None:
                fp_file_names = fp_data['file_name'].to_numpy()
                fp_labels = fp_data['label'].to_numpy()
            else:
                fp_file_names, fp_labels = np.array([]), np.array([])

            tp_class_file_names, fp_class_file_names = [], []
            for cl in self.classes:
                tp_class_file_names.append(list(tp_file_names[tp_labels == cl]))
                fp_class_file_names.append(list(fp_file_names[fp_labels == cl]))
        
        if data_num == "ALL":
            max_len = max(len(names) for names in class_file_names)
            for num in range(self.class_num):
                class_samples = class_file_names[num]
                if len(class_samples) > 0:
                    # datas.append([(name, False) for name in class_samples])

                    # Compute how many samples are missing to reach the max size
                    n_missing = max_len - len(class_samples)
                    duplicated = []

                    # Randomly duplicate existing samples (mark as should_augment=True)
                    for _ in range(n_missing):
                        chosen = random.choice(class_samples)
                        duplicated.append((chosen, True))

                    # Combine original samples (should_augment=False) and duplicated ones
                    augmented_list = [(name, False) for name in class_samples] + duplicated
                    datas.append(augmented_list)
                else:
                    datas.append([])
        else:
            data_num = int(data_num)
            for cl in range(self.class_num):
                class_samples = class_file_names[cl]

                if self.gen_type:
                    fp_class_samples = fp_class_file_names[cl]
                    tp_class_samples = tp_class_file_names[cl]

                    if len(fp_class_samples) > 0:
                        # class_data_num = 0.5 * int(data_num) if cl == 0 else int(data_num)
                        class_data_num = int(data_num)
                        if len(fp_class_samples) >= class_data_num:
                            samples = random.sample(fp_class_samples, int(class_data_num))
                            datas.append([(name, False) for name in samples])
                        else:
                            if len(fp_class_samples) + len(tp_class_samples) >= class_data_num:
                                # datas.append([(name, False) for name in fp_class_samples])
                                class_data_num = int(class_data_num - len(fp_class_samples))
                                samples = random.sample(tp_class_samples, class_data_num)
                                full_samples = [(name, False) for name in fp_class_samples] + [(name, False) for name in samples]
                                datas.append(full_samples)
                            else:
                                temp_samples = fp_class_samples + tp_class_samples
                                # Not enough samples, we need to augment
                                n_missing = int(class_data_num - len(fp_class_samples) - len(tp_class_samples))
                                duplicated = []

                                # Repeat samples with random rotation
                                for _ in range(n_missing):
                                    chosen = random.choice(class_samples)
                                    duplicated.append((chosen, True))  # (file_name, should_augment)

                                # Store the original and augmented names
                                augmented_list = [(name, False) for name in temp_samples] + duplicated
                                datas.append(augmented_list)
                            
                    else:
                        if len(tp_class_samples) > 0:
                            # class_data_num = 0.5 * int(data_num) if cl == 0 else int(data_num)
                            class_data_num = int(data_num)
                            if len(tp_class_samples) >= class_data_num:
                                samples = random.sample(tp_class_samples, int(class_data_num))
                                datas.append([(name, False) for name in samples])
                            else:
                                # Not enough samples, we need to augment
                                n_missing = int(class_data_num - len(tp_class_samples))
                                duplicated = []

                                # Repeat samples with random rotation
                                for _ in range(n_missing):
                                    chosen = random.choice(class_samples)
                                    duplicated.append((chosen, True))  # (file_name, should_augment)

                                # Store the original and augmented names
                                augmented_list = [(name, False) for name in tp_class_samples] + duplicated
                                datas.append(augmented_list)
                        else:
                            if len(class_file_names[cl]) > 0:
                                # class_data_num = 0.5 * int(data_num) if cl == 0 else int(data_num)
                                class_data_num = int(data_num)
                                if len(class_samples) >= class_data_num:
                                    samples = random.sample(class_samples, int(class_data_num))
                                    datas.append([(name, False) for name in samples])
                                else:
                                    # Not enough samples, we need to augment
                                    n_missing = int(class_data_num - len(class_samples))
                                    duplicated = []

                                    # Repeat samples with random rotation
                                    for _ in range(n_missing):
                                        chosen = random.choice(class_samples)
                                        duplicated.append((chosen, True))  # (file_name, should_augment)

                                    # Store the original and augmented names
                                    augmented_list = [(name, False) for name in class_samples] + duplicated
                                    datas.append(augmented_list)
                            else:
                                datas.append([])
                else:
                    if len(class_file_names[cl]) > 0:
                        # class_data_num = 0.5 * int(data_num) if cl == 0 else int(data_num)
                        class_data_num = int(data_num)
                        if len(class_samples) >= class_data_num:
                            samples = random.sample(class_samples, int(class_data_num))
                            datas.append([(name, False) for name in samples])
                        else:
                            # Not enough samples, we need to augment
                            n_missing = int(class_data_num - len(class_samples))
                            duplicated = []

                            # Repeat samples with random rotation
                            for _ in range(n_missing):
                                chosen = random.choice(class_samples)
                                duplicated.append((chosen, True))  # (file_name, should_augment)

                            # Store the original and augmented names
                            augmented_list = [(name, False) for name in class_samples] + duplicated
                            datas.append(augmented_list)
                    else:
                        datas.append([])
                
        # if self.check_overlap(*datas):
        #     print(f'Data overlap.')
        #     return

        data_file_names, data_labels = [], []

        for i in range(self.class_num):
            data_file_names += [x[0] for x in datas[i]]
            data_labels += [self.classes[i]] * len(datas[i])

        # Prepare train/val dataset
        train, val, test = [], [], []
        valid_test_percentage = valid_percentage + test_percentage
        for num in range(self.class_num):
            if len(datas[num]) > 0:
                if valid_test_percentage < 1:
                    train_, temp = train_test_split(datas[num], test_size=valid_test_percentage, random_state=0)
                else:
                    train_ = []
                    temp = datas[num]
                if test_percentage > 0:
                    val_, test_ = train_test_split(temp, test_size=test_percentage/valid_test_percentage, random_state=0)
                else:
                    val_ = temp
                    test_ = []
                
                train.append(train_)
                val.append(val_)
                test.append(test_)
            else:
                train.append([])
                val.append([])
                test.append([])

        # Flatten and extract file_name, label, and augment flag
        def extract_info(split_list, class_index):
            filenames = [x[0] for x in split_list]
            labels = [self.classes[class_index]] * len(split_list)
            augment_flags = [x[1] for x in split_list]
            return filenames, labels, augment_flags

        train_file_names, train_labels, train_augments = [], [], []
        val_file_names, val_labels, val_augments = [], [], []
        test_file_names, test_labels, test_augments = [], [], []

        for i in range(self.class_num):
            fn, lb, aug = extract_info(train[i], i)
            train_file_names += fn
            train_labels += lb
            train_augments += aug

            fn, lb, aug = extract_info(val[i], i)
            val_file_names += fn
            val_labels += lb
            val_augments += aug

            fn, lb, aug = extract_info(test[i], i)
            test_file_names += fn
            test_labels += lb
            test_augments += aug

        error_rate = self.file_paths['error_rate']  # Let 10% of data be wrong

        if error_rate > 0:
            n_samples = len(train_labels)
            n_errors = int(n_samples * error_rate)
            error_indices = random.sample(range(n_samples), n_errors)

            for i in error_indices:
                correct_label = train_labels[i]
                wrong_choices = [cl for cl in self.classes if cl != correct_label]
                train_labels[i] = random.choice(wrong_choices)

        Train = {
            "file_name": train_file_names,
            "label": train_labels,
            "augment": train_augments,
        }
        Val = {
            "file_name": val_file_names,
            "label": val_labels,
            "augment": val_augments,
        }
        Test = {
            "file_name": test_file_names,
            "label": test_labels,
            "augment": test_augments,
        }

        return Train, Val, Test

    def prepare_dataset(self, save_path, condition, gen, data_stage, wsi=None, mode=None, state=None, wsi_type=None, replay=False):
        train_data = []
        valid_data = []
        test_data = []
        other_valid_data = []

        train_datasets = []
        valid_datasets = []
        test_datasets = []
        other_valid_datasets = []

        if state == None:
            state = self.state
        if wsi_type == None:
            wsi_type = self.wsi_type
        if gen is None:
            gen_type = False
        else:
            gen_type = self.gen_type

        if replay:
            data_num = self.replay_data_num
        else:
            data_num = self.data_num

        print(f"Patches use for a WSI: {data_num}")

        if wsi == None:
            for h_wsi in self.hcc_old_wsis:
                selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{h_wsi}/{h_wsi}_patch_in_region_filter_2_v2.csv')
                Train, Valid, Test = self.split_datas(selected_data, self.data_num)
                h_train_dataset = self.TrainDataset(Train, f'{self.hcc_old_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "old")
                h_valid_dataset = self.ValidDataset(Valid, f'{self.hcc_old_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "old")
                h_test_dataset  = self.TestDataset(Test, f'{self.hcc_old_data_dir}/{h_wsi}',self.classes, self.test_tfm, state = "old", label_exist=False)

                train_datasets.append(h_train_dataset)
                valid_datasets.append(h_valid_dataset)
                test_datasets.append(h_test_dataset)
                
                train_data.extend(pd.DataFrame(Train).to_dict(orient='records'))
                valid_data.extend(pd.DataFrame(Valid).to_dict(orient='records'))
                test_data.extend(pd.DataFrame(Test).to_dict(orient='records'))

            for h_wsi in self.hcc_wsis:
                selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{h_wsi+91}/{h_wsi+91}_patch_in_region_filter_2_v2.csv')
                Train, Valid, Test = self.split_datas(selected_data, self.data_num)
                h_train_dataset = self.TrainDataset(Train, f'{self.hcc_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "new")
                h_valid_dataset = self.ValidDataset(Valid, f'{self.hcc_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "new")
                h_test_dataset  = self.TestDataset(Test, f'{self.hcc_data_dir}/{h_wsi}',self.classes, self.test_tfm, state = "new", label_exist=False)

                train_datasets.append(h_train_dataset)
                valid_datasets.append(h_valid_dataset)
                test_datasets.append(h_test_dataset)
                
                train_data.extend(pd.DataFrame(Train).to_dict(orient='records'))
                valid_data.extend(pd.DataFrame(Valid).to_dict(orient='records'))
                test_data.extend(pd.DataFrame(Test).to_dict(orient='records'))

            for c_wsi in self.cc_wsis:
                selected_data = pd.read_csv(f'{self.cc_csv_dir}/{c_wsi}/1{c_wsi:04d}_patch_in_region_filter_2_v2.csv')
                Train, Valid, Test = self.split_datas(selected_data, self.data_num)
                c_train_dataset = self.TrainDataset(Train, f'{self.cc_data_dir}/{c_wsi}', self.classes, self.train_tfm, state = "new")
                c_valid_dataset = self.ValidDataset(Valid, f'{self.cc_data_dir}/{c_wsi}', self.classes, self.train_tfm, state = "new")
                c_test_dataset  = self.TestDataset(Test, f'{self.cc_data_dir}/{c_wsi}',self.classes, self.train_tfm, state = "new", label_exist=False)

                train_datasets.append(c_train_dataset)
                valid_datasets.append(c_valid_dataset)
                test_datasets.append(c_test_dataset)

                train_data.extend(pd.DataFrame(Train).to_dict(orient='records'))
                valid_data.extend(pd.DataFrame(Valid).to_dict(orient='records'))
                test_data.extend(pd.DataFrame(Test).to_dict(orient='records'))

            train_dataset = ConcatDataset(train_datasets)
            valid_dataset = ConcatDataset(valid_datasets)
            test_dataset = ConcatDataset(test_datasets)

        else:
            if state == "old":
                if gen_type:
                    selected_data = pd.read_csv(f'{save_path}/{wsi}_Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}.csv')
                    # tp_data = pd.read_csv(f'{save_path}/{wsi}_Gen{gen}_ND_zscore_{mode}_tp_patches_by_Gen{gen-1}.csv')
                    # fp_data = pd.read_csv(f'{save_path}/{wsi}_Gen{gen}_ND_zscore_{mode}_fp_patches_by_Gen{gen-1}.csv')
                    # Train, Valid, Test = self.split_datas(selected_data, self.data_num, tp_data=tp_data, fp_data=fp_data)
                    Train, Valid, Test = self.split_datas(selected_data, data_num)
                else:
                    selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{wsi}/{wsi}_patch_in_region_filter_2_v2.csv')
                    Train, Valid, Test = self.split_datas(selected_data, data_num)
                train_dataset = self.TrainDataset(Train, f'{self.hcc_old_data_dir}/{wsi}', self.classes, self.train_tfm, state = "old")
                valid_dataset = self.ValidDataset(Valid, f'{self.hcc_old_data_dir}/{wsi}', self.classes, self.train_tfm, state = "old")
                test_dataset  = self.TestDataset(Test, f'{self.hcc_old_data_dir}/{wsi}',self.classes, self.test_tfm, state = "old", label_exist=False)
            
            elif wsi_type == "HCC":
                if gen_type:
                    selected_data = pd.read_csv(f'{save_path}/{wsi+91}_Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}.csv')
                    # tp_data = pd.read_csv(f'{save_path}/{wsi+91}_Gen{gen}_ND_zscore_{mode}_tp_patches_by_Gen{gen-1}.csv')
                    # fp_data = pd.read_csv(f'{save_path}/{wsi+91}_Gen{gen}_ND_zscore_{mode}_fp_patches_by_Gen{gen-1}.csv')
                    # Train, Valid, Test = self.split_datas(selected_data, self.data_num, tp_data=tp_data, fp_data=fp_data)
                    Train, Valid, Test = self.split_datas(selected_data, data_num)
                else:
                    selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{wsi+91}/{wsi+91}_patch_in_region_filter_2_v2.csv')
                    Train, Valid, Test = self.split_datas(selected_data, data_num)
                train_dataset = self.TrainDataset(Train, f'{self.hcc_data_dir}/{wsi}', self.classes, self.train_tfm, state = "new")
                valid_dataset = self.ValidDataset(Valid, f'{self.hcc_data_dir}/{wsi}', self.classes, self.train_tfm, state = "new")
                test_dataset  = self.TestDataset(Test, f'{self.hcc_data_dir}/{wsi}',self.classes, self.test_tfm, state = "new", label_exist=False)
            
            elif wsi_type == "CC":
                if gen_type:
                    selected_data = pd.read_csv(f'{save_path}/1{wsi:04d}_Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}.csv')
                    # tp_data = pd.read_csv(f'{save_path}/1{wsi:04d}_Gen{gen}_ND_zscore_{mode}_tp_patches_by_Gen{gen-1}.csv')
                    # fp_data = pd.read_csv(f'{save_path}/1{wsi:04d}_Gen{gen}_ND_zscore_{mode}_fp_patches_by_Gen{gen-1}.csv')
                    # Train, Valid, Test = self.split_datas(selected_data, self.data_num, tp_data=tp_data, fp_data=fp_data)
                    Train, Valid, Test = self.split_datas(selected_data, data_num)
                else:
                    selected_data = pd.read_csv(f'{self.cc_csv_dir}/{wsi}/1{wsi:04d}_patch_in_region_filter_2_v2.csv')
                    Train, Valid, Test = self.split_datas(selected_data, data_num)
                train_dataset = self.TrainDataset(Train, f'{self.cc_data_dir}/{wsi}', self.classes, self.train_tfm, state = "new")
                valid_dataset = self.ValidDataset(Valid, f'{self.cc_data_dir}/{wsi}', self.classes, self.train_tfm, state = "new")
                test_dataset  = self.TestDataset(Test, f'{self.cc_data_dir}/{wsi}',self.classes, self.train_tfm, state = "new", label_exist=False)
            else:
                if self.test_state == "old":
                    if self.gen_type:
                        selected_data = pd.read_csv(f'{save_path}/{wsi}_Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}.csv')
                        # tp_data = pd.read_csv(f'{save_path}/{wsi}_Gen{gen}_ND_zscore_{mode}_tp_patches_by_Gen{gen-1}.csv')
                        # fp_data = pd.read_csv(f'{save_path}/{wsi}_Gen{gen}_ND_zscore_{mode}_fp_patches_by_Gen{gen-1}.csv')
                        # Train, Valid, Test = self.split_datas(selected_data, self.data_num, tp_data=tp_data, fp_data=fp_data)
                        Train, Valid, Test = self.split_datas(selected_data, self.data_num)
                    else:
                        selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{wsi}/{wsi}_patch_in_region_filter_2_v2.csv')
                        Train, Valid, Test = self.split_datas(selected_data, self.data_num)
                    train_dataset = self.TrainDataset(Train, f'{self.hcc_old_data_dir}/{wsi}', self.classes, self.train_tfm, state = "old")
                    valid_dataset = self.ValidDataset(Valid, f'{self.hcc_old_data_dir}/{wsi}', self.classes, self.train_tfm, state = "old")
                    test_dataset  = self.TestDataset(Test, f'{self.hcc_old_data_dir}/{wsi}',self.classes, self.test_tfm, state = "old", label_exist=False)
                elif self.test_type == "HCC":
                    if self.gen_type:
                        selected_data = pd.read_csv(f'{save_path}/{wsi+91}_Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}.csv')
                        # tp_data = pd.read_csv(f'{save_path}/{wsi+91}_Gen{gen}_ND_zscore_{mode}_tp_patches_by_Gen{gen-1}.csv')
                        # fp_data = pd.read_csv(f'{save_path}/{wsi+91}_Gen{gen}_ND_zscore_{mode}_fp_patches_by_Gen{gen-1}.csv')
                        # Train, Valid, Test = self.split_datas(selected_data, self.data_num, tp_data=tp_data, fp_data=fp_data)
                        Train, Valid, Test = self.split_datas(selected_data, self.data_num)
                    else:
                        selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{wsi+91}/{wsi+91}_patch_in_region_filter_2_v2.csv')
                        Train, Valid, Test = self.split_datas(selected_data, self.data_num)
                    train_dataset = self.TrainDataset(Train, f'{self.hcc_data_dir}/{wsi}', self.classes, self.train_tfm, state = "new")
                    valid_dataset = self.ValidDataset(Valid, f'{self.hcc_data_dir}/{wsi}', self.classes, self.train_tfm, state = "new")
                    test_dataset  = self.TestDataset(Test, f'{self.hcc_data_dir}/{wsi}',self.classes, self.test_tfm, state = "new", label_exist=False)
                else:
                    if self.gen_type:
                        selected_data = pd.read_csv(f'{save_path}/1{wsi:04d}_Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}.csv')
                        # tp_data = pd.read_csv(f'{save_path}/1{wsi:04d}_Gen{gen}_ND_zscore_{mode}_tp_patches_by_Gen{gen-1}.csv')
                        # fp_data = pd.read_csv(f'{save_path}/1{wsi:04d}_Gen{gen}_ND_zscore_{mode}_fp_patches_by_Gen{gen-1}.csv')
                        # Train, Valid, Test = self.split_datas(selected_data, self.data_num, tp_data=tp_data, fp_data=fp_data)
                        Train, Valid, Test = self.split_datas(selected_data, self.data_num)
                    else:
                        selected_data = pd.read_csv(f'{self.cc_csv_dir}/{wsi}/1{wsi:04d}_patch_in_region_filter_2_v2.csv')
                        Train, Valid, Test = self.split_datas(selected_data, self.data_num)
                    train_dataset = self.TrainDataset(Train, f'{self.cc_data_dir}/{wsi}', self.classes, self.train_tfm, state = "new")
                    valid_dataset = self.ValidDataset(Valid, f'{self.cc_data_dir}/{wsi}', self.classes, self.train_tfm, state = "new")
                    test_dataset  = self.TestDataset(Test, f'{self.cc_data_dir}/{wsi}',self.classes, self.train_tfm, state = "new", label_exist=False)

            train_data.extend(pd.DataFrame(Train).to_dict(orient='records'))
            valid_data.extend(pd.DataFrame(Valid).to_dict(orient='records'))
            test_data.extend(pd.DataFrame(Test).to_dict(orient='records'))

        if self.other_validation and data_stage == "train":
            for h_wsi in self.valid_hcc_old_wsis:
                selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{h_wsi}/{h_wsi}_patch_in_region_filter_2_v2.csv')
                _, Valid, _ = self.split_datas(selected_data, self.other_valid_size, valid_percentage=1.0)
                h_valid_dataset = self.ValidDataset(Valid, f'{self.hcc_old_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "old")

                other_valid_datasets.append(h_valid_dataset)
                other_valid_data.extend(pd.DataFrame(Valid).to_dict(orient='records'))

            for h_wsi in self.valid_hcc_wsis:
                selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{h_wsi+91}/{h_wsi+91}_patch_in_region_filter_2_v2.csv')
                _, Valid, _ = self.split_datas(selected_data, self.other_valid_size, valid_percentage=1.0)
                h_valid_dataset = self.ValidDataset(Valid, f'{self.hcc_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "new")

                other_valid_datasets.append(h_valid_dataset)
                other_valid_data.extend(pd.DataFrame(Valid).to_dict(orient='records'))

            for c_wsi in self.valid_cc_wsis:
                selected_data = pd.read_csv(f'{self.cc_csv_dir}/{c_wsi}/1{c_wsi:04d}_patch_in_region_filter_2_v2.csv')
                _, Valid, _ = self.split_datas(selected_data, self.other_valid_size, valid_percentage=1.0)
                c_valid_dataset = self.ValidDataset(Valid, f'{self.cc_data_dir}/{c_wsi}', self.classes, self.train_tfm, state = "new")

                other_valid_datasets.append(c_valid_dataset)
                other_valid_data.extend(pd.DataFrame(Valid).to_dict(orient='records'))

            other_valid_dataset = ConcatDataset(other_valid_datasets)
        else:
            other_valid_dataset = None

        if data_stage == "train":
            pd.DataFrame(train_data).to_csv(f"{save_path}/{condition}_train.csv", index=False)
            pd.DataFrame(valid_data).to_csv(f"{save_path}/{condition}_valid.csv", index=False)
            pd.DataFrame(other_valid_data).to_csv(f"{save_path}/{condition}_other_valid.csv", index=False)
        elif data_stage == "test":
            pd.DataFrame(test_data).to_csv(f"{save_path}/{condition}_test.csv", index=False)

        return train_dataset, valid_dataset, other_valid_dataset, test_dataset

    def load_datasets(self, save_path, condition, data_stage, wsi, state=None, wsi_type=None):
        train_csv = f"{save_path}/{condition}_train.csv"
        valid_csv = f"{save_path}/{condition}_valid.csv"
        test_csv  = f"{save_path}/{condition}_test.csv"

        if state == None:
            state = self.state
        if wsi_type == None:
            wsi_type = self.wsi_type

        if data_stage == "train":
            if os.path.exists(train_csv) and os.path.exists(valid_csv):
                # read from existing files
                Train = pd.read_csv(train_csv).to_dict(orient="list")
                Valid = pd.read_csv(valid_csv).to_dict(orient="list")
                if state == "old":
                    train_dataset = self.TrainDataset(Train, f"{self.hcc_old_data_dir}/{wsi}", self.classes, self.train_tfm, state="old")
                    valid_dataset = self.ValidDataset(Valid, f"{self.hcc_old_data_dir}/{wsi}", self.classes, self.train_tfm, state="old")
                elif wsi_type == "HCC":
                    train_dataset = self.TrainDataset(Train, f"{self.hcc_data_dir}/{wsi}", self.classes, self.train_tfm, state="new")
                    valid_dataset = self.ValidDataset(Valid, f"{self.hcc_data_dir}/{wsi}", self.classes, self.train_tfm, state="new")
                else:
                    train_dataset = self.TrainDataset(Train, f"{self.cc_data_dir}/{wsi}", self.classes, self.train_tfm, state="new")
                    valid_dataset = self.ValidDataset(Valid, f"{self.cc_data_dir}/{wsi}", self.classes, self.train_tfm, state="new")
                return train_dataset, valid_dataset, None
            else:
                print(train_csv, valid_csv)
                raise FileNotFoundError("train/valid CSV not found")

        elif data_stage == "test":
            if os.path.exists(test_csv):
                Test = pd.read_csv(test_csv).to_dict(orient="list")
                if self.state == "old":
                    test_dataset = self.TestDataset(Test, f"{self.hcc_old_data_dir}/{wsi}", self.classes, self.train_tfm, state="old", label_exist=False)
                elif self.state == "HCC":
                    test_dataset = self.TestDataset(Test, f"{self.hcc_data_dir}/{wsi}", self.classes, self.train_tfm, state="new", label_exist=False)
                else:
                    test_dataset = self.TestDataset(Test, f"{self.cc_data_dir}/{wsi}", self.classes, self.train_tfm, state="new", label_exist=False)
                return None, None, test_dataset
            else:
                print(test_csv)
                raise FileNotFoundError("test CSV not found")

    def build_pl_dataset(self, wsi, gen, save_path, mode, labeled, test_state=None, test_type=None):
        '''
        selected_patches: patches that is in some class of contour, but the label of patch may not the same as the contour label.
        
        '''
        if test_state == None:
            test_state = self.test_state
        if test_type == None:
            test_type = self.test_type

        if test_state == "old":
            _wsi = wsi
        elif test_type == "HCC":
            _wsi = wsi + 91
        elif test_type == "CC":
            _wsi = f"1{wsi:04d}"

        if gen == 1:
            if labeled:
                df = pd.read_csv(f"{save_path}/TI/{_wsi}_{self.class_num}_class_patch_in_region_filter_2_v2_TI.csv")
            else:
                df = pd.read_csv(f"{save_path}/TI/{_wsi}_{self.class_num}_class_all_patches_filter_v2_TI.csv")
        else:
            if labeled:
                df = pd.read_csv(f"{save_path}/TI/{_wsi}_Gen{gen-1}_ND_zscore_{mode}_patches_by_Gen{gen-2}_patch_in_region_filter_2_v2_TI.csv")
            else:
                df = pd.read_csv(f"{save_path}/TI/{_wsi}_Gen{gen-1}_ND_zscore_{mode}_patches_by_Gen{gen-2}_all_patches_filter_v2_TI.csv")

        all_patches = df['file_name'].to_list()
        selected_columns = []
        for cl in self.classes:
            selected_columns.append(f"{cl}_pred")
        selected_data = df[selected_columns].to_numpy()

        ### Get (x, y, pseudo-label) of every patch ###
        all_pts = []
        for idx, img_name in enumerate(all_patches):
            if test_state == "old":
                match = re.search(r'-(\d+)-(\d+)-\d{5}x\d{5}', img_name)
                if match:
                    x = match.group(1)
                    y = match.group(2)
            else:
                x, y = img_name[:-4].split('_')
            
            row = selected_data[idx, :]
            max_col = np.argmax(row)
            all_pts.append([(int(x)) // self.patch_size, (int(y)) // self.patch_size, max_col])
        all_pts = np.array(all_pts)

        ### First sorted pts on x, then on y ###
        sorted_index = np.lexsort((all_pts[:, 1], all_pts[:, 0]))
        sorted_all_pts = all_pts[sorted_index] # x, y, label

        x_max = np.max(sorted_all_pts[:, 0])
        y_max = np.max(sorted_all_pts[:, 1])
        
        # convert to 1-based labels: N=1, H=2, C=3
        labels_1based = sorted_all_pts[:, 2] + 1
        original_image = np.zeros((y_max + 1, x_max + 1), dtype=np.uint8)
        for (x, y), label in zip(sorted_all_pts[:, :2], labels_1based):
            original_image[y, x] = label

        sorted_all_patches = [all_patches[i] for i in sorted_index]
        selected_patches = {
            'file_name': sorted_all_patches,
            'label': [self.classes[int(cl)] for cl in sorted_all_pts[:, 2]]
        }
        background = []

        for idx, cl in enumerate(self.classes):
            print(f"running for {cl} ...")

            ### Make components regions ###
            binary_img = np.zeros((y_max+1, x_max+1), np.uint8)  # for connected-components of HCC/Normal patches
            h, w = binary_img.shape

            cate_pts = sorted_all_pts[sorted_all_pts[:, 2] == idx]

            for pts in cate_pts:
                x, y = pts[0], pts[1]
                binary_img[y, x] = 1
                        
            if self.file_paths['flip_mode'] == "connect":
                from dataset import patch_flipping_connect
                                
                img_pad = np.zeros((h+2, w+2))
                img_pad[1:1+h, 1:1+w] = binary_img
                original_img_pad = np.zeros((h+2, w+2))
                original_img_pad[1:1+h, 1:1+w] = original_image

                num_labels, labels, stats = patch_flipping_connect.find_areas(img_pad)
                flip_pts = patch_flipping_connect.flip_patch(img_pad, original_img_pad, self.classes, num_labels, labels, stats, area_thresh=self.file_paths['area_thresh'])
                # img_flipped = img_pad_flipped[1:-1, 1:-1]
                for ptx, pty, cl_idx in flip_pts:
                    cl_text = "HCC" if cl == "H" else "Normal"
                    formatted_filename = (
                        f'C{wsi}_{cl_text}-{int(ptx * self.patch_size):05d}-{int(pty * self.patch_size):05d}-{self.patch_size:05d}x{self.patch_size:05d}.tif'
                        if self.state == "old"
                        else f'{int(ptx * self.patch_size)}_{int(pty * self.patch_size)}.tif'
                    )
                    # if (formatted_filename in all_patches) and (formatted_filename not in selected_patches['file_name']):
                    if formatted_filename in sorted_all_patches:
                        if cl_idx != 0:
                            patch_id = selected_patches['file_name'].index(formatted_filename)
                            selected_patches['label'][patch_id] = self.classes[cl_idx-1]
                            # selected_patches['file_name'].append(formatted_filename)
                            # selected_patches['label'].append(self.classes[cl_idx-1])
                        else:
                            # background.append(formatted_filename)
                            patch_id = selected_patches['file_name'].index(formatted_filename)
                            selected_patches['file_name'].pop(patch_id)
                            selected_patches['label'].pop(patch_id)  # keep labels aligned

            elif self.file_paths['flip_mode'] == "contour":
                from dataset import patch_flipping_contour
                
                contours, hierarchy = patch_flipping_contour.find_areas(binary_img)
                img_flipped = patch_flipping_contour.flip_patch(binary_img, contours, hierarchy, area_thresh=self.file_paths['area_thresh'])

                flip_patches = cv2.absdiff(img_flipped.astype(np.uint8), binary_img.astype(np.uint8))

                for pty in range(h):
                    for ptx in range(w):
                        if flip_patches[pty, ptx] == 1:
                            cl_text = "HCC" if cl == "H" else "Normal"
                            formatted_filename = (
                                f'C{wsi}_{cl_text}-{int(ptx * self.patch_size):05d}-{int(pty * self.patch_size):05d}-{self.patch_size:05d}x{self.patch_size:05d}.tif'
                                if self.state == "old"
                                else f'{int(ptx * self.patch_size)}_{int(pty * self.patch_size)}.tif'
                            )
                            if (formatted_filename in all_patches) and (formatted_filename not in selected_patches['file_name']):
                                selected_patches['file_name'].append(formatted_filename)
                                selected_patches['label'].append(cl)

        pd.DataFrame(selected_patches).to_csv(f"{save_path}/Data/{_wsi}_Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}.csv", index=False)

    class EfficientNetWithLinear(nn.Module):
        def __init__(self, output_dim, pretrain='efficientnet-b0'):
            super().__init__()
            # Load pre-trained EfficientNet model
            self.backbone = EfficientNet.from_name(pretrain)
            
            # Save the original last layer before Linear
            self.backbone._fc = nn.Identity()  # Remove the default classifier
            
            # New c1lassification layer
            self.fc = nn.Sequential(
                nn.Linear(1280, 2560),  # Assuming EfficientNet-B0 with 1280 features
                nn.ReLU(),
                nn.Linear(2560, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )

        def forward(self, x):
            # Extract features from backbone
            features = self.backbone(x)
            # Classifier output
            output = self.fc(features)
            
            return output

    class ViTWithLinear(nn.Module):
        def __init__(self, output_dim, pretrain=True):
            super().__init__()
            # Load pre-trained ViT
            weights = ViT_B_16_Weights.DEFAULT if pretrain else None
            self.backbone = vit_b_16(weights=weights)
            
            # Remove the default classifier
            self.backbone.heads = nn.Identity()  # Feature extractor
            
            # New classification head
            self.fc = nn.Sequential(
                nn.Linear(768, 2560),  # 768 is ViT-B/16 feature dim
                nn.ReLU(),
                nn.Linear(2560, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )

        def forward(self, x):
            features = self.backbone(x)
            output = self.fc(features)
            return output

    class ViTWithLinearTiny(nn.Module):
        def __init__(self, output_dim, pretrain=True):
            super().__init__()

            self.backbone = timm.create_model(
                'vit_tiny_patch16_224',
                pretrained=pretrain,
                num_classes=0 # remove classifier head
            )
            
            # New classification head
            self.fc = nn.Sequential(
                nn.Linear(192, 2560),  # 192 is ViT-Tiny feature dim
                nn.ReLU(),
                nn.Linear(2560, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )

        def forward(self, x):
            features = self.backbone(x)
            output = self.fc(features)
            return output

    def plot_loss_acc(self, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, save_path, condition):
        epochs = range(1, len(train_loss_list) + 1)
        fig, ax1 = plt.subplots(figsize=(8, 6))

        l1 = ax1.plot(epochs, train_loss_list, label="Train Loss", marker="o", linestyle="-", color="blue")
        l2 = ax1.plot(epochs, valid_loss_list, label="Valid Loss", marker="s", linestyle="--", color="red")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        l3 = ax2.plot(epochs, train_acc_list, label="Train Accuracy", marker="^", linestyle="-", color="green")
        l4 = ax2.plot(epochs, valid_acc_list, label="Valid Accuracy", marker="D", linestyle="--", color="orange")
        ax2.set_ylabel("Accuracy")
        ax2.tick_params(axis="y", labelcolor="green")

        lines = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")

        plt.title("Train vs Valid Loss & Accuracy")
        plt.savefig(f"{save_path}/{condition}_loss_and_accuracy_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def _train(self, model, modelName, criterion, optimizer, train_loader, val_loader, condition, model_save_path, loss_save_path, target_class=None, other_val_loader=None):
        n_epochs = self.file_paths['max_epoch']
        min_epoch = self.file_paths['min_epoch']
        max_notImprove = self.file_paths['max_notImprove']
        use_other_val = self.other_validation and (other_val_loader is not None)

        notImprove = 0
        min_loss = 1000.
        max_acc = 0

        ema_model = ema(model.parameters(), decay=self.file_paths['ema_decay'])
        scaler = torch.amp.GradScaler(device="cuda")

        train_loss_list = []
        train_acc_list = []
        valid_loss_list = []
        valid_acc_list = []
        other_valid_loss_list = []
        other_valid_acc_list = []
        iter_train_loss_list = []
        iter_valid_loss_list = []
        iter_other_valid_loss_list = []

        for epoch in range(1, n_epochs+1):
            # ---------- Training ----------
            # Make sure the model is in train mode before training.
            model.train()

            # These are used to record information in training.
            train_loss = []
            train_acc = []

            for idx, batch in enumerate(tqdm(train_loader)):
                # A batch consists of image data and corresponding labels.
                imgs, labels, _ = batch
                imgs = imgs.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                # ===== AMP forward =====
                # Forward the data. (Make sure data and model are on the same device.)
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    if target_class == None:
                        labels = torch.nn.functional.one_hot(labels.to(device), self.class_num).float().to(device)  # one-hot vector
                        logits = model(imgs.to(device))
                        loss = criterion(logits, labels)
                        preds = torch.nn.functional.one_hot((torch.sigmoid(logits)).argmax(dim=1), self.class_num).int()
                        # preds = (torch.sigmoid(logits) >= 0.5).int()
                    else:            
                        labels = (labels == target_class).to(device).unsqueeze(1).float()
                        logits = model(imgs.to(device))
                        loss = criterion(logits, labels)
                        preds = torch.nn.functional.one_hot((torch.sigmoid(logits)).argmax(dim=1), self.class_num).int()
                        # preds = (torch.sigmoid(logits) > 0.5).int()
                # acc = (preds.eq(labels.int()).all(dim=1)).float().mean()
                correct_per_sample = torch.all(preds == labels.int(), dim=1)  # True/False per sample
                acc = correct_per_sample.float().mean()

                # # Gradients stored in the parameters in the previous step should be cleared out first.
                # optimizer.zero_grad()
                # # Compute the gradients for parameters.
                # loss.backward()
                # # Update the parameters with computed gradients.
                # optimizer.step()                
                
                # ===== AMP backward =====
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # update EMA right after optimizer step
                ema_model.update(model.parameters())
                # Record the loss and accuracy.
                train_loss.append(loss.cpu().item())
                iter_train_loss_list.append(loss.cpu().item())
                train_acc.append(acc.cpu().item())
                torch.cuda.empty_cache()
                    
                train_avg_loss = sum(train_loss) / len(train_loss)
                train_avg_acc = sum(train_acc) / len(train_acc)
                
            train_loss_list.append(train_avg_loss)
            train_acc_list.append(train_avg_acc)
            msg = f"[ Train | {epoch:03d}/{n_epochs:03d} ] loss = {train_avg_loss:.5f}, acc = {train_avg_acc:.5f}"
            print(msg)

            torch.cuda.empty_cache()

            # ---------- Validation ----------
            # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
            model.eval()
            # swap EMA weights for validation
            ema_model.store()
            ema_model.copy_to(model.parameters())

            if epoch % self.model_save_freq == 0:
                torch.save(model.state_dict(), f"{model_save_path}/{condition}_Model_epoch{epoch}.ckpt")

            # These are used to record information in validation.
            valid_loss = []
            valid_acc = []
            # Iterate the validation set by batches.
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    for idx, batch in enumerate(tqdm(val_loader)):
                        # A batch consists of image data and corresponding labels.
                        imgs, labels, _ = batch
                        if target_class == None:
                            labels = torch.nn.functional.one_hot(labels.to(device), self.class_num).float().to(device)  # one-hot vector
                            logits = model(imgs.to(device))
                            loss = criterion(logits, labels)
                            preds = torch.nn.functional.one_hot((torch.sigmoid(logits)).argmax(dim=1), self.class_num).int()
                            # preds = (logits >= 0.5).int()
                        else:            
                            labels = (labels == target_class).to(device).unsqueeze(1).float()
                            logits = model(imgs.to(device))
                            loss = criterion(logits, labels)
                            preds = torch.nn.functional.one_hot((torch.sigmoid(logits)).argmax(dim=1), self.class_num).int()
                            # preds = (logits > 0.5).int()
                        # val_acc = (preds.eq(labels.int()).all(dim=1)).float().mean()
                        correct_per_sample = torch.all(preds == labels.int(), dim=1)  # True/False per sample
                        val_acc = correct_per_sample.float().mean()
                        
                        # Record the loss and accuracy.
                        valid_loss.append(loss.cpu().item())
                        iter_valid_loss_list.append(loss.cpu().item())
                        valid_acc.append(val_acc.cpu().item())
                        torch.cuda.empty_cache()

                        # The average loss and accuracy for entire validation set is the average of the recorded values.
                        valid_avg_loss = sum(valid_loss) / len(valid_loss)
                        valid_avg_acc = sum(valid_acc) / len(valid_acc)

            valid_loss_list.append(valid_avg_loss)
            valid_acc_list.append(valid_avg_acc)
            torch.cuda.empty_cache()

            # Print the information.
            msg = f"[ Valid | {epoch:03d}/{n_epochs:03d} ] loss = {valid_avg_loss:.5f}, acc = {valid_avg_acc:.5f}"
            print(msg)

            if use_other_val:
                other_valid_loss = []
                other_valid_acc = []
                with torch.no_grad():
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        for idx, batch in enumerate(tqdm(other_val_loader)):
                            imgs, labels, _ = batch
                            if target_class == None:
                                labels = torch.nn.functional.one_hot(labels.to(device), self.class_num).float().to(device)  # one-hot vector
                                logits = model(imgs.to(device))
                                loss = criterion(logits, labels)
                                preds = torch.nn.functional.one_hot((torch.sigmoid(logits)).argmax(dim=1), self.class_num).int()
                                # preds = (logits >= 0.5).int()
                                correct_per_sample = torch.all(preds == labels.int(), dim=1)  # True/False per sample
                                val_acc = correct_per_sample.float().mean()
                            else:            
                                labels = (labels == target_class).to(device).unsqueeze(1).float()
                                logits = model(imgs.to(device))
                                loss = criterion(logits, labels)
                                preds = torch.nn.functional.one_hot((torch.sigmoid(logits)).argmax(dim=1), self.class_num).int()
                                # preds = (logits > 0.5).int()
                                correct_per_sample = torch.all(preds == labels.int(), dim=1)  # True/False per sample
                                val_acc = correct_per_sample.float().mean()
                            
                            other_valid_loss.append(loss.cpu().item())
                            iter_other_valid_loss_list.append(loss.cpu().item())
                            other_valid_acc.append(val_acc.cpu().item())
                            torch.cuda.empty_cache()

                            other_valid_avg_loss = sum(other_valid_loss) / len(other_valid_loss)
                            other_valid_avg_acc = sum(other_valid_acc) / len(other_valid_acc)
                
                other_valid_loss_list.append(other_valid_avg_loss)
                other_valid_acc_list.append(other_valid_avg_acc)
                torch.cuda.empty_cache()

                msg = f"[ Other Valid | {epoch:03d}/{n_epochs:03d} ] loss = {other_valid_avg_loss:.5f}, acc = {other_valid_avg_acc:.5f}"
                print(msg)

            training_log_dict = {
                "train_loss": train_loss_list,
                "valid_loss": valid_loss_list,
                "train_acc": train_acc_list,
                "valid_acc": valid_acc_list,
            }

            if use_other_val:
                training_log_dict.update({
                    "other_valid_loss": other_valid_loss_list,
                    "other_valid_acc": other_valid_acc_list,
                })

            training_log = pd.DataFrame(training_log_dict)
            
            training_iteration_log = pd.DataFrame({
                "train_loss": iter_train_loss_list
            })
            validation_iteration_log = pd.DataFrame({
                "valid_loss": iter_valid_loss_list
            })
            if use_other_val:
                other_validation_iteration_log = pd.DataFrame({
                    "other_valid_loss": iter_other_valid_loss_list
                })

            training_log.to_csv(f"{loss_save_path}/{condition}_epoch_log.csv", index=False)
            training_iteration_log.to_csv(f"{loss_save_path}/{condition}_train_iteration_log.csv", index=False)
            validation_iteration_log.to_csv(f"{loss_save_path}/{condition}_valid_iteration_log.csv", index=False)
            if use_other_val:
                other_validation_iteration_log.to_csv(f"{loss_save_path}/{condition}_other_valid_iteration_log.csv", index=False)

            # torch.save(ema_model.state_dict(), f"{model_save_path}/{condition}_Model_epoch{epoch}.ckpt")
            update_loss = other_valid_avg_loss if use_other_val else valid_avg_loss

            if update_loss < min_loss:
            # if valid_avg_acc > max_acc:
                # Save model if your model improved
                min_loss = update_loss
                # max_acc = valid_avg_acc
                torch.save(model.state_dict(), f"{model_save_path}/{modelName}")
                # torch.save(ema_model.state_dict(), f"{model_save_path}/{modelName}")  # use EMA weights as "best"
                notImprove = 0
            else:
                notImprove = notImprove + 1

            ema_model.restore()

            if epoch == min_epoch:
                notImprove = 0
            if notImprove >= max_notImprove and epoch >= min_epoch:
                self.plot_loss_acc(train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, loss_save_path, condition)
                return

        self.plot_loss_acc(train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, loss_save_path, condition)

    def train_one_WSI(self, wsi):
        if self.state == "old":
            _wsi = wsi
        elif self.wsi_type == "HCC":
            _wsi = wsi + 91
        elif self.wsi_type == "CC":
            _wsi = f"1{wsi:04d}"

        if self.wsi_type == "HCC":
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"
            data_save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_wsi}/trial_{self.data_trial}"
        else:
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"
            data_save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_wsi}/trial_{self.data_trial}"

        condition = f"{_wsi}_{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
        print(f"WSI {_wsi} | {condition}")

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)

        if self.load_dataset:
            data_condition = f"{_wsi}_{self.num_wsi}WTC_LP{self.data_num}_2_class_trial_{self.data_trial}"
            train_dataset, valid_dataset, _ = self.load_datasets(f"{data_save_path}/Data", data_condition, "train", wsi=wsi)
        else:
            train_dataset, valid_dataset, _ = self.prepare_dataset(f"{save_path}/Data", condition, 0, "train", wsi = wsi)

        print(f"training data number: {len(train_dataset)}, validation data number: {len(valid_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)

        if self.backbone == "ViT":
            model = self.ViTWithLinear(output_dim=self.class_num)
        elif self.backbone == "ViT_tiny":
            model = self.ViTWithLinearTiny(output_dim=self.class_num)
        else:
            model = self.EfficientNetWithLinear(output_dim=self.class_num)
        if self.pretrain:
            pretrain_model_path = self.file_paths[f'{self.test_model}_model_path']
            model.load_state_dict(torch.load(pretrain_model_path, weights_only=True))

        optimizer = torch.optim.Adam(model.parameters(), lr=self.base_lr)
        model.to(device)
        modelName = f"{condition}_Model.ckpt"
        
        criterion = nn.BCEWithLogitsLoss()

        self._train(model, modelName, criterion, optimizer, train_loader, val_loader, condition, f"{save_path}/Model", f"{save_path}/Loss")

    def train(self):
        condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
        save_path = f"{self.save_path}/trial_{self.num_trial}"
        print(f"Trial {self.num_trial}")
        print(f"WSI number: {self.num_wsi}")

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)

        train_dataset, valid_dataset, other_valid_dataset, _ = self.prepare_dataset(f"{save_path}/Data", condition, 0, "train")
        print(f"training data number: {len(train_dataset)}, validation data number: {len(valid_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        other_val_loader = DataLoader(other_valid_dataset, batch_size=self.batch_size, shuffle=False) if self.other_validation else None

        # data_iter = iter(train_loader)
        # images, labels , file_names= next(data_iter)
        # img_grid = torchvision.utils.make_grid(images)

        # plt.imshow(img_grid.permute(1, 2, 0))
        # plt.show()
        
        # modelName = f"{condition}_Model_40.ckpt"
        modelName = f"{condition}_Model.ckpt"
        model_path = f"{save_path}/Model/{modelName}"

        if self.backbone == "ViT":
            model = self.ViTWithLinear(output_dim=self.class_num)
        elif self.backbone == "ViT_tiny":
            model = self.ViTWithLinearTiny(output_dim=self.class_num)
        else:
            model = self.EfficientNetWithLinear(output_dim=self.class_num)
        if self.pretrain:
            model.load_state_dict(torch.load(model_path))
        model.to(device)
        
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.base_lr)

        self._train(model, modelName, criterion, optimizer, train_loader, val_loader, condition, f"{save_path}/Model", f"{save_path}/Loss", other_val_loader=other_val_loader)

    def train_multi_model(self):
        condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
        if self.num_wsi == 1 and self.wsi_type == "HCC":
            save_path = f"{self.save_path}/{self.hcc_wsis[0]}/trial_{self.num_trial}"
        elif self.num_wsi == 1 and self.wsi_type == "CC":
            save_path = f"{self.save_path}/{self.cc_wsis[0]}/trial_{self.num_trial}"
        else:
            save_path = f"{self.save_path}/trial_{self.num_trial}"

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)

        train_dataset, valid_dataset, _ = self.prepare_dataset(f"{save_path}/Data", condition, 0, "train")
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        
        for c in self.classes:
            if self.backbone == "ViT":
                model = self.ViTWithLinear(output_dim=1)
            elif self.backbone == "ViT_tiny":
                model = self.ViTWithLinearTiny(output_dim=1)
            else:
                model = self.EfficientNetWithLinear(output_dim=1)
            model.to(device)
            modelName = f"{condition}_{c}_Model.ckpt"
            optimizer = torch.optim.Adam(model.parameters(), lr=self.base_lr)

            self._train(model, modelName, criterion, optimizer, train_loader, val_loader, condition, f"{save_path}/Model", f"{save_path}/Loss", target_class=self.classes.index(c))

    def train_generation_one_WSI(self, wsi, mode="ideal", labeled = True):
        if self.test_state == "old":
            _wsi = wsi
        elif self.test_type == "HCC":
            _wsi = wsi + 91
        elif self.test_type == "CC":
            _wsi = f"1{wsi:04d}"

        save_path = f'{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/{_wsi}/trial_{self.num_trial}'

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)

        for gen in range(1, self.generation+1):
            condition = f"Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}"
            print(condition)

            # if gen != 1:
            if labeled:
                self.test_TATI(wsi, gen-1, save_path, mode)
            else:
                self.test_all(wsi, gen-1, save_path, mode)
            self.build_pl_dataset(wsi, gen, save_path, mode, labeled)
            
            # Read TI.csv, prepare Dataframe
            train_dataset, valid_dataset, _ = self.prepare_dataset(f"{save_path}/Data", condition, gen, "train", wsi, mode)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        
            # Model setting and transfer learning or not
            if self.backbone == "ViT":
                model = self.ViTWithLinear(output_dim=self.class_num)
            elif self.backbone == "ViT_tiny":
                model = self.ViTWithLinearTiny(output_dim=self.class_num)
            else:
                model = self.EfficientNetWithLinear(output_dim=self.class_num)
            if gen == 1:
                if self.pretrain:
                    model_path = self.file_paths[f'{self.test_model}_model_path']
                    model.load_state_dict(torch.load(model_path, weights_only=True))
            else:
                model_path = f"{save_path}/Model/Gen{gen-1}_ND_zscore_{mode}_patches_by_Gen{gen-2}_1WTC.ckpt"
                model.load_state_dict(torch.load(model_path, weights_only=True))
            
            model.to(device)
            modelName = f"{condition}_1WTC.ckpt"

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.base_lr)

            print(f"Generation {gen}")
            self._train(model, modelName, criterion, optimizer, train_loader, val_loader, condition, f"{save_path}/Model", f"{save_path}/Loss")

    def prepare_pl_dataset_one_WSI(self, wsi, _wsi, gen, save_path, mode, labeled, condition, state, wsi_type):
        if labeled:
            self.test_TATI(wsi, gen-1, save_path, mode, test_state=state, test_type=wsi_type)
        else:
            self.test_all(wsi, gen-1, save_path, mode, test_state=state, test_type=wsi_type)
        self.build_pl_dataset(wsi, gen, save_path, mode, labeled, test_state=state, test_type=wsi_type)
        
        # Read TI.csv, prepare Dataframe
        train_dataset, valid_dataset, _ = self.prepare_dataset(f"{save_path}/Data", f"{_wsi}_{condition}", gen, "train", wsi, mode, state, wsi_type)

        return train_dataset, valid_dataset

    def train_generation(self, mode="ideal", labeled = True, replay = False):
        save_path = f'{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/trial_{self.num_trial}'

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)

        for gen in range(1, self.generation+1):
            condition = f"Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}"

            train_datasets = []
            valid_datasets = []

            for wsi in self.hcc_old_wsis:
                _wsi = wsi
                if self.load_dataset:
                    train_dataset, valid_dataset, _ = self.load_datasets(f"{save_path}/Data", f"{_wsi}_{condition}", "train", wsi, state='old', wsi_type='HCC')
                else:
                    train_dataset, valid_dataset = self.prepare_pl_dataset_one_WSI(wsi ,_wsi, gen, save_path, mode, labeled, condition, 'old', 'HCC')
                train_datasets.append(train_dataset)
                valid_datasets.append(valid_dataset)
            for wsi in self.hcc_wsis:
                _wsi = wsi + 91
                if self.load_dataset:
                    train_dataset, valid_dataset, _ = self.load_datasets(f"{save_path}/Data", f"{_wsi}_{condition}", "train", wsi, state='new', wsi_type='HCC')
                else:
                    train_dataset, valid_dataset = self.prepare_pl_dataset_one_WSI(wsi, _wsi, gen, save_path, mode, labeled, condition, 'new', 'HCC')
                train_datasets.append(train_dataset)
                valid_datasets.append(valid_dataset)
            for wsi in self.cc_wsis:
                _wsi = f"1{wsi:04d}"
                if self.load_dataset:
                    train_dataset, valid_dataset, _ = self.load_datasets(f"{save_path}/Data", f"{_wsi}_{condition}", "train", wsi, state='new', wsi_type='CC')
                else:
                    train_dataset, valid_dataset = self.prepare_pl_dataset_one_WSI(wsi, _wsi, gen, save_path, mode, labeled, condition, 'new', 'CC')
                train_datasets.append(train_dataset)
                valid_datasets.append(valid_dataset)

            if replay:
                for wsi in self.replay_hcc_old_wsis:
                    _wsi = wsi
                    if self.load_dataset:
                        train_dataset, valid_dataset, _ = self.load_datasets(f"{save_path}/Data", f"{_wsi}_{condition}", "train", wsi, state='old', wsi_type='HCC')
                    else:
                        train_dataset, valid_dataset, _ = self.prepare_dataset(f"{save_path}/Data", f"{_wsi}_{condition}", None, "train", wsi, mode, state='old', wsi_type='HCC', replay=True)
                    train_datasets.append(train_dataset)
                    valid_datasets.append(valid_dataset)
                for wsi in self.replay_hcc_wsis:
                    _wsi = wsi + 91
                    if self.load_dataset:
                        train_dataset, valid_dataset, _ = self.load_datasets(f"{save_path}/Data", f"{_wsi}_{condition}", "train", wsi, state='new', wsi_type='HCC')
                    else:
                        train_dataset, valid_dataset, _  = self.prepare_dataset(f"{save_path}/Data", f"{_wsi}_{condition}", None, "train", wsi, mode, state='new', wsi_type='HCC', replay=True)
                    train_datasets.append(train_dataset)
                    valid_datasets.append(valid_dataset)
                for wsi in self.replay_cc_wsis:
                    _wsi = f"1{wsi:04d}"
                    if self.load_dataset:
                        train_dataset, valid_dataset, _ = self.load_datasets(f"{save_path}/Data", f"{_wsi}_{condition}", "train", wsi, state='new', wsi_type='CC')
                    else:
                        train_dataset, valid_dataset, _  = self.prepare_dataset(f"{save_path}/Data", f"{_wsi}_{condition}", None, "train", wsi, mode, state='new', wsi_type='CC', replay=True)
                    train_datasets.append(train_dataset)
                    valid_datasets.append(valid_dataset)
            
            combined_train = ConcatDataset(train_datasets)
            combined_val = ConcatDataset(valid_datasets)

            print(f"training data number: {len(combined_train)}, validation data number: {len(combined_val)}")

            train_loader = DataLoader(combined_train, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(combined_val, batch_size=self.batch_size, shuffle=False)
        
            # Model setting and transfer learning or not
            if self.backbone == "ViT":
                model = self.ViTWithLinear(output_dim=self.class_num)
            elif self.backbone == "ViT_tiny":
                model = self.ViTWithLinearTiny(output_dim=self.class_num)
            else:
                model = self.EfficientNetWithLinear(output_dim=self.class_num)
            if gen == 1:
                if self.pretrain:
                    model_path = self.file_paths[f'{self.test_model}_model_path']
                    model.load_state_dict(torch.load(model_path, weights_only=True))
            else:
                model_path = f"{save_path}/Model/Gen{gen-1}_ND_zscore_{mode}_patches_by_Gen{gen-2}_{self.num_wsi}WTC.ckpt"
                model.load_state_dict(torch.load(model_path, weights_only=True))
            
            model.to(device)
            modelName = f"{condition}_{self.num_wsi}WTC.ckpt"

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.base_lr)

            print(f"Generation {gen}")
            self._train(model, modelName, criterion, optimizer, train_loader, val_loader, condition, f"{save_path}/Model", f"{save_path}/Loss")

    def model_test(self, test_dataset, model, save_path, condition, count_acc=True, classes=None, target_class=None):
        if classes is None:
            classes = self.classes
        
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model.eval()

        # Record Information
        all_fnames = []
        all_preds = []

        # with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            for imgs, fname in tqdm(test_loader):
                # Inference
                logits = model(imgs.to(device, non_blocking=True))
                preds = torch.sigmoid(logits)

                all_preds.append(preds.cpu())
                all_fnames.extend(fname)

        # Concatenate once outside the loop
        all_preds = torch.cat(all_preds).numpy()

        # Build Predictions dict
        Predictions = {"file_name": all_fnames}
        if target_class is None:
            for idx, class_name in enumerate(classes):
                Predictions[f"{class_name}_pred"] = all_preds[:, idx].tolist()
        else:
            Predictions[f"{target_class}_pred"] = all_preds[:, 0].tolist()
        
        pred_df = pd.DataFrame(Predictions)

        if count_acc:
            pred_df.to_csv(f"{save_path}/TI/{condition}_patch_in_region_filter_2_v2_TI.csv", index=False)
        else:
            pred_df.to_csv(f"{save_path}/TI/{condition}_all_patches_filter_v2_TI.csv", index=False)

    def evaluation(self, data_info_df, pred_df, save_path, condition, classes=None, target_class=None):
        if classes is None:
            classes = self.classes
        results_df = {"file_name":[]}
        all_labels, all_preds_labels, all_real_labels = [], [], []
        match_df  = data_info_df[data_info_df['file_name'].isin(pred_df['file_name'])]

        filename_inRegion = match_df['file_name'].to_list()
        label_inRegion = match_df['label'].to_list()

        for idx, filename in enumerate(tqdm(filename_inRegion)):
            results_df["file_name"].append(filename)
            row = pred_df[pred_df['file_name'] == filename]

            if target_class is None:
                if label_inRegion[idx] in classes:
                    label = classes.index(label_inRegion[idx])
                else:
                    label = len(classes) # unknown class
                preds = [row[f'{cl}_pred'].values[0] for cl in classes]
                over_threshold = [i for i, p in enumerate(preds) if p > 0.5]
                pred = over_threshold[0] if len(over_threshold) == 1 else -1
            else:
                label = int(label_inRegion[idx] == target_class)
                score = row[f'{target_class}_pred'].values[0]
                pred = 1 if score > 0.5 else 0
                real_label = classes.index(label_inRegion[idx])
                all_real_labels.append(real_label)

            all_labels.append(label)
            all_preds_labels.append(pred)

        if target_class is None:
            all_classes = ['unknown'] + classes
            text_labels = [self.classes[label] for label in all_labels]
            text_preds = [all_classes[pred+1] for pred in all_preds_labels]
        else:
            all_classes = ['others', target_class]
            text_labels = [classes[label] for label in all_real_labels]
            text_preds = [target_class if pred ==1 else 'others' for pred in all_preds_labels]

        results_df["true_label"] = text_labels
        results_df["pred_label"] = text_preds

        # Save to CSV
        pd.DataFrame(results_df).to_csv(f"{save_path}/Metric/{condition}_labels_predictions.csv", index=False)
        return all_labels, all_preds_labels

    def compute_metrics(self, all_labels, all_preds_labels, pred_df, save_path, condition, classes=None, target_class=None, compute_roc=True):
        if classes is None:
            classes = self.classes
        
        acc = accuracy_score(all_labels, all_preds_labels)
        print("Accuracy: {:.4f}".format(acc))

        Test_Acc = {"Condition": [condition], "Accuracy": [acc]}
        if target_class is None:
            all_classes = ['unknown'] + classes
            y_true_mapped = [l + 1 for l in all_labels]
            y_score_mapped = [p + 1 for p in all_preds_labels]

            cm = confusion_matrix(y_true_mapped, y_score_mapped, labels=range(len(all_classes)))
            cm_df = pd.DataFrame(
                cm,
                index=[f"True_{c}" for c in all_classes],
                columns=[f"Pred_{c}" for c in all_classes]
            )
            cm_df.to_csv(f"{save_path}/Metric/{condition}_confusion_matrix.csv")

            title = f"Confusion Matrix of {condition}"
            self.plot_confusion_matrix(cm, save_path, condition, all_classes, title)

            for i, class_name in enumerate(classes):
                cm_idx = i+1
                TP = cm[cm_idx, cm_idx]  # True Positives
                FN = cm[cm_idx, :].sum() - TP  # False Negatives
                FP = cm[:, cm_idx].sum() - TP  # False Positives
                TN = cm.sum() - (TP + FP + FN)  # True Negatives
                
                Test_Acc[f"{class_name}_TP"] = [TP]
                Test_Acc[f"{class_name}_FN"] = [FN]
                Test_Acc[f"{class_name}_TN"] = [TN]
                Test_Acc[f"{class_name}_FP"] = [FP]

            if compute_roc:
                y_true = np.array(all_labels)
                y_score = pred_df[[f"{cl}_pred" for cl in classes]].values

                per_class_auc = {}
                for i, class_name in enumerate(classes):
                    # One-vs-Rest labels
                    y_true_binary = (y_true == i).astype(int)
                    if len(np.unique(y_true_binary)) > 1:
                        auc_i = roc_auc_score(y_true_binary, y_score[:, i])
                    else:
                        auc_i = float('nan')  # AUC is not defined in this case
                    per_class_auc[class_name] = auc_i

                # Print results
                for class_name, auc_val in per_class_auc.items():
                    print(f"AUC for {class_name}: {auc_val}")

                self.plot_roc(y_true, y_score, save_path, condition)

                for class_name, auc_val in per_class_auc.items():
                    Test_Acc[f"{class_name}_AUC"] = [auc_val]
            
        else:
            all_classes = ['others', target_class]
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds_labels)

            cm = confusion_matrix(y_true, y_pred, labels=range(len(all_classes)))
            cm_df = pd.DataFrame(
                cm,
                index=[f"True_{c}" for c in all_classes],
                columns=[f"Pred_{c}" for c in all_classes]
            )
            cm_df.to_csv(f"{save_path}/Metric/{condition}_confusion_matrix.csv")

            title = f"Confusion Matrix of {condition} ({target_class} vs others)"
            self.plot_confusion_matrix(cm, save_path, condition, all_classes, title)

            TN, FP, FN, TP = cm.ravel()

            Test_Acc[f"{target_class}_TP"] = [TP]
            Test_Acc[f"{target_class}_FN"] = [FN]
            Test_Acc[f"{target_class}_TN"] = [TN]
            Test_Acc[f"{target_class}_FP"] = [FP]

            if compute_roc:
                y_score = pred_df[f"{target_class}_pred"].values
                if len(np.unique(y_true)) > 1:
                    auc = roc_auc_score(y_true, y_score)
                else:
                    auc = float('nan')
                self.plot_roc(y_true, y_score, save_path, condition, target_class)

                Test_Acc[f"{target_class}_AUC"] = [auc]

        # Save to CSV
        pd.DataFrame(Test_Acc).to_csv(f"{save_path}/Metric/{condition}_test_result.csv", index=False)

    def _test(self, test_dataset, data_info_df, model, save_path, condition, count_acc=True, classes=None, target_class=None):
        if classes is None:
            classes = self.classes

        self.model_test(test_dataset, model, save_path, condition, count_acc, classes, target_class)
        if not count_acc:
            return

        pred_df = pd.read_csv(f"{save_path}/TI/{condition}_patch_in_region_filter_2_v2_TI.csv")
        all_labels, all_preds_labels = self.evaluation(data_info_df, pred_df, save_path, condition, classes, target_class)
        
        self.compute_metrics(all_labels, all_preds_labels, pred_df, save_path, condition, classes, target_class)
        
    def test_small_set(self, wsi=None, model_wsi='one', test_state=None, test_type=None):
        if test_state == None:
            test_state = self.test_state
        if test_type == None:
            test_type = self.test_type
        
        if wsi is None:
            condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            save_path = f"{self.save_path}/trial_{self.num_trial}"
        else:
            if test_state == "old":
                _wsi = wsi
            elif test_type == "HCC":
                _wsi = wsi + 91
            elif test_type == "CC":
                _wsi = f"1{wsi:04d}"
            
            condition = f"{_wsi}_{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            if test_type == "HCC":
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"
                data_save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_wsi}/trial_{self.data_trial}"
            else:
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{wsi}/trial_{self.num_trial}"
                data_save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{wsi}/trial_{self.data_trial}"

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)
        
        if self.load_dataset:
            if wsi is None:
                data_condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.num_classes}_class_trial_{self.data_trial}"
                data_info_df = pd.read_csv(f"{data_save_path}/Data/{data_condition}_test.csv")
                _, _, test_dataset = self.load_datasets(f"{data_save_path}/Data", data_condition, "test")
            else:
                data_condition = f"{_wsi}_{self.num_wsi}WTC_LP{self.data_num}_2_class_trial_{self.data_trial}"
                data_info_df = pd.read_csv(f"{data_save_path}/Data/{data_condition}_test.csv")
                _, _, test_dataset = self.load_datasets(f"{data_save_path}/Data", data_condition, "test", wsi=wsi)
        else:
            if wsi is None:
                _, _, test_dataset = self.prepare_dataset(f"{save_path}/Data", condition, 0, "test")
            else:
                _, _, test_dataset = self.prepare_dataset(f"{save_path}/Data", condition, 0, "test", wsi=wsi)
            data_info_df = pd.read_csv(f"{save_path}/Data/{condition}_test.csv")

        print(f"testing data number: {len(test_dataset)}")

        if self.test_model == "self":
            # Prepare Model
            if model_wsi == 'one':
                model_path = f"{save_path}/Model/{_wsi}_{condition}_Model.ckpt"
            else:
                model_path = f"{save_dir}/Model/{condition}_Model.ckpt"

            if self.backbone == "ViT":
                model = self.ViTWithLinear(output_dim=self.class_num)
            elif self.backbone == "ViT_tiny":
                model = self.ViTWithLinearTiny(output_dim=self.class_num)
            else:
                model = self.EfficientNetWithLinear(output_dim=self.class_num)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.to(device)

            self._test(test_dataset, data_info_df, model, save_path, condition)
        
        elif self.test_model == "multi_wsi":
            for model_wsi in self.test_model_wsis:
                # Prepare Model
                if self.test_model_state == "old":
                    _model_wsi = model_wsi
                    model_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{model_wsi}/trial_{self.test_model_trial}"
                elif self.test_model_type == "HCC":
                    _model_wsi = model_wsi + 91
                    model_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_model_wsi}/trial_{self.test_model_trial}"
                elif self.test_model_type == "CC":
                    _model_wsi = f"1{model_wsi:04d}"
                    model_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{model_wsi}/trial_{self.test_model_trial}"

                modelName = f"{_model_wsi}_{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.test_model_trial}_Model.ckpt"
                model_path = f"{model_dir}/Model/{modelName}"

                if self.backbone == "ViT":
                    model = self.ViTWithLinear(output_dim=self.class_num)
                elif self.backbone == "ViT_tiny":
                    model = self.ViTWithLinearTiny(output_dim=self.class_num)
                else:
                    model = self.EfficientNetWithLinear(output_dim=self.class_num)

                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.to(device)

                _condition = f'{condition}_on_model_{model_wsi}'
                self._test(test_dataset, data_info_df, model, save_path, _condition)

        elif self.test_model == "multi_epoch":
            for ep in range(1, 21):
                if model_wsi == 'one':
                    model_path = f"{save_path}/Model/{_wsi}_{condition}_Model_epoch{ep}.ckpt"
                else:
                    model_path = f"{save_dir}/Model/{condition}_Model_epoch{ep}.ckpt"
                if not os.path.exists(model_path):
                    continue

                if self.backbone == "ViT":
                    model = self.ViTWithLinear(output_dim=self.class_num)
                elif self.backbone == "ViT_tiny":
                    model = self.ViTWithLinearTiny(output_dim=self.class_num)
                else:
                    model = self.EfficientNetWithLinear(output_dim=self.class_num)

                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.to(device)

                _condition = f'{condition}_for_epoch_{ep}'
                self._test(test_dataset, data_info_df, model, save_path, _condition)

        elif self.test_model == "multi_class":
            for c in self.classes:
                if model_wsi == 'one':
                    model_path = f"{save_path}/Model/{_wsi}_{condition}_{c}_Model.ckpt"
                else:
                    model_path = f"{save_dir}/Model/{condition}_{c}_Model.ckpt"
                if not os.path.isfile(model_path):
                    continue

                if self.backbone == "ViT":
                    model = self.ViTWithLinear(output_dim=1)
                elif self.backbone == "ViT_tiny":
                    model = self.ViTWithLinearTiny(output_dim=1)
                else:
                    model = self.EfficientNetWithLinear(output_dim=1)
                
                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.to(device)

                _condition = f'{condition}_for_class_{c}'
                self._test(test_dataset, data_info_df, model, save_path, _condition, model_type=c)
    
    def test(self):
        condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
        save_path = f"{self.save_path}/trial_{self.num_trial}"

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)
        
        _, _, test_dataset = self.prepare_dataset(f"{save_path}/Data", condition, 0, "test")
        print(f"testing data number: {len(test_dataset)}")
        
        # Prepare Model
        modelName = f"{condition}_Model.ckpt"
        model_path = f"{save_path}/Model/{modelName}"

        if self.backbone == "ViT":
            model = self.ViTWithLinear(output_dim=self.class_num)
        elif self.backbone == "ViT_tiny":
            model = self.ViTWithLinearTiny(output_dim=self.class_num)
        else:
            model = self.EfficientNetWithLinear(output_dim=self.class_num)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(device)

        data_info_df = pd.read_csv(f"{save_path}/Data/{condition}_test.csv")

        self._test(test_dataset, data_info_df, model, save_path, condition)

    def test_TATI(self, wsi, gen, save_path = None, mode = 'ideal', model_wsi = 'one', test_state=None, test_type=None):
        ### Multi-WTC Evaluation ###
        if test_state == None:
            test_state = self.test_state
        if test_type == None:
            test_type = self.test_type
        
        if test_state == "old":
            _wsi = wsi
        elif test_type == "HCC":
            _wsi = wsi + 91
        elif test_type == "CC":
            _wsi = f"1{wsi:04d}"

        # Dataset, Evaluation, Inference
        if test_state == "old":
            data_info_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset = self.TestDataset(data_info_df, f'{self.hcc_old_data_dir}/{wsi}', self.classes, self.test_tfm, state='old', label_exist=False)
        elif test_type == "HCC":
            data_info_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset = self.TestDataset(data_info_df, f'{self.hcc_data_dir}/{wsi}',self.classes,self.test_tfm, state='new', label_exist=False)
        elif test_type == "CC":
            data_info_df = pd.read_csv(f'{self.cc_csv_dir}/{wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset = self.TestDataset(data_info_df, f'{self.cc_data_dir}/{wsi}', self.classes,self.test_tfm, state='new', label_exist=False)
        
        if self.gen_type:
            if save_path == None:
                if model_wsi == 'one':
                    save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"
                else:
                    save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/trial_{self.num_trial}"
            if gen == 0:
                condition = f'{self.class_num}_class'
                model_path = self.file_paths[f'{self.test_model}_model_path']
            else:
                condition = f"Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}"
                if model_wsi == 'one':
                    model_path = f"{save_path}/Model/{condition}_1WTC.ckpt"
                else:
                    model_path = f"{save_path}/Model/{condition}_{self.num_wsi}WTC.ckpt"
            
            os.makedirs(f"{save_path}/Model", exist_ok=True)
            os.makedirs(f"{save_path}/Metric", exist_ok=True)
            os.makedirs(f"{save_path}/Loss", exist_ok=True)
            os.makedirs(f"{save_path}/TI", exist_ok=True)
            os.makedirs(f"{save_path}/Data", exist_ok=True)
            
            # Prepare Model
            model.load_state_dict(torch.load(model_path, weights_only = True))
            model.to(device)
            
            _condition = f'{_wsi}_{condition}'

            print(f"WSI {wsi} | {_condition}")
            print(self.classes)

            self._test(test_dataset, data_info_df, model, save_path, _condition)

        else:
            condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            if model_wsi == 'one':
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"
            else:
                save_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}"
                save_path = f"{save_dir}/{_wsi}"

            os.makedirs(f"{save_path}/Model", exist_ok=True)
            os.makedirs(f"{save_path}/Metric", exist_ok=True)
            os.makedirs(f"{save_path}/Loss", exist_ok=True)
            os.makedirs(f"{save_path}/TI", exist_ok=True)
            os.makedirs(f"{save_path}/Data", exist_ok=True)

            if self.test_model == "self":
                # Prepare Model
                if model_wsi == 'one':
                    model_path = f"{save_path}/Model/{_wsi}_{condition}_Model.ckpt"
                else:
                    model_path = f"{save_dir}/Model/{condition}_Model.ckpt"

                if self.backbone == "ViT":
                    model = self.ViTWithLinear(output_dim=self.class_num)
                elif self.backbone == "ViT_tiny":
                    model = self.ViTWithLinearTiny(output_dim=self.class_num)
                else:
                    model = self.EfficientNetWithLinear(output_dim=self.class_num)
                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.to(device)

                _condition = f'{_wsi}_{condition}'

                print(f"WSI {wsi} | {_condition}")

                self._test(test_dataset, data_info_df, model, save_path, _condition)
            
            elif self.test_model == "multi_wsi":
                for model_wsi in self.test_model_wsis:
                    # Prepare Model
                    if self.test_model_state == "old":
                        _model_wsi = model_wsi
                        model_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{model_wsi}/trial_{self.test_model_trial}"
                    elif self.test_model_type == "HCC":
                        _model_wsi = model_wsi + 91
                        model_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_model_wsi}/trial_{self.test_model_trial}"
                    elif self.test_model_type == "CC":
                        _model_wsi = f"1{model_wsi:04d}"
                        model_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{model_wsi}/trial_{self.test_model_trial}"

                    modelName = f"{_model_wsi}_{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.test_model_trial}_Model.ckpt"
                    model_path = f"{model_dir}/Model/{modelName}"

                    if self.backbone == "ViT":
                        model = self.ViTWithLinear(output_dim=self.class_num)
                    elif self.backbone == "ViT_tiny":
                        model = self.ViTWithLinearTiny(output_dim=self.class_num)
                    else:
                        model = self.EfficientNetWithLinear(output_dim=self.class_num)

                    model.load_state_dict(torch.load(model_path, weights_only=True))
                    model.to(device)

                    _condition = f'{_wsi}_{condition}_on_model_{model_wsi}'
                    
                    print(f"WSI {wsi} | {_condition}")

                    self._test(test_dataset, data_info_df, model, save_path, _condition)

            elif self.test_model == "multi_epoch":
                for ep in range(1, self.file_paths['max_epoch'] + 1):
                    if model_wsi == 'one':
                        model_path = f"{save_path}/Model/{_wsi}_{condition}_Model_epoch{ep}.ckpt"
                    else:
                        model_path = f"{save_dir}/Model/{condition}_Model_epoch{ep}.ckpt"
                    if not os.path.exists(model_path):
                        continue

                    if self.backbone == "ViT":
                        model = self.ViTWithLinear(output_dim=self.class_num)
                    elif self.backbone == "ViT_tiny":
                        model = self.ViTWithLinearTiny(output_dim=self.class_num)
                    else:
                        model = self.EfficientNetWithLinear(output_dim=self.class_num)

                    model.load_state_dict(torch.load(model_path, weights_only=True))
                    model.to(device)

                    _condition = f'{_wsi}_{condition}_for_epoch_{ep}'

                    print(f"WSI {wsi} | {_condition}")

                    self._test(test_dataset, data_info_df, model, save_path, _condition)

            elif self.test_model == "multi_class":
                for c in self.classes:
                    if model_wsi == 'one':
                        model_path = f"{save_path}/Model/{_wsi}_{condition}_{c}_Model.ckpt"
                    else:
                        model_path = f"{save_dir}/Model/{condition}_{c}_Model.ckpt"
                    if not os.path.isfile(model_path):
                        continue

                    if self.backbone == "ViT":
                        model = self.ViTWithLinear(output_dim=1)
                    elif self.backbone == "ViT_tiny":
                        model = self.ViTWithLinearTiny(output_dim=1)
                    else:
                        model = self.EfficientNetWithLinear(output_dim=1)
                    
                    model.load_state_dict(torch.load(model_path, weights_only=True))
                    model.to(device)

                    _condition = f'{_wsi}_{condition}_for_class_{c}'

                    print(f"WSI {wsi} | {_condition}")

                    self._test(test_dataset, data_info_df, model, save_path, _condition, model_type=c)

    def test_TATI_two_stage(self, wsi, gen, save_path = None, mode = 'ideal', model_wsi = 'one', test_state=None, test_type=None):
        if test_state == None:
            test_state = self.test_state
        if test_type == None:
            test_type = self.test_type
        
        if test_state == "old":
            _wsi = wsi
        elif test_type == "HCC":
            _wsi = wsi + 91
        elif test_type == "CC":
            _wsi = f"1{wsi:04d}"
        
        condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
        if model_wsi == 'one':
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"
        else:
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}"

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)

        # Dataset, Evaluation, Inference
        if test_state == "old":
            data_info_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset_1 = self.TestDataset(data_info_df, f'{self.hcc_old_data_dir}/{wsi}', self.classes, self.test_tfm, state='old', label_exist=False)
        elif test_type == "HCC":
            data_info_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset_1 = self.TestDataset(data_info_df, f'{self.hcc_data_dir}/{wsi}',self.classes,self.test_tfm, state='new', label_exist=False)
        elif test_type == "CC":
            data_info_df = pd.read_csv(f'{self.cc_csv_dir}/{wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset_1 = self.TestDataset(data_info_df, f'{self.cc_data_dir}/{wsi}', self.classes,self.test_tfm, state='new', label_exist=False)

        # First Stage
        model_path = self.file_paths['first_stage_model_path']
        if self.backbone == "ViT":
            model_1 = self.ViTWithLinear(output_dim=1)
        elif self.backbone == "ViT_tiny":
            model_1 = self.ViTWithLinearTiny(output_dim=1)
        else:
            model_1 = self.EfficientNetWithLinear(output_dim=1)
        
        model_1.load_state_dict(torch.load(model_path, weights_only=True))
        model_1.to(device)

        _condition = f'{_wsi}_{condition}_first_stage'
        print(f"WSI {wsi} | {_condition}")
        self._test(test_dataset_1, data_info_df, model_1, save_path, _condition, target_class=self.file_paths['first_stage_target_class'])
        
        # Second Stage
        pred_df = pd.read_csv(f"{save_path}/Metric/{_condition}_labels_predictions.csv")

        others_df = pred_df[pred_df['pred_label'] == 'others'].copy()
        others_df = others_df[['file_name', 'true_label']]
        others_df = others_df.rename(columns={"true_label": "label"})
        others_df = others_df.reset_index(drop=True)
        
        classes = self.classes.copy()
        classes.remove(self.file_paths['first_stage_target_class'])

        if test_state == "old":
            test_dataset_2 = self.TestDataset(others_df, f'{self.hcc_old_data_dir}/{wsi}', self.classes, self.test_tfm, state='old', label_exist=False)
        elif test_type == "HCC":
            test_dataset_2 = self.TestDataset(others_df, f'{self.hcc_data_dir}/{wsi}',self.classes,self.test_tfm, state='new', label_exist=False)
        elif test_type == "CC":
            test_dataset_2 = self.TestDataset(others_df, f'{self.cc_data_dir}/{wsi}', self.classes,self.test_tfm, state='new', label_exist=False)

        model_path = self.file_paths['second_stage_model_path']
        if self.backbone == "ViT":
            model_2 = self.ViTWithLinear(output_dim=2)
        elif self.backbone == "ViT_tiny":
            model_2 = self.ViTWithLinearTiny(output_dim=2)
        else:
            model_2 = self.EfficientNetWithLinear(output_dim=2)
        model_2.load_state_dict(torch.load(model_path, weights_only=True))
        model_2.to(device)

        _condition = f'{_wsi}_{condition}_second_stage'
        print(f"WSI {wsi} | {_condition}")
        self._test(test_dataset_2, others_df, model_2, save_path, _condition, classes=classes)

        _condition = f'{_wsi}_{condition}'
        
        df_pred_label = merge_labels(f"{save_path}", _wsi, _condition)
        df_pred_score = merge_TI(f"{save_path}", _wsi, _condition, self.classes, self.file_paths['first_stage_target_class'])

        all_labels = [self.classes.index(label) for label in df_pred_label['true_label'].to_list()]
        all_preds_labels = [
            self.classes.index(pred) if pred in self.classes else -1
            for pred in df_pred_label['pred_label'].to_list()
        ]
        
        self.compute_metrics(all_labels, all_preds_labels, df_pred_score, save_path, _condition, compute_roc=False)
    
    def test_flip(self, wsi, gen, save_path = None, mode = 'selected', model_wsi = 'one'):
        ### Multi-WTC Evaluation ###
        if self.test_state == "old":
            _wsi = wsi
        elif self.test_type == "HCC":
            _wsi = wsi + 91
        elif self.test_type == "CC":
            _wsi = f"1{wsi:04d}"

        if save_path == None:
            if model_wsi == 'one':
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"
            else:
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/trial_{self.num_trial}"

        condition = f"Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}"
        
        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)

        if self.test_type == "HCC":
            data_info_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
        elif self.test_type == "CC":
            data_info_df = pd.read_csv(f'{self.cc_csv_dir}/{wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
        
        _condition = f'{_wsi}_{condition}'
        pred_df = pd.read_csv(f'{save_path}/Data/{_condition}.csv')
        
        results_df = {"file_name":[]}
        all_labels, all_preds = [], []
        match_df  = data_info_df[data_info_df['file_name'].isin(pred_df['file_name'])]

        filename_inRegion = match_df['file_name'].to_list()
        label_inRegion = match_df['label'].to_list()

        for idx, filename in enumerate(tqdm(filename_inRegion)):
            label = self.classes.index(label_inRegion[idx])
            results_df["file_name"].append(filename)
            row = pred_df[pred_df['file_name'] == filename]
            pred_label = row['label'].values[0]
            pred = self.classes.index(pred_label)

            all_labels.append(label)
            all_preds.append(pred)

        text_labels = [self.classes[label] for label in all_labels]
        text_preds = [self.classes[pred] for pred in all_preds]

        results_df["true_label"] = text_labels
        results_df["pred_label"] = text_preds

        # Save to CSV
        pd.DataFrame(results_df).to_csv(f"{save_path}/Metric/{_condition}_flip_labels_predictions.csv", index=False)

        acc = accuracy_score(all_labels, all_preds)
        print("Accuracy: {:.4f}".format(acc))

        cm = confusion_matrix(all_labels, all_preds, labels=range(self.class_num))
        title = f"Confusion Matrix of {_condition}_flip"
        self.plot_confusion_matrix(cm, save_path, f"{_condition}_flip", title)

        Test_Acc = {"Condition": [_condition], "Accuracy": [acc]}
        for i, class_name in enumerate(self.classes):
            TP = cm[i, i]  # True Positives
            FN = cm[i, :].sum() - TP  # False Negatives
            FP = cm[:, i].sum() - TP  # False Positives
            TN = cm.sum() - (TP + FP + FN)  # True Negatives
            
            Test_Acc[f"{class_name}_TP"] = [TP]
            Test_Acc[f"{class_name}_FN"] = [FN]
            Test_Acc[f"{class_name}_TN"] = [TN]
            Test_Acc[f"{class_name}_FP"] = [FP]

        # Save to CSV
        pd.DataFrame(Test_Acc).to_csv(f"{save_path}/Metric/{_condition}_flip_test_result.csv", index=False)

    def test_all(self, wsi, gen, save_path = None, mode = 'selected', model_wsi = 'one', test_state=None, test_type=None):
        ### Multi-WTC Evaluation ###
        if test_state == None:
            test_state = self.test_state
        if test_type == None:
            test_type = self.test_type
        
        if test_state == "old":
            _wsi = wsi
        elif test_type == "HCC":
            _wsi = wsi + 91
        elif test_type == "CC":
            _wsi = f"1{wsi:04d}"
        
        if self.gen_type:
            if save_path == None:
                if model_wsi == 'one':
                    save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"
                else:
                    save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/trial_{self.num_trial}"
            if gen == 0:
                condition = f'{self.class_num}_class'
                model_path = self.file_paths[f'{self.test_model}_model_path']
            else:
                condition = f"Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}"
                if model_wsi == 'one':
                    model_path = f"{save_path}/Model/{condition}_1WTC.ckpt"
                else:
                    model_path = f"{save_path}/Model/{condition}_{self.num_wsi}WTC.ckpt"

        else:
            condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            save_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}"
            save_path = f"{save_dir}/{_wsi}" 

            modelName = f"{condition}_Model.ckpt"
            model_path = f"{save_dir}/Model/{modelName}"
        if self.backbone == "ViT":
            model = self.ViTWithLinear(output_dim=self.class_num)
        elif self.backbone == "ViT_tiny":
            model = self.ViTWithLinearTiny(output_dim=self.class_num)
        else:
            model = self.EfficientNetWithLinear(output_dim=self.class_num)

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)

        # Prepare Model
        model.load_state_dict(torch.load(model_path, weights_only = True))
        model.to(device)
        Sigmoid = nn.Sigmoid()

        # Dataset, Evaluation, Inference
        if test_type == "HCC":
            data_info_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_all_patches_filter_v2.csv')
            test_dataset = self.TestDataset(data_info_df, f'{self.hcc_data_dir}/{wsi}',self.classes,self.test_tfm, state='new', label_exist=False)
        elif test_type == "CC":
            data_info_df = pd.read_csv(f'{self.cc_csv_dir}/{wsi}/{_wsi}_all_patches_filter_v2.csv')
            test_dataset = self.TestDataset(data_info_df, f'{self.cc_data_dir}/{wsi}', self.classes,self.test_tfm, state='new', label_exist=False)
        
        _condition = f'{_wsi}_{condition}'

        print(f"WSI {wsi} | {_condition}")
        print(self.classes)

        self._test(test_dataset, data_info_df, model, save_path, _condition, count_acc=False)

    def plot_confusion_matrix(self, cm, save_path, condition, all_classes, title='Confusion Matrix'):
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap='Blues')
        fig.colorbar(cax)

        num_classes = len(all_classes)
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(all_classes, fontsize=14)
        ax.set_yticklabels(all_classes, fontsize=14)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color=color, fontsize=18)
        
        plt.title(title, fontsize=20, pad=20)
        ax.set_xlabel('Predicted Label', fontsize=16, labelpad=10)
        ax.set_ylabel('True Label', fontsize=16, labelpad=10)

        plt.subplots_adjust(top=0.85)
        plt.savefig(f"{save_path}/Metric/{condition}_confusion_matrix.png")
        plt.close()

    def plot_roc(self, y_true, y_score, save_path, condition, target_class=None):
        plt.figure(figsize=(8, 6))

        if target_class is not None:
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"{target_class} vs others (AUC = {roc_auc:.4f})")
        else:
            for i, class_name in enumerate(self.classes):
                # One-vs-Rest labels
                y_true_binary = (y_true == i).astype(int)
                if y_score.shape[1] <= i:
                    continue
                if len(np.unique(y_true_binary)) > 1:
                    # Compute ROC curve
                    fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
                    # Compute AUC
                    roc_auc = auc(fpr, tpr)
                    # Plot
                    plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.4f})")

        # Diagonal line for random guess
        plt.plot([0, 1], [0, 1], "k--", lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {condition}")
        plt.legend(loc="lower right")
        plt.tight_layout()

        # Save figure
        plt.savefig(f"{save_path}/Metric/{condition}_ROC_per_class.png")
        plt.close()

    def plot_TI_Result(self, wsi, gen, save_path = None, mode = 'ideal', model_wsi = 'one'):
        if save_path == None:
            _wsi = wsi+91 if (self.test_state == "new" and self.test_type == "HCC") else wsi
            __wsi = wsi if self.test_state == "old" else (wsi+91 if self.test_type == "HCC" else f"1{wsi:04d}")
        if self.gen_type:
            if model_wsi == 'one':
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/{__wsi}/trial_{self.num_trial}"
            else:
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/trial_{self.num_trial}"
            if gen == 0:
                condition = f'{self.class_num}_class'
            else:
                condition = f"Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}"
        else:
            condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            if self.num_wsi == 1:
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{__wsi}/trial_{self.num_trial}"
            else:
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}/{__wsi}"
        
        if self.test_model == "multi_wsi":
            _condition = f'{__wsi}_{condition}_for_wsi_{self.test_model_wsis[0]}'
        elif self.test_model == "multi_epoch":
            _condition = f'{__wsi}_{condition}_for_epoch_{self.file_paths["max_epoch"]}'
        else:
            _condition = f'{__wsi}_{condition}'
        df = pd.read_csv(f"{save_path}/Metric/{_condition}_labels_predictions.csv")
        
        all_patches = df['file_name'].to_list()

        ### Get (x, y, pseudo-label) of every patch ###
        all_pts = []
        for idx, img_name in enumerate(all_patches):
            # if self.test_state == "old":
            #     match = re.search(r'-(\d+)-(\d+)-\d{5}x\d{5}', img_name)
            #     if match:
            #         x = match.group(1)
            #         y = match.group(2)
            #     else:
            #         print("Style Error")
            # else:
            x, y = img_name[:-4].split('_')

            x = (int(x)) // self.patch_size
            y = (int(y)) // self.patch_size
            
            gt_label = self.classes.index(df['true_label'][idx])
            if self.test_state == "old":
                shift_map = self.file_paths["old_HCC_shift_map"]
                dx, dy = shift_map[gt_label]
                x += dx
                y += dy
            
            if df['pred_label'][idx] != 'unknown':
                pred_label  = self.classes.index(df['pred_label'][idx])
                
                if pred_label == gt_label:
                    label = pred_label + 1  # correct predict: encode as 1,2,3
                else:
                    label = 10 * (gt_label + 1) + (pred_label + 1)  # wrong predict：encode asss 11,12,13, 21,22,23, 31,32,33
            else:
                label = -1
            all_pts.append([x, y, label])

        all_pts = np.array(all_pts)

        x_max, y_max = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])

        image = np.zeros((y_max + 1, x_max + 1))
        for x, y, label in all_pts:
            image[y, x] = label

        if self.class_num == 3:
            color_map = {
                1: 'green',          # True Normal
                2: 'red',            # True HCC
                3: 'blue',           # True CC
                12: 'lightgreen',    # Normal -> HCC
                13: 'darkslategrey', # Normal -> CC
                21: 'lightsalmon',   # HCC -> Normal
                23: 'darkred',       # HCC -> CC
                31: 'lightblue',     # CC -> Normal
                32: 'indigo',        # CC -> HCC
                -1: 'grey'           # No Prediction
            }
            legend_specs = [
                (1,  'True Normal'),
                (2,  'True HCC'),
                (3,  'True CC'),

                (12, 'Normal -> HCC'),
                (13, 'Normal -> CC'),

                (21, 'HCC -> Normal'),
                (23, 'HCC -> CC'),

                (31, 'CC -> Normal'),
                (32, 'CC -> HCC'),

                (-1, 'No Prediction'),
            ]

        elif self.class_num == 2:
            if self.test_type == "HCC":
                color_map = {
                    1: 'green',         # True Normal
                    2: 'red',           # True HCC
                    12: 'lightgreen',   # Normal -> HCC
                    21: 'lightsalmon',  # HCC -> Normal
                }
                legend_specs = [
                    (1,  'True Normal'),
                    (2,  'True HCC'),
                    (12, 'Normal -> HCC'),
                    (21, 'HCC -> Normal'),
                ]

            elif self.test_type == "CC":
                color_map = {
                    1: 'green',            # True Normal
                    2: 'blue',             # True CC
                    12: 'darkslategrey',   # Normal -> CC
                    21: 'lightblue',       # CC -> Normal
                }
                legend_specs = [
                    (1,  'True Normal'),
                    (2,  'True CC'),
                    (12, 'Normal -> CC'),
                    (21, 'CC -> Normal'),
                ]

        legend_elements = [plt.Line2D([0], [0], color=color_map[k], lw=4, label=label) for k, label in legend_specs]
        plt.figure(figsize=(x_max // 20, y_max // 20))
        for label_value, color in color_map.items():
            plt.imshow(image == label_value, cmap=ListedColormap([[0,0,0,0], color]), interpolation='nearest', alpha=1)

        plt.title(f"Prediction vs Ground Truth of WSI {__wsi}", fontsize=20, pad=20)
        plt.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        plt.axis("off")

        plt.savefig(f"{save_path}/Metric/{_condition}_pred_vs_gt.png")
        plt.close()
        print(f"WSI {wsi} already plot the pred_vs_gt image")

    def plot_all_result(self, wsi, gen, save_path = None, mode = 'selected', plot_type = 'pred', model_wsi = 'one', plot_heatmap = False, plot_boundary = False):
        if save_path == None:
            _wsi = wsi+91 if (self.test_state == "new" and self.test_type == "HCC") else wsi
            __wsi = wsi if self.test_state == "old" else (wsi+91 if self.test_type == "HCC" else f"1{wsi:04d}")
        if self.gen_type:
            if model_wsi == 'one':
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/{__wsi}/trial_{self.num_trial}"
            else:
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/trial_{self.num_trial}"
            if gen == 0:
                condition = f'{self.class_num}_class'
            else:
                condition = f"Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}"
        else:
            condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            if self.test_model == "self":
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}/{__wsi}"
            else:
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}/{__wsi}"
        
        _condition = f'{__wsi}_{condition}'

        if self.test_type == "HCC":
            gt_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
        elif self.test_type == "CC":
            gt_df = pd.read_csv(f'{self.cc_csv_dir}/{wsi}/{__wsi}_patch_in_region_filter_2_v2.csv')
        
        if gen == 0 or plot_type == 'pred':
            df = pd.read_csv(f"{save_path}/TI/{_condition}_all_patches_filter_v2_TI.csv")
        elif plot_type == 'flip':
            df = pd.read_csv(f"{save_path}/Data/{_condition}.csv")

        all_patches = df['file_name'].to_list()

        ### Get (x, y, pseudo-label) of every patch ###
        all_pts = []
        all_preds = []
        if 'label' in df:
            for idx, img_name in enumerate(all_patches):
                x, y = img_name[:-4].split('_')
                x = (int(x)) // self.patch_size
                y = (int(y)) // self.patch_size

                if df['label'][idx] != 'unknown':
                    pred_label  = self.classes.index(df['label'][idx])
                else:
                    pred_label = -1
                all_pts.append([x, y, pred_label+1])

        else:
            pred_cols = [f"{cl}_pred" for cl in self.classes]

            for idx, img_name in enumerate(all_patches):
                x, y = img_name[:-4].split('_')
                row = df.iloc[idx][pred_cols].values
                # pred_label = int(np.argmax(row))   # 0=N, 1=H, 2=C
                over_threshold = [i for i, p in enumerate(row) if p > 0.5]

                if len(over_threshold) == 1:
                    pred_label = over_threshold[0]
                else:
                    pred_label = -1

                x = (int(x)) // self.patch_size
                y = (int(y)) // self.patch_size

                all_pts.append([x, y, pred_label+1])
                all_preds.append([x, y, row])

        all_pts = np.array(all_pts)
        x_max, y_max = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])

        image = np.zeros((y_max + 1, x_max + 1))
        for x, y, label in all_pts:
            image[y, x] = label

        gt_pts = []
        gt_patches = gt_df['file_name'].to_list()
        for idx, img_name in enumerate(gt_patches):
            x, y = img_name[:-4].split('_')
            gt_label  = self.classes.index(gt_df['label'][idx])
            x = (int(x)) // self.patch_size
            y = (int(y)) // self.patch_size
            gt_pts.append([x, y, gt_label + 1])

        gt_pts = np.array(gt_pts)
        gt_mask = np.zeros((y_max + 1, x_max + 1), np.uint8)
        for x, y, label in gt_pts:
            gt_mask[y, x] = label

        # --- Color map ---
        if self.class_num == 3:
            color_map = {
                0: 'grey', # No Prediction
                1: 'green',     # Pred Normal
                2: 'red',       # Pred HCC
                3: 'blue',      # Pred CC
            }
            legend_elements = [
                plt.Line2D([0], [0], color='green', lw=4, label='Pred Normal'),
                plt.Line2D([0], [0], color='red', lw=4, label='Pred HCC'),
                plt.Line2D([0], [0], color='blue', lw=4, label='Pred CC'),
                plt.Line2D([0], [0], color='grey', lw=4, label='No Prediction'),
            ]
        elif self.class_num == 2:
            if self.test_type == "HCC":
                color_map = {
                    1: 'green',     # Pred Normal
                    2: 'red',       # Pred HCC
                }
                legend_elements = [
                    plt.Line2D([0], [0], color='green', lw=4, label='Pred Normal'),
                    plt.Line2D([0], [0], color='red', lw=4, label='Pred HCC'),
                ]
            elif self.test_type == "CC":
                color_map = {
                    1: 'green',     # Pred Normal
                    2: 'blue',      # Pred CC
                }
                legend_elements = [
                    plt.Line2D([0], [0], color='green', lw=4, label='Pred Normal'),
                    plt.Line2D([0], [0], color='blue', lw=4, label='Pred CC'),
                ]
        plt.figure(figsize=(x_max/10, y_max/10))
        for label_value, color in color_map.items():
            plt.imshow(image == label_value, cmap=ListedColormap([[0,0,0,0], color]), interpolation='nearest', alpha=1)
        
        if plot_boundary:
            gt_boundaries = find_boundaries(gt_mask > 0, mode='inner')
            plt.contour(gt_boundaries, colors='black', linewidths=5.0)

        plt.title(f"Prediction of WSI {__wsi}", fontsize=20, pad=20)
        plt.legend(handles=legend_elements, loc='upper right', fontsize=18)
        plt.tight_layout()
        plt.axis("off")

        plt.savefig(f"{save_path}/Metric/{_condition}_{plot_type}.png")
        plt.close()
        print(f"WSI {wsi} already plot the {plot_type} image")

        if plot_heatmap and len(all_preds) > 0:
            heatmap = np.zeros((y_max + 1, x_max + 1, 3), dtype=np.float32)
            per_class_maps = {cl: np.zeros((y_max + 1, x_max + 1), dtype=np.float32) for cl in self.classes}

            for x, y, row in all_preds:
                for i, cl in enumerate(self.classes):
                    per_class_maps[cl][y, x] = row[i]

                if self.class_num == 2:
                    if self.test_type == "HCC":
                        heatmap[y, x, 0] = row[1]  # R = HCC
                        heatmap[y, x, 1] = row[0]  # G = Normal
                        heatmap[y, x, 2] = 0
                    elif self.test_type == "CC":
                        heatmap[y, x, 0] = 0
                        heatmap[y, x, 1] = row[0]  # G = Normal
                        heatmap[y, x, 2] = row[1]  # B = CC
                elif self.class_num == 3:
                    # --- Three-class (Normal, HCC, CC) ---
                    heatmap[y_patch, x_patch, 0] = row[1]  # R = HCC
                    heatmap[y_patch, x_patch, 1] = row[0]  # G = Normal
                    heatmap[y_patch, x_patch, 2] = row[2]  # B = CC
                
            plt.figure(figsize=(x_max/10, y_max/10))
            plt.imshow(heatmap, interpolation='nearest')
            plt.title(f'Combined probability heatmap for WSI {__wsi}', fontsize=20, pad=20)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{save_path}/Metric/{_condition}_combined_heatmap.png")
            plt.close()
            print(f"WSI {wsi} combined heatmap saved.")

            # ---------- Save per-class heatmaps ----------
            color_map = {
                'N': 'Greens',
                'H': 'Reds',
                'C': 'Blues'
            }
            for cl in self.classes:
                plt.figure(figsize=(x_max/10, y_max/10))
                cmap_name = color_map.get(cl, 'gray')
                plt.imshow(per_class_maps[cl], cmap=cmap_name, vmin=0, vmax=1, interpolation='nearest')
                plt.colorbar(label=f'{cl} probability', shrink=0.7)
                plt.title(f'{cl} heatmap for WSI {__wsi}', fontsize=20, pad=20)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f"{save_path}/Metric/{_condition}_{cl}_heatmap.png")
                plt.close()
                print(f"WSI {wsi} per-class heatmap for {cl} saved.")

    def plot_TI_Result_gt_boundary(self, wsi, gen, save_path):
        if save_path == None:
            _wsi = wsi+91 if (self.state == "new" and self.wsi_type == "HCC") else wsi
            __wsi = wsi if self.state == "old" else (wsi+91 if self.wsi_type == "HCC" else f"1{wsi:04d}")
        if self.gen_type:
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}/{_wsi}"
            if gen == 0:
                condition = f'{self.class_num}_class'
            else:
                condition = f"Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}"
        else:
            condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            save_dir = os.path.join(self.file_paths[f'{self.test_model}_model_path'], f"LP_{self.data_num}/trial_{self.num_trial}") 
            save_path = f"{save_dir}/{_wsi}" 

        df = pd.read_csv(f"{save_path}/Metric/{__wsi}_{condition}_labels_predictions.csv")
        # df = pd.read_csv(f"{save_path}/TI/{_wsi}_{condition}_patch_in_region_filter_2_v2_TI.csv")
        gt = pd.read_csv(f"{self.cc_csv_dir}/{wsi}/{__wsi}_patch_in_region_filter_2_v2.csv") if self.wsi_type == "CC" \
            else pd.read_csv(f"{self.hcc_csv_dir}/{wsi}/{__wsi}_patch_in_region_filter_2_v2.csv")
        
        pred_patches = df['file_name'].to_list() #patches_in_hcc_hulls
        gt_patches = gt['file_name'].to_list()

        ### Get (x, y, pseudo-label) of every patch ###
        pred_pts, gt_pts = [], []
        for idx, img_name in enumerate(pred_patches):
            if self.state == "old":
                match = re.search(r'-(\d+)-(\d+)-\d{5}x\d{5}', img_name)
                if match:
                    x = match.group(1)
                    y = match.group(2)
                else:
                    print("Style Error")
            else:
                x, y = img_name[:-4].split('_')

            pred_label  = self.classes.index(df['pred_label'][idx])  # N=0, H=1
            # label = 1 if df['H_pred'][idx] > df["N_pred"][idx] else 0

            x = (int(x)) // self.patch_size
            y = (int(y)) // self.patch_size
            
            pred_pts.append([x, y, pred_label])

        for idx, img_name in enumerate(gt_patches):
            if self.state == "old":
                match = re.search(r'-(\d+)-(\d+)-\d{5}x\d{5}', img_name)
                if match:
                    x = match.group(1)
                    y = match.group(2)
                else:
                    print("Style Error")
            else:
                x, y = img_name[:-4].split('_')

            gt_label  = self.classes.index(gt['label'][idx])  # N=0, H=1
            # label = 1 if df['H_pred'][idx] > df["N_pred"][idx] else 0

            x = (int(x)) // self.patch_size
            y = (int(y)) // self.patch_size
            
            gt_pts.append([x, y, gt_label])

        pred_pts = np.array(pred_pts)
        gt_pts = np.array(gt_pts)
        
        x_max = max(np.max(pred_pts[:, 0]), np.max(gt_pts[:, 0]))
        y_max = max(np.max(pred_pts[:, 1]), np.max(gt_pts[:, 1]))

        pred_labels = np.zeros((y_max + 1, x_max + 1), np.uint8)
        gt_labels = np.zeros((y_max + 1, x_max + 1), np.uint8)

        for x, y, label in pred_pts:
            pred_labels[y, x] = label + 1

        for x, y, label in gt_pts:
            gt_labels[y, x] = label + 1

        plt.imshow(pred_labels == 2, cmap=ListedColormap(['white', 'red']), interpolation='nearest', alpha=0.5)  # Pred - HCC
        plt.imshow(pred_labels == 1, cmap=ListedColormap(['white', 'green']), interpolation='nearest', alpha=0.5)  # Pred - Normal
        # plt.imshow(fib_patches_labels, cmap=ListedColormap(['white', 'blue']), interpolation='nearest', alpha=0.5)

        gt_HCC_boundaries = find_boundaries(gt_labels == 2, mode='inner')  # HCC (red)
        gt_Norm_boundaries = find_boundaries(gt_labels == 1, mode='inner')  # Normal (green)
        plt.contour(gt_HCC_boundaries, colors='red', linewidths=1.2)
        plt.contour(gt_Norm_boundaries, colors='green', linewidths=1.2)

        _wsi = wsi+91 if (self.state == "new" and self.wsi_type == "HCC") else wsi
        # plt.show()
        plt.savefig(f"{save_path}/Metric/{wsi}_pred_vs_gt.png")
        plt.tight_layout()
        plt.axis("off")
        plt.close()
