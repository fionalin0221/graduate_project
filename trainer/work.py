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
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import find_contours

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

torch.manual_seed(0)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = "cuda" if torch.cuda.is_available() else "cpu"

pts_ratio = 448

class Worker():
    def __init__(self, config):
        current_computer = config['current_computer']
        # only for 1 wsi
        self.type = config['type']
        self.state = config['state']
        self.gen_type = config['gen_type']
        self.generation = config["generation"]

        # for multi wsi
        self.file_paths = config['computers'][current_computer]['file_paths']
        class_list = config["class_list"]
        self.classes = [class_list[i] for i in self.file_paths['classes']]
        self.class_num = len(self.classes)
        self.batch_size = self.file_paths['batch_size']
        self.base_lr = float(self.file_paths['base_lr'])

        self.data_num = self.file_paths['data_num']
        self.num_trial = self.file_paths['num_trial']  
        self.data_trial = self.file_paths['data_trial'] 
        self.num_wsi = self.file_paths['num_wsi']
        self.test_model = self.file_paths['test_model']

        self.test_state = self.file_paths['test_state']
        self.test_type = self.file_paths['test_type']

        if self.gen_type:
            self.save_dir = self.file_paths[f'{self.type}_generation_save_path']
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = self.file_paths[f'{self.type}_WTC_result_save_path']
            self.save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}"
            # self.save_path = f"{self.save_dir}/100WTC_Result"
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.save_path, exist_ok=True)

        self.hcc_old_wsis = self.file_paths['HCC_old_wsis']
        self.hcc_wsis = self.file_paths['HCC_wsis']
        self.cc_wsis = self.file_paths['CC_wsis']

        self.hcc_data_dir = self.file_paths['HCC_new_patches_save_path']
        self.hcc_old_data_dir = self.file_paths['HCC_old_patches_save_path']
        self.cc_data_dir = self.file_paths['CC_patches_save_path']
        self.hcc_csv_dir = self.file_paths['HCC_csv_dir']
        self.cc_csv_dir = self.file_paths['CC_csv_dir']

        self.pseudo_label_type = "contour"  # zscore/contour

        self.train_tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomRotation(degrees=(0,360), expand=False),
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
            return len(self.data_dict["file_name"])
        
    def check_overlap(self, *lists):
        sets = [set(lst) for lst in lists]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                if sets[i] & sets[j]:
                    return True
        return False
    
    def split_datas(self, selected_data, data_num, tp_data=None, fp_data=None):
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
            for num in range(len(self.classes)):
                if len(class_file_names[num]) > 0:
                    # datas.append(class_file_names[num])
                    datas.append([(name, False) for name in class_file_names[num]])
                else:
                    datas.append([])
        else:
            data_num = int(data_num)
            for cl in range(len(self.classes)):
                class_samples = class_file_names[cl]

                if self.gen_type:
                    fp_class_samples = fp_class_file_names[cl]
                    tp_class_samples = tp_class_file_names[cl]

                    if len(fp_class_samples) > 0:
                        # class_data_num = 0.5 * int(data_num) if cl == 0 else int(data_num)
                        class_data_num = int(data_num)
                        if len(fp_class_samples) >= class_data_num:
                            print(f"{cl} is case 0")
                            samples = random.sample(fp_class_samples, int(class_data_num))
                            datas.append([(name, False) for name in samples])
                        else:
                            if len(fp_class_samples) + len(tp_class_samples) >= class_data_num:
                                print(f"{cl} is case 1")
                                # datas.append([(name, False) for name in fp_class_samples])
                                class_data_num = int(class_data_num - len(fp_class_samples))
                                samples = random.sample(tp_class_samples, class_data_num)
                                full_samples = [(name, False) for name in fp_class_samples] + [(name, False) for name in samples]
                                datas.append(full_samples)
                            else:
                                print(f"{cl} is case 2")
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
                                # print(f"{cl} is case 3")
                                samples = random.sample(tp_class_samples, int(class_data_num))
                                datas.append([(name, False) for name in samples])
                            else:
                                # print(f"{cl} is case 4")
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
                                # print(f"{cl} is case 5")
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
                        # version 2
                        # class_data_num = int(data_num * 0.5) if cl == 0 else int(data_num)
                        # datas.append(random.sample(class_file_names[cl], class_data_num))
                        
                        # version 1
                        # if self.type == "Mix" and self.classes[num] == "N":
                        #     datas.append(random.sample(class_file_names[num], int(data_num/2)))
                        # else:
                        #     datas.append(random.sample(class_file_names[num], int(data_num)))
                    else:
                        datas.append([])
                
        # if self.check_overlap(*datas):
        #     print(f'Data overlap.')
        #     return

        data_file_names, data_labels = [], []

        for i in range(self.class_num):
            # data_file_names += class_file_names[i]
            # data_labels += [self.classes[i]] * len(class_file_names[i])
            data_file_names += [x[0] for x in datas[i]]
            data_labels += [self.classes[i]] * len(datas[i])

        # Prepare train/val dataset
        train, val, test = [], [], []
        for num in range(len(self.classes)):
            if len(datas[num]) > 0:
                train_, val_ = train_test_split(datas[num], test_size=0.2, random_state=0)  # 80:20 split
                # val_, test_ = train_test_split(temp, test_size=0.5, random_state=0)  # 50:50 split on 20% = 10% each
                
                train.append(train_)
                val.append(val_)
                # test.append(test_)
            else:
                train.append([])
                val.append([])
                # test.append([])

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

            # fn, lb, aug = extract_info(test[i], i)
            # test_file_names += fn
            # test_labels += lb
            # test_augments += aug

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
        # Test = {
        #     "file_name": test_file_names,
        #     "label": test_labels,
        #     "augment": test_augments,
        # }
        Test = {}
        return Train, Val, Test

    def prepare_dataset(self, save_path, condition, gen, data_stage, wsi=None, mode=None):
        train_data = []
        valid_data = []
        test_data = []

        train_datasets = []
        valid_datasets = []
        test_datasets = []

        print(f"Patches use for a WSI: {self.data_num}")

        if wsi == None:
            for h_wsi in self.hcc_old_wsis:
                selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{h_wsi}/{h_wsi}_patch_in_region_filter_2_v2.csv')
                Train, Valid, Test = self.split_datas(selected_data, self.data_num)
                h_train_dataset = self.TrainDataset(Train, f'{self.hcc_old_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "old")
                h_valid_dataset = self.TrainDataset(Valid, f'{self.hcc_old_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "old")
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
                h_valid_dataset = self.TrainDataset(Valid, f'{self.hcc_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "new")
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
                c_valid_dataset = self.TrainDataset(Valid, f'{self.cc_data_dir}/{c_wsi}', self.classes, self.train_tfm, state = "new")
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
            if self.state == "old":
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
                valid_dataset = self.TrainDataset(Valid, f'{self.hcc_old_data_dir}/{wsi}', self.classes, self.train_tfm, state = "old")
                test_dataset  = self.TestDataset(Test, f'{self.hcc_old_data_dir}/{wsi}',self.classes, self.test_tfm, state = "old", label_exist=False)
            
            elif self.type == "HCC":
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
                valid_dataset = self.TrainDataset(Valid, f'{self.hcc_data_dir}/{wsi}', self.classes, self.train_tfm, state = "new")
                test_dataset  = self.TestDataset(Test, f'{self.hcc_data_dir}/{wsi}',self.classes, self.test_tfm, state = "new", label_exist=False)
            
            elif self.type == "CC":
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
                valid_dataset = self.TrainDataset(Valid, f'{self.cc_data_dir}/{wsi}', self.classes, self.train_tfm, state = "new")
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
                    valid_dataset = self.TrainDataset(Valid, f'{self.hcc_old_data_dir}/{wsi}', self.classes, self.train_tfm, state = "old")
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
                    valid_dataset = self.TrainDataset(Valid, f'{self.hcc_data_dir}/{wsi}', self.classes, self.train_tfm, state = "new")
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
                    valid_dataset = self.TrainDataset(Valid, f'{self.cc_data_dir}/{wsi}', self.classes, self.train_tfm, state = "new")
                    test_dataset  = self.TestDataset(Test, f'{self.cc_data_dir}/{wsi}',self.classes, self.train_tfm, state = "new", label_exist=False)

            train_data.extend(pd.DataFrame(Train).to_dict(orient='records'))
            valid_data.extend(pd.DataFrame(Valid).to_dict(orient='records'))
            test_data.extend(pd.DataFrame(Test).to_dict(orient='records'))

        if data_stage == "train":
            pd.DataFrame(train_data).to_csv(f"{save_path}/{condition}_train.csv", index=False)
            pd.DataFrame(valid_data).to_csv(f"{save_path}/{condition}_valid.csv", index=False)
        elif data_stage == "test":
            pd.DataFrame(test_data).to_csv(f"{save_path}/{condition}_test.csv", index=False)

        return train_dataset, valid_dataset, test_dataset
    

    def load_datasets(self, save_path, condition, data_stage, wsi):
        train_csv = f"{save_path}/{condition}_train.csv"
        valid_csv = f"{save_path}/{condition}_valid.csv"
        test_csv  = f"{save_path}/{condition}_test.csv"

        if data_stage == "train":
            if os.path.exists(train_csv) and os.path.exists(valid_csv):
                # read from existing files
                Train = pd.read_csv(train_csv).to_dict(orient="list")
                Valid = pd.read_csv(valid_csv).to_dict(orient="list")
                if self.state == "old":
                    train_dataset = self.TrainDataset(Train, f"{self.hcc_old_data_dir}/{wsi}", self.classes, self.train_tfm, state="old")
                    valid_dataset = self.TrainDataset(Valid, f"{self.hcc_old_data_dir}/{wsi}", self.classes, self.train_tfm, state="old")
                elif self.type == "HCC":
                    train_dataset = self.TrainDataset(Train, f"{self.hcc_data_dir}/{wsi}", self.classes, self.train_tfm, state="new")
                    valid_dataset = self.TrainDataset(Valid, f"{self.hcc_data_dir}/{wsi}", self.classes, self.train_tfm, state="new")
                else:
                    train_dataset = self.TrainDataset(Train, f"{self.cc_data_dir}/{wsi}", self.classes, self.train_tfm, state="new")
                    valid_dataset = self.TrainDataset(Valid, f"{self.cc_data_dir}/{wsi}", self.classes, self.train_tfm, state="new")
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

    def build_pl_dataset(self, wsi, gen, save_path, mode, labeled):
        '''
        selected_patches: patches that is in some class of contour, but the label of patch may not the same as the contour label.
        
        '''
        if self.test_state == "old":
            _wsi = wsi
        elif self.test_type == "HCC":
            _wsi = wsi + 91
        elif self.test_type == "CC":
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
            if self.state == "old":
                match = re.search(r'-(\d+)-(\d+)-\d{5}x\d{5}', img_name)
                if match:
                    x = match.group(1)
                    y = match.group(2)
            else:
                x, y = img_name[:-4].split('_')
            
            row = selected_data[idx, :]
            max_col = np.argmax(row)
            all_pts.append([(int(x)) // pts_ratio, (int(y)) // pts_ratio, max_col])
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
                        f'C{wsi}_{cl_text}-{int(ptx * pts_ratio):05d}-{int(pty * pts_ratio):05d}-{pts_ratio:05d}x{pts_ratio:05d}.tif'
                        if self.state == "old"
                        else f'{int(ptx * pts_ratio)}_{int(pty * pts_ratio)}.tif'
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
                                f'C{wsi}_{cl_text}-{int(ptx * pts_ratio):05d}-{int(pty * pts_ratio):05d}-{pts_ratio:05d}x{pts_ratio:05d}.tif'
                                if self.state == "old"
                                else f'{int(ptx * pts_ratio)}_{int(pty * pts_ratio)}.tif'
                            )
                            if (formatted_filename in all_patches) and (formatted_filename not in selected_patches['file_name']):
                                selected_patches['file_name'].append(formatted_filename)
                                selected_patches['label'].append(cl)

        # for idx, fname in enumerate(all_patches):
        #     row = selected_data[idx]
        #     mask = row > 0.5                # boolean array
        #     if mask.sum() == 1:             # exactly one True
        #         cl_idx = mask.argmax()  
        #         if fname not in selected_patches['file_name'] and fname not in background:
        #             selected_patches['file_name'].append(fname)
        #             selected_patches['label'].append(self.classes[cl_idx])

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
    
    def _train(self, model, modelName, criterion, optimizer, train_loader, val_loader, condition, model_save_path, loss_save_path, target_class):
        n_epochs = self.file_paths['max_epoch']
        min_epoch = self.file_paths['min_epoch']
        notImprove = 0
        min_loss = 1000.

        train_loss_list = []
        train_acc_list = []
        valid_loss_list = []
        valid_acc_list = []
        iter_train_loss_list = []
        iter_valid_loss_list = []

        for epoch in range(1, n_epochs+1):
            # ---------- Training ----------
            # Make sure the model is in train mode before training.
            model.train()

            # These are used to record information in training.
            train_loss = []
            train_acc = []
            train_bar = tqdm(train_loader)

            for idx, batch in enumerate(train_bar):
                # A batch consists of image data and corresponding labels.
                imgs, labels, _ = batch

                # Forward the data. (Make sure data and model are on the same device.)
                if target_class == None:
                    labels = torch.nn.functional.one_hot(labels.to(device), self.class_num).float().to(device)  # one-hot vector
                    logits = model(imgs.to(device))
                    loss = criterion(logits, labels)
                    preds = (torch.sigmoid(logits) >= 0.5).int()
                    # acc = (preds.eq(labels.int()).all(dim=1)).float().mean()
                    correct_per_sample = torch.all(preds == labels.int(), dim=1)  # True/False per sample
                    acc = correct_per_sample.float().mean()
                else:            
                    labels = (labels == target_class).to(device).unsqueeze(1).float()
                    logits = model(imgs.to(device))
                    loss = criterion(logits, labels)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    # acc = (preds.eq(labels.int()).all(dim=1)).float().mean()
                    correct_per_sample = torch.all(preds == labels.int(), dim=1)  # True/False per sample
                    acc = correct_per_sample.float().mean()

                # Gradients stored in the parameters in the previous step should be cleared out first.
                optimizer.zero_grad()
                # Compute the gradients for parameters.
                loss.backward()
                # Update the parameters with computed gradients.
                optimizer.step()
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

            # These are used to record information in validation.
            valid_loss = []
            valid_acc = []
            valid_bar = tqdm(val_loader)
            # Iterate the validation set by batches.
            with torch.no_grad():
                for idx, batch in enumerate(valid_bar):
                    # A batch consists of image data and corresponding labels.
                    imgs, labels, _ = batch
                    if target_class == None:
                        labels = torch.nn.functional.one_hot(labels.to(device), self.class_num).float().to(device)  # one-hot vector
                        logits = model(imgs.to(device))
                        # preds = Sigmoid(logits)
                        loss = criterion(logits, labels)
                        # val_acc = (logits.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean()
                        preds = (logits >= 0.5).int()
                        # val_acc = (preds.eq(labels.int()).all(dim=1)).float().mean()
                        correct_per_sample = torch.all(preds == labels.int(), dim=1)  # True/False per sample
                        val_acc = correct_per_sample.float().mean()
                    else:            
                        labels = (labels == target_class).to(device).unsqueeze(1).float()
                        logits = model(imgs.to(device))
                        # preds = Sigmoid(logits)
                        loss = criterion(logits, labels)
                        preds = (logits > 0.5).float()
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

            training_log = pd.DataFrame({
                "train_loss": train_loss_list,
                "valid_loss": valid_loss_list,
                "train_acc": train_acc_list,
                "valid_acc": valid_acc_list
            })

            training_iteration_log = pd.DataFrame({
                "train_loss": iter_train_loss_list
            })
            validation_iteration_log = pd.DataFrame({
                "valid_loss": iter_valid_loss_list
            })

            training_log.to_csv(f"{loss_save_path}/{condition}_epoch_log.csv", index=False)
            training_iteration_log.to_csv(f"{loss_save_path}/{condition}_train_iteration_log.csv", index=False)
            validation_iteration_log.to_csv(f"{loss_save_path}/{condition}_valid_iteration_log.csv", index=False)
            torch.save(model.state_dict(), f"{model_save_path}/{condition}_Model_epoch{epoch}.ckpt")

            if valid_avg_loss < min_loss:
                # Save model if your model improved
                min_loss = valid_avg_loss
                torch.save(model.state_dict(), f"{model_save_path}/{modelName}")
                notImprove = 0
            else:
                notImprove = notImprove + 1

            if epoch == min_epoch:
                notImprove = 0
            if notImprove >= 2 and epoch >= min_epoch:
                self.plot_loss_acc(train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, loss_save_path, condition)
                return

        self.plot_loss_acc(train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, loss_save_path, condition)

    def train_one_WSI(self, wsi):
        if self.state == "old":
            _wsi = wsi
        elif self.type == "HCC":
            _wsi = wsi + 91
        elif self.type == "CC":
            _wsi = f"1{wsi:04d}"

        if self.type == "HCC":
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"
            data_save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_wsi}/trial_{self.data_trial}"
        else:
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{wsi}/trial_{self.num_trial}"
            data_save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{wsi}/trial_{self.data_trial}"

        condition = f"{_wsi}_{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
        print(f"WSI {_wsi} | {condition}")

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)

        if self.file_paths['load_dataset']:
            data_condition = f"{_wsi}_{self.num_wsi}WTC_LP{self.data_num}_2_class_trial_{self.data_trial}"
            train_dataset, valid_dataset, _ = self.load_datasets(f"{data_save_path}/Data", data_condition, "train", wsi=wsi)
        else:
            train_dataset, valid_dataset, _ = self.prepare_dataset(f"{save_path}/Data", condition, 0, "train", wsi = wsi)

        print(f"training data number: {len(train_dataset)}, validation data number: {len(valid_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)

        model = self.EfficientNetWithLinear(output_dim=self.class_num)
        if self.file_paths['pretrain']:
            pretrain_model_path = self.file_paths['40WTC_model_path']
            model.load_state_dict(torch.load(pretrain_model_path, weights_only=True))

        optimizer = torch.optim.Adam(model.parameters(), lr=self.base_lr)
        model.to(device)
        modelName = f"{condition}_Model.ckpt"
        
        criterion = nn.BCEWithLogitsLoss()

        self._train(model, modelName, criterion, optimizer, train_loader, val_loader, condition, f"{save_path}/Model", f"{save_path}/Loss", target_class=None)

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

        train_dataset, valid_dataset, _ = self.prepare_dataset(f"{save_path}/Data", condition, 0, "train")
        print(f"training data number: {len(train_dataset)}, validation data number: {len(valid_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        # data_iter = iter(train_loader)
        # images, labels , file_names= next(data_iter)
        # img_grid = torchvision.utils.make_grid(images)

        # plt.imshow(img_grid.permute(1, 2, 0))
        # plt.show()
        
        # modelName = f"{condition}_Model_40.ckpt"
        modelName = f"{condition}_Model.ckpt"
        model_path = f"{save_path}/Model/{modelName}"

        model = self.EfficientNetWithLinear(output_dim=self.class_num)
        # model.load_state_dict(torch.load(model_path))
        model.to(device)
        
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.base_lr)

        self._train(model, modelName, criterion, optimizer, train_loader, val_loader, condition, f"{save_path}/Model", f"{save_path}/Loss", target_class=None)

    def train_multi_model(self):
        condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
        if self.num_wsi == 1 and self.type == "HCC":
            save_path = f"{self.save_path}/{self.hcc_wsis[0]}/trial_{self.num_trial}"
        elif self.num_wsi == 1 and self.type == "CC":
            save_path = f"{self.save_path}/{self.cc_wsis[0]}/trial_{self.num_trial}"
        else:
            save_path = f"{self.save_path}/trial_{self.num_trial}"

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)

        train_dataset, valid_dataset, _ = self.prepare_dataset(f"{save_path}/Data", 0)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        
        for c in self.classes:
            model = self.EfficientNetWithLinear(output_dim=1)
            model.to(device)
            modelName = f"{condition}_{c}_Model.ckpt"
            optimizer = torch.optim.Adam(model.parameters(), lr=self.base_lr)

            self._train(model, modelName, criterion, optimizer, train_loader, val_loader, f"{save_path}/Model", f"{save_path}/Loss", target_class=self.classes.index(c))

    def train_generation(self, wsi, mode="ideal", labeled = True):
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
            model = self.EfficientNetWithLinear(output_dim = self.class_num)
            if gen == 1:
                if self.file_paths['pretrain']:
                    model_path = self.file_paths[f'{self.num_wsi}WTC_model_path']
                    model.load_state_dict(torch.load(model_path, weights_only=True))
            else:
                model_path = f"{save_path}/Model/Gen{gen-1}_ND_zscore_{mode}_patches_by_Gen{gen-2}_1WTC.ckpt"
                model.load_state_dict(torch.load(model_path, weights_only=True))
            
            model.to(device)
            modelName = f"{condition}_1WTC.ckpt"

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.base_lr)

            self._train(model, modelName, criterion, optimizer, train_loader, val_loader, condition, f"{save_path}/Model", f"{save_path}/Loss", target_class=None)

    def _test(self, test_dataset, data_info_df, model, save_path, condition, count_acc = True):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model.eval()

        # Record Information
        all_fnames = []
        all_preds = []

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
        for idx, class_name in enumerate(self.classes):
            Predictions[f"{class_name}_pred"] = all_preds[:, idx].tolist()
        
        pred_df = pd.DataFrame(Predictions)

        if count_acc:
            pred_df.to_csv(f"{save_path}/TI/{condition}_patch_in_region_filter_2_v2_TI.csv", index=False)
        else:
            pred_df.to_csv(f"{save_path}/TI/{condition}_all_patches_filter_v2_TI.csv", index=False)
            return

        # pred_df = pd.read_csv(f"{save_path}/Metric/{condition}_pred_score.csv")
        # pred_df = pd.read_csv(f"{save_path}/TI/{condition}_patch_in_region_filter_2_v2_TI.csv")

        results_df = {"file_name":[]}
        all_labels, all_preds = [], []
        match_df  = data_info_df[data_info_df['file_name'].isin(pred_df['file_name'])]

        filename_inRegion = match_df['file_name'].to_list()
        label_inRegion = match_df['label'].to_list()

        for idx, filename in enumerate(tqdm(filename_inRegion)):
            label = self.classes.index(label_inRegion[idx])
            results_df["file_name"].append(filename)
            row = pred_df[pred_df['file_name'] == filename]
            preds = [row[f'{cl}_pred'].values[0] for cl in self.classes]

            over_threshold = [i for i, p in enumerate(preds) if p > 0.5]

            if len(over_threshold) == 1:
                pred = over_threshold[0]
            else:
                pred = -1

            all_labels.append(label)
            all_preds.append(pred)

        text_labels = [self.classes[label] for label in all_labels]
        text_preds = [self.classes[pred] for pred in all_preds]

        results_df["true_label"] = text_labels
        results_df["pred_label"] = text_preds

        # Save to CSV
        pd.DataFrame(results_df).to_csv(f"{save_path}/Metric/{condition}_labels_predictions.csv", index=False)

        acc = accuracy_score(all_labels, all_preds)
        print("Accuracy: {:.4f}".format(acc))

        cm = confusion_matrix(all_labels, all_preds, labels=range(len(self.classes)))
        title = f"Confusion Matrix of {condition}"
        self.plot_confusion_matrix(cm, save_path, condition, title)

        Test_Acc = {"Condition": [condition], "Accuracy": [acc]}
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
        pd.DataFrame(Test_Acc).to_csv(f"{save_path}/Metric/{condition}_test_result.csv", index=False)
        
    def test_one_WSI(self, wsi):
        if self.state == "old":
            _wsi = wsi
        elif self.type == "HCC":
            _wsi = wsi + 91
        elif self.type == "CC":
            _wsi = f"1{wsi:04d}"
        
        condition = f"{_wsi}_{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
        if self.type == "HCC":
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"
        else:
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{wsi}/trial_{self.num_trial}"

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)
        
        test_data = []
        if self.state == "old":
            data_info_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/old/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset = self.TestDataset(data_info_df, f'{self.hcc_old_data_dir}/{wsi}', self.classes, self.test_tfm, state='old', label_exist=False)
        elif self.type == "HCC":
            data_info_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset = self.TestDataset(data_info_df, f'{self.hcc_data_dir}/{wsi}',self.classes,self.test_tfm, state='new', label_exist=False)
        elif self.type == "CC":
            data_info_df = pd.read_csv(f'{self.cc_csv_dir}/{wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset = self.TestDataset(data_info_df, f'{self.cc_data_dir}/{wsi}', self.classes,self.test_tfm, state='new', label_exist=False)

        print(f"testing data number: {len(test_dataset)}")

        if self.test_model == "self":        
            # Prepare Model
            modelName = f"{condition}_Model.ckpt"
            model_path = f"{save_path}/Model/{modelName}"

            model = self.EfficientNetWithLinear(output_dim = self.class_num)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.to(device)

            self._test(test_dataset, data_info_df, model, save_path, condition)
        
        elif self.test_model == "multi":
            # model_wsis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 50, 51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 64, 66, 67, 68, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
            # for model_wsi in model_wsis:
            #     # Prepare Model
            #     if self.state == "old":
            #         _model_wsi = model_wsi
            #         # model_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{model_wsi}/trial_{self.num_trial}"
            #         if model_wsi in [1, 2, 3, 4, 5]:
            #             model_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{model_wsi}/trial_2"
            #             modelName = f"{_model_wsi}_{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_2_Model.ckpt"
            #         else:
            #             model_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{model_wsi}/trial_1"
            #             modelName = f"{_model_wsi}_{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_1_Model.ckpt"
            #     elif self.type == "HCC":
            #         _model_wsi = model_wsi + 91
            #         model_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_model_wsi}/trial_{self.num_trial}"
            #     elif self.type == "CC":
            #         _model_wsi = f"1{model_wsi:04d}"
            #         model_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{model_wsi}/trial_{self.num_trial}"

            #     # modelName = f"{_model_wsi}_{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}_Model.ckpt"
            #     model_path = f"{model_dir}/Model/{modelName}"

            #     model = self.EfficientNetWithLinear(output_dim = self.class_num)
            #     model.load_state_dict(torch.load(model_path, weights_only=True))
            #     model.to(device)

            #     _condition = f'{condition}_on_model_{model_wsi}'

            #     self._test(test_dataset, data_info_df, model, save_path, _condition)

            for ep in range(1, 21):
                modelName = f"{condition}_Model_epoch{ep}.ckpt"
                model_path = f"{save_path}/Model/{modelName}"
                if not os.path.exists(model_path):
                    continue

                model = self.EfficientNetWithLinear(output_dim = self.class_num)
                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.to(device)

                _condition = f'{condition}_for_epoch_{ep}'
                self._test(test_dataset, data_info_df, model, save_path, _condition)
    
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

        model = self.EfficientNetWithLinear(output_dim = len(self.classes))
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(device)

        data_info_df = pd.read_csv(f"{save_path}/Data/{condition}_test.csv")

        self._test(test_dataset, data_info_df, model, save_path, condition)

    def test_TATI(self, wsi, gen, save_path = None, mode = 'ideal'):
        ### Multi-WTC Evaluation ###
        if self.test_state == "old":
            _wsi = wsi
        elif self.test_type == "HCC":
            _wsi = wsi + 91
        elif self.test_type == "CC":
            _wsi = f"1{wsi:04d}"
        
        if self.gen_type:
            if save_path == None:
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"
            if gen == 0:
                condition = f'{self.class_num}_class'
                if self.test_model == "3_class_100WTC":
                    model_path = self.file_paths['100WTC_model_path']
                elif self.test_model == "3_class_40WTC":
                    model_path = self.file_paths['40WTC_model_path']
                model = self.EfficientNetWithLinear(output_dim = len(self.classes))
                # else:
                #     model_path = self.file_paths['HCC_100WTC_model_path']
                #     model = EfficientNet.from_name('efficientnet-b0')
                #     model._fc= nn.Linear(1280, 2)
            else:
                condition = f"Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}"
                model_path = f"{save_path}/Model/{condition}_1WTC.ckpt"
                model = self.EfficientNetWithLinear(output_dim = self.class_num)

        else:
            condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            save_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}"
            save_path = f"{save_dir}/{_wsi}" 

            modelName = f"{condition}_Model.ckpt"
            model_path = f"{save_dir}/Model/{modelName}"
            model = self.EfficientNetWithLinear(output_dim = len(self.classes))

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
        if self.test_state == "old":
            data_info_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset = self.TestDataset(data_info_df, f'{self.hcc_old_data_dir}/{wsi}', self.classes, self.test_tfm, state='old', label_exist=False)
        elif self.test_type == "HCC":
            data_info_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset = self.TestDataset(data_info_df, f'{self.hcc_data_dir}/{wsi}',self.classes,self.test_tfm, state='new', label_exist=False)
        elif self.test_type == "CC":
            data_info_df = pd.read_csv(f'{self.cc_csv_dir}/{wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset = self.TestDataset(data_info_df, f'{self.cc_data_dir}/{wsi}', self.classes,self.test_tfm, state='new', label_exist=False)
        
        _condition = f'{_wsi}_{condition}'

        print(f"WSI {wsi} | {_condition}")
        print(self.classes)

        self._test(test_dataset, data_info_df, model, save_path, _condition)
    
    def test_flip(self, wsi, gen, save_path = None, mode = 'selected'):
        ### Multi-WTC Evaluation ###
        if self.test_state == "old":
            _wsi = wsi
        elif self.test_type == "HCC":
            _wsi = wsi + 91
        elif self.test_type == "CC":
            _wsi = f"1{wsi:04d}"

        if save_path == None:
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"

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

        cm = confusion_matrix(all_labels, all_preds, labels=range(len(self.classes)))
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

    def test_all(self, wsi, gen, save_path = None, mode = 'selected'):
        ### Multi-WTC Evaluation ###
        if self.test_state == "old":
            _wsi = wsi
        elif self.test_type == "HCC":
            _wsi = wsi + 91
        elif self.test_type == "CC":
            _wsi = f"1{wsi:04d}"
        
        if self.gen_type:
            if save_path == None:
                save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"
            if gen == 0:
                condition = f'{self.class_num}_class'
                if self.test_model == "3_class_100WTC":
                    model_path = self.file_paths['100WTC_model_path']
                    model = self.EfficientNetWithLinear(output_dim = len(self.classes))
                elif self.test_model == "3_class_40WTC":
                    model_path = self.file_paths['40WTC_model_path']
                # else:
                #     model_path = self.file_paths['HCC_100WTC_model_path']
                #     model = EfficientNet.from_name('efficientnet-b0')
                #     model._fc= nn.Linear(1280, 2)
            else:
                condition = f"Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}"
                model_path = f"{save_path}/Model/{condition}_1WTC.ckpt"
                model = self.EfficientNetWithLinear(output_dim = self.class_num)

        else:
            condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            save_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}"
            save_path = f"{save_dir}/{_wsi}" 

            modelName = f"{condition}_Model.ckpt"
            model_path = f"{save_dir}/Model/{modelName}"
            model = self.EfficientNetWithLinear(output_dim = len(self.classes))

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
        if self.test_type == "HCC":
            data_info_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_all_patches_filter_v2.csv')
            test_dataset = self.TestDataset(data_info_df, f'{self.hcc_data_dir}/{wsi}',self.classes,self.test_tfm, state='new', label_exist=False)
        elif self.test_type == "CC":
            data_info_df = pd.read_csv(f'{self.cc_csv_dir}/{wsi}/{_wsi}_all_patches_filter_v2.csv')
            test_dataset = self.TestDataset(data_info_df, f'{self.cc_data_dir}/{wsi}', self.classes,self.test_tfm, state='new', label_exist=False)
        
        _condition = f'{_wsi}_{condition}'

        print(f"WSI {wsi} | {_condition}")
        print(self.classes)

        self._test(test_dataset, data_info_df, model, save_path, _condition, count_acc=False)

    def plot_confusion_matrix(self, cm, save_path, condition, title='Confusion Matrix'):
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap='Blues')
        fig.colorbar(cax)

        ax.set_xticks(np.arange(self.class_num))
        ax.set_yticks(np.arange(self.class_num))
        ax.set_xticklabels(self.classes, fontsize=14)
        ax.set_yticklabels(self.classes, fontsize=14)

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

    def plot_TI_Result(self, wsi, gen, save_path = None, mode = 'ideal'):
        if save_path == None:
            _wsi = wsi+91 if (self.test_state == "new" and self.test_type == "HCC") else wsi
            __wsi = wsi if self.test_state == "old" else (wsi+91 if self.test_type == "HCC" else f"1{wsi:04d}")
        if self.gen_type:
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/{__wsi}/trial_{self.num_trial}"
            if gen == 0:
                condition = f'{self.class_num}_class'
            else:
                condition = f"Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}"
        else:
            condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            if self.test_model == "self" or self.test_model == "multi":
                save_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{_wsi}/trial_{self.num_trial}"
                save_path = save_dir
            else:
                save_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}"
                save_path = f"{save_dir}/{__wsi}" 
        
        if self.test_model == "multi":
            _condition = f'{__wsi}_{condition}_for_epoch_20'
        else:
            _condition = f'{__wsi}_{condition}'
        df = pd.read_csv(f"{save_path}/Metric/{_condition}_labels_predictions.csv")

        shift_map = self.file_paths["old_HCC_shift_map"]
        
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

            x = (int(x)) // pts_ratio
            y = (int(y)) // pts_ratio
            
            if self.test_state == "old":
                dx, dy = shift_map[gt_label]
                x += dx
                y += dy
            
            gt_label = self.classes.index(df['true_label'][idx])
            if df['pred_label'][idx] != -1:
                pred_label  = self.classes.index(df['pred_label'][idx])
                
                if pred_label == gt_label:
                    label = pred_label + 1  # correct predict: encode as 1,2,3
                else:
                    label = 10 * (gt_label + 1) + (pred_label + 1)  # wrong predict：encode asss 11,12,13, 21,22,23, 31,32,33

                all_pts.append([x, y, label])

        all_pts = np.array(all_pts)

        x_max, y_max = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])

        image = np.zeros((y_max + 1, x_max + 1))
        for x, y, label in all_pts:
            image[y, x] = label

        if self.class_num == 3:
            color_map = {
                1: 'green',     # True Normal
                2: 'red',       # True HCC
                3: 'blue',      # True CC
                12: 'orange',   # Normal -> HCC
                13: 'yellow',   # Normal -> CC
                21: 'cyan',     # HCC -> Normal
                23: 'magenta',  # HCC -> CC
                31: 'purple',   # CC -> Normal
                32: 'brown'     # CC -> HCC
            }
            legend_elements = [
                plt.Line2D([0], [0], color='green', lw=4, label='True Normal'),
                plt.Line2D([0], [0], color='red', lw=4, label='True HCC'),
                plt.Line2D([0], [0], color='blue', lw=4, label='True CC'),
                plt.Line2D([0], [0], color='orange', lw=4, label='Normal -> HCC'),
                plt.Line2D([0], [0], color='yellow', lw=4, label='Normal -> CC'),
                plt.Line2D([0], [0], color='cyan', lw=4, label='HCC -> Normal'),
                plt.Line2D([0], [0], color='magenta', lw=4, label='HCC -> CC'),
                plt.Line2D([0], [0], color='purple', lw=4, label='CC -> Normal'),
                plt.Line2D([0], [0], color='brown', lw=4, label='CC -> HCC')
            ]
        elif self.class_num == 2:
            if self.test_type == "HCC":
                color_map = {
                    1: 'green',     # True Normal
                    2: 'red',       # True HCC
                    12: 'orange',   # Normal -> HCC
                    21: 'cyan',     # HCC -> Normal
                }
                legend_elements = [
                    plt.Line2D([0], [0], color='green', lw=4, label='True Normal'),
                    plt.Line2D([0], [0], color='red', lw=4, label='True HCC'),
                    plt.Line2D([0], [0], color='orange', lw=4, label='Normal -> HCC'),
                    plt.Line2D([0], [0], color='cyan', lw=4, label='HCC -> Normal'),
                ]
            elif self.test_type == "CC":
                color_map = {
                    1: 'green',     # True Normal
                    2: 'blue',      # True CC
                    12: 'yellow',   # Normal -> CC
                    21: 'purple',   # CC -> Normal
                }
                legend_elements = [
                    plt.Line2D([0], [0], color='green', lw=4, label='True Normal'),
                    plt.Line2D([0], [0], color='blue', lw=4, label='True CC'),
                    plt.Line2D([0], [0], color='yellow', lw=4, label='Normal -> CC'),
                    plt.Line2D([0], [0], color='purple', lw=4, label='CC -> Normal'),
                ]
        plt.figure(figsize=(x_max // 20, y_max // 20))
        for label_value, color in color_map.items():
            plt.imshow(image == label_value, cmap=ListedColormap([[0,0,0,0], color]), interpolation='nearest', alpha=1)

        plt.title(f"Prediction vs Ground Truth of WSI {__wsi}", fontsize=20, pad=20)
        plt.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        plt.axis("off")

        plt.savefig(f"{save_path}/Metric/{_condition}_pred_vs_gt.png")
        print(f"WSI {wsi} already plot the pred_vs_gt image")

    def plot_all_result(self, wsi, gen, save_path = None, mode = 'selected', plot_type = 'pred'):
        if save_path == None:
            _wsi = wsi+91 if (self.test_state == "new" and self.test_type == "HCC") else wsi
            __wsi = wsi if self.test_state == "old" else (wsi+91 if self.test_type == "HCC" else f"1{wsi:04d}")
        if self.gen_type:
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_LP_{self.data_num}/{__wsi}/trial_{self.num_trial}"
            if gen == 0:
                condition = f'{self.class_num}_class'
            else:
                condition = f"Gen{gen}_ND_zscore_{mode}_patches_by_Gen{gen-1}"
        else:
            condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            if self.test_model == "self":
                save_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}/{__wsi}"
                save_path = save_dir
            else:
                save_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}"
                save_path = f"{save_dir}/{__wsi}" 
        
        _condition = f'{__wsi}_{condition}'
        
        if gen == 0 or plot_type == 'pred':
            df = pd.read_csv(f"{save_path}/TI/{_condition}_all_patches_filter_v2_TI.csv")
        elif plot_type == 'flip':
            df = pd.read_csv(f"{save_path}/Data/{_condition}.csv")

        all_patches = df['file_name'].to_list()

        ### Get (x, y, pseudo-label) of every patch ###
        all_pts = []
        if 'label' in df:
            for idx, img_name in enumerate(all_patches):
                x, y = img_name[:-4].split('_')

                if df['label'][idx] != -1:
                    pred_label  = self.classes.index(df['label'][idx])  # N=0, H=1

                    x = (int(x)) // pts_ratio
                    y = (int(y)) // pts_ratio

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

                x = (int(x)) // pts_ratio
                y = (int(y)) // pts_ratio

                all_pts.append([x, y, pred_label+1])

        all_pts = np.array(all_pts)
        x_max, y_max = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])

        image = np.zeros((y_max + 1, x_max + 1))
        for x, y, label in all_pts:
            image[y, x] = label

        if self.class_num == 3:
            color_map = {
                1: 'green',     # True Normal
                2: 'red',       # True HCC
                3: 'blue',      # True CC
            }
            legend_elements = [
                plt.Line2D([0], [0], color='green', lw=4, label='Pred Normal'),
                plt.Line2D([0], [0], color='red', lw=4, label='Pred HCC'),
                plt.Line2D([0], [0], color='blue', lw=4, label='Pred CC'),
            ]
        elif self.class_num == 2:
            if self.test_type == "HCC":
                color_map = {
                    1: 'green',     # True Normal
                    2: 'red',       # True HCC
                }
                legend_elements = [
                    plt.Line2D([0], [0], color='green', lw=4, label='Pred Normal'),
                    plt.Line2D([0], [0], color='red', lw=4, label='Pred HCC'),
                ]
            elif self.test_type == "CC":
                color_map = {
                    1: 'green',     # True Normal
                    2: 'blue',      # True CC
                }
                legend_elements = [
                    plt.Line2D([0], [0], color='green', lw=4, label='Pred Normal'),
                    plt.Line2D([0], [0], color='blue', lw=4, label='Pred CC'),
                ]
        plt.figure(figsize=(x_max/10, y_max/10))
        for label_value, color in color_map.items():
            plt.imshow(image == label_value, cmap=ListedColormap([[0,0,0,0], color]), interpolation='nearest', alpha=1)

        plt.title(f"Prediction of WSI {__wsi}", fontsize=20, pad=20)
        plt.legend(handles=legend_elements, loc='upper right', fontsize=18)
        plt.tight_layout()
        plt.axis("off")

        plt.savefig(f"{save_path}/Metric/{_condition}_{plot_type}.png")
        print(f"WSI {wsi} already plot the {plot_type} image")

    def plot_TI_Result_gt_boundary(self, wsi, gen, save_path):
        if save_path == None:
            _wsi = wsi+91 if (self.state == "new" and self.type == "HCC") else wsi
            __wsi = wsi if self.state == "old" else (wsi+91 if self.type == "HCC" else f"1{wsi:04d}")
        if self.gen_type:
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}/{_wsi}"
            if gen == 0:
                condition = f'{self.class_num}_class'
            else:
                condition = f"Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}"
        else:
            condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            save_dir = os.path.join(self.file_paths[f'{self.type}_{self.num_wsi}WTC_model_path'], f"LP_{self.data_num}/trial_{self.num_trial}") 
            save_path = f"{save_dir}/{_wsi}" 

        df = pd.read_csv(f"{save_path}/Metric/{__wsi}_{condition}_labels_predictions.csv")
        # df = pd.read_csv(f"{save_path}/TI/{_wsi}_{condition}_patch_in_region_filter_2_v2_TI.csv")
        gt = pd.read_csv(f"{self.cc_csv_dir}/{wsi}/{__wsi}_patch_in_region_filter_2_v2.csv") if self.type == "CC" \
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

            x = (int(x)) // pts_ratio
            y = (int(y)) // pts_ratio
            
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

            x = (int(x)) // pts_ratio
            y = (int(y)) // pts_ratio
            
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

        _wsi = wsi+91 if (self.state == "new" and self.type == "HCC") else wsi
        # plt.show()
        plt.savefig(f"{save_path}/Metric/{wsi}_pred_vs_gt.png")
        plt.tight_layout()
        plt.axis("off")

    def contour_analysis(self, wsi, gen, save_path):
        if save_path == None:
            _wsi = wsi+91 if (self.state == "new" and self.type == "HCC") else wsi
            save_path = f'{self.save_dir}/100WTC_Result/{_wsi}/trial_{self.num_trial}'

        if gen == 0:
            condition = f'{self.class_num}_class'
        else:
            condition = f"Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}"

        df = pd.read_csv(f"{save_path}/Metric/{_wsi}_{condition}_labels_predictions.csv")
        # df = pd.read_csv(f"{save_path}/TI/{_wsi}_{condition}_patch_in_region_filter_2_v2_TI.csv")
        
        all_patches = df['file_name'].to_list() #patches_in_hcc_hulls

        ### Get (x, y, pseudo-label) of every patch ###
        all_pts = []
        for idx, img_name in enumerate(all_patches):
            if self.state == "old":
                match = re.search(r'-(\d+)-(\d+)-\d{5}x\d{5}', img_name)
                if match:
                    x = match.group(1)
                    y = match.group(2)
                else:
                    print("Style Error")
            else:
                x, y = img_name[:-4].split('_')

            label = self.classes.index(df['pred_label'][idx])  # N=0, H=1
            # label = 1 if df['H_pred'][idx] > df["N_pred"][idx] else 0

            x = (int(x)) // pts_ratio
            y = (int(y)) // pts_ratio
            
            all_pts.append([x, y, label])  #label 0,1

        all_pts = np.array(all_pts)
        
        ### First sorted pts on x, then on y ###
        sorted_index = np.lexsort((all_pts[:, 1], all_pts[:, 0]))
        sorted_all_pts = all_pts[sorted_index]

        x_max, y_max = np.max(sorted_all_pts[:, 0]), np.max(sorted_all_pts[:, 1])

        label_map = np.full((y_max + 1, x_max + 1), -1, dtype=np.int32)

        # Fill the label map with the labels from sorted_all_pts
        for x, y, label in sorted_all_pts:
            label_map[int(y), int(x)] = label

        # img = cv2.imread(f'{save_path}/test_img.png')
        # # resized_img = cv2.resize(img, (1500, 1500))
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # binary_img = np.where(gray_img > 127, 1, 0).astype(np.uint8)    
        # label_map = np.array(binary_img)

        for cl in range(len(self.classes)):
            # Create a binary mask for the target label
            binary_mask = (label_map == cl).astype(np.uint8)

            # Find contours using OpenCV
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Step 2: Analyze connected regions with opposite labels inside each contour
            results = []
            
            # Collect results for this contour
            contours_data = []

            # Define the opposite label (e.g., if target_label is 1, opposite_label is 0)
            opposite_label = 1 - cl

            # Plot the binary mask and contours
            plt.figure(figsize=(10, 10))
            plt.imshow(binary_mask, cmap='gray')
            plt.title(f"Contours for Class {cl}")
            plt.axis('off')
            plt.imsave(f"{save_path}/Metric/BinaryContour_Class{cl}.png", binary_mask, cmap='gray', format='png')
            plt.close()

            contour_idx = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > area_thresh:
                    
                    # Create a mask with the same size as the label_map
                    mask = np.zeros_like(label_map, dtype=np.uint8)
                    
                    # Fill the contour area in the mask and set it to 1
                    cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)

                    # Extract the region within the contour that has the opposite label
                    opposite_region = np.where((mask == 1) & (label_map == opposite_label), 1, 0)
                    
                    # Perform connected components analysis on the opposite region
                    num_connected_components, labels, stats, centroids = cv2.connectedComponentsWithStats(opposite_region.astype(np.uint8), connectivity=8)

                    # Collect results for the contour
                    for i in range(1, num_connected_components):  # Skip the background (label 0)
                        contours_data.append({
                            "contour_index": contour_idx,
                            "connected_region_index": i,
                            "connected_patch_size": stats[i, cv2.CC_STAT_AREA],
                            "connected_region_centroid_x": centroids[i][0],
                            "connected_region_centroid_y": centroids[i][1]
                        })
                    contour_idx += 1
            
            connected_patch_sizes = [data["connected_patch_size"] for data in contours_data]

            plt.figure(figsize=(10, 6))
            if connected_patch_sizes:
                max_patch_size = max(connected_patch_sizes)
                plt.hist(connected_patch_sizes, bins=max_patch_size, color='blue', alpha=0.7, edgecolor='black')
            else:
                plt.hist(connected_patch_sizes, bins=10, color='blue', alpha=0.7, edgecolor='black')

            plt.title(f"Distribution of Connected Patch Sizes for Class {cl}", fontsize=14)
            plt.xlabel("Patch Size", fontsize=12)
            plt.ylabel("Number", fontsize=12)

            plt.tight_layout()
            plt.savefig(f"{save_path}/Metric/ConnectedPatchSize_Distribution_Class{cl}.png")

            # Save results for this class to a CSV
            pd.DataFrame(contours_data).to_csv(f"{save_path}/Metric/ContourAnalysis_Class{cl}.csv", index=False)
            print(f"Contour analysis results for class {cl} saved")

    def contour_analysis_multi(self, gen):
        file_list_normal, file_list_hcc = [], []
        for wsi in self.hcc_wsis:
            _wsi = wsi+91 if (self.state == "new" and self.type == "HCC") else wsi
            save_path = f'{self.save_dir}/100WTC_Result/{_wsi}/trial_{self.num_trial}'

            if gen == 0:
                condition = f'{self.class_num}_class'
            else:
                condition = f"Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}"
            
            file_list_normal.append(f"{save_path}/Metric/ContourAnalysis_Class0.csv")
            file_list_hcc.append(f"{save_path}/Metric/ContourAnalysis_Class1.csv")

        
        def safe_read_csv(file_name):
            if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
                try:
                    return pd.read_csv(file_name)
                except pd.errors.EmptyDataError:
                    print(f"File is empty: {file_name}")
                    return pd.DataFrame()
            else:
                print(f"File not found or is empty: {file_name}")
                return pd.DataFrame()
        
        save_path = f'{self.save_dir}/100WTC_Result'
        all_sizes_normal, all_sizes_hcc = [], []
        # Read Normal category files
        for file_name in file_list_normal:
            data = safe_read_csv(file_name)
            if not data.empty and 'connected_patch_size' in data.columns:
                all_sizes_normal.extend(data['connected_patch_size'].tolist())

        # Read HCC category files
        for file_name in file_list_hcc:
            data = safe_read_csv(file_name)
            if not data.empty and 'connected_patch_size' in data.columns:
                all_sizes_hcc.extend(data['connected_patch_size'].tolist())
            
        if all_sizes_normal:
            all_sizes_normal = np.array(all_sizes_normal)

            sorted_sizes = np.sort(all_sizes_normal)
            cumulative_distribution = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)

            plt.figure(figsize=(10, 6))
            plt.plot(sorted_sizes, cumulative_distribution, marker='o', linestyle='-', color='b')
            plt.title("Cumulative Distribution of HCC Connected Component Sizes", fontsize=14)
            plt.xlabel("Connected Component Size", fontsize=12)
            plt.ylabel("Cumulative Probability", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(f"{save_path}/cumulative_distribution_plot_normal.png")
            # plt.show()
        else:
            print("No data of normal available for plotting.")

        if all_sizes_hcc:
            all_sizes_hcc = np.array(all_sizes_hcc)

            sorted_sizes = np.sort(all_sizes_hcc)
            cumulative_distribution = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)

            plt.figure(figsize=(10, 6))
            plt.plot(sorted_sizes, cumulative_distribution, marker='o', linestyle='-', color='b')
            plt.title("Cumulative Distribution of Normal Connected Component Sizes", fontsize=14)
            plt.xlabel("Connected Component Size", fontsize=12)
            plt.ylabel("Cumulative Probability", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(f"{save_path}/cumulative_distribution_plot_hcc.png")
            # plt.show()
        else:
            print("No data of hcc available for plotting.")


