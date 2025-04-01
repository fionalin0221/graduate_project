import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import re
import cv2
import random
import yaml
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
area_thresh = 288

class Worker():
    def __init__(self, config):
        current_computer = config['current_computer']
        self.type = config['type']
        self.file_paths = config['computers'][current_computer]['file_paths']
        self.state = config['state']
        class_list = config["class_list"]
        self.classes = [class_list[i] for i in self.file_paths['classes']]
        self.class_num = len(self.classes)

        self.gen_type = config['gen_type']
        self.generation = config["generation"]

        self.data_num = self.file_paths['data_num']
        self.num_trial = self.file_paths['num_trial']   
        self.num_wsi = self.file_paths['num_wsi']
        self.test_model = self.file_paths['test_model']

        if self.gen_type:
            self.save_dir = self.file_paths[f'{self.type}_generation_save_path']
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = self.file_paths[f'{self.type}_WTC_result_save_path']
            self.save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}"
            # self.save_path = f"{self.save_dir}/100WTC_Result"
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.save_path, exist_ok=True)

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

            return image, label, img_path

        def __len__(self):
            return len(self.data_dict["file_name"])
    
    class TestDataset(Dataset):
        def __init__(self, data_dir, data_dict, classes, transform, state, label_exist=True):
            
            self.data_dir = data_dir
            self.classes = classes
            self.transform = transform
            self.label_exist = label_exist
            self.data_dict = data_dict

            # if self.label_exist:
            self.label = self.data_dict['label']
            self.filename = self.data_dict['file_name']
            self.state = state
            
            
        def __getitem__(self, index):
            label_text = self.label[index]
            label = self.classes.index(label_text)
        
            if self.state == 'old':
                if label == 'H':
                    image_path = os.path.join(self.data_dir, 'HCC', self.filename[index])
                if label == 'N':
                    image_path = os.path.join(self.data_dir, 'Normal', self.filename[index])
                image = Image.open(image_path)
                image = self.transform(image)

            if self.state == 'new':
                image_path = os.path.join(self.data_dir, self.filename[index])
                image = Image.open(image_path)
                image = self.transform(image)

            
            if self.label_exist:
                return image, label, self.filename[index]
            
            else:
                return image, self.filename[index]
        
        def __len__(self):
            return len(self.filename)
        
    def check_overlap(self, *lists):
        sets = [set(lst) for lst in lists]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                if sets[i] & sets[j]:
                    return True
        return False
    
    def split_datas(self, selected_data, data_num):
        file_names = selected_data['file_name'].to_numpy()
        labels = selected_data['label'].to_numpy()        
        datas = []

        class_file_names = []
        for cl in self.classes:
            class_file_names.append(list(file_names[labels == cl]))
        
        if data_num == "ALL":
            for num in range(len(self.classes)):
                if len(class_file_names[num]) > 0:
                    datas.append(class_file_names[num])
                else:
                    datas.append([])
        else:
            data_num = int(data_num)
            for num in range(len(self.classes)):
                if len(class_file_names[num]) > 0:
                    class_data_num = int(data_num * 0.5) if num == 0 else int(data_num * 0.5)
                    datas.append(random.sample(class_file_names[num], class_data_num))
                    # if self.type == "Mix" and self.classes[num] == "N":
                    #     datas.append(random.sample(class_file_names[num], int(data_num/2)))
                    # else:
                    #     datas.append(random.sample(class_file_names[num], int(data_num)))
                else:
                    datas.append([])
                
        if self.check_overlap(*datas):
            print(f'Data overlap.')
            return

        data_file_names, data_labels = [], []

        for i in range(self.class_num):
            data_file_names += class_file_names[i]
            data_labels += [self.classes[i]] * len(class_file_names[i])

        # Prepare train/val dataset
        train, val, test = [], [], []
        for num in range(len(self.classes)):
            if len(datas[num]) > 0:
                train_, temp = train_test_split(datas[num], test_size=0.2, random_state=0)  # 80:20 split
                val_, test_ = train_test_split(temp, test_size=0.5, random_state=0)  # 50:50 split on 20% = 10% each
                
                train.append(train_)
                val.append(val_)
                test.append(test_)
            else:
                train.append([])
                val.append([])
                test.append([])

        train_file_names, train_labels, val_file_names, val_labels, test_file_names, test_labels = [], [], [], [], [], []
        for i in range(self.class_num):
            if len(datas[i]) > 0:
                train_file_names += train[i]
                train_labels += [self.classes[i]] * len(train[i])
                val_file_names += val[i]
                val_labels += [self.classes[i]] * len(val[i])
                test_file_names += test[i]
                test_labels += [self.classes[i]] * len(test[i])

        Train = {
            "file_name": train_file_names,
            "label": train_labels
        }
        Val = {
            "file_name": val_file_names,
            "label": val_labels
        }
        Test = {
            "file_name": test_file_names,
            "label": test_labels
        }
        return Train, Val, Test

    def prepare_dataset(self, save_path, condition, gen, data_stage):
        train_data = []
        valid_data = []
        test_data = []

        train_datasets = []
        valid_datasets = []
        test_datasets = []

        print(f"Patches use for a WSI: {self.data_num}")

        for h_wsi in self.hcc_wsis:
            if self.state == "old":
                if self.gen_type:
                    selected_data = pd.read_csv(f'{save_path}/{h_wsi}_Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}.csv')
                else:
                    selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{h_wsi}/{h_wsi}_patch_in_region_filter_2_v2.csv')
                Train, Valid, Test = self.split_datas(selected_data, self.data_num)
                h_train_dataset = self.TrainDataset(Train, f'{self.hcc_old_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "old")
                h_valid_dataset = self.TrainDataset(Valid, f'{self.hcc_old_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "old")
                h_test_dataset  = self.TestDataset(f'{self.hcc_old_data_dir}/{h_wsi}',Test, self.classes, self.test_tfm, state = "old", label_exist=False)

            else:
                if self.gen_type:
                    # selected_data = pd.read_csv(f'{save_path}/{h_wsi+91}_Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}.csv')
                    selected_data = pd.read_csv(f'{save_path}/{h_wsi+91}_Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}.csv')
                else:
                    selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{h_wsi+91}/{h_wsi+91}_patch_in_region_filter_2_v2.csv')
                Train, Valid, Test = self.split_datas(selected_data, self.data_num)
                h_train_dataset = self.TrainDataset(Train, f'{self.hcc_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "new")
                h_valid_dataset = self.TrainDataset(Valid, f'{self.hcc_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "new")
                h_test_dataset  = self.TestDataset(f'{self.hcc_data_dir}/{h_wsi}',Test, self.classes, self.test_tfm, state = "new", label_exist=False)

            train_datasets.append(h_train_dataset)
            valid_datasets.append(h_valid_dataset)
            test_datasets.append(h_test_dataset)
            
            train_data.extend(pd.DataFrame(Train).to_dict(orient='records'))
            valid_data.extend(pd.DataFrame(Valid).to_dict(orient='records'))
            test_data.extend(pd.DataFrame(Test).to_dict(orient='records'))

        for c_wsi in self.cc_wsis:
            print(c_wsi)
            if self.gen_type:
                selected_data = pd.read_csv(f'{save_path}/1{c_wsi:04d}_Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}.csv')
            else:
                selected_data = pd.read_csv(f'{self.cc_csv_dir}/{c_wsi}/1{c_wsi:04d}_patch_in_region_filter_2_v2.csv')
            Train, Valid, Test = self.split_datas(selected_data, self.data_num)
            c_train_dataset = self.TrainDataset(Train, f'{self.cc_data_dir}/{c_wsi}', self.classes, self.train_tfm, state = "new")
            c_valid_dataset = self.TrainDataset(Valid, f'{self.cc_data_dir}/{c_wsi}', self.classes, self.train_tfm, state = "new")
            c_test_dataset  = self.TestDataset(f'{self.cc_data_dir}/{c_wsi}',Test, self.classes, self.train_tfm, state = "new", label_exist=False)

            train_datasets.append(c_train_dataset)
            valid_datasets.append(c_valid_dataset)
            test_datasets.append(c_test_dataset)

            train_data.extend(pd.DataFrame(Train).to_dict(orient='records'))
            valid_data.extend(pd.DataFrame(Valid).to_dict(orient='records'))
            test_data.extend(pd.DataFrame(Test).to_dict(orient='records'))

        train_dataset = ConcatDataset(train_datasets)
        valid_dataset = ConcatDataset(valid_datasets)
        test_dataset = ConcatDataset(test_datasets)

        if data_stage == "train":
            pd.DataFrame(train_data).to_csv(f"{save_path}/{condition}_train.csv", index=False)
            pd.DataFrame(valid_data).to_csv(f"{save_path}/{condition}_valid.csv", index=False)
        elif data_stage == "test":
            pd.DataFrame(test_data).to_csv(f"{save_path}/{condition}_test.csv", index=False)

        return train_dataset, valid_dataset, test_dataset
   
    def build_pl_dataset(self, wsi, gen, save_path):
        '''
        selected_patches: patches that is in some class of contour, but the label of patch may not the same as the contour label.
        
        '''
        _wsi = wsi+91 if (self.state == "new" and self.type == "HCC") else wsi

        if gen == 1:
            df = pd.read_csv(f"{save_path}/TI/{_wsi}_{self.class_num}_class_patch_in_region_filter_2_v2_TI.csv")
        else:
            df = pd.read_csv(f"{save_path}/TI/{_wsi}_Gen{gen-1}_ND_zscore_selected_patches_by_Gen{gen-2}_patch_in_region_filter_2_v2_TI.csv")
            # df = pd.read_csv(f"{save_path}/TI/{_wsi}_Gen{gen-1}_ND_zscore_ideal_patches_by_Gen{gen-2}_patch_in_region_filter_2_v2_TI.csv")

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
        sorted_all_pts = all_pts[sorted_index]

        selected_patches = {'file_name': [], 'label': []}
        
        cl = "H" if self.type == "HCC" else "C"
        selected_patches, patches_in_cancer_regions, tp_in_cancer_regions, fp_in_cancer_regions = \
            find_contours.find_contour(wsi, sorted_all_pts, self.state, cl, area_thresh, all_patches, selected_patches)
        # print(len(selected_patches['file_name']))
        
        selected_patches, patches_in_norm_regions, tn_in_norm_regions, fn_in_norm_regions = \
            find_contours.find_contour(wsi, sorted_all_pts, self.state, "N", area_thresh, all_patches, selected_patches)
        # print(len(selected_patches['file_name']))

        pd.DataFrame(selected_patches).to_csv(f"{save_path}/Data/{_wsi}_Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}.csv", index=False)
        
        ideal_patches, pl_cancer_contour_df, pl_norm_contour_df = find_contours.zscore_filter(
            tp_in_cancer_regions,
            fp_in_cancer_regions,
            tn_in_norm_regions,
            fn_in_norm_regions,
            patches_in_cancer_regions,
            patches_in_norm_regions,
            cl
        )
        pl_cancer_contour_df.to_csv(f"{save_path}/Data/{_wsi}_Gen{gen}_tpfp_ND_zscore_filtered_contour_by_Gen{gen-1}.csv")
        pl_norm_contour_df.to_csv(f"{save_path}/Data/{_wsi}_Gen{gen}_tnfn_ND_zscore_filtered_contour_by_Gen{gen-1}.csv")
        pd.DataFrame(ideal_patches).to_csv(f"{save_path}/Data/{_wsi}_Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}.csv", index=False)

    class EfficientNetWithLinear(nn.Module):
        def __init__(self, output_dim, pretrain='efficientnet-b0'):
            nn.Module.__init__(self)
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

    def plot_loss_acc(self, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, save_path):
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
        plt.savefig(f"{save_path}/loss_and_accuracy_curve.png", dpi=300, bbox_inches="tight")
    
    def _train(self, model, modelName, criterion, optimizer, train_loader, val_loader, condition, model_save_path, loss_save_path, target_class):
        n_epochs = 20
        min_epoch = 20
        notImprove = 0
        min_loss = 1000.

        train_loss_list = []
        train_acc_list = []
        valid_loss_list = []
        valid_acc_list = []

        log_file = f"{loss_save_path}/{condition}_log.yaml"

        for epoch in range(1, n_epochs):
            # ---------- Training ----------
            # Make sure the model is in train mode before training.
            model.train()

            # These are used to record information in training.
            train_loss = []
            train_acc = []
            train_bar = tqdm(train_loader)
            for idx, batch in enumerate(train_bar):
                # A batch consists of image data and corresponding labels.
                imgs, labels, file_names = batch

                # Forward the data. (Make sure data and model are on the same device.)
                if target_class == None:
                    labels = torch.nn.functional.one_hot(labels.to(device), self.class_num).float().to(device)  # one-hot vector
                    logits = model(imgs.to(device))
                    # preds = Sigmoid(logits)
                    loss = criterion(logits, labels)
                    acc = (logits.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean()
                else:            
                    labels = (labels == target_class).to(device).unsqueeze(1).float()
                    logits = model(imgs.to(device))
                    # preds = Sigmoid(logits)
                    loss = criterion(logits, labels)
                    preds = (logits > 0.5).float()
                    acc = (preds == labels).float().mean()

                # Gradients stored in the parameters in the previous step should be cleared out first.
                optimizer.zero_grad()
                # Compute the gradients for parameters.
                loss.backward()
                # Update the parameters with computed gradients.
                optimizer.step()
                # Record the loss and accuracy.
                train_loss.append(loss.cpu().item())
                train_acc.append(acc.cpu().item())
                torch.cuda.empty_cache()
                    
                train_avg_loss = sum(train_loss) / len(train_loss)
                train_avg_acc = sum(train_acc) / len(train_acc)

                if idx % 100 == 0:
                    msg = f"[ Train | Epoch{epoch} Batch{idx} ] loss = {train_avg_loss:.5f}, acc = {train_avg_acc:.5f}"
                    tqdm.write(msg)
                    with open(log_file, "a") as f:
                        yaml.dump(msg, f, default_flow_style=False)

            train_loss_list.append(train_avg_loss)
            train_acc_list.append(train_avg_acc)
            msg = f"[ Train | {epoch:03d}/{n_epochs:03d} ] loss = {train_avg_loss:.5f}, acc = {train_avg_acc:.5f}"
            print(msg)
            with open(log_file, "a") as f:
                yaml.dump(msg, f, default_flow_style=False)
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
                    imgs, labels, file_names = batch
                    if target_class == None:
                        labels = torch.nn.functional.one_hot(labels.to(device), self.class_num).float().to(device)  # one-hot vector
                        logits = model(imgs.to(device))
                        # preds = Sigmoid(logits)
                        loss = criterion(logits, labels)
                        val_acc = (logits.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean()
                    else:            
                        labels = (labels == target_class).to(device).unsqueeze(1).float()
                        logits = model(imgs.to(device))
                        # preds = Sigmoid(logits)
                        loss = criterion(logits, labels)
                        preds = (logits > 0.5).float()
                        val_acc = (preds == labels).float().mean()
                    
                    # Record the loss and accuracy.
                    valid_loss.append(loss.cpu().item())
                    valid_acc.append(val_acc.cpu().item())
                    torch.cuda.empty_cache()

                    # The average loss and accuracy for entire validation set is the average of the recorded values.
                    valid_avg_loss = sum(valid_loss) / len(valid_loss)
                    valid_avg_acc = sum(valid_acc) / len(valid_acc)

                    if idx % 100 == 0:
                        msg = f"[ Valid | Epoch{epoch} Batch{idx} ] loss = {valid_avg_loss:.5f}, acc = {valid_avg_acc:.5f}"
                        tqdm.write(msg)
                        with open(log_file, "a") as f:
                            yaml.dump(msg, f, default_flow_style=False)

            valid_loss_list.append(valid_avg_loss)
            valid_acc_list.append(valid_avg_acc)
            torch.cuda.empty_cache()

            # Print the information.
            msg = f"[ Valid | {epoch:03d}/{n_epochs:03d} ] loss = {valid_avg_loss:.5f}, acc = {valid_avg_acc:.5f}"
            print(msg)
            with open(log_file, "a") as f:
                yaml.dump(msg, f, default_flow_style=False)

            training_log = pd.DataFrame({
                "train_loss": train_loss_list,
                "valid_loss": valid_loss_list,
                "train_acc": train_acc_list,
                "valid_acc": valid_acc_list
            })

            training_log.to_csv(f"{loss_save_path}/{condition}_epoch_log.csv", index=False)

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
                self.plot_loss_acc(train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, loss_save_path)
                break

    def train_one_WSI(self, wsi):
        if self.state == "old":
            _wsi = wsi
        elif self.type == "HCC":
            _wsi = wsi + 91
        elif self.type == "CC":
            _wsi = f"1{wsi:04d}"

        save_path = f"{self.save_path}/{wsi}/trial_{self.num_trial}"
        condition = f"{_wsi}_{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
        print(f"WSI {wsi} | {condition}")

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)

        # train_dataset, valid_dataset, _ = self.prepare_dataset(f"{save_path}/Data", condition, 0, "train")
        train_data, valid_data = [], []
        if self.state == "old":
            data_dir = f'{self.hcc_old_data_dir}/{wsi}'
            selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
        elif self.type == "HCC":
            data_dir = f'{self.hcc_data_dir}/{wsi}'
            selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
        elif self.type == "CC":
            data_dir = f'{self.cc_data_dir}/{wsi}'
            selected_data = pd.read_csv(f'{self.cc_csv_dir}/{wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
        Train, Valid, _ = self.split_datas(selected_data, self.data_num)
        train_dataset = self.TrainDataset(Train, data_dir, self.classes, self.train_tfm, state = self.state)
        valid_dataset = self.TrainDataset(Valid, data_dir, self.classes, self.train_tfm, state = self.state)
        train_data.extend(pd.DataFrame(Train).to_dict(orient='records'))
        valid_data.extend(pd.DataFrame(Valid).to_dict(orient='records'))

        pd.DataFrame(train_data).to_csv(f"{save_path}/Data/{condition}_train.csv", index=False)
        pd.DataFrame(valid_data).to_csv(f"{save_path}/Data/{condition}_valid.csv", index=False)
        print(f"training data number: {len(train_dataset)}, validation data number: {len(valid_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

        model = self.EfficientNetWithLinear(output_dim=self.class_num)
        model.to(device)
        modelName = f"{condition}_Model.ckpt"
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

        # data_iter = iter(train_loader)
        # images, labels , file_names= next(data_iter)
        # img_grid = torchvision.utils.make_grid(images)

        # plt.imshow(img_grid.permute(1, 2, 0))
        # plt.show()

        model = self.EfficientNetWithLinear(output_dim=self.class_num)
        model.to(device)
        modelName = f"{condition}_Model.ckpt"
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        
        for c in self.classes:
            model = self.EfficientNetWithLinear(output_dim=1)
            model.to(device)
            modelName = f"{condition}_{c}_Model.ckpt"
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

            self._train(model, modelName, criterion, optimizer, train_loader, val_loader, f"{save_path}/Model", f"{save_path}/Loss", target_class=self.classes.index(c))

    def train_generation(self):
        wsis = {"HCC": self.hcc_wsis, "CC": self.cc_wsis}.get(self.type)

        wsi = wsis[0]
        _wsi = wsi+91 if (self.state == "new" and self.type == "HCC") else wsi
        save_path = f'{self.save_dir}/{_wsi}/trial_{self.num_trial}'

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)

        for gen in range(2, self.generation):
            # condition = f"Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}"
            condition = f"Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}"
            print(condition)

            self.test_TATI(wsi, gen-1, save_path)
            self.build_pl_dataset(wsi, gen, save_path)
            
            # Read TI.csv, prepare Dataframe
            train_dataset, valid_dataset, _ = self.prepare_dataset(f"{save_path}/Data", condition, gen)

            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
        
            # Model setting and transfer learning or not
            model = self.EfficientNetWithLinear(output_dim = 2)
            model.to(device)
            modelName = f"{condition}_1WTC.ckpt"

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

            self._train(model, modelName, criterion, optimizer, train_loader, val_loader, condition, f"{save_path}/Model", f"{save_path}/Loss", target_class=None)

    def _test(self, test_dataset, data_info_df, model, save_path, condition, test_type):
        # Record Information
        Predictions = {"file_name": []}
        for class_name in self.classes:
            Predictions[f"{class_name}_pred"] = []

        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

        model.eval()
        Sigmoid = nn.Sigmoid()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                imgs, fname = batch
                logits = model(imgs.to(device))
                preds = Sigmoid(logits)
                
                # Inference
                Predictions["file_name"].extend(fname)
                for idx, class_name in enumerate(self.classes):
                    Predictions[f"{class_name}_pred"].extend(preds.cpu().numpy()[:, idx])

        pred_df = pd.DataFrame(Predictions)
        if test_type == "Metric":
            pred_df.to_csv(f"{save_path}/Metric/{condition}_pred_score.csv")
        elif test_type == "TI":
            pred_df.to_csv(f"{save_path}/TI/{condition}_patch_in_region_filter_2_v2_TI.csv", index=False)

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
            preds = []
            for cl in self.classes:
                preds.append(row[f'{cl}_pred'].values[0])
            pred = np.argmax(preds)

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
        if self.type == "HCC":
            _wsi = wsi + 91
            condition = f"{_wsi}_{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            save_path = f"{self.save_path}/{_wsi}/trial_{self.num_trial}"
        elif self.type == "CC":
            _wsi = f"1{wsi:04d}"
            condition = f"{_wsi}_{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            save_path = f"{self.save_path}/{wsi}/trial_{self.num_trial}"

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)
        
        # _, _, test_dataset = self.prepare_dataset(f"{save_path}/Data", condition, 0, "test")
        test_data = []
        if self.state == "old":
            _wsi = wsi
            data_dir = f'{self.hcc_old_data_dir}/{_wsi}'
            selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
        elif self.type == "HCC":
            _wsi = wsi + 91
            data_dir = f'{self.hcc_data_dir}/{wsi}'
            selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
        elif self.type == "CC":
            _wsi = f"1{wsi:04d}"
            data_dir = f'{self.cc_data_dir}/{wsi}'
            selected_data = pd.read_csv(f'{self.cc_csv_dir}/{wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
        
        _, _, Test = self.split_datas(selected_data, self.data_num)
        test_dataset = self.TestDataset(data_dir, Test, self.classes, self.test_tfm, state = self.state, label_exist=False)
        test_data.extend(pd.DataFrame(Test).to_dict(orient='records'))
        pd.DataFrame(test_data).to_csv(f"{save_path}/Data/{condition}_test.csv", index=False)

        print(f"testing data number: {len(test_dataset)}")
        
        # Prepare Model
        modelName = f"{condition}_Model.ckpt"
        model_path = f"{save_path}/Model/{modelName}"

        model = self.EfficientNetWithLinear(output_dim = 2)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        data_info_df = pd.read_csv(f"{save_path}/Data/{condition}_test.csv")

        self._test(test_dataset, data_info_df, model, save_path, condition, "Metric")
    
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

        model = self.EfficientNetWithLinear(output_dim = 2)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        data_info_df = pd.read_csv(f"{save_path}/Data/{condition}_test.csv")

        self._test(test_dataset, data_info_df, model, save_path, condition, "Metric")

    def test_TATI(self, wsi, gen, save_path):
        ### Multi-WTC Evaluation ###
        if save_path == None:
            if self.state == "old":
                _wsi = wsi
            elif self.type == "HCC":
                _wsi = wsi + 91
            elif self.type == "CC":
                _wsi = f"1{wsi:04d}"
                
        if self.gen_type:
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}/{_wsi}"
            if gen == 0:
                condition = f'{self.class_num}_class'
                model_path = self.file_paths['HCC_100WTC_model_path']
                model = EfficientNet.from_name('efficientnet-b0')
                model._fc= nn.Linear(1280, 2)
            else:
                # condition = f"Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}"
                condition = f"Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}"
                model_path = f"{save_path}/Model/{condition}_1WTC.ckpt"
                model = self.EfficientNetWithLinear(output_dim = 2)

        else:
            if self.test_model == "self":
                condition = f"{_wsi}_{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
                save_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/{wsi}/trial_{self.num_trial}"
                save_path = save_dir
            else:
                condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
                save_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}"
                save_path = f"{save_dir}/{_wsi}" 

            modelName = f"{condition}_Model.ckpt"
            model_path = f"{save_dir}/Model/{modelName}"
            model = self.EfficientNetWithLinear(output_dim = 2)

        os.makedirs(f"{save_path}/Model", exist_ok=True)
        os.makedirs(f"{save_path}/Metric", exist_ok=True)
        os.makedirs(f"{save_path}/Loss", exist_ok=True)
        os.makedirs(f"{save_path}/TI", exist_ok=True)
        os.makedirs(f"{save_path}/Data", exist_ok=True)

        print(f"WSI {wsi} | {condition}")
        print(self.classes) #class0 = Normal

        # Prepare Model
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        Sigmoid = nn.Sigmoid()

        # Dataset, Evaluation, Inference
        if self.state == "old":
            _wsi = wsi
            data_info_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_all_patches_filter_v2.csv')
            test_dataset = self.TestDataset(f'{self.hcc_old_data_dir}/{wsi}', data_info_df, self.classes, self.test_tfm, state='old', label_exist=False)
        elif self.type == "HCC":
            _wsi = wsi + 91
            data_info_df = pd.read_csv(f'{self.hcc_csv_dir}/{_wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset = self.TestDataset(f'{self.hcc_data_dir}/{wsi}', data_info_df, self.classes,self.test_tfm, state='new', label_exist=False)
        else:
            _wsi = f'1{wsi:04d}'
            data_info_df = pd.read_csv(f'{self.cc_csv_dir}/{wsi}/{_wsi}_patch_in_region_filter_2_v2.csv')
            test_dataset = self.TestDataset(f'{self.cc_data_dir}/{wsi}', data_info_df, self.classes,self.test_tfm, state='new', label_exist=False)
        
        _condition = f'{_wsi}_{condition}'
        self._test(test_dataset, data_info_df, model, save_path, _condition, "TI")

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

    def plot_TI_Result(self, wsi, gen, save_path):
        if save_path == None:
            _wsi = wsi+91 if (self.state == "new" and self.type == "HCC") else wsi
            __wsi = wsi if self.state == "old" else (wsi+91 if self.type == "HCC" else f"1{wsi:04d}")
        if self.gen_type:
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}/{_wsi}"
            if gen == 0:
                condition = f'{self.class_num}_class'
            else:
                condition = f"Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}"
        else:
            condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
            if self.test_model == "self":
                save_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}/{wsi}"
                save_path = save_dir
            else:
                save_dir = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}"
                save_path = f"{save_dir}/{__wsi}" 

        df = pd.read_csv(f"{save_path}/Metric/{__wsi}_{condition}_labels_predictions.csv")
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

            pred_label  = self.classes.index(df['pred_label'][idx])  # N=0, H=1
            gt_label = self.classes.index(df['true_label'][idx])

            x = (int(x)) // pts_ratio
            y = (int(y)) // pts_ratio

            if pred_label == 0 and gt_label == 0:
                label = 1
            elif pred_label == 1 and gt_label == 0:
                label = 2
            elif pred_label == 1 and gt_label == 1:
                label = 3
            elif pred_label == 0 and gt_label == 1:
                label = 4
            else:
                label = 0
            
            all_pts.append([x, y, label])

        all_pts = np.array(all_pts)

        x_max, y_max = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])

        image = np.zeros((y_max + 1, x_max + 1))
        for x, y, label in all_pts:
            image[y, x] = label
        
        legend_elements = [
            plt.Line2D([0], [0], color='green', lw=4, label='True Negative'),   # GT - Normal, Pred Normal
            plt.Line2D([0], [0], color='orange', lw=4, label='Flase Negative'),   # GT - Normal, Pred - HCC
            plt.Line2D([0], [0], color='red', lw=4, label='True Positive'),     # GT - HCC, Pred - HCC
            plt.Line2D([0], [0], color='blue', lw=4, label='False Positive'), # GT - HCC, Pred -Normal
        ]

        plt.figure(figsize=(x_max // 20, y_max // 20))
        plt.imshow(image == 1, cmap=ListedColormap(['white', 'green']), interpolation='nearest', alpha=0.5)
        plt.imshow(image == 3, cmap=ListedColormap(['white', 'red']), interpolation='nearest', alpha=0.5)
        plt.imshow(image == 2, cmap=ListedColormap(['white', 'orange']), interpolation='nearest', alpha=0.5)
        plt.imshow(image == 4, cmap=ListedColormap(['white', 'blue']), interpolation='nearest', alpha=0.5)
        
        # plt.imshow(image, cmap=cmap, interpolation='nearest')
        plt.title(f"Prediction vs Ground Truth of WSI {__wsi}", fontsize=20, pad=20)
        plt.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        plt.axis("off")

        # _wsi = wsi+91 if (self.state == "new" and self.type == "HCC") else wsi
        plt.savefig(f"{save_path}/Metric/{wsi}_pred_vs_gt.png")
        print(f"WSI {wsi} already plot the pred_vs_gt image")
        # plt.show()

    def plot_TI_Result_gt_boundary(self, wsi, gen, save_path):
        if save_path == None:
            _wsi = wsi+91 if (self.state == "new" and self.type == "HCC") else wsi
            __wsi = wsi if self.state == "old" else (wsi+91 if self.type == "HCC" else f"1{wsi:04d}")
        if self.gen_type:
            save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP_{self.data_num}/trial_{self.num_trial}/{_wsi}"
            if gen == 0:
                condition = f'{self.class_num}_class'
            else:
                condition = f"Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}"
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


