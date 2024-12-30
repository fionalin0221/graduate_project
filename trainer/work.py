import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import re
import random
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import torchvision
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import find_contours

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt

torch.manual_seed(0)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = "cuda" if torch.cuda.is_available() else "cpu"

class Worker():
    def __init__(self, config):
        current_computer = config['current_computer']
        self.type = config['type']
        self.file_paths = config['computers'][current_computer]['file_paths']
        self.state = config['state']
        class_list = config["class_list"]
        self.classes = [class_list[i] for i in self.file_paths['classes']]
        self.class_num = len(self.classes)

        self.gen = config['gen']
        self.generation = config["generation"]

        self.data_num = self.file_paths['data_num']
        self.num_trial = self.file_paths['num_trial']
        self.num_wsi = len(self.file_paths['HCC_wsis']) if self.type != "CC" else len(self.file_paths['CC_wsis'])    

        if self.gen:
            self.save_dir = self.file_paths[f'{self.type}_generation_save_path']
        else:
            self.save_dir = self.file_paths[f'{self.type}_WTC_result_save_path']
        
        self.save_path = f"{self.save_dir}/{self.num_wsi}WTC_Result/LP{self.data_num}"
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)

        self.hcc_wsis = self.file_paths['HCC_wsis']
        self.cc_wsis = self.file_paths['CC_wsis']
        self.hcc_data_dir = self.file_paths['HCC_patches_save_path']
        self.hcc_old_data_dir = self.file_paths['HCC_old_patches_save_path']
        self.cc_data_dir = self.file_paths['CC_patches_save_path']
        self.hcc_csv_dir = self.file_paths['HCC_csv_dir']
        self.cc_csv_dir = self.file_paths['CC_csv_dir']

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
        
        for num in range(len(self.classes)):
            if len(class_file_names[num]) > 0:
                if self.type == "Mix" and self.classes[num] == "N":
                    datas.append(random.sample(class_file_names[num], int(data_num/2)))
                else:
                    datas.append(random.sample(class_file_names[num], int(data_num)))
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

    def prepare_dataset(self, save_path, condition, gen):
        train_data = []
        valid_data = []
        test_data = []

        train_datasets = []
        valid_datasets = []
        test_datasets = []

        print(f"Patches use for a WSI: {self.data_num}")

        for h_wsi in self.hcc_wsis:
            if self.state == "old":
                if self.gen:
                    selected_data = pd.read_csv(f'{self.save_dir}/{h_wsi}/Data/{h_wsi}_Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}.csv')
                else:
                    selected_data = pd.read_csv(f'{self.hcc_csv_dir}/{h_wsi}/{h_wsi}_patch_in_region_filter_2_v2.csv')
                Train, Valid, Test = self.split_datas(selected_data, self.data_num)
                h_train_dataset = self.TrainDataset(Train, f'{self.hcc_old_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "old")
                h_valid_dataset = self.TrainDataset(Valid, f'{self.hcc_old_data_dir}/{h_wsi}', self.classes, self.train_tfm, state = "old")
                h_test_dataset  = self.TestDataset(f'{self.hcc_old_data_dir}/{h_wsi}',Test, self.classes, self.test_tfm, state = "old", label_exist=False)

            else:
                if self.gen:
                    selected_data = pd.read_csv(f'{self.save_dir}/{h_wsi+91}/Data/{h_wsi+91}_Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}.csv')
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
            if self.gen:
                selected_data = pd.read_csv(f'{self.save_dir}/{c_wsi}/Data/1{c_wsi:04d}_Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}.csv')
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

        pd.DataFrame(train_data).to_csv(f"{save_path}/{condition}_train.csv", index=False)
        pd.DataFrame(valid_data).to_csv(f"{save_path}/{condition}_valid.csv", index=False)
        pd.DataFrame(test_data).to_csv(f"{save_path}/{condition}_test.csv", index=False)

        return train_dataset, valid_dataset, test_dataset    
   
    def build_pl_dataset(self, wsi, gen):
        pts_ratio = 448
        num_thresh = 288
        _wsi = wsi+91 if (self.state == "new" and self.type == "HCC") else wsi

        if gen == 1:
            df = pd.read_csv(f"{self.save_dir}/{_wsi}/TI/trial_{self.num_trial}/{_wsi}_2_class_all_patches_filter_v2_TI.csv")
        else:
            df = pd.read_csv(f"{self.save_dir}/{_wsi}/TI/trial_{self.num_trial}/{_wsi}_Gen{gen-1}_LP_All_ideal_patches_by_Gen{gen-2}_all_patches_filter_v2_TI.csv")

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

        selected_patches = {'filename': [], 'label': []}

        cl = "H" if self.type == "HCC" else "C"
        selected_patches, patches_in_cancer_regions, tp_in_cancer_regions, fp_in_cancer_regions = \
            find_contours.find_contours(wsi, sorted_all_pts, self.state, cl, num_thresh, all_patches, selected_patches)

        selected_patches, patches_in_norm_regions, tn_in_norm_regions, fn_in_norm_regions = \
            find_contours.find_contours(wsi, sorted_all_pts, self.state, "N", num_thresh, all_patches, selected_patches)
        
        pos_num, neg_num = 0,0
        positive_cases = {"contour_key": [], "number": [], "density": []}
        for key in tp_in_cancer_regions.keys():
            num = tp_in_cancer_regions[key] + fp_in_cancer_regions[key]
            den = tp_in_cancer_regions[key] / (tp_in_cancer_regions[key] + fp_in_cancer_regions[key])
            positive_cases["contour_key"].append(key)
            positive_cases["number"].append(num)
            positive_cases["density"].append(den)
            pos_num += num
        pos_df = pd.DataFrame(positive_cases)

        negative_cases = {"contour_key": [], "number": [], "density": []}
        for key in tn_in_norm_regions.keys():
            num = tn_in_norm_regions[key] + fn_in_norm_regions[key]
            den = tn_in_norm_regions[key] / (tn_in_norm_regions[key] + fn_in_norm_regions[key])
            negative_cases["contour_key"].append(key)
            negative_cases["number"].append(num)
            negative_cases["density"].append(den)
            neg_num += num
        neg_df = pd.DataFrame(negative_cases)

        # num_filter_pos_df = pos_df[pos_df["number"] >= num_thresh]
        # num_filter_neg_df = neg_df[neg_df["number"] >= num_thresh]

        ideal_patches = {'filename': [], 'label': []}
        
        if pos_num >0:
            pl_cancer_contour_df = find_contours.ND_zscore_filter(contour_df=pos_df, weight=[1, 1])
            pl_cancer_contour_df.to_csv(f"{self.save_dir}/{_wsi}/Data/trial_{self.num_trial}/{_wsi}_Gen{gen}_tpfp_ND_zscore_filtered_contour_by_Gen{gen-1}.csv")
            # Filter keys where the sum of z-scores is greater than or equal to 0
            pl_cancer_filtered_keys = pl_cancer_contour_df[pl_cancer_contour_df['zscore_sum'] >= 0]['contour_key'].to_list()
            for pl_cancer_key in pl_cancer_filtered_keys:
                ideal_patches['filename'].extend(patches_in_cancer_regions[pl_cancer_key])
                ideal_patches['label'].extend([cl] * len(patches_in_cancer_regions[pl_cancer_key]))

        else:
            pl_cancer_contour_df = pos_df
            pl_cancer_contour_df.to_csv(f"{self.save_dir}/{_wsi}/Data/trial_{self.num_trial}/{_wsi}_Gen{gen}_tpfp_ND_zscore_filtered_contour_by_Gen{gen-1}.csv")
            print('NO cancer')

        if neg_num >0:
            pl_norm_contour_df = find_contours.ND_zscore_filter(contour_df=neg_df, weight=[1, 1])
            pl_norm_contour_df.to_csv(f"{self.save_dir}/{_wsi}/Data/trial_{self.num_trial}/{_wsi}_Gen{gen}_tnfn_ND_zscore_filtered_contour_by_Gen{gen-1}.csv")
            pl_norm_filtered_keys = pl_norm_contour_df[pl_norm_contour_df['zscore_sum'] >= 0]['contour_key'].to_list()
            for pl_norm_key in pl_norm_filtered_keys:
                ideal_patches['filename'].extend(patches_in_norm_regions[pl_norm_key])
                ideal_patches['label'].extend(['N'] * len(patches_in_norm_regions[pl_norm_key]))

        else:
            pl_norm_contour_df=neg_df
            pl_norm_contour_df.to_csv(f"{self.save_dir}/{_wsi}/Data/trial_{self.num_trial}/{_wsi}_Gen{gen}_tnfn_ND_zscore_filtered_contour_by_Gen{gen-1}.csv")
            print('NO Normal')
        
        pd.DataFrame(selected_patches).to_csv(f"{self.save_dir}/{_wsi}/Data/trial_{self.num_trial}/{_wsi}_Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}.csv", index=False)
        pd.DataFrame(ideal_patches).to_csv(f"{self.save_dir}/{_wsi}/Data/trial_{self.num_trial}/{_wsi}_Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}.csv", index=False)

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

    def _train(self, model, modelName, criterion, optimizer, train_loader, val_loader, condition, model_save_path, loss_save_path, target_class):
        n_epochs = 100
        notImprove = 0
        min_loss = 1000.

        train_loss_list = []
        train_acc_list = []
        valid_loss_list = []
        valid_acc_list = []

        for epoch in range(1, n_epochs):
            # ---------- Training ----------
            # Make sure the model is in train mode before training.
            model.train()

            # These are used to record information in training.
            train_loss = []
            train_acc = []
            train_bar = tqdm(train_loader)
            for batch in train_bar:
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

            train_loss_list.append(train_avg_loss)
            train_acc_list.append(train_avg_acc)
            print(f"[ Train | {epoch:03d}/{n_epochs:03d} ] loss = {train_avg_loss:.5f}, acc = {train_avg_acc:.5f}")
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
                for batch in valid_bar:
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

            valid_loss_list.append(valid_avg_loss)
            valid_acc_list.append(valid_avg_acc)
            torch.cuda.empty_cache()

            if valid_avg_loss < min_loss:
                # Save model if your model improved
                min_loss = valid_avg_loss
                torch.save(model.state_dict(), f"{model_save_path}/{modelName}")
                notImprove = 0
            else:
                notImprove = notImprove + 1

            # Print the information.
            print(f"[ Valid | {epoch:03d}/{n_epochs:03d} ] loss = {valid_avg_loss:.5f}, acc = {valid_avg_acc:.5f}")
            if epoch == 2:
            #     notImprove = 0
            # if notImprove >= 2 and epoch >= 2:
                break 

        train_loss_arr = np.array(train_loss_list).reshape(1,len(train_loss_list))
        valid_loss_arr = np.array(valid_loss_list).reshape(1,len(train_loss_list))
        train_acc_arr = np.array(train_acc_list).reshape(1,len(train_loss_list))
        valid_acc_arr = np.array(valid_acc_list).reshape(1,len(train_loss_list))

        training_log = np.append(train_loss_arr,valid_loss_arr,axis=0)
        training_log = np.append(training_log,train_acc_arr,axis=0)
        training_log = np.append(training_log,valid_acc_arr,axis=0)
        
        np.savetxt(f"{loss_save_path}/{condition}_train_log.csv", training_log, delimiter=",")

    def train(self):
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

        train_dataset, valid_dataset, _ = self.prepare_dataset(f"{save_path}/Data", condition, 0)
        
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

        for gen in range(self.generation):
            condition = f"Gen{gen}_LP_All_ideal_patches_by_Gen{gen-1}"
            print(condition)

            self.test_TATI(wsi, gen, save_path)
            self.build_pl_dataset(wsi, gen)
            
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



    def test(self):
        condition = f"{self.num_wsi}WTC_LP{self.data_num}_{self.class_num}_class_trial_{self.num_trial}"
        _, _, test_dataset = self.prepare_dataset()
        
        modelName = f"{condition}_Model.ckpt"
        model_path = f"{self.save_path}/{modelName}"

        # Prepare Model
        model = self.EfficientNetWithLinear()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        
        Sigmoid = nn.Sigmoid()

        # Record Information
        Predictions = {"file_name": []}
        for class_name in self.classes:
            Predictions[f"{class_name}_pred"] = []

        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

        model.eval()
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
        pred_df.to_csv(f"{self.save_path}/{self.condition}_pred_score.csv")

        data_info_df = pd.read_csv(f"{self.save_path}/{self.condition}_test.csv")

        all_labels, all_preds = [], []
        match_df  = data_info_df[data_info_df['file_name'].isin(pred_df['file_name'])]
        filename_inRegion = match_df['file_name'].to_list()
        label_inRegion = match_df['label'].to_list()

        for idx, filename in enumerate(tqdm(filename_inRegion)):
            label = self.classes.index(label_inRegion[idx])

            row = pred_df[pred_df['file_name'] == filename]
            preds = []
            for cl in self.classes:
                preds.append(row[f'{cl}_pred'].values[0])
            pred = np.argmax(preds)

            all_labels.append(label)
            all_preds.append(pred)
            
        acc = accuracy_score(all_labels, all_preds)
        print("Accuracy: {:.4f}".format(acc))

        cm = confusion_matrix(all_labels, all_preds, labels=range(len(self.classes)))

        recall_per_class = []
        precision_per_class = []
        f1_per_class = []
        for i, class_name in enumerate(self.classes):
            TP = cm[i, i]  # True Positives for class i
            FN = cm[i, :].sum() - TP  # False Negatives for class i
            FP = cm[:, i].sum() - TP  # False Positives for class i
            
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0  # Avoid division by zero
            recall_per_class.append(recall)

            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            precision_per_class.append(precision)

            # Calculate F1 Score
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            f1_per_class.append(f1)

        # Calculate Macro Metrics
        macro_recall = np.mean(recall_per_class)
        macro_precision = np.mean(precision_per_class)
        macro_f1 = np.mean(f1_per_class)

        print("Macro Recall: {:.4f}".format(macro_recall))
        print("Macro Precision: {:.4f}".format(macro_precision))
        print("Macro F1 Score: {:.4f}".format(macro_f1))

        Test_Acc = {
            "Condition": [],
            "Accuracy": [],
            "Macro_Recall": [],
            "Macro_Precision": [],
            "Macro_F1": [],
        }

        # Dynamically add entries for precision, recall, and F1 for each class
        for i, class_name in enumerate(self.classes):
            Test_Acc[f"{class_name}_Recall"] = []
            Test_Acc[f"{class_name}_Precision"] = []
            Test_Acc[f"{class_name}_F1"] = []

        # Example of populating the dictionary
        Test_Acc["Condition"].append(self.condition)
        Test_Acc["Accuracy"].append(acc)
        Test_Acc["Macro_Recall"].append(macro_recall)
        Test_Acc["Macro_Precision"].append(macro_precision)
        Test_Acc["Macro_F1"].append(macro_f1)

        for i, class_name in enumerate(self.classes):
            Test_Acc[f"{class_name}_Recall"].append(recall_per_class[i])
            Test_Acc[f"{class_name}_Precision"].append(precision_per_class[i])
            Test_Acc[f"{class_name}_F1"].append(f1_per_class[i])

        # Save to CSV
        pd.DataFrame(Test_Acc).to_csv(f"{self.save_path}/{self.condition}_test_acc_all.csv", index=False)

    def test_1WSI(self):
        wsis = {"HCC": self.hcc_wsis, "CC": self.cc_wsis}.get(self.type)
        data_dir = (
            self.hcc_old_data_dir if (self.type == "HCC" and self.state == "old")
            else self.hcc_data_dir if (self.type == "HCC")
            else self.cc_data_dir
        )
        
        # for wsi in wsis:
        #     modelName = f"{self.condition}_Model.ckpt"
        #     model_path = f"{self.save_path}/{modelName}"

        #     # Prepare Model
        #     model = self.EfficientNetWithLinear()
        #     model.load_state_dict(torch.load(model_path))
        #     model.to(device)
        #     Sigmoid = nn.Sigmoid()

        #     # Record Information
        #     Predictions = {"file_name": []}
        #     for class_name in self.classes:
        #         Predictions[f"{class_name}_pred"] = []

        #     Test = pd.read_csv(f"{self.save_dir}/{wsi}/Data/trial_{self.num_trial}/{self.condition}_test.csv")
            
        #     test_dataset = self.TestDataset(f'{data_dir}/{wsi}',Test, self.test_tfm, label_exist=False)
        #     test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

        #     with torch.no_grad():
        #         for batch in tqdm(test_loader):
        #             imgs, fname = batch
        #             logits = model(imgs.to(device))
        #             preds = Sigmoid(logits)
                    
        #             # Inference
        #             Predictions["file_name"].extend(fname)
        #             for idx, class_name in enumerate(self.classes):
        #                 Predictions[f"{class_name}_pred"].extend(preds.cpu().numpy()[:, idx])

        #     pred_df = pd.DataFrame(Predictions)

        #     all_labels, all_preds = [], []
        #     match_df  = data_info_df[data_info_df['file_name'].isin(pred_df['file_name'])]
        #     filename_inRegion = match_df['file_name'].to_list()
        #     label_inRegion = match_df['label'].to_list()

        #     for idx, filename in enumerate(tqdm(filename_inRegion)):
        #         label = classes.index(label_inRegion[idx])

        #         row = pred_df[pred_df['file_name'] == filename]
        #         preds = []
        #         for cl in classes:
        #             preds.append(row[f'{cl}_pred'].values[0])
        #         pred = np.argmax(preds)

        #         all_labels.append(label)
        #         all_preds.append(pred)
                
        #     acc = accuracy_score(all_labels, all_preds)
        #     print("Accuracy: {:.4f}".format(acc))

        #     Test_Acc = {"WSI": [], "Accuracy": []}
        #     if type == "HCC":
        #         Test_Acc["WSI"].append(wsi+91)
        #     elif type == "CC":
        #         Test_Acc["WSI"].append(10000+wsi)
        #     Test_Acc["Accuracy"].append(acc)

        #     pd.DataFrame(Test_Acc).to_csv(f"{save_dir}/{wsi}/Loss/trial_{num_trial}/{condition}_test_acc.csv", index=False)

    def test_TATI(self, wsi, gen, save_path):
        ### Multi-WTC Evaluation ###
        if gen == 0:
            condition = '100WTC_DB_8_1_1_Model'
            model_path = self.file_paths['100WTC_model_path']
            model = EfficientNet.from_name('efficientnet-b0')
            model._fc= nn.Linear(1280, 2)
        else:
            condition = f"Gen{gen}_LP_All_ideal_patches_by_Gen{gen-1}"
            model_path = f"{self.save_dir}/{wsi}/Model/trial_{self.num_trial}/{condition}_1WTC.ckpt"
            model = self.EfficientNetWithLinear(output_dim = 2)

        print(f"WSI {wsi} | {condition}")

        # Prepare Model
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        Sigmoid = nn.Sigmoid()
        print("debug")

        # Record Information
        Predictions = {"file_name": []}
        for cl in self.classes:
            Predictions[f"{cl}_pred"] = []

        # Dataset, Evaluation, Inference
        if self.state == "old":
            test_set = self.TestDataset(f'{self.hcc_old_data_dir}/{wsi}', f'{self.hcc_csv_dir}/{wsi}/{wsi}_all_patches_filter_v2.csv', self.test_tfm, state='old', label_exist=False)
        elif self.type == "HCC":
            test_set = self.TestDataset(f'{self.hcc_data_dir}/{wsi}', f'{self.hcc_csv_dir}/{wsi+91}/{wsi+91}_all_patches_filter_v2.csv', self.test_tfm, state='new', label_exist=False)
        else:
            test_set = self.TestDataset(f'{self.cc_data_dir}/{wsi}', f'{self.cc_csv_dir}/{wsi}/{wsi}_all_patches_filter_v2.csv', self.test_tfm, state='new', label_exist=False)
        
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                imgs, fname = batch
                logits = model(imgs.to(device))
                preds = Sigmoid(logits)
                
                # Inference
                Predictions["file_name"].extend(fname)
                Predictions[f"{self.classes[0]}_pred"].extend(preds.cpu().numpy()[:, 0])
                Predictions[f"{self.classes[1]}_pred"].extend(preds.cpu().numpy()[:, 1])
        _wsi = wsi+91 if (self.state == "new" and self.type == "HCC") else wsi
        pd.DataFrame(Predictions).to_csv(f"{save_path}/{_wsi}_{condition}_all_patches_filter_v2_TI.csv", index=False)