import xml.etree.ElementTree as ET
import numpy as np
import os
from shutil import copy
from tqdm import tqdm
import pandas as pd
import matplotlib.path as mpath
import yaml

def Ray_Segment(point, s_point, e_point):
    if s_point[1] == e_point[1]:
        return False
    if s_point[1] > point[1] and e_point[1] > point[1]:
        return False
    if s_point[1] < point[1] and e_point[1] < point[1]:
        return False
    if s_point[1] == point[1] and e_point[1] > point[1]:
        return False
    if e_point[1] == point[1] and s_point[1] > point[1]:
        return False
    if s_point[0] < point[0] and e_point[1] < point[1]:
        return False
    
    xseg = e_point[0]-(e_point[0]-s_point[0])*(e_point[1]-point[1])/(e_point[1]-s_point[1])
    
    if xseg < point[0]:
        return False
    
    return True


def Point_in_Region(point, region):
    xmin = np.min(region[:, 0])
    ymin = np.min(region[:, 1])
    xmax = np.max(region[:, 0])
    ymax = np.max(region[:, 1])
    if (point[0] > xmax or point[0] < xmin or point[1] > ymax or point[1] < ymin):
        return False
    
    count = 0
    for i in range(len(region)-1):
        if Ray_Segment(point, region[i], region[i+1]) == True:
            count += 1

    if count % 2 == 1:
        return True
    else:
        return False

config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'config.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
current_computer = config['current_computer']
type = config['type']
file_paths = config['computers'][current_computer]['type'][type]['file_paths']
class_list = config["class_list"]
classes = [class_list[i] for i in file_paths['classes']]
print(classes)

wsis = file_paths['wsis']
for wsi in wsis:
    print(f'{type} WSI : ', wsi)
    if type == "HCC":
        xml_name = "LIVER_{:05d}.xml".format(wsi)
        tree = ET.parse(os.path.join(file_paths['ndpi_path'], xml_name), parser=ET.XMLParser(encoding="utf-8"))
    elif type == "CC":
        xml_name = "LIVER_1{:04d}.xml".format(wsi)
        tree = ET.parse(os.path.join(file_paths['ndpi_path'], xml_name), parser=ET.XMLParser(encoding="utf-8"))
    print(xml_name)
    root = tree.getroot()

    all_region = {}
    for child in root.iter():
        # print(child.tag, child.attrib)
        if child.tag == 'Region':
            label_id = child.attrib['Text'] + child.attrib['Id']
            all_region[label_id] = []
        if child.tag == 'Vertex':
            x = int(float(child.attrib['X']))
            y = int(float(child.attrib['Y']))
            all_region[label_id].append((x, y))
    if type == "HCC":
        csv_dir = os.path.join(file_paths['csv_dir'],f"{wsi+91}")
    elif type == "CC":
        csv_dir = os.path.join(file_paths['csv_dir'],f"{wsi}")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    data_info = {}
    data_info['file_name'] = []
    data_info['label'] = []
    nums = [0] * len(classes)

    patches_path = os.path.join(file_paths['patches_save_path'],f"{wsi}")
    dir = os.listdir(patches_path)

    for f in tqdm(dir):
        fx , fy = f.split('_')
        fy = fy[:-4]
        left_up = [int(fx), int(fy)]
        right_up = [int(fx) + 448, int(fy)]
        left_down = [int(fx), int(fy) + 448]
        right_down = [int(fx) + 448, int(fy) + 448]

        for k, v in all_region.items():
            if k[0] in classes:
                region = np.array(v)
                if ((Point_in_Region(left_up, region)==True) and (Point_in_Region(right_up, region)==True) 
                    and (Point_in_Region(left_down, region)==True) and (Point_in_Region(right_down, region)==True)):
                    data_info['file_name'].append(f)
                    data_info['label'].append(k[0])

                    nums[classes.index(k[0])] += 1

    df = pd.DataFrame(data_info)
    if type == "HCC":
        df.to_csv(os.path.join(csv_dir, f'{wsi+91}_patch_in_region_filter_{len(classes)}_v2.csv'), index=False)
    elif type == "CC":
        df.to_csv(os.path.join(csv_dir, f'1{wsi:04d}_patch_in_region_filter_{len(classes)}_v2.csv'), index=False)
    print(nums[0], nums[1])
