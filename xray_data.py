import pandas
import torch
import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils import data

#  현재 경로 기준, 한 칸 위로 가서 Data 폴더 접근
DATA_PATH = '../Data/RSNA/'

def read_data():
    path = DATA_PATH
    sub_path = 'stage_2_train_images/'
    csv = pandas.read_csv(path + 'stage_2_detailed_class_info.csv')
    class_label = {'Normal': 0,
                   'No Lung Opacity / Not Normal': 1, 'Lung Opacity': 2}
    patient_dict = {}
    for i in range(csv.shape[0]):
        name = csv['patientId'][i]
        target = csv['class'][i]
        target = class_label[target]
        patient_dict[name] = target

    for file_name in tqdm(os.listdir(path + sub_path)):
        patient = file_name.split('.')[0]
        target = patient_dict.get(patient, None)
        if target is None:
            continue
        if target == 0:
            os.system(f'cp {path}{sub_path}{file_name} {path}normal/')
        elif target == 1:
            os.system(f'cp {path}{sub_path}{file_name} {path}not_normal/')
        elif target == 2:
            os.system(f'cp {path}{sub_path}{file_name} {path}lung_opacity/')


class Xray(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None):
        super(Xray, self).__init__()
        self.transform = transform if transform is not None else lambda x: x
        self.labels = []
        self.slices = []

        for label in os.listdir(main_path):
            if label not in ['0', '1']:
                continue
            for file_name in tqdm(os.listdir(os.path.join(main_path, label))):
                full_path = os.path.join(main_path, label, file_name)
                try:
                    if file_name.endswith('.dcm'):
                        data = sitk.ReadImage(full_path)
                        data = sitk.GetArrayFromImage(data).squeeze()
                        img = Image.fromarray(data).convert('L')
                    elif file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img = Image.open(full_path).convert('L')
                    else:
                        continue

                    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
                    self.slices.append(img)
                    self.labels.append(int(label))
                except Exception as e:
                    print(f"❌ 견찰 파일: {file_name} → {e}")
                    continue

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.slices)

def get_xray_dataloader(bs, workers, dtype='train', img_size=64, dataset='rsna'):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset == 'rsna':
        base_path = '../Data/RSNA'
    elif dataset == 'pedia':
        base_path = '../Data/Pediatric/CellData/chest_xray'

    if dtype == 'train':
        path = os.path.join(base_path, 'train')
    else:
        path = os.path.join(base_path, dtype)   # 여기서 test1, test2, test3 다 가능하게 수정

    dset = Xray(main_path=path, transform=transform, img_size=img_size)
    train_flag = True if dtype == 'train' else False
    dataloader = data.DataLoader(dset, bs, shuffle=train_flag,
                                 drop_last=train_flag, num_workers=workers, pin_memory=True)

    return dataloader
# def get_xray_dataloader(bs, workers, dtype='train', img_size=64, dataset='rsna'):
#     transform = transforms.Compose([
#         transforms.Resize(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])

#     if dataset == 'rsna':
#         path = '../Data/RSNA/'
#     elif dataset == 'pedia':
#         path = '../Data/Pediatric/CellData/chest_xray/'

#     path = os.path.join(path, dtype)  # train or test 폴더로 이동

#     dset = Xray(main_path=path, transform=transform, img_size=img_size)
#     train_flag = True if dtype == 'train' else False
#     dataloader = data.DataLoader(dset, bs, shuffle=train_flag,
#                                  drop_last=train_flag, num_workers=workers, pin_memory=True)

#     return dataloader


if __name__ == '__main__':
    read_data()