import torch,os,random,glob,math
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader

class my_dataset(Dataset):
    def __init__(self, rootA_in, rootA_label, rootA_imap, rootA_rmap, rootB_in, rootB_label, rootB_imap, rootB_rmap, rootC_in, rootC_label, rootC_imap, rootC_rmap, crop_size=256,
                 fix_sample_A=500, fix_sample_B=500, fix_sample_C=500, regular_aug =False):
        super(my_dataset,self).__init__()

        self.regular_aug = regular_aug  
        self.fix_sample_A = fix_sample_A 

        in_files_A = os.listdir(rootA_in)
        if self.fix_sample_A > len(in_files_A):
            self.fix_sample_A = len(in_files_A)
        in_files_A = random.sample(in_files_A, self.fix_sample_A)
        self.imgs_in_A = [os.path.join(rootA_in, k) for k in in_files_A]
        self.imgs_imap_A = [os.path.join(rootA_imap, k) for k in in_files_A]
        self.imgs_rmap_A = [os.path.join(rootA_rmap, k) for k in in_files_A]
        self.imgs_gt_A = [os.path.join(rootA_label, k) for k in in_files_A]

        len_imgs_in_A = len(self.imgs_in_A)
        self.length = len_imgs_in_A
        self.r_l_rate = 1 
        self.r_l_rate1 = 1  

        in_files_B = os.listdir(rootB_in)
        self.fix_sample_B = fix_sample_B
        if self.fix_sample_B >len(in_files_B):
            self.fix_sample_B = len(in_files_B)
        in_files_B = random.sample(in_files_B, self.fix_sample_B)
        self.imgs_in_B = [os.path.join(rootB_in, k) for k in in_files_B]
        self.imgs_imap_B = [os.path.join(rootB_imap, k) for k in in_files_B]
        self.imgs_rmap_B = [os.path.join(rootB_rmap, k) for k in in_files_B]
        self.imgs_gt_B = [os.path.join(rootB_label, k) for k in in_files_B]  

        len_imgs_in_B_ori = len(self.imgs_in_B )
        self.imgs_in_B = self.imgs_in_B * (self.r_l_rate + math.ceil(len_imgs_in_A / len_imgs_in_B_ori))  
        self.imgs_in_B = self.imgs_in_B[0: self.r_l_rate * len_imgs_in_A]
        self.imgs_imap_B = self.imgs_imap_B * (self.r_l_rate + math.ceil(len_imgs_in_A / len_imgs_in_B_ori))
        self.imgs_imap_B = self.imgs_imap_B[0: self.r_l_rate * len_imgs_in_A]
        self.imgs_rmap_B = self.imgs_rmap_B * (self.r_l_rate + math.ceil(len_imgs_in_A / len_imgs_in_B_ori))
        self.imgs_rmap_B = self.imgs_rmap_B[0: self.r_l_rate * len_imgs_in_A]
        self.imgs_gt_B = self.imgs_gt_B * (self.r_l_rate + math.ceil(len_imgs_in_A / len_imgs_in_B_ori))
        self.imgs_gt_B = self.imgs_gt_B[0: self.r_l_rate * len_imgs_in_A]

        in_files_C = os.listdir(rootC_in)
        self.fix_sample_C = fix_sample_C
        if self.fix_sample_C > len(in_files_C):
            self.fix_sample_C = len(in_files_C)
        in_files_C = random.sample(in_files_C, self.fix_sample_C)
        self.imgs_in_C = [os.path.join(rootC_in, k) for k in in_files_C]
        self.imgs_imap_C = [os.path.join(rootC_imap, k) for k in in_files_C]
        self.imgs_rmap_C = [os.path.join(rootC_rmap, k) for k in in_files_C]
        self.imgs_gt_C = [os.path.join(rootC_label, k) for k in in_files_C]  

        len_imgs_in_C_ori = len(self.imgs_in_C) 
        self.imgs_in_C = self.imgs_in_C * (self.r_l_rate1 + math.ceil(len_imgs_in_A / len_imgs_in_C_ori))  
        self.imgs_in_C = self.imgs_in_C[0: self.r_l_rate1 * len_imgs_in_A]
        self.imgs_imap_C = self.imgs_imap_C * (self.r_l_rate + math.ceil(len_imgs_in_A / len_imgs_in_C_ori))
        self.imgs_imap_C = self.imgs_imap_C[0: self.r_l_rate * len_imgs_in_A]
        self.imgs_rmap_C = self.imgs_rmap_C * (self.r_l_rate + math.ceil(len_imgs_in_A / len_imgs_in_C_ori))
        self.imgs_rmap_C = self.imgs_rmap_C[0: self.r_l_rate * len_imgs_in_A]
        self.imgs_gt_C = self.imgs_gt_C * (self.r_l_rate1 + math.ceil(len_imgs_in_A / len_imgs_in_C_ori))
        self.imgs_gt_C = self.imgs_gt_C[0: self.r_l_rate1 * len_imgs_in_A]

        self.crop_size = crop_size

    def __getitem__(self, index):
        data_IN_A, data_GT_A, data_IMAP_A, data_RMAP_A, img_name_A = self.read_imgs_pair(self.imgs_in_A[index], self.imgs_gt_A[index], self.imgs_imap_A[index], self.imgs_rmap_A[index], self.train_transform, self.crop_size)
        data_IN_B, data_GT_B, data_IMAP_B, data_RMAP_B, img_name_B = self.read_imgs_pair(self.imgs_in_B[index], self.imgs_gt_B[index], self.imgs_imap_B[index], self.imgs_rmap_B[index], self.train_transform, self.crop_size)
        data_IN_C, data_GT_C, data_IMAP_C, data_RMAP_C, img_name_C = self.read_imgs_pair(self.imgs_in_C[index], self.imgs_gt_C[index], self.imgs_imap_C[index], self.imgs_rmap_C[index], self.train_transform, self.crop_size)

        data_A = [data_IN_A, data_GT_A, data_IMAP_A, data_RMAP_A, img_name_A]
        data_B = [data_IN_B, data_GT_B, data_IMAP_B, data_RMAP_B, img_name_B]
        data_C = [data_IN_C, data_GT_C, data_IMAP_C, data_RMAP_C, img_name_C]
        return data_A, data_B, data_C

    def read_imgs_pair(self, in_path, gt_path, imap_path, rmap_path, transform, crop_size):
        in_img_path_A = in_path  
        img_name_A = in_img_path_A.split('/')[-1]

        in_img_A = np.array(Image.open(in_img_path_A))
        gt_img_path_A = gt_path  
        imap_img_path_A = imap_path
        imap_img_A = np.array(Image.open(imap_img_path_A))

        rmap_img_path_A = rmap_path
        rmap_img_A = np.array(Image.open(rmap_img_path_A))

        gt_img_A = np.array(Image.open(gt_img_path_A))
        data_IN_A, data_GT_A, data_IMAP_A, data_RMAP_A = transform(in_img_A, gt_img_A, imap_img_A, rmap_img_A, crop_size)

        return data_IN_A, data_GT_A, data_IMAP_A, data_RMAP_A, img_name_A

    def train_transform(self, img, label, imap, rmap, patch_size=256):
        ih, iw,_ = img.shape

        patch_size = patch_size
        ix = random.randrange(0, max(0, iw - patch_size))
        iy = random.randrange(0, max(0, ih - patch_size))
        img = img[iy:iy + patch_size, ix: ix + patch_size]
        imap = imap[iy:iy + patch_size, ix: ix + patch_size]
        rmap = rmap[iy:iy + patch_size, ix: ix + patch_size]
        label = label[iy:iy + patch_size, ix: ix + patch_size]

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img = transform(img)
        imap = transform(imap)
        rmap = transform(rmap)
        label = transform(label)

        return img, label, imap, rmap

    def __len__(self):
        return len(self.imgs_in_A)

class my_dataset_eval(Dataset):
    def __init__(self, root_in, root_label, root_imap, root_rmap, transform=None, fix_sample=100):
        super(my_dataset_eval, self).__init__()

        self.fix_sample = fix_sample
        in_files = os.listdir(root_in)
        if self.fix_sample > len(in_files):
            self.fix_sample = len(in_files)
        in_files = random.sample(in_files, self.fix_sample)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        self.imgs_imap = [os.path.join(root_imap, k) for k in in_files]
        self.imgs_rmap = [os.path.join(root_rmap, k) for k in in_files]
        self.imgs_gt = [os.path.join(root_label, k) for k in in_files]

        self.transform = transform

    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        img_name =in_img_path.split('/')[-1]

        in_img = Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]

        imap_img_path = self.imgs_imap[index]
        imap_img = Image.open(imap_img_path)

        rmap_img_path = self.imgs_rmap[index]
        rmap_img = Image.open(rmap_img_path)

        gt_img = Image.open(gt_img_path)
        trans_eval = transforms.Compose(
            [
                transforms.ToTensor()
            ])

        data_IN = trans_eval(in_img)
        data_GT = trans_eval(gt_img)
        data_IMAP = self.transform(imap_img)
        data_RMAP = self.transform(rmap_img)

        _, h, w = data_GT.shape
        if (h % 16 != 0) or (w % 16 != 0):
            data_GT = transforms.Resize(((h // 16) * 16, (w // 16) * 16))(data_GT)
            data_IMAP = transforms.Resize(((h // 16) * 16, (w // 16) * 16))(data_IMAP)
            data_RMAP = transforms.Resize(((h // 16) * 16, (w // 16) * 16))(data_RMAP)
            data_IN = transforms.Resize(((h // 16) * 16, (w // 16) * 16))(data_IN)

        return data_IN, data_GT, data_IMAP, data_RMAP, img_name

    def __len__(self):
        return len(self.imgs_in)

class DatasetForInference(Dataset):
    def __init__(self, dir_path):
        self.image_paths = glob.glob(os.path.join(dir_path, '*') )
        self.transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
        ]) 
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        input_path = self.image_paths[index]
        input_image = Image.open(input_path).convert('RGB')
        input_image = self.transform(input_image)
        _, h, w = input_image.shape
        if (h % 16 != 0) or (w % 16 != 0):
            input_image = transforms.Resize(((h//16)*16, (w//16)*16))(input_image)
        return input_image 