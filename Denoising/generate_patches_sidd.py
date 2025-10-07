import cv2
import torch
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx
import argparse
import imageio.v2 as imageio

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='dataset path')
parser.add_argument('--dataset', type=str, default='SIDD', help='dataset')
args = parser.parse_args()


src = os.path.join(args.path, args.dataset)
tar = 'data2/nbuhn/Datasets/train/SIDD'
vtar = 'data2/nbuhn/Datasets/val/SIDD'
ttar = 'data2/nbuhn/Datasets/test/SIDD'


lr_tar = os.path.join(tar, args.dataset, 'raw')
hr_tar = os.path.join(tar, args.dataset, 'gt')

vallr_tar = os.path.join(vtar, args.dataset, 'raw')
valhr_tar = os.path.join(vtar, args.dataset, 'gt')

testlr_tar = os.path.join(ttar, args.dataset, 'raw')
testhr_tar = os.path.join(ttar, args.dataset, 'gt')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)
os.makedirs(vallr_tar, exist_ok=True)
os.makedirs(valhr_tar, exist_ok=True)
os.makedirs(testlr_tar, exist_ok=True)
os.makedirs(testhr_tar, exist_ok=True)
files = natsorted(glob(os.path.join(src, '*', '*.tif')))

lr_files, hr_files = [], []

for file_ in files:
    if 'gt' in file_:
        hr_files.append(file_)
    if 'raw' in file_:
        lr_files.append(file_)




files = [(i, j) for i, j in zip(lr_files, hr_files)]
nfiles = len(files)
ntrain = nfiles*9//10
train_files = files[:ntrain]
train_files = train_files[:-5]
val_files = train_files[-5:]
test_files = files[ntrain:]

print('test_files', len(test_files))
print('train_files', len(train_files))
print('val_files', len(val_files))




def save_files(file_):
    lr_file, hr_file = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = imageio.imread(lr_file)
    lr_img = np.asarray(lr_img)
    hr_img = imageio.imread(hr_file)
    hr_img = np.asarray(hr_img)
    print('hr_img_shape orig', hr_img.shape)
    print('hr_img_shape', hr_img.shape)
    lr_savename = os.path.join(lr_tar, filename + '.tif')
    hr_savename = os.path.join(hr_tar, filename + '.tif')
        
    imageio.imwrite(lr_savename, lr_img)
    imageio.imwrite(hr_savename, hr_img)

def save_val_files(file_):
    lr_file, hr_file = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = imageio.imread(lr_file)
    lr_img = np.asarray(lr_img)
    hr_img = imageio.imread(hr_file)
    hr_img = np.asarray(hr_img)
    print('hr_img_shape orig', hr_img.shape)
    print('hr_img_shape', hr_img.shape)
    lr_savename = os.path.join(vallr_tar, filename + '.tif')
    hr_savename = os.path.join(valhr_tar, filename + '.tif')
                
    imageio.imwrite(lr_savename, lr_img)
    imageio.imwrite(hr_savename, hr_img)
def save_test_files(file_):
    lr_file, hr_file = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = imageio.imread(lr_file)
    lr_img = np.asarray(lr_img)
    hr_img = imageio.imread(hr_file)
    hr_img = np.asarray(hr_img)
    print('hr_img_shape orig', hr_img.shape)
    print('hr_img_shape', hr_img.shape)
    lr_savename = os.path.join(testlr_tar, filename + '.tif')
    hr_savename = os.path.join(testhr_tar, filename + '.tif')
                
    imageio.imwrite(lr_savename, lr_img)
    imageio.imwrite(hr_savename, hr_img)


from joblib import Parallel, delayed
import multiprocessing
num_cores = 10
Parallel(n_jobs=num_cores)(delayed(save_files)(file_) for file_ in tqdm(train_files))
Parallel(n_jobs=num_cores)(delayed(save_val_files)(file_) for file_ in tqdm(val_files))
Parallel(n_jobs=num_cores)(delayed(save_test_files)(file_) for file_ in tqdm(test_files))