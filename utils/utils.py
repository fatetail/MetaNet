import os 
import cv2
import numpy as np

def get_all_image_mean_and_std(data_path):
    wk = os.walk(data_path)
    image_path_list = []
    for path, dir_list, file_list in wk:
        for file_name in file_list:
            if file_name.split('.')[-1] == 'jpg':
                image_path_list.append(os.path.join(path, file_name))
    R_mean = 0
    G_mean = 0
    B_mean = 0

    R_std = 0
    G_std = 0
    B_std = 0
    img_nums = 0
    for img_path in image_path_list:
        img = cv2.imread(img_path) / 255
        tmp_R_mean, tmp_R_std = cv2.meanStdDev(img[:, :, 2])
        tmp_G_mean, tmp_G_std = cv2.meanStdDev(img[:, :, 1])
        tmp_B_mean, tmp_B_std = cv2.meanStdDev(img[:, :, 0])
        R_mean += tmp_R_mean
        G_mean += tmp_G_mean
        B_mean += tmp_B_mean

        R_std += tmp_R_std
        G_std += tmp_G_std
        B_std += tmp_B_std
        img_nums += 1

    R_mean /= img_nums
    G_mean /= img_nums
    B_mean /= img_nums

    R_std /= img_nums
    G_std /= img_nums
    B_std /= img_nums
    print(R_mean)
    print('xxxx')
    print(G_mean)
    print('----')
    print(B_mean)
    print('11111')
    print(R_std)
    print('222')
    print(G_std)
    print('----')
    print(B_std)
get_all_image_mean_and_std('./../../Data/ISIC2019/ISIC2019_Train_Total_Data/')