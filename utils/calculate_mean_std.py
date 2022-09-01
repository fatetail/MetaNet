import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import PIL.Image as Image
from torchvision import transforms, utils


image_dir = './ISIC2019_Train_Total_Data/'
image_list = []
for file in os.listdir(image_dir):
    image_list.append(os.path.join(image_dir, file))


class MyDataset(Dataset):
    def __init__(self):
        self.data = image_list

    def __getitem__(self, index):
        x = Image.open(self.data[index])
        x = transforms.ToTensor()(x)
        return x

    def __len__(self):
        return len(self.data)


def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data in loader:
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


dataset = MyDataset()
loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=10,
    shuffle=False
)

mean, std = online_mean_and_sd(loader)
