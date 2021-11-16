
import os
import imghdr
import torch.utils.data as data
from PIL import Image


class ImageDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, recursive_search=False, nc=3):
        super(ImageDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.transform = transform
        self.imgpaths = self.__load_imgpaths_from_dir(self.data_dir, walk=recursive_search)
        self.nc = nc

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, index):
        img = Image.open(self.imgpaths[index])

        if self.nc == 3:
            img = img.convert('RGB')
        elif self.nc == 1:
            pass
        else:
            raise ValueError('nc should be 1 or 3')

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __is_imgfile(self, filepath):
        filepath = os.path.expanduser(filepath)
        if os.path.isfile(filepath) and imghdr.what(filepath):
            return True
        return False

    def __load_imgpaths_from_dir(self, dirpath, walk=False):
        imgpaths = []
        dirpath = os.path.expanduser(dirpath)
        if walk:
            for (root, _, files) in os.walk(dirpath):
                for file in files:
                    file = os.path.join(root, file)
                    if self.__is_imgfile(file):
                        imgpaths.append(file)
        else:
            for path in os.listdir(dirpath):
                path = os.path.join(dirpath, path)
                if not self.__is_imgfile(path):
                    continue
                imgpaths.append(path)
        return imgpaths
