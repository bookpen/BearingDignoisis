import os
from PIL import Image
from torch.utils.data import Dataset
from config import label

class ImageData(Dataset):
    def __init__(self,file_path,transform=None):
        super(ImageData, self).__init__()
        self.transform = transform
        self.data = []
        self.label = []
        self.file_path = file_path
        for i in os.listdir(file_path):
            for jpg in os.listdir(os.path.join(file_path,i)):
                datapath = os.path.join(file_path, i,jpg)
                cache = Image.open(datapath)
                self.data.append(cache.copy())
                if "g" in jpg.split(".")[0]:
                    self.label.append(0)
                if "b" in jpg.split(".")[0]:
                    self.label.append(label[i])
                cache.close()

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform:
            image = self.transform(image)

        return image,self.label[index]

    def __len__(self):
        return len(self.data)