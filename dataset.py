from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.nn.functional import one_hot
import os
import pandas as pd

class CaptchaDataSet(Dataset):
    def __init__(self,img_dir,annotation_path,character_set,transform=None,transform_target=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotation_path)
        self.transform = transform
        self.character_set = character_set

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,self.img_labels.iloc[index,0])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)

        # 将字符串类型标签转为适合ctcloss的标签
        label = [self.character_set.find(c) for c in self.img_labels.iloc[index,1]]

        return image,label