import torch
import collections
import os
from xml.etree.ElementTree import Element as ET_Element

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import  verify_str_arg
from torchvision import datasets

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from typing import Any, Callable, Dict, Optional, Tuple, List

from PIL import Image

from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import transforms as T
from trainner.utils import collate_fn
from utils import tensor_to_numpy_cv2 , re_labeling
import cv2
import numpy as np


font = cv2.FONT_HERSHEY_SIMPLEX
# Dictionary containing some colors
colors = {'blue': (0, 0, 255), 'green': (0, 255, 0), 'red': (255, 0, 0), 'orange': (223, 70, 14),
          'yellow': (255, 255, 0),
          'magenta': (255, 0, 255), 'cyan': (0, 255, 255), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220), 'red 1': (255, 82, 82),
          'red 2': (255, 82, 82), 'red 3': (255, 82, 82)}
# labels_dict = ['targetobject', 'hand']

# no contact 0 self=1, other person =2 portable object=3 , non portable =4
#new labeling Hand_free_R =1 ,Hand_free_L=2,Hand_cont_R=3,Hand_cont_L=4  ,person_=5 ,person_L=6 ,person_LR=7,portable_R =8 ,portable_L=9, portable_LR=10
# labels_dict = ['Hand_free_R','Hand_free_L','Hand_cont_R','Hand_cont_L' ,'person_R','person_L' ,'person_LR','portable_R' ,'portable_L', 'portable_LR']
# labels_dict={'Hand_free_R':1,'Hand_free_L':2,'Hand_cont_R':3,'Hand_cont_L':4 ,
#              'person_R':5,'person_L':6 ,'person_LR':7,'portable_R':8 ,
#              'portable_L':9, 'portable_LR':10,'non-portable_R':11,'non-portable_L':12,'non-portable_LR':13}

labels_dict={'Hand_free':1,'Hand_cont':2,'object':3,'person':4}

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



class _VOCBase(VisionDataset):
    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str

    def __init__(
        self,
        root: str,
        image_set: str = "train",


        transforms: Optional[Callable] = None,
    ):

        super().__init__(root,transforms)


        valid_image_sets = ["train", "trainval", "val"]

        self.image_set = verify_str_arg(image_set, "image_set", valid_image_sets)

        voc_root=self.root
        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, self._TARGET_DIR)
        self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]

        assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)

class VOCDetection(_VOCBase):

    _SPLITS_DIR = "Main"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"

    @property
    def annotations(self) -> List[str]:
        return self.targets


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target_dict = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

        boxes, labels=re_labeling(target_dict)

        image_id = torch.tensor([index])
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # target["dict"]=target_dict
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOCDetection.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def image_folder_loader(image_dir,batch_size):
    transofmer=transforms.Compose([transforms.ToTensor()])
    data = datasets.ImageFolder(root=image_dir, transform=transofmer)
    image_loader = torch.utils.data.DataLoader(data,batch_size=batch_size,num_workers=4)
    return image_loader


def dataloader(batch_size,data_path):

    train_dataset = VOCDetection(root=data_path, image_set='train', transforms=get_transform(train=True))
    #train with some of the dataset
    # start_point=7992
    # end_point=23976
    start_point = 0
    end_point = 79921
    print('start data',start_point,'end at',end_point,'total trainning set',len(train_dataset))
    train_subset=list(range(start_point,end_point))
    train_subset=torch.utils.data.Subset(train_dataset,train_subset)

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                            shuffle=True, num_workers=4,collate_fn=collate_fn)
    val_dataset =VOCDetection(root=data_path, image_set='val',transforms=get_transform(train=True))
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4,collate_fn=collate_fn)
    print('validation set length', len(val_dataset))
    trainval_dataset = VOCDetection(root=data_path, image_set='trainval', transforms=get_transform(train=True))

    trainval_loader = DataLoader(trainval_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4,collate_fn=collate_fn)
    return train_loader,trainval_loader ,val_loader