import transforms as T
import  json
import argparse
from utils_local import tensor_to_numpy_cv2 , re_labeling
import cv2
import numpy as np
from dataloader import dataloader

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
labels_dict={'Hand_free_R':1,'Hand_free_L':2,'Hand_cont_R':3,'Hand_cont_L':4 ,
             'person_R':5,'person_L':6 ,'person_LR':7,'portable_R':8 ,
             'portable_L':9, 'portable_LR':10,'non-portable_R':11,'non-portable_L':12,'non-portable_LR':13}

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == '__main__':
    file = 'config.json'
    with open(file) as json_data_file:
        config = json.load(json_data_file)
    root = config["data root"]
    checkpoint = root + 'torch_trained_fasterrcnn_100p.pth'
    a = argparse.ArgumentParser()
    a.add_argument("--dataset", help="PSCAL VOC2007 format folder",
                   default=root + 'pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007')
    a.add_argument("--scale", type=int, help="input image scale", default=0.6)
    a.add_argument("--output", help="path to output folder", default=root + 'output/')
    a.add_argument("--batch", type=int, help="batch size", default=1)
    a.add_argument("--checkpoint", help="train model weight", default=checkpoint)
    args = a.parse_args()
    train_loader, trainval_loader, val_loader = dataloader(args.batch, args.dataset)
    for batch in val_loader:
        images, targets = batch

        for target,image in zip(targets,images):
            # print(target['dict']['annotation']['object'])
            # print(len(target['dict']),len(target['boxes']))
            image=tensor_to_numpy_cv2(image)
            draw = np.copy(image)
            # print('target',target['dict'])
            for bbox ,cls in zip(target['boxes'],target['labels']):
                bbox = np.array(bbox).astype(int)
                # print('label',cls,labels_dict[cls-1])

                #targetoject
                if cls==labels_dict['Hand_free_L']:
                    color = (0, 0, 225)
                    # cv2.putText(draw, f'{labels_dict[cls-1]:s} ', bbox[:2], font, 1, color, 2, cv2.LINE_AA)
                else :
                    color=(225,0,0)
                cv2.rectangle(draw, bbox[:2], bbox[2:4], color, 2)
                # cv2.putText(draw, f'{labels_dict[cls-1]:s} ', bbox[:2], font, 1, color, 2, cv2.LINE_AA)
                cv2.putText(draw, f'{list(labels_dict.keys())[cls-1]} ', bbox[:2], font, 1, color, 2, cv2.LINE_AA)
            cv2.imshow('image',draw)
            cv2.waitKey(0)
