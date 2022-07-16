import torch
from torchvision import models
from timeit import default_timer as timer
from datetime import timedelta
from PIL import ImageDraw,Image
import random
from bbox import BBox
from utils_local import tensor_to_PIL
import argparse
from torchvision import transforms
import json
from frRCNN import FrRCNN
import cv2
from utils_local import image_resize


#labels_dict=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','dog','chair','cow','diningtable','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
labels_dict = ['targetobject','hand']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    file = 'config.json'
    with open(file) as json_data_file:
        config = json.load(json_data_file)
    root = config["data root"]
    checkpoint = root + 'torch_trained_fasterrcnn_100p.pth'
    a = argparse.ArgumentParser()
    a.add_argument("--checkpoint", help="the weight dirctory for the trained model",default=checkpoint)
    args = a.parse_args()


    model=FrRCNN(checkpoint)

    #loading/checking data....
    batch_size=1
    print('batch size',batch_size)
    #image= demo_data_image()
    file = 'demo_input.png'
    image=cv2.imread(file)
    tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(tensor)
    model.predict(root, tensors=[tensor],save=True,images=[image],show=True )
