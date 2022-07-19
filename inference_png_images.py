import argparse
from torchvision import transforms
import glob
import cv2
from utils import image_resize
import json
from frRCNN import FrRCNN
if __name__ == '__main__':
    file='config.json'
    with open(file) as json_data_file:
        config= json.load(json_data_file)
    root = config["data root"]
    checkpoint=config["checkpoint"]

    a = argparse.ArgumentParser()
    a.add_argument("--images", help="path to input images", default=root+'images/')
    a.add_argument("--scale",type=int, help="input image scale", default=0.6)
    a.add_argument("--output", help="path to output folder", default=root+'output/')
    a.add_argument("--batch",type=int, help="batch size", default=60)
    a.add_argument("--checkpoint", help="train model weight", default=checkpoint)
    args = a.parse_args()
    #loading model
    model=FrRCNN(checkpoint)
    #loading/checking data....
    print('batch size',args.batch)
    imdir=args.images
    image_list = []
    tensor_list=[]
    image_batch=[]
    print('loadin images form path[', imdir,'].....')
    ext = ['png', 'jpg', 'gif']  # Add image formats here
    files = []
    [files.extend(glob.glob(imdir + '*.' + e))for e in ext]
    files = list(sorted(files))
    image_list = [cv2.imread(file) for file in files]
    count=1
    for i in range(len(image_list)):
        image=image_resize(image_list[i],args.scale)
        tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform=transforms.Compose([transforms.ToTensor()])
        tensor=transform(tensor)
        tensor_list.append(tensor)
        image_batch.append(image)
        if len(tensor_list)==args.batch or i ==len(image_list):
            # inference_and_save_mobilnet_full_data(model, args.output,image_batch,tensor_list,count, labels_dict)
            model.predict(args.output, tensors=tensor_list,save=True,count=count,images=image_batch)
            count+=args.batch
            tensor_list = []
            image_batch=[]