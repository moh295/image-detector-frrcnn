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
    checkpoint = root + 'torch_trained_fasterrcnn_100p.pth'

    a = argparse.ArgumentParser()
    a.add_argument("--video", help="path to input images", default=root+'input.mp4')
    a.add_argument("--scale",type=int, help="input image scale", default=0.6)
    a.add_argument("--output", help="path to output folder", default=root+'output.mp4')
    a.add_argument("--batch",type=int, help="batch size", default=60)
    a.add_argument("--checkpoint", help="train model weight", default=checkpoint)
    args = a.parse_args()
    #loading model
    model=FrRCNN(checkpoint)
    #loading/checking data....
    print('batch size',args.batch)

    image_list = []
    tensor_list=[]
    image_batch=[]
    count = 1
    success = True
    init_vid=True
    vidcap = cv2.VideoCapture(args.video)
    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        if not success: break
        image=image_resize(image,args.scale)
        image_list.append(image)
        tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform=transforms.Compose([transforms.ToTensor()])
        tensor=transform(tensor)
        tensor_list.append(tensor)
        image_batch.append(image)

        # for first frame only
        if init_vid:
            init_vid = False
            height, width, layers = image.shape
            fps = 30
            print('initiat vid size', width, height, fps, args.output)
            model.create_video(width, height, fps, args.output)


        if len(tensor_list)==args.batch:
            frame=model.predict(args.output, tensors=tensor_list,save_vid=True,count=count,images=image_batch)

            count+=args.batch
            tensor_list = []
            image_batch=[]
    #check if there is still images in the list in case the prvieos loop break befor len(tensor_list)==args.batch
    if len(tensor_list):
        model.predict(args.output, tensors=tensor_list, save_vid=True, count=count, images=image_batch)

    model.realse_vid()