import argparse
from torchvision import transforms
import cv2
from utils import image_resize
import json
from frRCNN import FrRCNN
from tracker import Grasp_tracker


class get_frame:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        # get total number of frames
        self.totalFrames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)


    def __getitem__(self, idx: int):

        # check for valid frame number
        if idx >= 0 & idx <= self.totalFrames:
            # set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret: return frame


if __name__ == '__main__':
    file='config.json'
    with open(file) as json_data_file:
        config= json.load(json_data_file)
    root = config["data root"]
    # checkpoint = root + 'torch_trained_fasterrcnn_100p.pth'
    checkpoint=config["checkpoint"]

    a = argparse.ArgumentParser()
    a.add_argument("--video", help="path to input images", default=root+'input.mp4')
    a.add_argument("--input_scale",type=float, help="input image scale", default=0.6)
    a.add_argument("--output_scale", type=float, help="input image scale", default=1.0)
    a.add_argument("--output", help="path to output folder", default=root+'output.mp4')
    a.add_argument("--batch",type=int, help="batch size", default=30)
    a.add_argument("--skip", type=int, help="skip each number of frames (default 1 for not skipping)", default=1)
    a.add_argument("--fps", type=int, help="output video frame rate /s", default=30)
    a.add_argument("--checkpoint", type=str, help="train model weight", default=checkpoint)
    args = a.parse_args()
    #loading model
    model=FrRCNN(args.checkpoint)
    #loading/checking data....
    print('batch size',args.batch)

    image_list = []
    tensor_list=[]
    image_batch=[]
    grasp_tracker = Grasp_tracker()
    count = 1
    success = True
    init_vid=True
    images=get_frame(args.video)
    number_of_frames=int(images.totalFrames)
    print('number of frame',number_of_frames)
    for i in  range(number_of_frames):
        print(f'processing frame {i} / {number_of_frames} -- batches` {int(i/args.batch)}/{int(number_of_frames/args.batch)}')
        if i%args.skip==0 and images[i] is not None:
            image=image_resize(images[i],args.input_scale)
            image_list.append(image)
            tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform=transforms.Compose([transforms.ToTensor()])
            tensor=transform(tensor)
            tensor_list.append(tensor)
            image_batch.append(image)

            # for first frame only
            if init_vid:
                init_vid = False
                height, width, layers = image_resize( image,args.output_scale).shape
                fps = args.fps
                print('initiat vid size', width, height, fps, args.output)
                model.create_video(width, height, fps, args.output)

            if len(tensor_list)==args.batch:
                frame=model.predict(args.output, tensors=tensor_list,save_vid=True,count=count,images=image_batch,output_scale=args.output_scale,tracker=grasp_tracker)
                count+=args.batch
                tensor_list = []
                image_batch=[]
    #check if there is still images in the list in case the previous loop break befor len(tensor_list)==args.batch
    if len(tensor_list):
        model.predict(args.output, tensors=tensor_list, save_vid=True, count=count, images=image_batch,output_scale=args.output_scale,tracker=grasp_tracker)

    model.realse_vid()