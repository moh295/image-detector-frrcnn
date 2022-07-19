import argparse
from torchvision import transforms
import cv2
from utils import image_resize
import json
from frRCNN import FrRCNN


if __name__ == '__main__':
    file = 'config.json'
    with open(file) as json_data_file:
        config = json.load(json_data_file)
    root = config["data root"]
    # checkpoint = root + 'torch_trained_fasterrcnn_100p.pth'
    checkpoint=config["checkpoint"]
    a = argparse.ArgumentParser()
    a.add_argument("--cam",type=int, help="webcam number e.g: 0 , 1", default=1)
    a.add_argument("--scale",type=int, help="input image scale", default=0.6)
    a.add_argument("--output", help="path to output folder", default=root+'output/')
    a.add_argument("--batch",type=int, help="batch size", default=1)
    a.add_argument("--checkpoint", help="train model weight", default=checkpoint)
    args = a.parse_args()
    model=FrRCNN(checkpoint)
    #open cam
    cap = cv2.VideoCapture(args.cam)
    print('batch size',args.batch)
    image_list = []
    tensor_list=[]
    image_batch=[]

    count=1
    while True:

        # Capture frame-by-frame
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ret, frame = cap.read()
        # # Display the resulting frame
        # cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        image=image_resize(frame,args.scale)
        tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform=transforms.Compose([transforms.ToTensor()])
        tensor=transform(tensor)
        tensor_list.append(tensor)
        image_batch.append(image)
        if len(tensor_list)==args.batch :
            model.predict(args.output, tensors=tensor_list, images=image_batch, show_vid=True)
            count+=args.batch
            tensor_list = []
            image_batch=[]

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()