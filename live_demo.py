import torch
from torchvision import models
from timeit import default_timer as timer
from datetime import timedelta
import random
from bbox import BBox
import argparse
from torchvision import transforms
import glob
import cv2
import numpy as np
from utils_local import image_resize
#labels_dict=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','dog','chair','cow','diningtable','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
labels_dict = ['targetobject','hand']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
font = cv2.FONT_HERSHEY_SIMPLEX
# Dictionary containing some colors
colors = {'blue': (0, 0, 255), 'green': (0, 255, 0), 'red': (255, 0, 0), 'orange':(223,70,14),'yellow': (255, 255, 0),
          'magenta': (255, 0, 255), 'cyan': (0, 255, 255), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220),'red 1': (255, 82, 82),'red 2': (255, 82, 82) ,'red 3': (255, 82, 82)}

def inference_and_save_mobilnet_full_data(model,save_dir,images,tensors,count,labels_dict):
    # apply model on images and save the result

    prob_thresh = 0.1
    cnt = count
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start = timer()

    start_pred = timer()
    tensors = list(tensor.to(device) for tensor in tensors)
    # print('prediction started')
    predictions = model(tensors)
    end = timer()
    elapsed = timedelta(seconds=end - start_pred)
    print(f'prediction takes {elapsed}')
    path_to_output_image = save_dir


    for data, image in zip(predictions,images):
        detection_bboxes, detection_classes, detection_probs = data['boxes'].cpu().detach().numpy(), \
                                                               data['labels'].cpu().detach().numpy(), data[
                                                                   'scores'].cpu().detach().numpy()

        # print(detection_probs)
        # print('detection_classes',detection_classes)
        kept_indices = detection_probs > prob_thresh
        detection_bboxes = detection_bboxes[kept_indices]
        detection_classes = detection_classes[kept_indices]
        detection_probs = detection_probs[kept_indices]
        draw = np.copy(image)
        for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
            bbox = np.array(bbox).astype(int)
            category = labels_dict[cls - 1]
            intensity=int(200-200*prob)
            color = (intensity,intensity,255) if cls == 1 else (225,intensity,intensity)
            if cls==1:
                cv2.rectangle(draw, bbox[:2], bbox[2:4], color, 2)
                cv2.putText(draw, f'{category:s} {prob:.3f}', bbox[:2], font, 1, color, 2, cv2.LINE_AA)
            elif prob>0.85:
                cv2.rectangle(draw, bbox[:2], bbox[2:4], color, 2)
                cv2.putText(draw, f'{category:s} {prob:.3f}', bbox[:2], font, 1, color, 2, cv2.LINE_AA)

        # cv2.imwrite(path_to_output_image + str(cnt) + '.png',draw)
        cv2.imshow('output image',image_resize(draw,2))
        cv2.waitKey(1)

        print(f'Output image is saved to {path_to_output_image}{cnt}.png')
        cnt += 1
        # image.show()
    end = timer()
    elapsed = timedelta(seconds=end-start)
    print(f'full prediction process takes {elapsed}')

if __name__ == '__main__':
    checkpoint = '/App/data/torch_trained_fasterrcnn.pth'
    a = argparse.ArgumentParser()
    a.add_argument("--device",type=int, help="webcam number e.g: 0 , 1", default=0)
    a.add_argument("--scale",type=int, help="input image scale", default=0.6)
    a.add_argument("--output", help="path to output folder", default='data/output/')
    a.add_argument("--batch",type=int, help="batch size", default=60)
    a.add_argument("--checkpoint", help="train model weight", default=checkpoint)
    args = a.parse_args()
    #loading model
    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    #open cam
    cap = cv2.VideoCapture(args.device)
    print('batch size',args.batch)
    image_list = []
    tensor_list=[]
    image_batch=[]



    count=1
    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # Display the resulting frame
        cv2.imshow('frame', frame)
        # cv2.imshow('gray', gray)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        image=image_resize(frame,args.scale)
        tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform=transforms.Compose([transforms.ToTensor()])
        tensor=transform(tensor)
        tensor_list.append(tensor)
        image_batch.append(image)
        if len(tensor_list)==args.batch :
            inference_and_save_mobilnet_full_data(model, args.output,image_batch,tensor_list,count, labels_dict)
            count+=args.batch
            tensor_list = []
            image_batch=[]

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()