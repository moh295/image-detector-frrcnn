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
from utils_local import image_resize ,tensor_to_PIL
from dataloader import dataloader
#labels_dict=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','dog','chair','cow','diningtable','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
labels_dict = ['targetobject','hand']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
font = cv2.FONT_HERSHEY_SIMPLEX
# Dictionary containing some colors
colors = {'blue': (0, 0, 255), 'green': (0, 255, 0), 'red': (255, 0, 0), 'orange':(223,70,14),'yellow': (255, 255, 0),
          'magenta': (255, 0, 255), 'cyan': (0, 255, 255), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220),'red 1': (255, 82, 82),'red 2': (255, 82, 82) ,'red 3': (255, 82, 82)}

def inference_and_save_mobilnet_full_data(model,save_dir,dataloder):
    # apply model on images and save the result

    prob_thresh = 0.9
    cnt = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start = timer()
    for images in dataloder:
        # start_pred = timer()
        images = list(image.to(device) for image in images[0])
        start_pred = timer()

        # print('prediction started')
        predictions = model(images)
        end = timer()
        elapsed = timedelta(seconds=end - start_pred)
        print(f'prediction takes {elapsed}')
        path_to_output_image = save_dir


        for data, image in zip(predictions,images):
            image = tensor_to_PIL(image, normlized=False).convert('RGB')
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
                cv2.rectangle(draw, bbox[:2], bbox[2:4], color, 2)
                cv2.putText(draw, f'{category:s} {prob:.3f}', bbox[:2], font, 1, color, 2, cv2.LINE_AA)
            cv2.imwrite(path_to_output_image + str(cnt) + '.png',draw)
            print(f'Output image is saved to {path_to_output_image}{cnt}.png')
            cnt += 1
            # image.show()
        end = timer()
        elapsed = timedelta(seconds=end-start)
        print(f'full prediction process takes {elapsed}')

if __name__ == '__main__':
    checkpoint = '/App/data/torch_trained_fasterrcnn.pth'
    a = argparse.ArgumentParser()
    a.add_argument("--dataset", help="PSCAL VOC2007 format folder", default='data/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007')
    a.add_argument("--scale", help="input image scale", default=0.6)
    a.add_argument("--output", help="path to output folder", default='data/output/')
    a.add_argument("--batch", help="batch size", default=60)
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
    #loading/checking data....
    print('batch size',args.batch)
    train_loader,trainval_loader ,val_loader=dataloader(args.batch,args.dataset)
    inference_and_save_mobilnet_full_data(model, args.output,val_loader)
