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
#labels_dict=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','dog','chair','cow','diningtable','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
labels_dict = ['targetobject','hand']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
font = cv2.FONT_HERSHEY_SIMPLEX
# Dictionary containing some colors
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}
def inference_and_save_mobilnet_full_data(model,save_dir,images,tensors,count,labels_dict):
    # apply model on images and save the result
    scale = 1
    prob_thresh = 0.65
    cnt = count
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start = timer()

    # start_pred = timer()
    tensors = list(tensor.to(device) for tensor in tensors)
    # print('prediction started')
    predictions = model(tensors)
    # end = timer()
    # elapsed = timedelta(seconds=end - start_pred)
    # print(f'prediction takes {elapsed}')
    path_to_output_image = save_dir


    for data, image in zip(predictions,images):
        detection_bboxes, detection_classes, detection_probs = data['boxes'].cpu().detach().numpy(), \
                                                               data['labels'].cpu().detach().numpy(), data[
                                                                   'scores'].cpu().detach().numpy()
        detection_bboxes /= scale
        # print(detection_probs)
        # print('detection_classes',detection_classes)
        kept_indices = detection_probs > prob_thresh
        detection_bboxes = detection_bboxes[kept_indices]
        detection_classes = detection_classes[kept_indices]
        detection_probs = detection_probs[kept_indices]
        draw = np.copy(image)

        for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
            bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
            print(bbox)
            category = labels_dict[cls-1]
            color=colors['rand']
            cv2.rectangle(draw,bbox, color, 2)
            cv2.putText(draw, f'{category:s} {prob:.3f}', (bbox.left, bbox.top), font,1, color,2, cv2.LINE_AA)


        cv2.imwrite(path_to_output_image + str(cnt) + '_demo_output.png',draw)
        print(f'Output image is saved to {path_to_output_image}{cnt}.png')
        cnt += 1
        # image.show()


    end = timer()
    elapsed = timedelta(seconds=end-start)
    print(f'full prediction process takes {elapsed}')



if __name__ == '__main__':
    checkpoint = '/App/data/torch_trained_fasterrcnn.pth'
    a = argparse.ArgumentParser()
    a.add_argument("--images", help="path to input images", default='data/images/')
    a.add_argument("--output", help="path to output folder", default='data/output/')
    a.add_argument("--batch", help="batch size", default=30)
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
    batch_size=1
    print('batch size',batch_size)
    imdir=args.images
    image_list = []
    tensor_list=[]
    image_batch=[]
    print('loadin images form path[', imdir,'].....')
    ext = ['png', 'jpg', 'gif']  # Add image formats here
    files = []
    [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
    image_list = [cv2.imread(file) for file in files]
    print('image list length',len(image_list))
    count=1
    for i in range(len(image_list)):
        tensor = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB)
        transform=transforms.Compose([transforms.ToTensor()])
        tensor=transform(tensor)
        tensor_list.append(tensor)
        image_batch.append(image_list[i])
        if len(tensor_list)==batch_size or i ==len(image_list):
            inference_and_save_mobilnet_full_data(model, args.output, image_batch,tensor_list,count, labels_dict)
            count+=batch_size
            tensor_list = []
            image_batch=[]