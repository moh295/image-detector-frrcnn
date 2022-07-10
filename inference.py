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
import glob
#labels_dict=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','dog','chair','cow','diningtable','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
labels_dict = ['hand','targetobject']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def inference_and_save_mobilnet_full_data(model,save_dir,images,count,labels_dict):
    # apply model on images and save the result
    scale = 1
    prob_thresh = 0.65
    cnt = count
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start = timer()

    # start_pred = timer()
    images = list(image.to(device) for image in images)
    # print('prediction started')
    predictions = model(images)
    # end = timer()
    # elapsed = timedelta(seconds=end - start_pred)
    # print(f'prediction takes {elapsed}')
    path_to_output_image = save_dir


    for data, image in zip(predictions, images):
        # print('result',data['scores'])
        image=tensor_to_PIL(image,normlized=False)
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
        draw = ImageDraw.Draw(image)

        for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
            color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
            bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
            # print('catogory',cls)
            category = labels_dict[cls-1]
            # print('catogory', cls,category)
            draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
            draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)
            # draw.text((bbox.left, bbox.top), text=f'{prob:.3f}', fill=color)
        image.save(path_to_output_image + str(cnt) + '_demo_output.png')
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
    path=args.images+'*.png'
    image_list = []
    tensor_list=[]
    print('path', path)
    for filename in glob.glob(path):
        im = Image.open(filename)
        image_list.append(im)
    print('image list length',len(image_list))
    count=1
    for i in range(len(image_list)):
        image = image_list[i].convert('RGB')
        transform=transforms.Compose([transforms.ToTensor()])
        image=transform(image)
        tensor_list.append(image)
        if len(tensor_list)==batch_size or i ==len(image_list):
            inference_and_save_mobilnet_full_data(model, args.output, tensor_list,count, labels_dict)
            count+=batch_size
            tensor_list = []
