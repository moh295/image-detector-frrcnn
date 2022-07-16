
import argparse
import json
from utils_local import tensor_to_numpy_cv2
from dataloader import dataloader
from frRCNN import FrRCNN

if __name__ == '__main__':
    file = 'config.json'
    with open(file) as json_data_file:
        config = json.load(json_data_file)
    root = config["data root"]
    checkpoint = root + 'torch_trained_fasterrcnn_100p.pth'
    a = argparse.ArgumentParser()
    a.add_argument("--dataset", help="PSCAL VOC2007 format folder", default=root+'pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007')
    a.add_argument("--scale",type=int, help="input image scale", default=0.6)
    a.add_argument("--output", help="path to output folder", default=root+'output/')
    a.add_argument("--batch",type=int, help="batch size", default=60)
    a.add_argument("--checkpoint", help="train model weight", default=checkpoint)
    args = a.parse_args()
    #loading model
    model = FrRCNN(checkpoint)
    #loading/checking data....
    print('batch size',args.batch)
    train_loader,trainval_loader ,val_loader=dataloader(args.batch,args.dataset)
    count=1
    for tensors in val_loader:
        images_lsit = [tensor_to_numpy_cv2(image)for image in tensors[0]]
        model.predict(args.output, tensors=tensors[0],save=True,count=count,images=images_lsit)
        count+=args.batch
