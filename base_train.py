import torch
from dataloader import dataloader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from timeit import default_timer as timer
from datetime import timedelta
from base_trainner.engine import train_one_epoch, evaluate ,eval_one_epoch
import argparse
import json
import pickle
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def obj_detcetion_training(model,num_epochs,data_loader,data_loader_test,print_freq):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    start = timer()
    elapsed = 0
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    train_loss_list=[]
    precision_list=[]
    test_loss_list=[]
    for epoch in range(num_epochs):
        _, train_epoch_loss=train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=print_freq)
        train_loss_list.append(train_epoch_loss)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset

        _,test_epoch_loss=eval_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=print_freq)
        test_loss_list.append(test_epoch_loss)

        #coco precision and recall evaluation
        _,_,precision= evaluate(model, data_loader_test, device=device)
        precision_list.append(precision)


    end = timer()
    elapsed = timedelta(seconds=end - start)
    print('Finished Training....duration :', elapsed)
    with open('data/train_loss_list.pickle', 'wb') as handle:
        pickle.dump(train_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/test_loss_list.pickle', 'wb') as handle:
        pickle.dump(test_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/precision_list.pickle', 'wb') as handle:
        pickle.dump(precision_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model.state_dict()

if __name__ == '__main__':

    file = 'config.json'
    with open(file) as json_data_file:
        config = json.load(json_data_file)
    root = config["data root"]

    new_checkpoint = root+'retrain_fasterrcnn_80k_4c_test.pth'
    a = argparse.ArgumentParser()
    a.add_argument("--checkpoint", type=str,help="the weight dirctory for the trained model",
                   default=False)
    a.add_argument("--output",type=str, help="new checkpoint file", default=new_checkpoint)

    a.add_argument("--batch",type=int, help="batch size", default=20)
    a.add_argument("--epoch", type=int, help="number of epoch", default=3)
    a.add_argument("--print_freq", type=int, help="printing training status frequency", default=50)
    a.add_argument("--dataset",type=str, help="PSCAL VOC2007 format folder",
                   default=root+'pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007')
    args = a.parse_args()

    #loading model
    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device(device)))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #loading/checking data....
    print('batch size',args.batch,'training for ',args.epoch,'epoch')
    train_loader, trainval_loader, val_loader= dataloader(args.batch,args.dataset)
    #trainging ....

    stat_dic=obj_detcetion_training(model,args.epoch,train_loader,trainval_loader,args.print_freq)
    print('saving checkpoint to ', args.output)
    torch.save(stat_dic, args.output)
