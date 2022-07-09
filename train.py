import torch
from dataloader import dataloader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from timeit import default_timer as timer
from datetime import timedelta
from engine import train_one_epoch, evaluate

checkpoint= '/App/data/torch_trained_fasterrcnn.pth'
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

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=print_freq)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    end = timer()
    elapsed = timedelta(seconds=end - start)
    print('Finished Training....duration :', elapsed)
    return model.state_dict()



if __name__ == '__main__':
    #loading model
    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(device)
    # model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True).to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #loading/checking data....
    batch_size=5
    print('batch size',batch_size)
    train_loader, trainval_loader, val_loader= dataloader()
    #trainging ....
    epochs=20
    print_freq=100
    stat_dic=obj_detcetion_training(model,epochs,train_loader,val_loader,print_freq)
    print('saving checkpoint to ', checkpoint)
    torch.save(stat_dic, checkpoint)
