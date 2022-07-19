"""
USAGE

Training on custom ResNet without mosaic augmentations:
python train.py --model fasterrcnn_custom_resnet --epochs 2 --config data_configs/voc.yaml --no-mosaic --batch-size 16
Training on ResNet50 FPN with custom project folder name and visualizing transformed images before training begins:
python train.py --model fasterrcnn_resnet5-_fpn --epochs 2 --config data_configs/voc.yaml -vt --project-name resnet50fpn_voc --no-mosaic --batch-size 16
"""

from trainner.engine import  train_one_epoch, evaluate


from trainner.general_utils import (
    set_training_dir, Averager,
    save_model, save_train_loss_plot,
    show_tranformed_image,
    save_mAP, save_model_state
)
from trainner.logger import (
    log, set_log, coco_log,
    set_summary_writer, tensorboard_loss_log,
    tensorboard_map_log,
    csv_log
)

import torch
import argparse
import numpy as np
from torchvision import models
import json
from dataloader import dataloader

torch.multiprocessing.set_sharing_strategy('file_system')

# For same annotation colors each time.
np.random.seed(42)


def parse_opt():
    file = 'config.json'
    with open(file) as json_data_file:
        config = json.load(json_data_file)
    root = config["data root"]

    new_checkpoint = root + 'retrain_fasterrcnn_80k_2.pth'
    a = argparse.ArgumentParser()
    a.add_argument("--checkpoint", type=str, help="the weight dirctory for the trained model",
                   default=False)
    a.add_argument("--output", type=str, help="project name", default='new_project')

    a.add_argument("--batch", type=int, help="batch size", default=30)
    a.add_argument("--epoch", type=int, help="number of epoch", default=8)
    a.add_argument("--print_freq", type=int, help="printing training status frequency", default=100)

    a.add_argument("--dataset", type=str, help="PSCAL VOC2007 format folder",
                   default=root + 'pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007')

    args = a.parse_args()
    return args


def main(args):


    # Settings/parameters/constants.

    CLASSES = labels_dict = ['__background__','targetobject', 'hand']
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS = args.epoch
    SAVE_VALID_PREDICTIONS = True
    BATCH_SIZE = args.batch
    VISUALIZE_TRANSFORMED_IMAGES = False
    OUT_DIR = set_training_dir(args.output)
    COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))
    # Set logging file.
    set_log(OUT_DIR)
    writer = set_summary_writer(OUT_DIR)


    print('batch size', args.batch, 'training for ', args.epoch, 'epoch')
    train_loader, trainval_loader, valid_loader = dataloader(BATCH_SIZE, args.dataset)


    if VISUALIZE_TRANSFORMED_IMAGES:
        show_tranformed_image(train_loader, DEVICE, CLASSES, COLORS)

    # Initialize the Averager class.
    train_loss_hist = Averager()
    # Train and validation loss lists to store loss values of all
    # iterations till ena and plot graphs for all iterations.
    train_loss_list = []
    train_loss_list_epoch = []
    val_map_05 = []
    val_map = []
    start_epochs = 0
    # loading model
    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False).to(DEVICE)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device(DEVICE)))
    model.eval()
    # keys = list(args.checkpoint['model_state_dict'].keys())
    # for i in range(len(keys) - 14):
    #     print(i)
    #     print(keys[i])
    #     print(checkpoint['model_state_dict'][keys[i]].weight)

    # counter = 0
    # with torch.no_grad():
    #     for i in range(len(model.backbone)):
    #         if hasattr(model.backbone[i], 'weight'):
    #             model.backbone[i].weight.copy_(args.checkpoint['model_state_dict'][keys[2 * counter]])
    #             model.backbone[i].bias.copy_(args.checkpoint['model_state_dict'][keys[2 * counter + 1]])
    #             counter += 1
        # THIS SHOULD GO INTO RESUME TRAINING NOT LOAD CHECKPONT.
        # LOAD CHECKPOINT CAN ALSO BE FROM A COCO TRAINED MODEL.
        # if checkpoint['epoch']:
        #     start_epochs = checkpoint['epoch']
        #     print(f"Resuming from epoch {start_epochs}...")
        # if checkpoint['train_loss_list']:
        #     train_loss_list = checkpoint['train_loss_list']

    print(model)
    # model = model.to(DEVICE)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Get the model parameters.
    params = [p for p in model.parameters() if p.requires_grad]
    # Define the optimizer.
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)

    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005,
    #                             momentum=0.9, weight_decay=0.0005)
    # # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.1)

    # if args['cosine_annealing']:
    #     # LR will be zero as we approach `steps` number of epochs each time.
    #     # If `steps = 5`, LR will slowly reduce to zero every 5 epochs.
    #     steps = NUM_EPOCHS + 10
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #         optimizer,
    #         T_0=steps,
    #         T_mult=1,
    #         verbose=False
    #     )
    # else:
    #     scheduler = None
    scheduler = None
    for epoch in range(start_epochs, NUM_EPOCHS):
        train_loss_hist.reset()

        _, batch_loss_list = train_one_epoch(
            model,
            optimizer,
            train_loader,
            DEVICE,
            epoch,
            train_loss_hist,
            print_freq=100,
            scheduler=scheduler
        )

        coco_evaluator, stats = evaluate(
            model,
            valid_loader,
            device=DEVICE,
            save_valid_preds=SAVE_VALID_PREDICTIONS,
            out_dir=OUT_DIR,
            classes=CLASSES,
            colors=COLORS
        )

        # Add the current epoch's batch-wise lossed to the `train_loss_list`.
        train_loss_list.extend(batch_loss_list)
        train_loss_list_epoch.append(train_loss_hist.value)
        val_map_05.append(stats[1])
        val_map.append(stats[0])

        # Save the current epoch model state. This can be used
        # to resume training. It saves model state dict, number of
        # epochs trained for, optimizer state dict, and loss function.
        save_model(epoch, model, optimizer, train_loss_list, OUT_DIR)
        # Save the model dictionary only.
        save_model_state(model, OUT_DIR)

        # Save loss plot for batch-wise list.
        save_train_loss_plot(OUT_DIR, train_loss_list)
        # Save loss plot for epoch-wise list.
        save_train_loss_plot(
            OUT_DIR,
            train_loss_list_epoch,
            'epochs',
            'train loss',
            save_name='train_loss_epoch'
        )

        # Save mAP plots.
        save_mAP(OUT_DIR, val_map_05, val_map)
        # Save batch-wise train loss plot using TensorBoard.
        # tensorboard_loss_log('Train loss', np.array(train_loss_list), writer)
        # Save epoch-wise train loss plot using TensorBoard.
        tensorboard_loss_log('Train loss', np.array(train_loss_list_epoch), writer)
        # Save mAP plot using TensorBoard.
        tensorboard_map_log(
            name='mAP',
            val_map_05=np.array(val_map_05),
            val_map=np.array(val_map),
            writer=writer
        )

        coco_log(OUT_DIR, stats)
        csv_log(OUT_DIR, stats, epoch)


# if __name__ == '__main__':
def start():
    args = parse_opt()
    main(args)