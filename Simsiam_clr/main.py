import os
import numpy as np
import torch
import torchvision.models as models
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from tools import Logger, knn_monitor
from datasets import get_dataset
from datetime import datetime
from models.siamese import Siamese
# from apex import amp
from optimizers.lr_scheduler import adjust_learning_rate
from models.backbones import cifar_resnet_1 as resnet
from linear_eval import main as linear_eval


def main(args):
    device = args.device
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=True, **args.aug_kwargs), 
            train=True,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=True,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # define model
    # model = SimSiam(backbone=resnet.__dict__[args.model.backbone]()).to(device)
    model = Siamese(model=resnet.__dict__[args.model.backbone](pretrained=False)).to(device)

    # define initial learning rate optimizer
    opt_params = [
        {'params': model.backbone.parameters(), 'fix_lr':False},
        {'params': model.projector.parameters(), 'fix_lr':False},
        {'params': model.predictor.parameters(), 'fix_lr':True}
    ]
    init_lr = args.train.base_lr * args.train.batch_size / 128
    optimizer = torch.optim.SGD(
        opt_params,
        lr=init_lr,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay
    )
    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    
    global_progress = tqdm(range(args.train.stop_at_epoch), desc=f'Training')

    start_time = datetime.now().strftime('%m%d')

    for epoch in global_progress:
        # training
        model.train()
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.train.num_epochs}', disable=True) 
        for idx, ((image1, image2), labels) in enumerate(local_progress):

            lr = adjust_learning_rate(optimizer, init_lr, epoch, args, warmup=False)
            optimizer.zero_grad()
            loss = model.forward(  # if pin_memory=True => non_blocking=True, in order to speed up 
                image1.to(device, non_blocking=True),
                image2.to(device, non_blocking=True),
                # image3.to(device, non_blocking=True)
                )

            loss.backward()
            optimizer.step()
            
            data_dict = ({'lr':lr, 'loss':loss})


            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)
        if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
            accuracy = knn_monitor(model, memory_loader, test_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=True)

        # update training info
        epoch_dict = {"epoch":epoch, "accuracy":accuracy}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)    #　update scalers without max accuracy
    linear_eval(args, model)


if __name__ == "__main__":
    args = get_args()
    main(args=args)

    # To train, run:
    # python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_cifar.yaml --ckpt_dir ~/.cache/
