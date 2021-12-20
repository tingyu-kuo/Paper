import os
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from datetime import datetime
from models.simsiam import SimSiam

def main(device, args):

    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=True, **args.aug_kwargs), 
            train=True,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.eval.batch_size,
        **args.dataloader_kwargs
    )

    # define model
    model = SimSiam(backbone=models.__dict__[args.name](zero_init_residual=True), num_classes=100)
    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/128, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=args.train.base_lr*args.train.batch_size/128,
    #     momentum=args.train.optimizer.momentum,
    #     weight_decay=args.train.optimizer.weight_decay
    # )

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/128, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/128, args.train.final_lr*args.train.batch_size/128, 
        len(train_loader),
        constant_predictor_lr=True
    )

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    

    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    
    for epoch in global_progress:
        # training
        model.train()
        train_acc = 0 
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2), labels) in enumerate(local_progress):
            
            model.zero_grad()
            data_dict, acc = model.forward(
                # if pin_memory=True => non_blocking=True, in order to speed up 
                images1.to(device, non_blocking=True),
                images2.to(device, non_blocking=True),
                labels.to(device)
                )
            loss = data_dict['loss'].mean() # ddp
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            train_acc += acc

            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)

        train_acc = (train_acc / (idx + 1))*100
        
        # testing
        model.eval()
        test_acc = 0
        max_acc = 0
        for idx, (images, labels) in enumerate(test_loader):

            with torch.no_grad():
                acc = model.valid(images.to(device, non_blocking=True), labels.to(device))
            test_acc += acc

        test_acc = (test_acc / (idx + 1))*100
        if test_acc > max_acc:
            max_acc = test_acc
            # Save checkpoint
            model_path = os.path.join(args.ckpt_dir, f"{args.name}_{datetime.now().strftime('%m%d')}.pth")
            torch.save({
                'epoch': epoch+1,
                'state_dict':model.backbone.state_dict()
            }, model_path)

        # update training info
        epoch_dict = {"epoch":epoch, "train_acc":train_acc, "test_acc":test_acc, "max":max_acc}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers({k:epoch_dict[k] for k in ["epoch", "train_acc", "test_acc"]})    #　update scalers without max accuracy


        
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')



if __name__ == "__main__":
    args = get_args()

    main(device=args.device, args=args)

    # # error
    # completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    # os.rename(args.log_dir, completed_log_dir)
    # print(f'Log file has been saved to {completed_log_dir}')

    # train, run:
    # python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_cifar.yaml --ckpt_dir ~/.cache/ --hide_progress

