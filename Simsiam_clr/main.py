import os
import torch
import torchvision.models as models
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
# from optimizers import get_optimizer, LR_Scheduler
from tools import Logger, knn_monitor
from datasets import get_dataset
from datetime import datetime
from models.simsiam import SimSiam
from apex import amp
from optimizers.lr_scheduler import adjust_learning_rate
from models import get_model
from models.backbones import cifar_resnet_1


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
    model = SimSiam(backbone=cifar_resnet_1.resnet18()).to(device) # zero_init_residual=True

    # define initial learning rate optimizer
    # If using DataParallel (for multi gpu), then params needed 'model.module.parameters()', else 'model.parameters()' 
    init_lr = args.train.base_lr * args.train.batch_size / 128
    optim_params = [{'params': model.backbone.parameters(), 'fix_lr': False},
                    {'params': model.projector.parameters(), 'fix_lr': False},
                    {'params': model.predictor.parameters(), 'fix_lr': True}]
    # optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
    #                 {'params': model.module.predictor.parameters(), 'fix_lr': True}]

    optimizer = torch.optim.SGD(
        optim_params,
        lr=init_lr,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay
    )
    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    start_time = datetime.now().strftime('%m%d')
    max_acc = 0

    for epoch in global_progress:
        # training
        model.train()
        train_acc = 0 
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.train.num_epochs}', disable=True) 
        for idx, ((images1, images2), labels) in enumerate(local_progress):

            lr = adjust_learning_rate(optimizer, init_lr, epoch, args, warmup=False)
            optimizer.zero_grad()
            loss, l, acc = model.forward(  # if pin_memory=True => non_blocking=True, in order to speed up 
                images1.to(device, non_blocking=True),
                images2.to(device, non_blocking=True),
                labels.to(device)
                )
            
            # loss, acc = model.baseline(
            #     images1.float().to(device, non_blocking=True),
            #     labels.to(device)
            # )


            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            optimizer.step()
            
            data_dict = ({'lr':lr, 'loss':loss})
            train_acc += acc

            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)

        train_acc = (train_acc / (idx + 1))*100

        # testing
        model.eval()
        test_acc = 0
        for idx, (images, labels) in enumerate(test_loader):

            with torch.no_grad():
                acc = model.valid(images.to(device, non_blocking=True), labels.to(device))
            test_acc += acc

        test_acc = (test_acc / (idx + 1))*100
        if test_acc > max_acc:
            max_acc = test_acc
            # Save checkpoint
            model_path = os.path.join(args.ckpt_dir, f"{args.name}_{start_time}.pth")
            torch.save({
                'epoch': epoch+1,
                'accuracy': f'{max_acc:.1f}',
                'state_dict':model.backbone.state_dict()
            }, model_path)

        # update training info
        epoch_dict = {"epoch":epoch, "train_acc":train_acc, "test_acc":test_acc, "best_acc":max_acc, "Simsiam":l[0].item(), "Xent":l[1].item()}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers({k:epoch_dict[k] for k in ["epoch", "train_acc", "test_acc"]})    #　update scalers without max accuracy

    os.rename(model_path, f"{args.name}_{start_time}_{max_acc}.pth")

if __name__ == "__main__":
    args = get_args()
    main(args=args)

    # To train, run:
    # python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_cifar.yaml --ckpt_dir ~/.cache/
