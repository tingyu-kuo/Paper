import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from tools import AverageMeter
from datasets import get_dataset
# from optimizers import get_optimizer, LR_Scheduler
from optimizers.lr_scheduler import adjust_learning_rate

def main(args, model):

    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs), 
            train=True, 
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=True,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **args.dataloader_kwargs
    )

    model = model.to(args.device)

    # get model output size
    with torch.no_grad():
        x = torch.rand(1,3,224,224).to(args.device)
        output = model.encoder_eval(x)
    out_dim = output.size(1)

    classifier = nn.Linear(in_features=out_dim, out_features=100, bias=True).to(args.device)

    init_lr = args.train.base_lr * args.train.batch_size / 128
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=init_lr,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay
    )

    loss_meter = AverageMeter(name='Loss')
    acc_meter = AverageMeter(name='Accuracy')

    # Start training
    global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
    for epoch in global_progress:
        loss_meter.reset()
        model.eval()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=True)
        
        for idx, (images, labels) in enumerate(local_progress):
            lr = adjust_learning_rate(optimizer, init_lr, epoch, args, warmup=False)
            classifier.zero_grad()
            with torch.no_grad():
                feature = model.encoder_eval(images.to(args.device))

            preds = classifier(feature)

            loss = F.cross_entropy(preds, labels.to(args.device))

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            local_progress.set_postfix({'lr':lr, "loss":loss_meter.val, 'loss_avg':loss_meter.avg})

    classifier.eval()
    correct, total = 0, 0
    acc_meter.reset()
    for idx, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            feature = model.encoder_eval(images.to(args.device))
            preds = classifier(feature).argmax(dim=1)
            correct = (preds == labels.to(args.device)).sum().item()
            acc_meter.update(correct/preds.shape[0])
    print(f'Accuracy = {acc_meter.avg*100:.2f}')




if __name__ == "__main__":
    main(args=get_args())
    # to evaluate, run:
    # python linear_eval.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_cifar_eval_sgd.yaml --ckpt_dir ~/.cache/ --hide_progress --eval_from ~/.cache/simsiam-cifar100-experiment-resnet18_cifar_variant1_1125170759.pth
















