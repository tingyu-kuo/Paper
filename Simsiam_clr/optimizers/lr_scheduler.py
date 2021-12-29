import torch
import numpy as np
import math

def adjust_learning_rate(optimizer, init_lr, epoch, args, warmup=True):
    final_lr = args.train.final_lr
    # Setting  schedule function
    if warmup:
        warmup_epochs = args.train.warmup_epochs
        warmup_lr = args.train.warmup_lr
        if epoch < warmup_epochs:
            cur_lr = warmup_lr + (init_lr - warmup_lr) * ((epoch + 1) / warmup_epochs)
        else:
            cur_lr = final_lr + (init_lr - final_lr) * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (args.train.num_epochs - warmup_epochs)))
    else:
        cur_lr = final_lr + (init_lr - final_lr) * 0.5 * (1. + math.cos(math.pi * epoch / args.train.num_epochs))

    # Checking if fix_lr
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    return cur_lr


if __name__ == "__main__":
    # python optimizers/lr_scheduler.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_cifar.yaml --ckpt_dir ~/.cache/
    # Testing
    import torchvision
    from arguments import get_args
    import matplotlib.pyplot as plt

    model = torchvision.models.resnet50()
    optimizer = torch.optim.SGD(model.parameters(), lr=999)
    args = get_args()
    n_iter = 390
    
    lrs = []
    for epoch in range(args.train.num_epochs):
        for it in range(n_iter):
            lr = adjust_learning_rate(optimizer, 0.01, epoch, args)
            lrs.append(lr)
    plt.plot(lrs)
    plt.show()
