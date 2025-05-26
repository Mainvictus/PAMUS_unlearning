import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Subset

from data_utils import getDatasets
from nn_utils import do_epoch
from grad_utils import getGradObjs

from nn_utils import manual_seed

import torch.nn.functional as F
from grad_utils import getGradObjs, gradNorm, getHessian, getVectorizedGrad, getOldPandG
import copy
import pandas as pd

def DisableBatchNorm(model):
    for name ,child in (model.named_children()):
        if name.find('BatchNorm') != -1:
            pass
        else:
            child.eval()
            child.track_running_stats=False

    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    outString = 'trained_models/'+args.dataset+"_"+args.model+'_'+str(args.used_training_size)+'_seed_' + str(args.train_seed)+'_epochs_' + str(args.epochs)+'_lr_' + str(args.learning_rate)+'_wd_' + str(args.weight_decay)+'_bs_' + str(args.batch_size)+'_optim_' + str(args.optim)
    if args.data_augment:
        outString = outString + "_transform"
    else:
        outString = outString + "_notransform"
    print(outString)

    manual_seed(args.train_seed)
    ordering = np.random.permutation(args.orig_trainset_size)
    selection = ordering[:args.used_training_size]

    train_dataset, val_dataset = getDatasets(name=args.dataset, include_indices=selection, exclude_indices=args.exclude_indices, data_augment=args.data_augment)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4)
    
    print("Training set length: ", len(train_dataset))
    print("Validation set length: ", len(val_dataset))

    exec("from models import %s" % args.model)
    model = eval(args.model)().to(device)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    if device=='cuda':
        model = torch.nn.DataParallel(model)
    
    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        error('unknown optimizer')


    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[80,120], gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=10, verbose=True)
    
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss, train_accuracy = do_epoch(model, train_loader, criterion, epoch, args.epochs, optim=optim, device=device, outString=outString)

        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(model, val_loader, criterion, epoch, args.epochs, optim=None, device=device, outString=outString)

        tqdm.write(f'{args.model} EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        #if val_accuracy > best_accuracy:
        #    print('Saving model...')
        #    best_accuracy = val_accuracy
        #    torch.save(model.state_dict(), 'trained_models/test.pt')

        lr_scheduler.step()          # For MultiStepLR
        # lr_scheduler.step()          # For CosineAnnealingLR
        # lr_scheduler.step(val_loss)  # For ReduceLROnPlateau 

    print('Saving model...')
    torch.save(model.state_dict(), outString + '.pt')
    
        # 记录样本损失
    if args.log_loss:
        model.eval()
        log_loss_file = outString
        log_loss_file = log_loss_file.replace('trained_models/','loss_log/')
        log_loss_file = log_loss_file + '_loss.csv'
        tmp = {'max':[], 'max_diff':[],'loss':[],'gradnorm':[],'y_true':[], 'y_pred':[]}
        log_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, pin_memory=True,
                                              shuffle=False, num_workers=4)
        for x, y_true in tqdm(log_loader):
            tmp['y_true'].append(y_true.item())
            x, y_true = x.to(device), y_true.to(device)
            logits = model(x)
            loss = criterion(logits, y_true)
            y_pred = F.softmax(logits, dim=1)
            tmp['max'].append(y_pred[0].max().detach().cpu().item())
            tmp['loss'].append(loss.detach().cpu().item())
            top_probs, y_top = torch.topk(y_pred, 1)
            tmp['y_pred'].append(y_top[0][0].detach().cpu().item())
            top_probs, _ = torch.topk(y_pred, 2)
            max_diff = top_probs[0][0] - top_probs[0][1]
            tmp['max_diff'].append(max_diff.detach().cpu().item())
            
            ############ Sample Forward Pass ########
            model_copy = copy.deepcopy(model)
            model_copy.train()
            model_copy = DisableBatchNorm(model_copy)

            y_pred = model_copy(x)
            sample_loss_before = criterion(y_pred, y_true)
            # print('Sample Loss Before: ', sample_loss_before)

            ####### Sample Gradient
            optim.zero_grad()
            sample_loss_before.backward()

            fullprevgradnorm = gradNorm(model_copy)
            tmp['gradnorm'].append(fullprevgradnorm)      
        df = pd.DataFrame(tmp)
        df.to_csv(log_loss_file, header=True, index=False)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network')
    arg_parser.add_argument('--train_seed', type=int, default=0)
    arg_parser.add_argument('--orig_trainset_size', type=int, default=50000)
    arg_parser.add_argument('--used_training_size', type=int, default=1000)
    arg_parser.add_argument('--dataset', type=str, default='cifar10')
    arg_parser.add_argument('--model', type=str, default='CIFAR10Net')
    arg_parser.add_argument('--batch_size', type=int, default=128)
    arg_parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'])
    arg_parser.add_argument('--epochs', type=int, default=200)
    arg_parser.add_argument('--n_classes', type=int, default=10)
    arg_parser.add_argument('--learning_rate', type=float, default=0.1)
    arg_parser.add_argument('--momentum', type=float, default=0.9)
    arg_parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay, or l2_regularization for SGD')
    arg_parser.add_argument('--exclude_indices', type=list, default=None, help='list of indices to leave out of training.')
    arg_parser.add_argument('--data_augment', default=False, action='store_true')
    arg_parser.add_argument('--log_loss', default=False, action='store_true')
    args = arg_parser.parse_args()
    main(args)
