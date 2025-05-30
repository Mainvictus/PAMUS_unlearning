import argparse

import copy
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import time

from data_utils import getDatasets
from nn_utils import do_epoch, manual_seed

from scrub_tools import scrubSample, inp_perturb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def scrubMany(args):

    # outString = 'trained_models/'+args.dataset+"_"+args.model+'_epochs_' + str(args.train_epochs)
    outString = 'trained_models/'+args.dataset+"_"+args.model+'_'+str(args.used_training_size)+'_seed_' + str(args.train_seed)+'_epochs_' + str(args.train_epochs)+'_lr_' + str(args.train_lr)+'_wd_' + str(args.train_wd)+'_bs_' + str(args.train_bs)+'_optim_' + str(args.train_optim)
    if args.data_augment:
        outString = outString + "_transform"
    else:
        outString = outString + "_notransform"
    outfile = args.outfile
    outfile = outfile.replace('.csv', f'_{args.selectionType}.csv')
    
    tmp = {}
    tmp['dataset'] = [args.dataset]
    tmp['model'] = [args.model]
    tmp['train_epochs'] = [args.train_epochs]
    tmp['selectionType'] = [args.selectionType]
    tmp['order'] = [args.order]
    tmp['HessType'] = [args.HessType]
    tmp['approxType'] = [args.approxType]
    tmp['run'] = [args.run]
    tmp['orig_trainset_size'] = [args.orig_trainset_size]
    tmp['delta'] = [args.delta]
    tmp['epsilon'] = [args.epsilon]
    tmp['l2_reg'] = [args.l2_reg]


    exec("from models import %s" % args.model)
    model = eval(args.model)().to(device)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # reload model to trained state
    model.load_state_dict(torch.load(outString+".pt"))

    criterion = torch.nn.CrossEntropyLoss()

    manual_seed(args.train_seed)
    selection = np.random.permutation(args.orig_trainset_size)
    selection = selection[:args.used_training_size]

    full_dataset, val_dataset = getDatasets(name=args.dataset, data_augment=False, include_indices=selection)
        
    print('Validation Set Size: ', len(val_dataset))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=1)
    with torch.no_grad():
        val_loss, val_accuracy = do_epoch(model, val_loader, criterion, 0, 0, optim=None, device=device)
        print(f'Model:{args.model} Before: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        tmp['val_acc_before'] = [val_accuracy]
        tmp['val_loss_before'] = [val_loss]

    prev_val_acc = val_accuracy

    manual_seed(args.run)
    if args.unlearning_attack == "white_box":
        outfile = outfile.replace('.csv', f'_{args.attack_type}.csv')
        log_loss_file = outString
        log_loss_file = log_loss_file.replace('trained_models/','loss_log/')
        log_loss_file = log_loss_file + '_loss.csv'
        loss_df = pd.read_csv(log_loss_file)
        if args.attack_type == 'max':
            loss_df.sort_values('max', inplace=True)
        elif args.attack_type == 'max_diff':
            loss_df.sort_values('max_diff', inplace=True)
        elif args.attack_type == 'max_gradnorm':
            loss_df.sort_values('gradnorm', inplace=True, ascending=False)
        else:
            loss_df.sort_values('loss', inplace=True, ascending=False)
        ordering = loss_df.index.tolist()
    elif args.unlearning_attack == "black_box":
        ordering = np.random.permutation(args.used_training_size)
        pass
    else:   # random forget
        ordering = np.random.permutation(args.used_training_size)
    scrubbed_list = []
    i = 0
    j = 0
    while i < args.n_removals:
    #for i in range(args.n_removals):
        # print ('######################## GPU Memory Allocated {} MB'.format(torch.cuda.memory_allocated(device=device)/1024./1024.))

        if args.scrub_batch_size is not None:

            # select samples to scrub
            scrub_list = []
            while len(scrub_list) < args.scrub_batch_size and (i+len(scrub_list)) < args.n_removals:
                scrubee = ordering[j]
                if full_dataset[scrubee][1] == args.removal_class:
                    scrub_list.append(scrubee)
                j += 1

        else:
            # randomly select samples to scrub
            scrubee = ordering[j]
            scrub_list = [ordering[j]]
            j += 1

        #scrub_dataset = Subset(full_dataset, [scrubee])
        scrub_dataset = Subset(full_dataset, scrub_list)

        #scrubbed_list.append(scrubee)
        scrubbed_list.extend(scrub_list)

        # residual_dataset, _ = getDatasets(name=args.dataset, val_also=False, exclude_indices=scrubbed_list)
        # residual_loader = torch.utils.data.DataLoader(residual_dataset, batch_size=args.batch_size,
        #                                          shuffle=False, num_workers=1)
        
        # print('Residual dataset size: ', len(residual_dataset))
        #print('Removing: ', i, scrubee)
        # print('Removing: ', i, scrub_list)

        #tmp['scrubee'] = [scrubee]
        tmp['scrub_list'] = [scrub_list]
        tmp['n_removals'] = [i]


        # loops once for now, maybe more in future
        foci_val = 1
        while foci_val > args.cheap_foci_thresh:

            if args.unlearning_attack:
                prev_statedict_fname = outString +'_' + args.selectionType + '_' +args.unlearning_attack+'_'+args.attack_type + '_prevSD.pt'
            else:
                prev_statedict_fname = outString + '_' + args.selectionType + '_prevSD.pt'
            torch.save(model.state_dict(), prev_statedict_fname)

            # because we reload the model
            optim = torch.optim.SGD(model.parameters(), lr=args.lr)

            foci_val, updatedSD, samplossbefore, samplossafter, gradnormbefore, gradnormafter, slices_to_update = inp_perturb(model, scrub_dataset, criterion, args, optim, device, outString=outString)

            # reload for deepcopy
            # apply new weights
            # without this cannot deepcopy later
            model = eval(args.model)().to(device)
            model.load_state_dict(updatedSD)

            # for future
            #print('FOCI Value: ', foci_val)

            
            with torch.no_grad():
                val_loss, val_accuracy = do_epoch(model, val_loader, criterion, 0, 0, optim=None, device=device)

            #print(f'After: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

            # print(f'\t Previous Val Acc: {prev_val_acc}')
            # print(f'\t New Val Acc: {val_accuracy}')


            if prev_val_acc - val_accuracy > args.val_gap_skip:
                # print('########## BAD SAMPLE BATCH DETECTED, REVERTING MODEL #######')
                model = eval(args.model)().to(device)
                model.load_state_dict(torch.load(prev_statedict_fname))
                tmp['bad_sample'] = 1
            else:
                prev_val_acc = val_accuracy
                tmp['bad_sample'] = 0
                i += len(scrub_list)

            tmp['time'] = time.time()

            tmp['val_acc_after'] = [val_accuracy]
            tmp['val_loss_after'] = [val_loss]

            #tmp['foci_single_val'] = [foci_val]

            tmp['sample_loss_before'] = [samplossbefore.detach().cpu().item()]
            tmp['sample_loss_after'] = [samplossafter.detach().cpu().item()]
            tmp['sample_gradnorm_before'] = [gradnormbefore]
            tmp['sample_gradnorm_after'] = [gradnormafter]

            # resid_loss, resid_accuracy, resid_gradnorm = do_epoch(model, residual_loader, criterion, 0, 0, optim=None, device=device, compute_grads=True)
            # print('Residual Gradnorm:', resid_gradnorm)
            # tmp['residual_loss_after'] = [resid_loss]
            # tmp['residual_acc_after'] = [resid_accuracy]
            # tmp['residual_gradnorm_after'] = [resid_gradnorm]
                
            tmp['slices_to_update'] = [slices_to_update]
            tmp['y_true'] =[full_dataset[scrubee][1]]

            df = pd.DataFrame(tmp)
            if os.path.isfile(outfile):
                df.to_csv(outfile, mode='a', header=False, index=False)
            else:
                df.to_csv(outfile, mode='a', header=True, index=False)

            
    return model

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Scrub a sample')
    arg_parser.add_argument('--dataset', type=str, default='cifar10')
    arg_parser.add_argument('--model', type=str, default='resnet')
    # arg_parser.add_argument('--MODEL_FILE', type=str, default="trained_models/full.pt", help='A model in trained_models, trained using train.py')
    arg_parser.add_argument('--train_epochs', type=int, default=10, help='Number of epochs model was originally trained for, used to get last two gradients')
    arg_parser.add_argument('--interp_size', type=int, default=32, help='Size of input image to interpolate for hypercolumns')
    #arg_parser.add_argument('--scrub_index', type=int, default=None, help='Index of an example to scrub.')
    arg_parser.add_argument('--n_removals', type=int, default=10000, help='number of samples to scrub')
    arg_parser.add_argument('--orig_trainset_size', type=int, default=None, help='size of orig training set')
    arg_parser.add_argument('--used_training_size', type=int, default=None, help='size of orig training set')
    arg_parser.add_argument('--epsilon', type=float, default=0.1, help='scrubbing rate')
    arg_parser.add_argument('--delta', type=float, default=0.01, help='scrubbing rate')
    arg_parser.add_argument('--l2_reg', type=float, default=0.001, help='weight_decay or l2_reg, used for noisy return and hessian smoothing')
    arg_parser.add_argument('--lr', type=float, default=1.0, help='scrubbing rate')
    arg_parser.add_argument('--batch_size', type=int, default=128)
    arg_parser.add_argument('--scrubType', type=str, default='IP', choices=['IP','HC'])
    arg_parser.add_argument('--HessType', type=str, default='Sekhari', choices=['Sekhari','CR'])
    arg_parser.add_argument('--approxType', type=str, default='FD', choices=['FD','Fisher'])
    arg_parser.add_argument('--n_perturbations', type=int, default=1000)
    arg_parser.add_argument('--order', type=str, default='Hessian', choices=['BP','Hessian'])
    arg_parser.add_argument('--selectionType', type=str, default='FOCI', choices=['Full', 'FOCI', 'Random', 'One'])
    arg_parser.add_argument('--FOCIType', type=str, default='full', choices=['full','cheap'])
    arg_parser.add_argument('--cheap_foci_thresh', type=float, default=0.05, help='threshold for codec2 calls in cheap_foci')
    arg_parser.add_argument('--run', type=int, default=1, help='Repitition index.')
    arg_parser.add_argument('--outfile', type=str, default="scrub_ablate_results.csv", help='output file name to append to')
    arg_parser.add_argument('--updatedmodelname', type=str, help='output file name to append to')
    arg_parser.add_argument('--hessian_device', type=str, default='cpu', help='Device for Hessian computation')

    arg_parser.add_argument('--val_gap_skip', type=float, default=0.05, help='validation drop for skipping a sample to remove (should retrain)')

    arg_parser.add_argument('--scrub_batch_size', type=int, default=None)
    arg_parser.add_argument('--removal_class', type=int, default=0)
    
    # Added for new outstring
    arg_parser.add_argument('--train_seed', type=int, default=0)
    arg_parser.add_argument('--train_lr', type=float, default=0.001, help="training learning rate")
    arg_parser.add_argument('--train_wd', type=float, default=0.01, help="training weight decay")
    arg_parser.add_argument('--train_bs', type=int, default=256, help="training batch size")
    arg_parser.add_argument('--train_optim', type=str, default='sgd', choices=['sgd', 'adam'], help="training optimizer")
    #arg_parser.add_argument('--data_augment', default=False, action='store_true')
    arg_parser.add_argument('--data_augment', type=int, default=0, help='whether to augment or not') 
    arg_parser.add_argument('--unlearning_attack', type=str, default='black_box', choices=['white_box', 'black_box']) 
    arg_parser.add_argument('--as_epsilon', type=float, default=2.5) 
    arg_parser.add_argument('--attack_type', type=str, default='max', choices=['max', 'max_diff', 'loss', 'max_gradnorm']) 
    arg_parser.add_argument('--pgd_steps', type=int, default=50)
    
    args = arg_parser.parse_args()

    manual_seed(args.run)

    scrubMany(args)


