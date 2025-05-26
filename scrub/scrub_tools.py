import argparse
import random
import copy
import numpy as np
import pandas as pd
import os
import torch
from torch.nn.utils import parameters_to_vector as p2v
from torch.nn.utils import vector_to_parameters as v2p
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.datasets import CelebA
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm

from hypercolumn import HyperC, ActivationsHook, NLP_ActivationsHook
from grad_utils import getGradObjs, gradNorm, getHessian, getVectorizedGrad, getOldPandG

import sys
# sys.path.append('../')
sys.path.append('/root/unlearning/unlearning-attack/unlearning-attack/LCODEC-deep-unlearning')
#from codec import foci, cheap_foci
from codec import torch_foci as foci
# import codec.torch_foci as foci

# TODO：
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image - epsilon*sign_data_grad    # 注意符号，+和-都可以试试
    # 添加剪切以维持[0,1]范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image

def pgd_attack(model, data, target, epsilon, steps, criterion=None):
    """
    生成通过PGD方法扰动的数据。

    Args:
    - model: 模型。
    - data: 原始数据。
    - target: 真实标签。
    - epsilon: 每步扰动的幅度。
    - steps: 迭代步数。
    - criterion: 损失函数。

    Returns:
    - perturbed_data: 扰动后的数据。
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    
    perturbed_data = data.clone().detach().requires_grad_(True)
    step_epsilon = epsilon / steps

    for _ in range(steps):
        model.zero_grad()
        output = model(perturbed_data)
        loss = criterion(output, target)
        loss.backward()

        # 使用梯度的符号来获取扰动方向，并更新扰动数据
        data_grad = perturbed_data.grad.data
        perturbed_data = perturbed_data + step_epsilon * data_grad.sign()
        perturbed_data = torch.clamp(perturbed_data, 0, 1)  # 假设数据已归一化

        # 为下一步迭代准备，去除梯度信息
        perturbed_data = perturbed_data.detach().requires_grad_(True)
    # print("---------------------------------------------------")
    return perturbed_data


def entropy_loss(outputs):
    """
    使用熵作为损失函数，熵越大，模型对分类的不确定性越高。
    """
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    log_probabilities = torch.nn.functional.log_softmax(outputs, dim=1)
    entropy = -torch.sum(probabilities * log_probabilities, dim=1).mean()
    return -entropy  # Minimizing negative entropy maximizes entropy

def iterative_entropy_attack(model, data, epsilon, steps, criterion=None):
    """
    生成通过迭代式方法扰动的数据，目标是增大输出的熵，使得模型对于多个类别的预测概率更加接近。

    Args:
    - model: 模型。
    - data: 原始数据。
    - target: 真实标签。
    - epsilon: 每步扰动的幅度。
    - steps: 迭代步数。
    - criterion: 损失函数。

    Returns:
    - perturbed_data: 扰动后的数据。
    """
    if criterion is None:
        criterion = entropy_loss
    
    perturbed_data = data.clone().detach().requires_grad_(True)
    step_epsilon = epsilon / steps

    for _ in range(steps):
        model.zero_grad()
        output = model(perturbed_data)
        loss = criterion(output)
        loss.backward()

        # 使用梯度的符号来获取扰动方向，并更新扰动数据
        data_grad = perturbed_data.grad.data
        perturbed_data = perturbed_data + step_epsilon * data_grad.sign()
        perturbed_data = torch.clamp(perturbed_data, 0, 1)  # 假设数据已归一化

        # 为下一步迭代准备，去除梯度信息
        perturbed_data = perturbed_data.detach().requires_grad_(True)

    return perturbed_data



def DisableBatchNorm(model):
    for name ,child in (model.named_children()):
        if name.find('BatchNorm') != -1:
            pass
        else:
            child.eval()
            child.track_running_stats=False

    return model

def stats(model):

    print('params norm:')
    print(sum([p.norm(2) for p in model.parameters()]))
    print('params max:')
    print(np.max([p.norm(2).item() for p in model.parameters()]))
    print('params min:')
    print(np.min([p.norm(2).item() for p in model.parameters()]))

    return


def LBFGSTorch(fresh_model, datapoint, criterion, params, device):

    model = copy.deepcopy(fresh_model)

    x, y_true = datapoint[0][0], datapoint[0][1]
    x, y_true = torch.Tensor(x).to(device), torch.Tensor([y_true]).type(torch.long).to(device)

    optim= torch.optim.LBFGS(model.parameters())

    print('inside deepcopy')
    stats(model)

    def closure():
        optim.zero_grad()
        output = model(x)
        loss = criterion(output, y_true)
        loss.backward()
        return loss

    optim.step(closure)

    
    print('inside deepcopy')
    stats(model)

    return

        


def inp_perturb(model, dataset, criterion, params, optim, device, outString):
    '''
        Works for slice selection of Linear (columns) and Convolution (filters) layers. 
    '''
    if params.scrub_batch_size is not None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=params.scrub_batch_size,
                                                    shuffle=False, num_workers=1)
        x, y_true = next(iter(data_loader))
        x = x.to(device)
        y_true = y_true.to(device)
    else:
        x, y_true = dataset[0][0], dataset[0][1]
        x, y_true = torch.Tensor(x).to(device), torch.Tensor([y_true]).type(torch.long).to(device)
        x.unsqueeze_(0)
        
        if params.unlearning_attack == 'black_box':
            model_copy = copy.deepcopy(model)
            data, target = dataset[0][0], dataset[0][1]
            data, target = torch.Tensor(data).to(device), torch.Tensor([target]).type(torch.long).to(device)

            data.requires_grad = True
            # # cifar用，添加维度
            # if data.dim() == 3:
            #     data = data.unsqueeze(0)  # Check and add batch dimension if necessary
            # 通过模型前向传递数据
            output = model(data)
            # 计算损失
            loss = criterion(output, target)
            # 将所有现有的渐变归零
            model.zero_grad()
            # 计算后向传递模型的梯度
            loss.backward()
            # 收集datagrad
            data_grad = data.grad.data
            # 进行攻击
            # perturbed_data = fgsm_attack(data, params.as_epsilon, data_grad)
            # perturbed_data = pgd_attack(model, data, target, params.as_epsilon, params.pgd_steps)
            perturbed_data = iterative_entropy_attack(model, data, params.as_epsilon, params.pgd_steps)
            data = torch.Tensor(perturbed_data.tolist()).to(device)
            x = torch.Tensor(perturbed_data.tolist()).to(device)


    model_copy = copy.deepcopy(model)

    myActs = ActivationsHook(model)

    torchLayers = myActs.getLayers()

    activations = []
    layers = None # same for all, filled in by loop below
    losses = []

    model.eval()
    if params.selectionType == 'Random' or params.selectionType == 'FOCI':
        # print("Starting input perturbation")
        for m in range(params.n_perturbations):

            tmpdata = x + (0.1)*torch.randn(x.shape).to(device)
            acts, out = myActs.getActivations(tmpdata.to(device))
            loss = criterion(out, y_true)
            vec_acts = p2v(acts)

            activations.append(vec_acts.detach())
            losses.append(loss.detach())

        acts = torch.vstack(activations)
        losses = torch.Tensor(losses).to(device)
    else:
            tmpdata = x + (0.1)*torch.randn(x.shape).to(device)
            acts, out = myActs.getActivations(tmpdata.to(device))
            vec_acts = p2v(acts)
        
    # descructor is not called on return for this
    # call it manually
    myActs.clearHooks()

    # run selection
    if params.selectionType == 'Full':
        selectedActs = np.arange(len(vec_acts)).tolist()

    elif params.selectionType == 'Random':
        foci_result, _ = foci(acts, losses, earlyStop=True, verbose=False)
        selectedActs = np.random.permutation(len(vec_acts))[:int(len(foci_result))]
        
    elif params.selectionType == 'One':
        selectedActs = [np.random.permutation(len(vec_acts))[0]]

    elif params.selectionType == 'FOCI':
        # print('Running FOCI...')
        if params.FOCIType == 'full':
            # print('Running full FOCI...')
            selectedActs, scores = foci(acts, losses, earlyStop=True, verbose=False)
        elif params.FOCIType == 'cheap':
            # print('Running cheap FOCI...')
            selectedActs, scores = cheap_foci(acts, losses)
        else:
            error('unknown foci type')

    else: 
        error('unknown scrub type')

    # create mask for update
    # params_mask = [1 if i in params else 0 for i in range(vec_acts.shape[1])]

    slices_to_update = reverseLinearIndexingToLayers(selectedActs, torchLayers)
    # print('Selected model blocks to update:')
    # print(slices_to_update)

    ############ Sample Forward Pass ########
    model.train()
    model = DisableBatchNorm(model)
    total_loss = 0
    total_accuracy = 0

    
    y_pred = model(x)
    sample_loss_before = criterion(y_pred, y_true)
    # print('Sample Loss Before: ', sample_loss_before)

    ####### Sample Gradient
    optim.zero_grad()
    sample_loss_before.backward()

    fullprevgradnorm = gradNorm(model)
    # print('Sample Gradnorm Before: ', fullprevgradnorm)

    sampGrad1, _ = getGradObjs(model)
    vectGrad1, vectParams1, reverseIdxDict = getVectorizedGrad(sampGrad1, model, slices_to_update, device)
    #vectGrad1full = p2v(sampGrad1)
    #vectGrad1 = [vectGrad1full[i] for i in params]
    model.zero_grad()

    if params.order == 'Hessian':

        # old hessian
        #second_last_name = outString + '_epoch_'  + str(params.train_epochs-2) + "_grads.pt"
        #dwtlist = torch.load(second_last_name)
        #delwt, vectPOld, _ = getVectorizedGrad(dwtlist, model, slices_to_update, device)
        delwt, vectPOld = getOldPandG(outString, params.train_epochs-2, model, slices_to_update, device)

        #one_last_name = outString + '_epoch_'  + str(params.train_epochs-1) + "_grads.pt"
        #dwtm1list = torch.load(one_last_name)
        #delwtm1, vectPOld_1, _ = getVectorizedGrad(dwtm1list, model, slices_to_update, device)
        delwtm1, vectPOld_1 = getOldPandG(outString, params.train_epochs-1, model, slices_to_update, device)

        oldHessian = getHessian(delwt, delwtm1, params.approxType, w1=vectPOld, w2=vectPOld_1, hessian_device=params.hessian_device)

        # sample hessian
        model_copy.train()
        model_copy = DisableBatchNorm(model_copy)

        # for finite diff use a small learning rate
        # default adam is 0.001/1e-3, so use it here
        optim_copy = torch.optim.SGD(model_copy.parameters(), lr=1e-3)

        y_pred = model_copy(x)
        loss = criterion(y_pred, y_true)
        optim_copy.zero_grad()
        loss.backward()

        # step to get model at next point, compute gradients
        optim_copy.step()

        y_pred = model_copy(x)
        loss = criterion(y_pred, y_true)
        optim_copy.zero_grad()
        loss.backward()

        # print('Sample Loss after Step for Hessian: ', loss)

        sampGrad2, _ = getGradObjs(model_copy)
        vectGrad2, vectParams2, _ = getVectorizedGrad(sampGrad1, model_copy, slices_to_update, device)

        sampleHessian = getHessian(vectGrad1, vectGrad2, params.approxType, w1=vectParams1, w2=vectParams2, hessian_device=params.hessian_device)

        if params.HessType == 'Sekhari':
            # Sekhari unlearning update
            n = params.orig_trainset_size
            combinedHessian = (1/(n-1))*(n*oldHessian.to(params.hessian_device) - sampleHessian.to(params.hessian_device))

            updatedParams = NewtonScrubStep(vectParams1, vectGrad1, combinedHessian, n, l2lambda=params.l2_reg, hessian_device=params.hessian_device)
            updatedParams = NoisyReturn(updatedParams, nsamps=n, m=1, lamb=params.l2_reg, epsilon=params.epsilon, delta=params.delta, device=device)

        elif params.HessType == 'CR':
            updatedParams = CR_NaiveNewton(vectParams1, vectGrad1, sampleHessian, l2lambda=params.l2_reg, hessian_device=params.hessian_device)

        else:
            error('Unknown Hessian Update Type.')

    elif params.order == 'BP':
        updatedParams = vectParams1 + params.lr*vectGrad1

    else:
        error('unknown scrubtype')

    with torch.no_grad():
        updateModelParams(updatedParams, reverseIdxDict, model)

    y_pred = model(x)
    loss2 = criterion(y_pred, y_true)
    # print('Sample Loss After: ', loss2)
    optim.zero_grad()
    loss2.backward()

    fullscrubbedgradnorm = gradNorm(model)
    # print('Sample Gradnorm After: ', fullscrubbedgradnorm)

    model.zero_grad()

    # for future multiple scrubbing for a single sample
    #if params.FOCIType == 'cheap':
    #    foci_val = scores[0]
    #else:
    #    foci_val = 1
    foci_val = 0
 
    return foci_val, model.state_dict(), sample_loss_before, loss2, fullprevgradnorm, fullscrubbedgradnorm, len(slices_to_update)


def reid_inp_perturb(model, dataset, engine, params, device, outString):
    '''
        Works for slice selection of Linear (columns) and Convolution (filters) layers. 
    '''

    #print('prior lbfgs')
    #stats(model)
    #LBFGSTorch(model, datapoint, criterion, params, device)
    #print('post lbfgs')
    #stats(model)

    if params.scrub_batch_size is not None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=params.scrub_batch_size,
                                                 shuffle=False, num_workers=1)
        x, y_true = next(iter(data_loader))
        x = x.to(device)
        y_true = y_true.to(device)
    else:
        x, y_true = dataset[0]['img'], dataset[0]['pid']
        x, y_true = torch.Tensor(x).to(device), torch.Tensor([y_true]).type(torch.long).to(device)
        x.unsqueeze_(0)

    model_copy = copy.deepcopy(model)

    myActs = ActivationsHook(model)

    torchLayers = myActs.getLayers()

    activations = []
    layers = None # same for all, filled in by loop below
    losses = []

    # model.eval()
    model.train()
    model = DisableBatchNorm(model)
    print("Starting input perturbation")
    for m in range(params.n_perturbations):

        tmpdata = x + (0.1)*torch.randn(x.shape).to(device)
        acts, out = myActs.getActivations(tmpdata.to(device))
        loss = engine.compute_loss(engine.criterion, out, y_true)
        vec_acts = p2v(acts)

        activations.append(vec_acts.detach())
        losses.append(loss.detach())

    acts = torch.vstack(activations)
    losses = torch.Tensor(losses).to(device)

    # descructor is not called on return for this
    # call it manually
    myActs.clearHooks()

    # run selection
    if params.selectionType == 'Full':
        selectedActs = np.arange(len(vec_acts)).tolist()

    elif params.selectionType == 'Random':
        foci_result, _ = foci(acts, losses, earlyStop=True, verbose=False)
        selectedActs = np.random.permutation(len(vec_acts))[:10]

    elif params.selectionType == 'One':
        selectedActs = [np.random.permutation(len(vec_acts))[0]]

    elif params.selectionType == 'FOCI':
        print('Running FOCI...')
        if params.FOCIType == 'full':
            print('Running full FOCI...')
            selectedActs, scores = foci(acts, losses, earlyStop=True, verbose=False)
        elif params.FOCIType == 'cheap':
            print('Running cheap FOCI...')
            selectedActs, scores = cheap_foci(acts, losses)
        else:
            error('unknown foci type')

    else: 
        error('unknown scrub type')

    # create mask for update
    # params_mask = [1 if i in params else 0 for i in range(vec_acts.shape[1])]

    slices_to_update = reverseLinearIndexingToLayers(selectedActs, torchLayers)
    print('Selected model blocks to update:')
    print(slices_to_update)

    ############ Sample Forward Pass ########
    model.train()
    model = DisableBatchNorm(model)
    total_loss = 0
    total_accuracy = 0

    y_pred = model(x)
    sample_loss_before = engine.compute_loss(engine.criterion, y_pred, y_true)
    print('Sample Loss Before: ', sample_loss_before)

    ####### Sample Gradient
    engine.optimizer.zero_grad()
    sample_loss_before.backward()

    fullprevgradnorm = gradNorm(model)
    print('Sample Gradnorm Before: ', fullprevgradnorm)

    sampGrad1, _ = getGradObjs(model)
    vectGrad1, vectParams1, reverseIdxDict = getVectorizedGrad(sampGrad1, slices_to_update, device)
    #vectGrad1full = p2v(sampGrad1)
    #vectGrad1 = [vectGrad1full[i] for i in params]
    model.zero_grad()

    if params.order == 'Hessian':

        # old hessian
        #second_last_name = outString + '_epoch_'  + str(params.train_epochs-2) + "_grads.pt"
        #dwtlist = torch.load(second_last_name)
        #delwt, vectPOld, _ = getVectorizedGrad(dwtlist, slices_to_update, device)
        delwt, vectPOld = getOldPandG(outString, params.train_epochs-2, slices_to_update, device)

        #one_last_name = outString + '_epoch_'  + str(params.train_epochs-1) + "_grads.pt"
        #dwtm1list = torch.load(one_last_name)
        #delwtm1, vectPOld_1, _ = getVectorizedGrad(dwtm1list, slices_to_update, device)
        delwtm1, vectPOld_1 = getOldPandG(outString, params.train_epochs-1, slices_to_update, device)

        oldHessian = getHessian(delwt, delwtm1, params.approxType, w1=vectPOld, w2=vectPOld_1, hessian_device=params.hessian_device)

        # sample hessian
        model_copy.train()
        model_copy = DisableBatchNorm(model_copy)

        # for finite diff use a small learning rate
        # default adam is 0.001/1e-3, so use it here
        optim_copy = torch.optim.SGD(model_copy.parameters(), lr=1e-3)

        y_pred = model_copy(x)
        loss = engine.compute_loss(engine.criterion, y_pred, y_true)
        optim_copy.zero_grad()
        loss.backward()

        # step to get model at next point, compute gradients
        optim_copy.step()

        y_pred = model_copy(x)
        loss = engine.compute_loss(engine.criterion, y_pred, y_true)
        optim_copy.zero_grad()
        loss.backward()

        print('Sample Loss after Step for Hessian: ', loss)

        sampGrad2, _ = getGradObjs(model_copy)
        vectGrad2, vectParams2, _ = getVectorizedGrad(sampGrad1, slices_to_update, device)

        sampleHessian = getHessian(vectGrad1, vectGrad2, params.approxType, w1=vectParams1, w2=vectParams2, hessian_device=params.hessian_device)

        if params.HessType == 'Sekhari':
            # Sekhari unlearning update
            n = params.orig_trainset_size
            combinedHessian = (1/(n-1))*(n*oldHessian.to(params.hessian_device) - sampleHessian.to(params.hessian_device))

            updatedParams = NewtonScrubStep(vectParams1, vectGrad1, combinedHessian, n, l2lambda=params.l2_reg, hessian_device=params.hessian_device)
            updatedParams = NoisyReturn(updatedParams, nsamps=n, m=1, lamb=params.l2_reg, epsilon=params.epsilon, delta=params.delta, device=device)

        elif params.HessType == 'CR':
            updatedParams = CR_NaiveNewton(vectParams1, vectGrad1, sampleHessian, l2lambda=params.l2_reg, hessian_device=params.hessian_device)

        else:
            error('Unknown Hessian Update Type.')

    elif params.order == 'BP':
        updatedParams = vectParams1 + params.lr*vectGrad1

    else:
        error('unknown scrubtype')

    with torch.no_grad():
        updateModelParams(updatedParams, reverseIdxDict, model)

    y_pred = model(x)
    loss2 = engine.compute_loss(engine.criterion, y_pred, y_true)
    print('Sample Loss After: ', loss2)
    engine.optimizer.zero_grad()
    loss2.backward()

    fullscrubbedgradnorm = gradNorm(model)
    print('Sample Gradnorm After: ', fullscrubbedgradnorm)

    model.zero_grad()

    return model.state_dict(), sample_loss_before, loss2, fullprevgradnorm, fullscrubbedgradnorm


def getHyperColumn(model, datapoint, params, device):
    # all columns for now
    #model = copy.deepcopy(model)
    myHyperC = HyperC(model, interp_size=params.interp_size, interpolate=True).float()

    hypercolumn, layers = myHyperC.getHC(torch.Tensor(datapoint).unsqueeze_(0).to(device))
    print('Hypercolumn Shape:', hypercolumn.shape)
    return hypercolumn, layers

def myFOCI(torchInput, torchHypercolumn):
    # grayscale input
    Y = torchInput.mean(dim=0).view(-1, 1).numpy()
    X = np.transpose(torchHypercolumn.view(torchHypercolumn.shape[1], -1).cpu().detach().numpy())

    layers, scores = foci(X,Y, earlyStop=True, verbose=False)

    return layers, scores

def reverseLinearIndexingToLayers(selectedSlices, torchLayers):
    ind_list = []
    for myslice in selectedSlices:
        prevslicecnt = 0
        if isinstance(torchLayers[0], torch.nn.Conv2d):
            nextslicecnt = torchLayers[0].out_channels
        elif isinstance(torchLayers[0], torch.nn.Linear):
            nextslicecnt = torchLayers[0].out_features
        else:
            print(f'cannot reverse process layer: {torchLayers[0]}')
            return NotImplementedError

        for l in range(len(torchLayers)):
            if myslice < nextslicecnt:
                modslice = myslice - prevslicecnt
                ind_list.append([torchLayers[l], modslice])
                break

            prevslicecnt = nextslicecnt

            if isinstance(torchLayers[l+1], torch.nn.Conv2d):
                nextslicecnt += torchLayers[l+1].out_channels
            elif isinstance(torchLayers[l+1], torch.nn.Linear):
                nextslicecnt += torchLayers[l+1].out_features
            else:
                print(f'cannot reverse process layer: {torchLayers[l+1]}')
                return NotImplementedError

    return ind_list

def reverseIndexingToLayers(selectedFilters, torchLayers):
    ind_list = []# for i in range(len(torchLayers))
    for filt in selectedFilters:
        prevfiltcnt = 0
        nextfiltcnt = torchLayers[0].out_channels
        for l in range(len(torchLayers)):
            if filt < nextfiltcnt:
                modfilt = filt - prevfiltcnt
                ind_list.append([torchLayers[l], modfilt])
                break
            prevfiltcnt = nextfiltcnt
            nextfiltcnt += torchLayers[l+1].out_channels
        
    return ind_list


######## DEPRECATED
def scrubSample(model, train_data, criterion, params, optim, device, outString):
    exit(0)

def updateModelParams(updatedParams, reversalDict, model):
    for key in reversalDict.keys():
        layername, weightbias, uu = key
        start_idx, end_idx, orig_shape, param = reversalDict[key]

        # slice this update
        vec_w = updatedParams[start_idx:end_idx]

        # reshape
        reshaped_w = vec_w.reshape(orig_shape)

        #tmp = [t for t in model.named_parameters()]
        #print(layername, weightbias, uu)
        #print('param:', id(param))
        #print('model:', id(tmp[0][1]))

        # apply position update
        #print(param[uu])
        param[uu] = reshaped_w.clone().detach()
        #print(param[uu])

    return


def NewtonScrubStep(weight, grad, hessian, n, l2lambda=0, hessian_device='cpu'):

    # smoothhessian = hessian + (l2lambda)*torch.eye(hessian.shape[0]).to(hessian.device)
    # newton = torch.linalg.solve(smoothhessian, grad)

    original_device = weight.device
    smoothhessian = hessian.to(hessian_device) + (l2lambda)*torch.eye(hessian.shape[0]).to(hessian_device)
    newton = torch.linalg.solve(smoothhessian, grad.to(hessian_device))
    newton = newton.to(original_device)

    # removal, towards positive gradient direction
    new_weight = weight + (1/(n-1))*newton

    return new_weight


def CR_NaiveNewton(weight, grad, hessian, l2lambda=0, hessian_device='cpu'):
    original_device = weight.device

    smoothhessian = hessian.to(hessian_device) + (l2lambda)*torch.eye(hessian.shape[0]).to(hessian_device)
    newton = torch.linalg.solve(smoothhessian, grad.to(hessian_device)).to(original_device)

    # removal, towards positive gradient direction
    new_weight = weight + newton

    return new_weight

def NoisyReturn(weights, epsilon=0.1, delta=0.01, m=1, nsamps=50000, M=0.25, L=1.0, lamb=0.01, device='cpu'):
    # func params default for cross entropy
    # cross entropy not strongly convex, pick a small number

    gamma = 2*M*(L**2)*(m**2)/((nsamps**2)*(lamb**3))
    sigma = (gamma/epsilon)*np.sqrt(2*np.log(1.25/delta))
    # print('std for noise:', sigma)

    noise = torch.normal(torch.zeros(len(weights)), sigma).to(device)
    return weights + noise


 
