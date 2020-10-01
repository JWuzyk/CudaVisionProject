import time
import torch.nn as nn
import torch.nn.functional as F

from tv_loss import isotropicTVLoss
from tv_loss import anisotropicTVLoss

def criterion(output, target,mode,device,l1=1,l2=1,l3=0,l4=0,verbose = False):
    '''
        mse loss for detection weighted by l1
        pixelwise Nll loss for segmentation with errors on lines weighted more
        TV loss weighted by l3(keep small)
    '''
    t0 = time.time()
    nll = nn.NLLLoss()
    
    loss = 0 
    if mode == 'both':
        det,seg = target[0] , target[1] #.to(device).float(),target[1].to(device).float()
        loss = l1*F.mse_loss(output[0], det) \
                + l2*nll(F.log_softmax(output[1]), seg.long()) \
                + l3*(isotropicTVLoss(output[0]) +isotropicTVLoss(output[1][:,2:3,:,:])+isotropicTVLoss(output[1][:,0:1,:,:])) \
                + l4*(anisotropicTVLoss(output[0]) +anisotropicTVLoss(output[1][:,2:3,:,:])+anisotropicTVLoss(output[1][:,0:1,:,:]))
    if mode == 'det':
        det = target #.to(device).float()
        loss = l1*F.mse_loss(output[0], det) \
                + l3*(isotropicTVLoss(output[0]))\
                + l4*(anisotropicTVLoss(output[0]))
    if mode == 'seg':
        seg = target #.to(device).float()
        loss = l2*nll(F.log_softmax(output[1]), seg.long()) \
                + l3*(isotropicTVLoss(output[1][:,2:3,:,:])+isotropicTVLoss(output[1][:,0:1,:,:]))\
                + l4*(anisotropicTVLoss(output[1][:,2:3,:,:])+anisotropicTVLoss(output[1][:,0:1,:,:]))

        
    if verbose:
        with torch.no_grad():
            #Segmentation metrics
            z= y[1].argmax(axis=1)
            print(f'acc: {z[z==seg].size()[0]/seg.view((-1)).size()[0]:.4f}')
            #IOU=[0,0,0]
            #seg[seg==0]=1,seg[seg==1]=0,seg[seg==2]=0
            #print(y[0][0])
            #print('L1: ', (torch.sum(torch.abs(y[0]-det))/y[0].size()[0]).item())
            #print(f'MSE loss: {l1*F.mse_loss(y[0], det)} NNL Loss: {l2*nll(F.log_softmax(y[1]), seg.long())} TV Loss: {l3*(isotropicTVLoss(y[0])+isotropicTVLoss(y[1][:,1,:,:][:,None,:,:]))}')
            #score(y[0][0].cpu().T.numpy(),det[0].cpu().T.numpy())
        
    return loss
