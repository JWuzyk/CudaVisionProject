#https://discuss.pytorch.org/t/yet-another-post-on-custom-loss-functions/14552
import torch

def anisotropicTVLoss(input):
    #bsize, chan, height, width = input.size()
    dy = torch.abs(input[:,:,1:,:] - input[:,:,:-1,:])
    dx = torch.abs(input[:,:,:,1:] - input[:,:,:,:-1])
    error = torch.sum(dy)+torch.sum(dx)
    return error 

def isotropicTVLoss(input):
    #bsize, chan, height, width = input.size()
    dx = input - torch.cat((input[:,:,:,1:],input[:,:,:,-1:]),3)
    dy = input - torch.cat((input[:,:,1:,:],input[:,:,-1:,:]),2)
    
    dxdy = torch.cat((dx[None,:,:,:,:],dy[None,:,:,:,:]))
    error = torch.norm(dxdy,p=2,dim=0).mean()
    # The two lines above are equivalent to error = (dx.pow(2) + dy.pow(2)).sqrt().mean()
    # However, autograd does not work with sqrt(). Cause is exploding gradient
    return error
