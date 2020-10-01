import time
import numpy as np
import torch
import skimage.io as io
from criterion import criterion
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import time
from datetime import datetime
import mode_transforms

def train(args, model, device, train_loader, optimizer, epoch, mode, logger):
    t_1 = time.time()
    model.train()
    results = np.zeros((3,3))
    toDevice = mode_transforms.ToDevice(mode = mode, device = device)
    t0=time.time()
    for batch_idx, sample in enumerate(train_loader):
        data, target = toDevice(sample)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target,mode,device,args['l1'],args['l2'],args['l3'])
        loss.backward()
        optimizer.step()

        if batch_idx % args['log_interval'] == 0:
            t1=time.time()
            print(f'Train Epoch {mode}: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item()/len(data):.6f}, time: {t1-t0}')
            t0=time.time()

        if logger != None:
            logger.log.loc[0,f'Train Loss {mode}']=loss.item()/args['batch_size']
    t_2 = time.time()
    if logger != None:
        logger.log.loc[0,'time']= t_2-t_1


def train_concurrent(args, model, device, train_loader, optimizer, epoch, logger):
    t_1 = time.time()
    model.train()
    results = np.zeros((3,3))
    toDevice_det = mode_transforms.ToDevice(mode = 'det', device = device)
    toDevice_seg = mode_transforms.ToDevice(mode = 'seg', device = device)
    t0=time.time()
    for batch_idx, (sample_det, sample_seg) in enumerate(train_loader):
        im_det,target_det = toDevice_det(sample_det)
        im_seg,target_seg= toDevice_seg(sample_seg)
        optimizer.zero_grad()
        output_det = model(im_det)
        output_seg = model(im_seg)
        loss = criterion(output_det,target_det,'det',device,args['l1'],args['l2'],args['l3'])+  criterion(output_seg,target_seg,'seg',device,args['l1'],args['l2'],args['l3'])
        loss.backward()
        optimizer.step()

        batch_size = im_det.shape[0]
        if batch_idx % args['log_interval'] == 0:
            t1=time.time()
            print(f'Train Epoch concurrent: {epoch} [{batch_idx * batch_size}/{len(train_loader.dataset)}({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item()/batch_size:.6f}, time: {t1-t0}')
            t0=time.time()
        logger.log.loc[0,f'Train Loss']=loss.item()/batch_size
    t_2 = time.time()
    if logger != None:
        logger.log.loc[0,'time']= t_2-t_1
