import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_run(im,ground,output,mode):
    with torch.no_grad():
        (det,seg) = output
        im = im.cpu().numpy().transpose((1,2,0))
        if mode == 'both':
            #(ground_det,ground_seg)= ground
            #print(ground_det,ground_seg)
            ground_seg=ground[1]
            ground_det=ground[0].numpy().transpose((1,2,0))

        if mode == 'seg':
            ground_seg=ground
            ground_det=np.zeros(im.shape)

        if mode == 'det':
            ground_det=ground.numpy().transpose((1,2,0))
            ground_seg=np.zeros(im.shape)

        det = det -  torch.min(det)
        det = det/torch.max(det)
        det= det.cpu().numpy().transpose((1,2,0))

        im = (im - im.reshape(-1, im.shape[-1]).min(axis=0))
        im = (im/im.reshape(-1, im.shape[-1]).max(axis=0))

        seg = seg.cpu().argmax(axis=0)

        fig,[[ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(2,3, figsize=(15,10))
        ax1.imshow(im)
        ax2.imshow(ground_det)
        ax3.imshow(ground_seg)
        ax5.imshow(det)
        ax6.imshow(seg)
        ax4.axis('off')
        plt.show()
