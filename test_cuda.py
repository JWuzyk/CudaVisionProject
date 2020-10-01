import numpy as np
import torch
from PIL import ImageFile

from metric_detection import get_score, score
from metric_segmentation import IOU_run
from visualize import visualize_run
from criterion import criterion

import mode_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(args, model, device, test_loader,mode,logger):
    model.eval()
    test_loss = 0
    correct = 0
    results = np.zeros((3,3))
    accuracy = np.zeros(3)
    IOU = np.zeros(3)
    cnt = 0
    toDevice = mode_transforms.ToDevice(mode = mode, device = device)
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = toDevice(sample)
            output = model(data)
            test_loss += criterion(output,target,mode,device,args['l1'],args['l2'],args['l3'])

            if mode == 'det':
                results+= score(output[0],target)

            if mode == 'seg':
                IOU_0, accuracy_0 = IOU_run(output[1],target)
                IOU+=IOU_0
                accuracy+= accuracy_0
                cnt+=data.shape[0]

            if mode == 'both':
                results+= score(output[0],target[0])
                IOU_0, accuracy_0 = IOU_run(output[1],target[1])
                IOU+=IOU_0
                accuracy+= accuracy_0
                cnt+=data.shape[0]


    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: {mode} Average loss: {test_loss:.4f}\n')
    if logger:
        logger.log.loc[0,f'Test Loss {mode}']=test_loss.item()

    if mode == 'det':
        score_b = get_score(results[0])
        score_g = get_score(results[1])
        score_r = get_score(results[2])
        print(f'Balls: F1: {score_b[0]:.4f}, accuracy: {score_b[1]:.4f}, recall: {score_b[2]:.4f}, precision: {score_b[3]:.4f}, FDR: {score_b[4]:.4f},')
        print(f'Goalposts: F1: {score_g[0]:.4f}, accuracy: {score_g[1]:.4f}, recall: {score_g[2]:.4f}, precision: {score_g[3]:.4f}, FDR: {score_g[4]:.4f},')
        print(f'Robots: F1: {score_r[0]:.4f}, accuracy: {score_r[1]:.4f}, recall: {score_r[2]:.4f}, precision: {score_r[3]:.4f}, FDR: {score_r[4]:.4f},')
        if logger:
            logger.log.loc[0,'Balls correct':'Balls FP']=results[0]
            logger.log.loc[0,'Goals correct':'Goals FP']=results[1]
            logger.log.loc[0,'Robots correct':'Robots FP']=results[2]
       # visualize_run(data.cpu()[0],target.cpu()[0],(output[0].cpu()[0],output[1].cpu()[0]),mode='det')


    if mode == 'seg':
        # IOU metrics
        IOU=IOU/cnt
        accuracy=accuracy/cnt
        print(f'Background: IOU: {IOU[0]:.4f}, accuracy: {accuracy[0]:.4f}')
        print(f'Lines: IOU: {IOU[1]:.4f}, accuracy: {accuracy[1]:.4f}')
        print(f'Field: IOU: {IOU[2]:.4f}, accuracy: {accuracy[2]:.4f}')
        print(f'Total: IOU: {(IOU[0]+IOU[2]+IOU[1])/3:.4f}, accuracy: {(accuracy[0]+accuracy[2]+accuracy[1])/3:.4f}')
        if logger:
            logger.log.loc[0,'Background IOU':'Field IOU']= IOU
            logger.log.loc[0,'Background Acc':'Field Acc']= accuracy
#        visualize_run(data.cpu()[0],target.cpu()[0],(output[0].cpu()[0],output[1].cpu()[0]),mode='seg')

    if mode == 'both':
        score_b = get_score(results[0])
        score_g = get_score(results[1])
        score_r = get_score(results[2])
        print(f'Balls: F1: {score_b[0]:.2f}, accuracy: {score_b[1]:.2f}, recall: {score_b[2]:.2f}, precision: {score_b[3]:.2f}, FDR: {score_b[4]:.2f},')
        print(f'Goalposts: F1: {score_g[0]:.2f}, accuracy: {score_g[1]:.2f}, recall: {score_g[2]:.2f}, precision: {score_g[3]:.2f}, FDR: {score_g[4]:.2f},')
        print(f'Robots: F1: {score_r[0]:.2f}, accuracy: {score_r[1]:.2f}, recall: {score_r[2]:.2f}, precision: {score_r[3]:.2f}, FDR: {score_r[4]:.2f},')
        if logger:
            logger.log.loc[0,'Balls correct':'Balls FP']=results[0]
            logger.log.loc[0,'Goals correct':'Goals FP']=results[1]
            logger.log.loc[0,'Robots correct':'Robots FP']=results[2]

        IOU=IOU/cnt
        accuracy=accuracy/cnt
        print(f'Background: IOU: {IOU[0]:.2f}, accuracy: {accuracy[0]:.2f}')
        print(f'Lines: IOU: {IOU[1]:.2f}, accuracy: {accuracy[1]:.2f}')
        print(f'Field: IOU: {IOU[2]:.2f}, accuracy: {accuracy[2]:.2f}')
        print(f'Total: IOU: {(IOU[0]+IOU[2]+IOU[1])/3:.2f}, accuracy: {(accuracy[0]+accuracy[2]+accuracy[1])/3:.2f}')
        if logger:
            logger.log.loc[0,'Background IOU':'Field IOU']= IOU
            logger.log.loc[0,'Background Acc':'Field Acc']= accuracy

        visualize_run(data.cpu()[0],(target[0].cpu()[0],target[1].cpu()[0]),(output[0].cpu()[0],output[1].cpu()[0]),mode='both')
