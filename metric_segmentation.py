import numpy as np
import torch
import matplotlib.pyplot as plt

def count(array):
    return np.concatenate(((array == 0).sum(axis = 1)[:,None],
                           (array == 1).sum(axis = 1)[:,None],
                           (array == 2).sum(axis =1)[:,None]), axis=1)

def IOU_run(seg,seg_true):

    if str(seg.device) == 'cpu':
        seg = seg.argmax(axis = 1).numpy()
    else:
        seg = seg.argmax(axis = 1).cpu().numpy()
    if str(seg_true.device) == 'cpu':
        seg_true = seg_true.numpy()
    else:
        seg_true = seg_true.cpu().numpy()

    number_predictions = seg_true.shape[1]*seg_true.shape[2]

    seg_true = seg_true.reshape((seg_true.shape[0], number_predictions)).astype(np.int)
    seg = seg.reshape((seg.shape[0], number_predictions)).astype(np.int)

    categories = ['background','lines','field']
    true = {}
    pred = {}
    true_positives_imagewise = {}
    true_negatives_imagewise = {}
    false_positives_imagewise = {}
    false_negatives_imagewise = {}
    IOU_imagewise= np.empty((seg.shape[0],3))
    accuracy_imagewise= np.empty((seg.shape[0],3))
    for i,cat in enumerate(categories):
        pred[cat] = seg == i
        true[cat] = seg_true == i

        true_positives_imagewise[cat] = np.logical_and(pred[cat],true[cat]).sum(axis = 1)
        true_negatives_imagewise[cat] = np.logical_and(np.logical_not(pred[cat]),np.logical_not(true[cat])).sum(axis = 1)
        false_positives_imagewise[cat] = np.logical_and(pred[cat],np.logical_not(true[cat])).sum(axis = 1)
        false_negatives_imagewise[cat] = np.logical_and(np.logical_not(pred[cat]),true[cat]).sum(axis = 1)

        IOU_imagewise[:,i] = true_positives_imagewise[cat]/(true_positives_imagewise[cat] + false_positives_imagewise[cat] + false_negatives_imagewise[cat])
      
        accuracy_imagewise[:,i] = (true_positives_imagewise[cat] + true_negatives_imagewise[cat])/number_predictions

    IOU_imagewise[np.isnan(IOU_imagewise)] = 0 # Ignore values equal to infinite

    return IOU_imagewise.sum(axis = 0), accuracy_imagewise.sum(axis = 0)
