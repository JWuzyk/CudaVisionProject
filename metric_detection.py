import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import cv2 as cv

def get_results(arr_pred,arr_target,min_d):
    '''
        This returns (0,0,0) if the network makes no predictions or there are no true positives
        probably should be changed to None.
    '''

    if len(arr_target)==0 or len(arr_pred)==0:
        return np.array([0,len(arr_target),0])

    targets = [0]*len(arr_target)
    for i in range(len(targets)):
        targets[i]=i
    correct = 0 #True positive TP
    wrong = 0
    for i in range(len(arr_pred)):
        for j in targets:
            d = (arr_pred[i][0]-arr_target[j][0])**2+(arr_pred[i][1]-arr_target[j][1])**2
            if d <= min_d**2:
                correct+=1
                targets.remove(j)
                break

    wrong = len(arr_pred)-correct #False Positive FP
    FN = len(targets) #False Negatives FN
  
    return np.array([correct,wrong,FN])

def get_max(im):
    im = np.float32(im)
    ret, thresh = cv.threshold(im, 0.2, 1, 0)
    thresh=np.uint8(thresh)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    peaks = []
    try:    
        peak=np.mean(contours[0].reshape(-1,2),0)
        peaks.append(peak)
    except:
        return([])
    for j,k in enumerate(contours[1:]):
        peak=np.mean(k.reshape(-1,2),0)
        peaks.append(peak)
    return peaks
    '''
    peaks = peak_local_max(im,min_distance=4,exclude_border = True,threshold_abs=0.2)

    rem = []
    for i in range(len(peaks)):
        for j in range(i+1,len(peaks)):
            d= (peaks[i][0]-peaks[j][0])**2+(peaks[i][0]-peaks[j][0])**2
            if d < 16 and d>0:
                rem.append(j)            
    peaks[rem]=-99
    peaks=peaks[ peaks!=np.array([-99,-99])]
    peaks=peaks.reshape(-1,2)
    return peaks
    '''
def score(im,target,min_d=8):

    # Input: 2 tensors of dim (batch,channels,height,width)
    # Output: array containing the number of true positive, False Positive and False Negatives for balls,goals and robots in the images 
    with torch.no_grad():
        im,target=im.cpu().numpy(),target.cpu().numpy()
        result_b,result_g,result_r = np.array([0,0,0],dtype='float64'),np.array([0,0,0],dtype='float64'),np.array([0,0,0],dtype='float64') # TP,FP,FN

        for i in range(im.shape[0]):
            im_maxs_b = get_max(im[i,0,:,:]) #balls
            im_maxs_g = get_max(im[i,1,:,:]) #goals
            im_maxs_r = get_max(im[i,2,:,:]) #robots
            target_maxs_b = get_max(target[i,0,:,:]) #balls
            target_maxs_g = get_max(target[i,1,:,:]) #goals
            target_maxs_r = get_max(target[i,2,:,:]) #robots
            result_b += get_results(im_maxs_b,target_maxs_b,min_d)
            result_g += get_results(im_maxs_g,target_maxs_g,min_d)
            result_r += get_results(im_maxs_r,target_maxs_r,min_d)

    return np.array([result_b,result_g,result_r])

def get_score(arr):
    correct,wrong,FN = arr[0],arr[1],arr[2]

    precision = correct/(correct+wrong)
    recall = correct/(correct+FN)
    FDR = 1 - precision
    if (recall+precision)==0:
        F1 = 0
    else:
        F1 = 2 * recall*precision/(recall+precision)
    accuracy = correct/(correct+wrong+FN)
    return [F1,accuracy,recall,precision,FDR]
