import numpy as np
import xml.etree.ElementTree as ET
from scipy.stats import multivariate_normal
import skimage.io as io
from skimage.transform import rescale

def xml_to_im(filename):
    ''' return a 3 channel array with the correpsonding bounding boxes filled in 
    channel 0: ball
    channel 1: goalpost
    channel 2: robot
    scaled down by 4
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    balls=[]
    goalposts=[]
    robots=[]
    w = int(root.find('size').find('width').text)
    h = int(root.find('size').find('height').text)

    for child in root.iter('object'):
        for subchild in child.iter('bndbox'):
            if child.find('name').text == 'ball':
                balls.append([int(subchild[0].text),int(subchild[1].text),int(subchild[2].text),int(subchild[3].text)])
            if child.find('name').text == 'goalpost':
                goalposts.append([int(subchild[0].text),int(subchild[1].text),int(subchild[2].text),int(subchild[3].text)])
            if child.find('name').text == 'robot':
                robots.append([int(subchild[0].text),int(subchild[1].text),int(subchild[2].text),int(subchild[3].text)])

    target = np.zeros((int(h/4),int(w/4),3))
    for i in balls:
        rv = multivariate_normal([int((i[3]+i[1])/8),int((i[2]+i[0])/8)], [[4, 0], [0, 4]])
        x, y = np.mgrid[0:int(h/4):1, 0:int(w/4):1]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        target[:,:,0] +=  np.sqrt((2*np.pi)**2*16)*rv.pdf(pos)
    for i in goalposts:
        rv = multivariate_normal([int((i[1]+i[3])/8),int((i[0])/4)], [[4, 0], [0, 4]])
        x, y = np.mgrid[0:int(h/4):1, 0:int(w/4):1]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        target[:,:,1] +=  np.sqrt((2*np.pi)**2*16)*rv.pdf(pos)
    for i in robots:
        rv = multivariate_normal([int((i[1]+i[3])/8),int((i[0])/4)], [[4, 0], [0, 4]])
        x, y = np.mgrid[0:int(h/4):1, 0:int(w/4):1]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        target[:,:,2] +=  2*np.sqrt((2*np.pi)**2*16)*rv.pdf(pos) #Weight robots more since the network struggles with them

    return target

def seg_to_im(seg_loc):
    
    '''
    0-background
    1-lines
    2-field
    '''
    
    seg=np.array(io.imread(seg_loc))
    if seg.shape[-1]==4:
        seg=seg/128
        seg = seg[:,:,0]+seg[:,:,1]
        seg = rescale(seg, 0.25, anti_aliasing=False)
        seg=np.round(seg)
    else:
        seg[seg==1]=3 # assign ball to be field
        seg[seg==2]=1
        seg[seg==3]=2
        seg = rescale(seg, 0.25, anti_aliasing=False)
        seg=seg*2/(np.max(seg))
        seg=np.round(seg)
    return seg
