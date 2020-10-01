import numpy as np
import xml.etree.ElementTree as ET
from scipy.stats import multivariate_normal
from PIL import Image
from torchvision import transforms

def multivariat_noise(size,position,covariance = [[128, 0], [0, 128]]):
    rv = multivariate_normal([int((position[3]+position[1])/2),int((position[2]+position[0])/2)], covariance)
    x, y = np.mgrid[0:size[0]:1, 0:size[1]:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    return np.sqrt((2*np.pi)**2*16)*rv.pdf(pos)
    
def xml_to_pil(filename):
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

    target_size = (h,w)
    target = np.zeros((h,w,3))
    for i in balls:
        target[:,:,0] +=  multivariat_noise(target_size,i)
    for i in goalposts:
        target[:,:,1] += multivariat_noise(target_size,i)
    for i in robots:
        target[:,:,2] += multivariat_noise(target_size,i) #Does not weight robots more.

    if target.max() != 0:
        target = target/target.max()
    return transforms.ToPILImage()((target * 255).astype(np.uint8))

def seg_to_pil(seg_loc):
    seg_pil = Image.open(seg_loc)
    seg_np = np.array(seg_pil)

    # Input coding
    background = seg_np == 0
    ball = seg_np == 1
    lines = seg_np == 2
    field = seg_np == 3

    # Change encoding
    seg_np[ball] = 2 # ball set to field
    seg_np[lines] = 1 # set lines encoding from 2 to 1
    seg_np[field] = 2 # set field encoding from 3 to 2

    return Image.fromarray(seg_np)
