import os
import pandas as pd
import numpy as np
import torch
import csv
from torch.utils.data import Dataset
from skimage.transform import resize,rescale
import skimage.io as io
from torchvision import transforms
from PIL import Image

# Local functions
from seg_xml_to_im import seg_to_im, xml_to_im
from seg_det_to_pil import seg_to_pil, xml_to_pil
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    
class RobotsDataset(Dataset):
    """
    mode indicates whether the Dataset returns just dectetion target, segmentation target or both
    """

    def __init__(self, data_loc, seg_loc ='segmentation/', im_loc='image/', xml_loc='detection/',
            dec_loc='detection_pil/', download = False, mode = 'both', precomputed=False, transform=None):
        """
        Args:
            data_loc (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_loc = data_loc
        self.seg_loc=seg_loc
        self.im_loc=im_loc
        self.xml_loc = xml_loc


        self.dec_loc = dec_loc
        dec_path = os.path.join(self.data_loc,self.dec_loc)
        if not os.path.isdir(data_loc) and download:
            print('Downloading data')
            self.download()
        if not os.path.isfile(data_loc+'index_both.csv') and not os.path.isfile(data_loc+'index_det.csv') and not os.path.isfile(data_loc+'index_seg.csv'):
            self.create_index()
        if not os.path.isdir(dec_path):
            os.mkdir(dec_path)
            print("Directory " , dec_path ,  " Created ")

        self.index_both = pd.read_csv(data_loc+'index_both.csv')
        self.index_det = pd.read_csv(data_loc+'index_det.csv')
        self.index_seg = pd.read_csv(data_loc+'index_seg.csv')
        self.transform = transform
        self.mode=mode


    def __len__(self):
        if self.mode == 'both':
            return len(self.index_both)
        if self.mode == 'seg':
            return len(self.index_seg)
        if self.mode == 'det':
            return len(self.index_det)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.mode == 'both':           
            name = self.index_both.iloc[idx, 1]
        if self.mode == 'seg':           
            name = self.index_seg.iloc[idx, 1]
        if self.mode == 'det':           
            name = self.index_det.iloc[idx, 1]
            
        img_path = os.path.join(self.data_loc,self.im_loc,name)
        xml_path = os.path.join(self.data_loc,self.xml_loc,name)
        dec_path = os.path.join(self.data_loc,self.dec_loc,name) #Change
        seg_path = os.path.join(self.data_loc,self.seg_loc,name)
        
        if os.path.isfile(img_path+'.jpg'):
            image = Image.open(img_path+'.jpg')
        elif os.path.isfile(img_path+'.png'):
            image = Image.open(img_path+'.png')
        else:
            print('error reading ' + img_path)
              
        if self.mode=='both':
            seg = seg_to_pil(seg_path + '.png')
            #det = xml_to_pil(xml_path + '.xml')
            if os.path.exists(dec_path + '.png'): #Change
                det = Image.open(dec_path + '.png')
            else:
                det = xml_to_pil(xml_path + '.xml')
                det.save(dec_path + '.png')

            sample = (image,(det,seg))
        if self.mode=='det':
            if os.path.exists(dec_path + '.png'): #Change
                det = Image.open(dec_path + '.png')
            else:
                det = xml_to_pil(xml_path + '.xml')
                det.save(dec_path + '.png')

            sample = (image,det)
        if self.mode=='seg':
            seg = seg_to_pil(seg_path + '.png')
            sample = (image,seg)
            
        if self.transform:
            sample = self.transform(sample)
        return sample

    def create_index(self):
        i=0
        img_path = os.path.join(self.data_loc,self.im_loc)
        xml_path = os.path.join(self.data_loc,self.xml_loc)
        seg_path = os.path.join(self.data_loc,self.seg_loc)

        with open(self.data_loc + 'index_both.csv', mode='w') as index:
            index_writer = csv.writer(index, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            index_writer.writerow(['index','name'])



            gen = (name for name in os.listdir(img_path) if os.path.isfile(os.path.join(xml_path, name[:-4]+'.xml')) and os.path.isfile(os.path.join(seg_path, name[:-4]+'.png')))
            for name in gen:
                index_writer.writerow([i,name[0:-4]])
                i+=1
            print(f'Generated both csv with {i} entries')
            
        i=0            
        with open(self.data_loc + 'index_seg.csv', mode='w') as index:
            index_writer = csv.writer(index, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            index_writer.writerow(['index','name'])


            gen = (name for name in os.listdir(img_path) if os.path.isfile(os.path.join(seg_path, name[:-4]+'.png')))
            for name in gen:
                index_writer.writerow([i,name[0:-4]])
                i+=1
            print(f'Generated seg csv with {i} entries')
            
        i=0            
        with open(self.data_loc + 'index_det.csv', mode='w') as index:
            index_writer = csv.writer(index, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            index_writer.writerow(['index','name'])
            gen = (name for name in os.listdir(img_path) if os.path.isfile(os.path.join(xml_path, name[:-4]+'.xml')))
            for name in gen:
                index_writer.writerow([i,name[0:-4]])
                i+=1
            print(f'Generated det csv with {i} entries')

    def download(self):
        url = 'https://github.com/SoloJacobs/cudavisionfinalproject/blob/master/small_data.zip?raw=true'
        fpath= './'
        urllib.request.urlretrieve(url, fpath)
        with ZipFile('./small_data.zip', 'r') as zipObj:
            zipObj.extractall()
            
    def set_mode(mode):
        if mode == 'both' or mode =='det' or mode =='seg':
            self.mode=mode
        else:
            print('invalid mode')

class RobotsDatasetConcurrent(Dataset):
    """
    mode indicates whether the Dataset returns just dectetion target, segmentation target or both
    """

    def __init__(self, data_loc, seg_loc ='segmentation/', im_loc='image/', xml_loc='detection/',
            dec_loc='detection_pil/', download = False,transform_dict=None):
        """
        Args:
            data_loc (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
                
            mode is one of : both, seg, det, concurrent
        """
        self.data_loc = data_loc
        self.seg_loc=seg_loc
        self.im_loc=im_loc
        self.xml_loc = xml_loc

        self.dec_loc = dec_loc
        dec_path = os.path.join(self.data_loc,self.dec_loc)
        if not os.path.isdir(data_loc) and download:
            print('Downloading data')
            self.download()
        if not os.path.isfile(data_loc+'index_both.csv')and not os.path.isfile(data_loc+'index_det.csv') and not os.path.isfile(data_loc+'index_seg.csv'):
            self.create_index()
        if not os.path.isdir(dec_path):
            os.mkdir(dec_path)
            print("Directory " , dec_path ,  " Created ")

        self.index_both = pd.read_csv(data_loc+'index_both.csv')
        self.index_det = pd.read_csv(data_loc+'index_det.csv')
        self.index_seg = pd.read_csv(data_loc+'index_seg.csv')
        self.transform_dict = transform_dict


    def __len__(self):
        return max(len(self.index_det),len(self.index_seg))

    def __getitem__(self, idx):
        len_det = len(self.index_det)
        len_seg = len(self.index_seg)
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name_seg = self.index_seg.iloc[idx % len_seg, 1]
        name_det = self.index_det.iloc[idx % len_det, 1]            
            
        img_path_seg = os.path.join(self.data_loc,self.im_loc,name_seg)
        img_path_det = os.path.join(self.data_loc,self.im_loc,name_det)

        xml_path = os.path.join(self.data_loc,self.xml_loc,name_det)
        seg_path = os.path.join(self.data_loc,self.seg_loc,name_seg)
        dec_path = os.path.join(self.data_loc,self.dec_loc,name_det) #Change
        
        if os.path.isfile(img_path_seg+'.jpg'):
            image_seg = Image.open(img_path_seg+'.jpg')
        elif os.path.isfile(img_path_seg+'.png'):
            image_seg = Image.open(img_path_seg+'.png')
        else:
            print('error reading ' + img_path_seg)
            
        if os.path.isfile(img_path_det+'.jpg'):
            image_det = Image.open(img_path_det+'.jpg')
        elif os.path.isfile(img_path_det+'.png'):
            image_det = Image.open(img_path_det+'.png')
        else:
            print('error reading ' + img_path_det)
        
        seg = seg_to_pil(seg_path + '.png')
        if os.path.exists(dec_path + '.png'): #Change
            det = Image.open(dec_path + '.png')
        else:
            det = xml_to_pil(xml_path + '.xml')
            det.save(dec_path + '.png')
        

        if self.transform_dict:
            sample_det = self.transform_dict['det']((image_det,det))
            sample_seg = self.transform_dict['seg']((image_seg,seg))
        return sample_det, sample_seg

    def create_index(self):
        i=0
        img_path = os.path.join(self.data_loc,self.im_loc)
        xml_path = os.path.join(self.data_loc,self.xml_loc)
        seg_path = os.path.join(self.data_loc,self.seg_loc)
        with open(self.data_loc + 'index_both.csv', mode='w') as index:
            index_writer = csv.writer(index, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            index_writer.writerow(['index','name'])



            gen = (name for name in os.listdir(img_path) if os.path.isfile(os.path.join(xml_path, name[:-4]+'.xml')) and os.path.isfile(os.path.join(seg_path, name[:-4]+'.png')))
            for name in gen:
                index_writer.writerow([i,name[0:-4]])
                i+=1
            print(f'Generated both csv with {i} entries')
            
        i=0            
        with open(self.data_loc + 'index_seg.csv', mode='w') as index:
            index_writer = csv.writer(index, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            index_writer.writerow(['index','name'])


            gen = (name for name in os.listdir(img_path) if os.path.isfile(os.path.join(seg_path, name[:-4]+'.png')))
            for name in gen:
                index_writer.writerow([i,name[0:-4]])
                i+=1
            print(f'Generated seg csv with {i} entries')
            
        i=0            
        with open(self.data_loc + 'index_det.csv', mode='w') as index:
            index_writer = csv.writer(index, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            index_writer.writerow(['index','name'])
            gen = (name for name in os.listdir(img_path) if os.path.isfile(os.path.join(xml_path, name[:-4]+'.xml')))
            for name in gen:
                index_writer.writerow([i,name[0:-4]])
                i+=1
            print(f'Generated det csv with {i} entries')

    def download(self):
        url = 'https://github.com/SoloJacobs/cudavisionfinalproject/blob/master/small_data.zip?raw=true'
        fpath= './'
        urllib.request.urlretrieve(url, fpath)
        with ZipFile('./small_data.zip', 'r') as zipObj:
            zipObj.extractall()

import os
import pickle
import tqdm
import pandas as pd
import math
from RobotData import RobotsDataset
import mode_transforms

class MetaData:
    def __init__(self,dataset_dir,split_df = None):
        self.pd_path = os.path.join(dataset_dir,'meta.pkl')
        self.split_path = os.path.join(dataset_dir,'split.pkl')
        self.path = {}
        self.subdirs = ['segmentation','detection','image']
        for subdir in self.subdirs:
            self.path[subdir] = os.path.join(dataset_dir,subdir)

        if not os.path.isfile(self.pd_path):
            print('Initializing new meta file.')

            for subdir in self.subdirs:
                if os.path.isdir(self.path[subdir]):
                    raise ValueError('Directory ' + self.path[subdir] + ' already exists. Can not inialize new meta file.')
                else:
                    os.makedirs(self.path[subdir])

            if os.path.isfile(self.split_path):
                raise ValueError('File ' + self.split_path + ' already exists. Can not inialize new meta file.')
            if split_df is None:
                split_df = pd.DataFrame(columns=['split','mode']).rename_axis('name')
            split_df.to_pickle(self.split_path)

            columns = ['index','name','input_dimension','split','device','seg_loc','det_loc','im_loc']
            index = pd.DataFrame(columns=columns)
            index.to_pickle(self.pd_path)

        else:
            for subdir in self.subdirs:
                if not os.path.isdir(self.path[subdir]):
                    raise ValueError('Directory ' + self.path[subdir] + ' does not exist.')
            if not os.path.isfile(self.split_path):
                raise ValueError('File ' + self.split_path + ' does not exist.')

    def add_directory(self,data_dir,device,input_dimension, split = None):
        data_subdir = {}
        for subdir in self.subdirs:
                data_subdir = os.path.join(data_dir,subdir)
        meta_data = self.data()
        split_df = self.split()
        modes = ['seg','det','both']
        dataset = {}
        dataloader = {}
        for mode in modes:
            transform = transforms.Compose([
                mode_transforms.Resize(input_dimension, mode=mode),
                mode_transforms.ToTensor(mode = mode),
                mode_transforms.Normalize(),
                mode_transforms.ToDevice(mode = mode, device=device)
            ])
            dataset[mode] = RobotsDataset(data_loc = data_dir,mode = mode,transform=transform)

        index = {'det': dataset['det'].index_det,
                 'seg': dataset['seg'].index_seg,
                 'both': dataset['both'].index_both
                }

        meta_data_red = meta_data[(meta_data['input_dimension'] == input_dimension)]
        meta_data_red = meta_data_red[(meta_data_red['device'] == device)] # Only these entries are relevant.
        meta_data_has = {}
        has_seg = ~meta_data_red['seg_loc'].isnull()
        has_det = ~meta_data_red['det_loc'].isnull()
        meta_data_has['seg'] = meta_data_red[has_seg]
        meta_data_has['det'] = meta_data_red[has_det]
        meta_data_has['both'] = meta_data_red[has_det | has_seg]

        final_index = {}
        for mode in modes:
            final_index[mode] = index[mode].merge(meta_data_has[mode], on=['name'], how='left', indicator=True)
            final_index[mode] = final_index[mode][final_index[mode]['_merge'] == 'left_only'].drop(['_merge'], axis=1) # Drop already generated examples
        final_index['seg'] = final_index['seg'].merge(final_index['both']['name'], on=['name'], how='left', indicator=True)
        final_index['seg'] = final_index['seg'][final_index['seg']['_merge'] == 'left_only'].drop(['_merge'], axis=1)
        final_index['det'] = final_index['det'].merge(final_index['both']['name'], on=['name'], how='left', indicator=True)
        final_index['det'] = final_index['det'][final_index['det']['_merge'] == 'left_only'].drop(['_merge'], axis=1)

        if len(meta_data) == 0:
            available_index_dataset = 0
        else:
            available_index_dataset = meta_data['index'].max() + 1
        progress =  tqdm.tqdm_notebook(total=len(final_index['det']) + len(final_index['seg']) + len(final_index['both']))
        for index_data, name, index_dataset in final_index['det'].filter(['index_x','name','index_y']).itertuples(index=False):
            if not name in split_df.index:
                split_df.loc[name] = [split]
            if math.isnan(index_dataset):
                index_dataset = available_index_dataset
                available_index_dataset = available_index_dataset + 1
                det_loc = os.path.join(self.path['detection'],'detection' + str(index_dataset) + '.pt')
                im_loc = os.path.join(self.path['image'],'image' + str(index_dataset) + '.pt')
                image, detection = dataset['det'].__getitem__(index_data)
                torch.save(detection, det_loc)
                torch.save(image, im_loc)
                meta_data = meta_data.append({'index': index_dataset,
                                               'name': name,
                                               'input_dimension': input_dimension,
                                               'split': split_df.loc[name,'split'],
                                               'device': device,
                                               'seg_loc': None,
                                               'det_loc': det_loc,
                                               'im_loc': im_loc,
                                              }, ignore_index=True)
            else:
                location = meta_data['index'] == index_dataset
                det_loc = os.path.join(self.path['detection'],'detection' + str(index_dataset) + '.pt')
                im_loc = os.path.join(self.path['image'],'image' + str(index_dataset) + '.pt')
                _, detection = dataset['det'].__getitem__(index_data)
                meta_data.loc[location,'det_loc'] = det_loc
                torch.save(detection, det_loc)
            progress.update()

        for index_data, name, index_dataset in final_index['seg'].filter(['index_x','name','index_y']).itertuples(index=False):
            if not name in split_df.index:
                split_df.loc[name] = [split]
            if math.isnan(index_dataset):
                index_dataset = available_index_dataset
                available_index_dataset = available_index_dataset + 1
                seg_loc = os.path.join(self.path['segmentation'],'segmentation' + str(index_dataset) + '.pt')
                im_loc = os.path.join(self.path['image'],'image' + str(index_dataset) + '.pt')
                image, segmentation = dataset['seg'].__getitem__(index_data)
                torch.save(segmentation, seg_loc)
                torch.save(image, im_loc)
                meta_data = meta_data.append({'index': index_dataset,
                                               'name': name,
                                               'input_dimension': input_dimension,
                                               'split': split_df.loc[name,'split'],
                                               'device': device,
                                               'seg_loc': seg_loc,
                                               'det_loc': None,
                                               'im_loc': im_loc,
                                              }, ignore_index=True)
            else:
                location = meta_data['index'] == index_dataset
                seg_loc = os.path.join(self.path['segmentation'],'segmentation' + str(index_dataset) + '.pt')
                im_loc = os.path.join(self.path['image'],'image' + str(index_dataset) + '.pt')
                _, segmentation = dataset['seg'].__getitem__(index_data)
                meta_data.loc[location,'seg_loc'] = seg_loc
                torch.save(segmentation, seg_loc)
            progress.update()

        for index_data, name, index_dataset in final_index['both'].filter(['index_x','name','index_y']).itertuples(index=False):
            if not name in split_df.index:
                split_df.loc[name] = [split]
            if math.isnan(index_dataset):
                index_dataset = available_index_dataset
                available_index_dataset = available_index_dataset + 1
                det_loc = os.path.join(self.path['detection'],'detection' + str(index_dataset) + '.pt')
                seg_loc = os.path.join(self.path['segmentation'],'segmentation' + str(index_dataset) + '.pt')
                im_loc = os.path.join(self.path['image'],'image' + str(index_dataset) + '.pt')
                image, (detection,segmentation) = dataset['both'].__getitem__(index_data)
                torch.save(detection, det_loc)
                torch.save(segmentation, seg_loc)
                torch.save(image, im_loc)
                meta_data = meta_data.append({'index': index_dataset,
                                               'name': name,
                                               'input_dimension': input_dimension,
                                               'split': split_df.loc[name,'split'],
                                               'device': device,
                                               'seg_loc': seg_loc,
                                               'det_loc': det_loc,
                                               'im_loc': im_loc,
                                              }, ignore_index=True)
            else:
                raise ValueError("Image entry without any targets")
            progress.update()

        meta_data.to_pickle(self.pd_path)
        pickle.dump(split_df, open(self.split_path, "wb" ))

    def data(self):
        return pd.read_pickle(self.pd_path)

    def split(self):
        return pickle.load(open(self.split_path, "rb" ))

    def delete(self,input_dimension,device):
        meta_data = self.data()
        input_dimension = meta_data['input_dimension']
        device_data = meta_data['device']
        location = (meta_data['input_dimension'] == input_dimension) & (meta_data['device'] == device)
        meta_data_to_delete = meta_data.loc[location]
        for f in meta_data_to_delete['seg_loc']:
            if f:
                os.remove(f)
        for f in meta_data_to_delete['det_loc']:
            if f:
                os.remove(f)
        for f in meta_data_to_delete['im_loc']:
            if f:
                os.remove(f)
        meta_data.drop(meta_data_to_delete.index, inplace=True)
        meta_data.to_pickle(self.pd_path)

from contextlib import contextmanager

@contextmanager
def meta_handle(dataset_dir,data_dir,device,input_dimension, split = None, delete = False):
    meta = MetaData(dataset_dir)
    meta.add_directory(data_dir,device,input_dimension, split = split)
    try:
        yield meta
    finally:
        if delete:
            meta.delete(input_dimension=input_dimension,device=device)

from torch.utils.data import Dataset

class RobotsDatasetPrecomputed(Dataset):
    """
    mode indicates whether detection, segmentation or both is returned
    """

    def __init__(self, meta, mode, input_dimension, device, split, transform=None):
        """
        Args:
        """
        self.meta = meta
        self.mode = mode
        self.transform = transform
        
        meta_data = self.meta.data()
        location = (meta_data['input_dimension'] == input_dimension) & (meta_data['device'] == device) & (meta_data['split'] == split)
        self.index = None
        if self.mode == 'both':
            location = location & meta_data['seg_loc'].notnull() & meta_data['det_loc'].notnull()
            self.index = meta_data[location]
            self.len = len(self.index)
        if self.mode == 'det':
            location = location & meta_data['det_loc'].notnull()
            self.index = meta_data[location]
            self.len = len(self.index)
        if self.mode == 'seg':
            location = location & meta_data['seg_loc'].notnull()
            self.index = meta_data[location]
            self.len = len(self.index)
        if self.mode == 'concurrent':
            location_det = location & meta_data['det_loc'].notnull()
            location_seg = location & meta_data['seg_loc'].notnull()
            self.index_det = meta_data[location_det]
            self.index_seg = meta_data[location_seg]
            self.len_det = len(self.index_det)
            self.len_seg = len(self.index_seg)
            self.len = max(self.len_det,self.len_seg)
            
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): 
            idx = idx.tolist() 
        if self.mode == 'both':
            im_path = self.index['im_loc'].iloc[idx]
            det_path = self.index['det_loc'].iloc[idx]
            seg_path = self.index['seg_loc'].iloc[idx]
            im = torch.load(im_path)
            target = (torch.load(det_path),torch.load(seg_path))
            return im, target
        if self.mode == 'det':
            im_path = self.index['im_loc'].iloc[idx]
            det_path = self.index['det_loc'].iloc[idx]
            im = torch.load(im_path)
            target = torch.load(det_path)
            return im, target
        if self.mode == 'seg':
            im_path = self.index['im_loc'].iloc[idx]
            seg_path = self.index['seg_loc'].iloc[idx]
            im = torch.load(im_path)
            target = torch.load(seg_path)
            return im, target
        if self.mode == 'concurrent':
            seg_path = self.index_seg['seg_loc'].iloc[idx % self.len_seg]
            det_path = self.index_det['det_loc'].iloc[idx % self.len_det]            
            im_path_seg = self.index_seg['im_loc'].iloc[idx  % self.len_seg]
            im_path_det = self.index_det['im_loc'].iloc[idx % self.len_det]
            sample_seg = (torch.load(im_path_seg),torch.load(seg_path))
            sample_det = (torch.load(im_path_det),torch.load(det_path))
            return sample_det, sample_seg
        raise ValueError('Mode does not exist.')
