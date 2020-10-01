import urllib.request
import re
import os
import sys


def download(target_dir):
    print('Beginning file download with urllib2...')

    url = 'http://bigcuda5.informatik.uni-bonn.de:8686'
    folder_names = ['/blob/dataset/','/blob/forceTest/','/blob/forceTrain/','/segmentation/dataset/image/','/segmentation/dataset/target/','/segmentation/forceTrain/image/','/segmentation/forceTrain/target/']
    max_number_files = sys.maxsize
    for folder_name in folder_names:
        i = 0
        target_folder = os.path.join(target_dir,folder_name[1:])
        print('Creating ' + target_folder)
        current_url = url + folder_name
        urllib.request.urlretrieve(current_url, 'index.html')
        f = open("index.html", "r")
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # Iterate through each line in the html file
        # If there is a link, download whatever is there and save it folder_name/
        for line in f:
            extract = re.search('href="(.*)"',line)
            if extract:
                i = i + 1
                filename = extract.group(1)
                if filename != '.ipynb_checkpoints/':
                    urllib.request.urlretrieve(current_url + filename, target_folder + filename)
                if i >= max_number_files:
                    break
        f.close()
        os.remove('index.html')
