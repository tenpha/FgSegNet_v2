import numpy as np
import os, glob, sys
from keras.preprocessing import image as kImage
from instance_normalization import InstanceNormalization
from  FgSegNet_v2_module import loss,acc
#from skimage.transform import pyramid_gaussian
from keras.models import load_model
from scipy.misc import imsave#, imresize
import gc


def getFiles(scene_input_path):
    inlist = glob.glob(os.path.join(scene_input_path,'*.png'))
    return np.asarray(inlist)


training_sets_path = os.path.join('..','training_sets','CDnet2014_train','baseline','highway25')
save_removed_results_path = os.path.join('FgSegNet_v2', 'results_removed','baseline','highway' )
th_path = os.path.join('FgSegNet_v2', 'results25_th0.7','baseline','highway' )
gt_list = getFiles(training_sets_path)
th_list = getFiles(th_path)
if (gt_list is None):
    raise ValueError('X_list is None')

for i in th_list:
    th_name = os.path.basename(i).replace('bin','gt')
    for j in gt_list:
        gt_name = os.path.basename(j)
        if th_name==gt_name:
            os.remove(i)

#python processFolder.py ../../datasets/CDnet2014_dataset ../FgSegNet_v2/results25_th0.7