import numpy as np
import os, glob, sys
from keras.preprocessing import image as kImage
from instance_normalization import InstanceNormalization
from FgSegNet_v2_module import loss, acc
# from skimage.transform import pyramid_gaussian
from keras.models import load_model
from scipy.misc import imsave  # , imresize
import gc


def getFiles(scene_input_path):
    inlist = glob.glob(os.path.join(scene_input_path, '*.png'))
    return np.asarray(inlist)


dataset = {
    # 'baseline':['highway', 'pedestrians', 'office', 'PETS2006'],
    'baseline':['highway',],
    # 'cameraJitter':['badminton', 'traffic', 'boulevard', 'sidewalk'],
    # 'badWeather':['skating', 'blizzard', 'snowFall', 'wetSnow'],
    # 'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass'],
    # 'intermittentObjectMotion':['abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],
    # 'lowFramerate': ['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
    # 'nightVideos':['bridgeEntry', 'busyBoulvard', 'fluidHighway', 'streetCornerAtNight', 'tramStation', 'winterStreet'],
    # 'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
    # 'shadow':['backdoor', 'bungalows', 'busStation', 'copyMachine', 'cubicle', 'peopleInShade'],
    # 'thermal':['corridor', 'diningRoom', 'lakeSide', 'library', 'park'],
    # 'turbulence':['turbulence0', 'turbulence1', 'turbulence2', 'turbulence3']
}

for category, scene_list in dataset.items():
    for scene in scene_list:
        print('\n->>> ' + category + ' / ' + scene)
        training_sets_path = os.path.join('..', 'training_sets', 'CDnet2014_train', category, scene)

        th_path = os.path.join('FgSegNet_v2', 'results25_th0.7', category, scene)
        gt_list = getFiles(training_sets_path)
        th_list = getFiles(th_path)
        if (gt_list is None):
            raise ValueError('X_list is None')

        for i in th_list:
            th_name = os.path.basename(i).replace('bin', 'gt')
            for j in gt_list:
                gt_name = os.path.basename(j)
                if th_name == gt_name:
                    os.remove(i)
        print('\n\t- remove training frames in '+th_path)

# python processFolder.py ../../datasets/CDnet2014_dataset ../FgSegNet_v2/results25_th0.7
