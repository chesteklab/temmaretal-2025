import pdb

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import torch
from matplotlib import font_manager
import seaborn as sns
import os
import yaml
#some basic text parameters for figures
mpl.rcParams['font.family'] = "Atkinson Hyperlegible" # if installed but not showing up, rebuild mpl cache
mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlelocation'] = 'left'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['pdf.fonttype'] = 42

# Sets up some parameters used throughout the project

# set up standard plot colors
hcColor = np.asarray([0.5, 0.5, 0.5])
kfColor = np.asarray([64, 64, 192]) / 255
dsColor = np.asarray([64, 192, 64]) / 255
tcnColor = np.asarray([192, 64, 64]) / 255
rnnColor = np.asarray([128, 0, 96]) / 255
tcnColorB = np.asarray([192, 96, 192]) / 255

noregColor = np.asarray([64, 192, 192]) / 255
nobnColor = np.asarray([192, 64, 192]) / 255
nodpColor = np.asarray([128, 128, 224]) / 255


offline_palette = np.stack((kfColor, dsColor, tcnColor, rnnColor, tcnColorB))
offline_order = ('rr','ds','tcn','rnn')
online_palette = np.stack((hcColor, kfColor, tcnColor, rnnColor))
online_palette_w = np.stack((hcColor, kfColor, tcnColor, tcnColorB))
offline_variance_palette = {'hc':hcColor, 'TCN':tcnColor, 'TCN_nodp':nodpColor,
                          'TCN_nobn':nobnColor, 'TCN_noreg':noregColor}

variance_models = ('TCN', 'TCN_nobn', 'TCN_nodp', 'TCN_noreg')
varianceLabels = ('tcfnn', 'dropout only (noBN)', 'batchnorm only (noDP)', 'noReg')
varianceTicks = ('tcFNN', 'noBN', 'noDP', 'noReg')

context_palette = np.asarray([[255, 200, 50],
                             [255, 155, 40],
                             [255, 100, 45],
                             [225, 25, 30],
                             [60, 200, 255],
                             [128, 128, 128]])/255

context_group_palette = np.asarray([[250, 190, 0],
                                   [155, 20, 115],
                                   [60, 200, 255],
                                   [128, 128, 128]])/255

context_order = ['Normal', 'Spring', 'Wrist','SprWrst', 'Mixed','Mixed_Full']
onlineTracesPalette = context_group_palette[0:2,:]

dsmap = mpl.colors.LinearSegmentedColormap.from_list('dsmap',[[0,0,0],dsColor])
# standard binsize (when not already used)
binsize = 50
numChans = 96
batch_size = 64

_dir = os.path.dirname(__file__)
with open(os.path.join(_dir, 'config.yaml')) as f:
    _paths = yaml.safe_load(f)

raw_data_dir = _paths['raw_data_dir']
model_dir = _paths['model_dir']
results_dir = _paths['results_dir']
data_dir = _paths['data_dir']
architectures_path = os.path.join(_dir, _paths['architectures_path'])
training_params_path = os.path.join(_dir, _paths['training_params_path'])

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
dtype = torch.float