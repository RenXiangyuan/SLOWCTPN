# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

import config
import utils

from model import CTPN

import os
#
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
#
model = CTPN()
# data
print('loading data ...')
data_train = utils.get_files_with_ext(config.dir_images_train, 'png')[:10]

data_valid = utils.get_files_with_ext(config.dir_images_valid, 'png')[:200]
print('load finished.')
# train
model.train_and_valid(data_train, data_valid, load_model = False)
#




