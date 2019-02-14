# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

import config
import utils

from model import CTPN


import os
#
os.environ['CUDA_VISIBLE_DEVICES'] = '2' #使用 GPU 0
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
#


#
model = CTPN()
#

#
# predict
model.prepare_for_prediction()
#
# list_images_valid = utils.get_files_with_ext(config.train_valid, 'png')
list_images_valid = utils.get_files_with_ext(config.dir_images_valid, 'png')[:20]
for img_file in list_images_valid:
    #
    # img_file = './data_test/images/bkgd_1_0_generated_0.png'
    #
    print(img_file)
    #
    conn_bbox, text_bbox, conf_bbox = model.predict(img_file=img_file, out_dir='./valid_results_prediction')
    # conn_bbox, text_bbox, conf_bbox = model.predict(img_file=img_file, out_dir = './results_prediction')
    #

