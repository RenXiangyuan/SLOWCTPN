# -*- coding: utf-8 -*-
"""
@author: limingfan

"""


# data for train
dir_data_train = '/data/nfsdata/table_detect/train'
dir_images_train = dir_data_train + '/images'
dir_contents_train = dir_data_train + '/labels'

# data for validation
dir_data_valid = '/data/nfsdata/table_detect/test'
dir_images_valid = dir_data_valid + '/images'
dir_contents_valid = dir_data_valid + '/labels'
dir_results_valid = dir_data_valid + '/results'
#
train_valid = "./data/train_test"

#
model_detect_dir = './model_detect'
model_detect_name = 'model_detect'
model_detect_pb_file = model_detect_name + '.pb'
#
# anchor_heights = [6, 12, 24, 36]
anchor_heights = [60, 120, 240, 360]

#

threshold = 0.5  #
#

