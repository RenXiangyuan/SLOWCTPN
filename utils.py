# -*- coding: utf-8 -*-

import os

from PIL import Image,ImageDraw
import numpy as np
import json
from math import ceil, floor


'''
#
dir_data = './data_train'
dir_images = dir_data + '/images'
dir_contents = dir_data + '/contents'
#
'''

#
def get_files_with_ext(path, str_ext, limit = -1):
    file_list = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)  
        if file_path.endswith(str_ext):  
            file_list.append(file_path)
            if limit > 0 and len(file_list)  >= limit: break
            #print(file_path)
    return file_list
    #
def get_target_json_file(img_file):
    #
    pre_dir = os.path.abspath(os.path.dirname(img_file)+os.path.sep+"..")
    txt_dir = os.path.join(pre_dir, 'labels')
    #
    filename = os.path.basename(img_file)
    arr_split = os.path.splitext(filename)
    filename = arr_split[0] + '.json'
    #
    txt_file = os.path.join(txt_dir, filename)
    #
    return txt_file

#
def get_target_txt_file(img_file):
    #
    pre_dir = os.path.abspath(os.path.dirname(img_file)+os.path.sep+"..")
    txt_dir = os.path.join(pre_dir, 'contents')
    #
    filename = os.path.basename(img_file)
    arr_split = os.path.splitext(filename)
    filename = arr_split[0] + '.txt'
    #
    txt_file = os.path.join(txt_dir, filename)
    #
    return txt_file
    #
#
def get_list_contents(content_file):
    #
    contents = []
    #
    if not os.path.exists(content_file): return contents
    #
    with open(content_file, 'r') as fp:
        lines = fp.readlines()
    #
    for line in lines:
        arr_str = line.split('|')
        item = list(map(lambda x: int(x), arr_str[0].split('-')))
        #
        contents.append([item, arr_str[1]])
        #
    return contents
#
#
def calculate_targets_at_use_json(anchor_center, txt_list, anchor_heights):
    anchor_width = 32
    #
    ash = 32  # anchor stride - height
    asw = 32  # anchor stride - width
    #

    #
    hc = anchor_center[0]
    wc = anchor_center[1]
    #
    maxIoU = 0
    anchor_posi = 0
    text_bbox = []
    #
    for item in txt_list:
        #
        # width: if more than half of the anchor is text, positive;
        # height: if more than half of the anchor is text, positive;
        # heigth_IoU: of the 4 anchors, choose the one with max height_IoU;
        #
        bbox = item
        #
        # horizontal
        flag = 0
        #
        if (bbox[0] < wc and wc <= bbox[2]):
            flag = 1
        elif (wc < bbox[0] and bbox[2] < wc + asw):
            if (bbox[0] - wc < wc + asw - bbox[2]):
                flag = 1
        elif (wc - asw < bbox[0] and bbox[2] < wc):
            if (bbox[2] - wc <= wc - asw - bbox[0]):
                flag = 1
        #
        if flag == 0: continue
        #
        # vertical
        #
        bcenter = (bbox[1] + bbox[3]) / 2.0
        #
        d0 = abs(hc - bcenter)
        dm = abs(hc - ash - bcenter)
        dp = abs(hc + ash - bcenter)
        #
        if (d0 < ash and d0 <= dm and d0 < dp):
            pass
        else:
            continue
            #
        #
        posi = 0
        #
        for ah in anchor_heights:
            #
            hah = ah // 2  # half_ah
            #
            IoU = 1.0 * (min(hc + hah, bbox[3]) - max(hc - hah, bbox[1])) \
                  / (max(hc + hah, bbox[3]) - min(hc - hah, bbox[1]))
            #
            if IoU > maxIoU:
                maxIoU = IoU
                anchor_posi = posi
                text_bbox = bbox
            #
            posi += 1
            #
        #
        break
    #
    # no text
    if maxIoU <= 0:  #
        #
        num_anchors = len(anchor_heights)
        #
        cls = [0, 0] * num_anchors
        ver = [0, 0] * num_anchors
        hor = [0, 0] * num_anchors
        #
        return cls, ver, hor
    #
    # text
    cls = []
    ver = []
    hor = []
    #
    for idx, ah in enumerate(anchor_heights):
        #
        if not idx == anchor_posi:
            cls.extend([0, 0])  #
            ver.extend([0, 0])
            hor.extend([0, 0])
            continue
        #
        cls.extend([1, 1])  #
        #
        half_ah = ah // 2
        half_aw = anchor_width // 2
        #
        anchor_bbox = [wc - half_aw, hc - half_ah, wc + half_aw, hc + half_ah]
        #
        ratio_bbox = [0, 0, 0, 0]
        #
        ratio = (text_bbox[0] - anchor_bbox[0]) / anchor_width
        if abs(ratio) < 1:
            ratio_bbox[0] = ratio
        #
        # print(ratio)
        #
        ratio = (text_bbox[2] - anchor_bbox[2]) / anchor_width
        if abs(ratio) < 1:
            ratio_bbox[2] = ratio
        #
        # print(ratio)
        #
        ratio_bbox[1] = (text_bbox[1] - anchor_bbox[1]) / ah
        ratio_bbox[3] = (text_bbox[3] - anchor_bbox[3]) / ah
        #
        # print(ratio_bbox)
        #
        ver.extend([ratio_bbox[1], ratio_bbox[3]])
        hor.extend([ratio_bbox[0], ratio_bbox[2]])
        #
    #
    return cls, ver, hor
#
def calculate_targets_at(anchor_center, txt_list, anchor_heights):
    #
    # anchor_center = [hc, wc]
    #
    #
    # anchor width:  8,
    # anchor height: 6, 12, 24, 36, ...
    #
    # anchor stride: 12,8
    #

    #
    anchor_width = 8
    #
    ash = 12  # anchor stride - height
    asw = 8   # anchor stride - width
    #
    
    #
    hc = anchor_center[0]
    wc = anchor_center[1]
    #
    maxIoU = 0
    anchor_posi = 0
    text_bbox = []
    #
    for item in txt_list:
        #
        # width: if more than half of the anchor is text, positive;
        # height: if more than half of the anchor is text, positive;        
        # heigth_IoU: of the 4 anchors, choose the one with max height_IoU;
        #
        bbox = item[0]
        #
        # horizontal
        flag = 0    
        #
        if (bbox[0] < wc and wc <= bbox[2]):
            flag = 1
        elif (wc < bbox[0] and bbox[2] < wc+asw):
            if (bbox[0] - wc < wc+asw - bbox[2]):
                flag = 1
        elif (wc-asw < bbox[0] and bbox[2] < wc):
            if (bbox[2] - wc <= wc-asw - bbox[0]):
                flag = 1
        #
        if flag == 0: continue
        #
        # vertical
        #
        bcenter = (bbox[1] + bbox[3]) / 2.0
        #
        d0 = abs(hc - bcenter)
        dm = abs(hc-ash - bcenter)
        dp = abs(hc+ash - bcenter)
        #
        if (d0 < ash and d0 <= dm and d0 < dp):
            pass
        else:
            continue        
        #
        #
        posi = 0
        #
        for ah in anchor_heights:
            #
            hah = ah //2  # half_ah
            #
            IoU = 1.0* (min(hc+hah, bbox[3])-max(hc-hah, bbox[1])) \
                      /(max(hc+hah, bbox[3])-min(hc-hah, bbox[1]))
            #
            if IoU > maxIoU:
                maxIoU = IoU
                anchor_posi = posi
                text_bbox = bbox
            #
            posi += 1
            #
        #
        break
    #
    # no text
    if maxIoU <= 0:  #
        #
        num_anchors = len(anchor_heights)
        #
        cls = [0, 0] * num_anchors
        ver = [0, 0] * num_anchors
        hor = [0, 0] * num_anchors
        #
        return cls, ver, hor
    #
    # text
    cls = []
    ver = []
    hor = []
    #
    for idx, ah in enumerate(anchor_heights):
        #
        if not idx == anchor_posi:
            cls.extend([0, 0])  #
            ver.extend([0, 0])
            hor.extend([0, 0])
            continue
        #
        cls.extend([1, 1])  #
        #
        half_ah = ah //2
        half_aw = anchor_width //2
        #
        anchor_bbox = [wc - half_aw, hc - half_ah, wc + half_aw, hc + half_ah]
        #
        ratio_bbox = [0, 0, 0, 0]
        #
        ratio = (text_bbox[0]-anchor_bbox[0]) /anchor_width
        if abs(ratio) < 1: 
            ratio_bbox[0] = ratio
        #
        # print(ratio)
        #
        ratio = (text_bbox[2]-anchor_bbox[2]) /anchor_width
        if abs(ratio) < 1:
            ratio_bbox[2] = ratio
        #
        # print(ratio)
        #
        ratio_bbox[1] = (text_bbox[1]-anchor_bbox[1]) /ah
        ratio_bbox[3] = (text_bbox[3]-anchor_bbox[3]) /ah
        #
        # print(ratio_bbox)
        #
        ver.extend([ratio_bbox[1], ratio_bbox[3]])
        hor.extend([ratio_bbox[0], ratio_bbox[2]]) 
        #
    #
    return cls, ver, hor
    #
#
# util function
#
def get_list_contents_use_json(txt_file, width):
    bboxes = []
    if not os.path.exists(txt_file):
        return bboxes
    with open(txt_file, 'r') as fp:
        data = json.load(fp)
        for boxes in data:
            # resized_box = [round(boxes['x0'] ),
            #                round(boxes['y0'] ),
            #                round(boxes['x1'] ),
            #                round(boxes['y1'] )]
            resized_box = [round(boxes['y0']),#x0
                           round(width - boxes['x1']),
                           round(boxes['y1']),
                           round(width - boxes['x0'])]
            bboxes.append(resized_box)
    return bboxes
#
def get_image_and_targets(img_file, txt_file, anchor_heights):
    
    # img_data
    img = Image.open(img_file)

    origin_width = img.size[0]
    # print("img size: ", img.size)
    #
    # # counter-clockwise 90
    img = img.transpose(Image.ROTATE_90)
    img_data = np.array(img, dtype = np.float32)/255
    # height, width, channel
    #
    img_data = img_data[:,:,0:3]  # rgba
    #
    
    # texts
    # txt_list = get_list_contents(txt_file)

    txt_list = get_list_contents_use_json(txt_file, origin_width)
    # print("One Label: ", txt_list[0])
    #
    
    # targets
    img_size = img_data.shape  # height, width, channel
    # print("img size: ", img_size)
    #
    # ///2, ///2, //3, -255
    # ///2, ///2, ///2,    
    #
    height_feat = ceil(ceil(ceil(ceil(ceil(img_size[0]/2.0)/2.0)/2.0)/2)/2)
    width_feat = ceil(ceil(ceil(ceil(ceil(img_size[1]/2.0)/2.0)/2.0)/2)/2)
    #
    
    #
    num_anchors = len(anchor_heights)
    #
    target_cls = np.zeros((height_feat, width_feat, 2*num_anchors))
    target_ver = np.zeros((height_feat, width_feat, 2*num_anchors))
    target_hor = np.zeros((height_feat, width_feat, 2*num_anchors))
    #
    
    #
    # detection
    #
    # [3,1; 1,1],
    # [9,2; 3,2], [9,2; 3,2], [9,2; 3,2]
    # [18,4; 6,4], [18,4; 6,4], [18,4; 6,4]
    # [36,8; 12,8], [36,8; 12,8], [36,8; 12,8], 
    #
    # anchor width:  8,
    # anchor height: 12, 24, 36, 48,
    #
    # feature_layer --> receptive_field
    # [0,0] --> [0:36, 0:8]
    # [0,1] --> [0:36, 8:8+8]
    # [i,j] --> [12*i:36+12*i, 8*j:8+8*j]
    #
    # feature_layer --> anchor_center
    # [0,0] --> [18, 4]
    # [0,1] --> [18, 4+8]
    # [i,j] --> [18+12*i, 4+8*j]
    #
    
    #
    # anchor_width = 8
    #
    ash = 32  # anchor stride - height
    asw = 32   # anchor stride - width
    #
    hc_start = 16
    wc_start = 16
    #
    
    for h in range(height_feat):
        #
        hc = hc_start + ash * h  # anchor height center
        #
        for w in range(width_feat):
            #
            cls,ver,hor = calculate_targets_at_use_json([hc, wc_start + asw * w], txt_list, anchor_heights)
            #
            target_cls[h, w] = cls
            target_ver[h, w] = ver
            target_hor[h, w] = hor
            #
    #
    return [img_data], [height_feat, width_feat], target_cls, target_ver, target_hor
    #
#
def trans_results(r_cls, r_ver, r_hor, anchor_heights, threshold):
    anchor_positivenesses, anchor_vertical_overlap_ratios, anchor_horizontal_overlap_ratios =r_cls, r_ver, r_hor

    anchor_width = 32
    anchor_stride_height = 32
    anchor_stride_width = 32
    height_center_start = 16
    width_center_start = 16

    feature_map_height, feature_map_width, _ = anchor_positivenesses.shape

    output_bboxes = []
    output_confidences = []
    for y in range(feature_map_height):
        for x in range(feature_map_width):
            if max(anchor_positivenesses[y, x, :]) < threshold:
                continue
            anchor_posi = np.argmax(anchor_positivenesses[y, x, :])
            anchor_idx = anchor_posi // 2
            anchor_height = anchor_heights[anchor_idx]
            anchor_posi = anchor_idx * 2
            height_center = height_center_start + anchor_stride_height * y
            width_center = width_center_start + anchor_stride_width * x
            half_anchor_height = anchor_height // 2
            half_anchor_width = anchor_width // 2

            anchor_bbox = [max(0, width_center - half_anchor_width), max(0,height_center - half_anchor_height),
                           width_center + half_anchor_width, height_center + half_anchor_height]

            text_bbox = [0, 0, 0, 0]
            text_bbox[0] = anchor_bbox[0] + anchor_width * anchor_horizontal_overlap_ratios[y, x, anchor_posi]
            text_bbox[1] = anchor_bbox[1] + anchor_height * anchor_vertical_overlap_ratios[y, x, anchor_posi]
            text_bbox[2] = anchor_bbox[2] + anchor_width * anchor_horizontal_overlap_ratios[y, x, anchor_posi + 1]
            text_bbox[3] = anchor_bbox[3] + anchor_height * anchor_vertical_overlap_ratios[y, x, anchor_posi + 1]

            output_bboxes.append(text_bbox)

            # print(anchor_idx)
            # print(anchor_bbox)
            # print("anchor_horizontal_overlap_ratios",anchor_horizontal_overlap_ratios[y, x, anchor_posi])
            # print("anchor_vertical_overlap_ratios",anchor_vertical_overlap_ratios[y, x, anchor_posi])
            # print()
            # output_confidences.append(max(anchor_positivenesses[y, x, :]))
            # return output_bboxes, output_confidences
    return output_bboxes, output_confidences
    #

def do_nms_and_connection(list_bbox, list_conf):
    max_margin = 50
    len_list_box = len(list_bbox)
    conn_bbox = []
    head = tail = 0
    for i in range(1, len_list_box):
        distance_i_j = abs(list_bbox[i][0] - list_bbox[i - 1][0])
        overlap_i_j = overlap(list_bbox[i][1], list_bbox[i][3], list_bbox[i - 1][1], list_bbox[i - 1][3])
        if distance_i_j < max_margin and overlap_i_j > 0.7:
            tail = i
            if i == len_list_box-1:
                this_test_box = [list_bbox[head][0], list_bbox[head][1], list_bbox[tail][2], list_bbox[tail][3]]
                conn_bbox.append(this_test_box)
                head = tail = i
        else:
            this_test_box = [list_bbox[head][0], list_bbox[head][1], list_bbox[tail][2], list_bbox[tail][3]]
            conn_bbox.append(this_test_box)
            head = tail = i

    return conn_bbox


def overlap(h_up1, h_dw1, h_up2, h_dw2):
    """
    :param h_up1:
    :param h_dw1:
    :param h_up2:
    :param h_dw2:
    :return:
    """
    overlap_value = (min(h_dw1, h_dw2) - max(h_up1, h_up2)) \
                    / (max(h_dw1, h_dw2) - min(h_up1, h_up2))
    return overlap_value
        
#
def draw_text_boxes(img_file, text_bbox):
    #
    #打开图片，画图
    img_draw = Image.open(img_file)
    width = img_draw.size[0]
    #
    draw = ImageDraw.Draw(img_draw)
    #
    for item in text_bbox:
        #
        # xs = width - item[3]
        # ys = item[0]
        # xe = width - item[1]
        # ye = item[2]
        xs = item[0]
        ys = item[1]
        xe = item[2]
        ye = item[3]

        line_width = 1 # round(text_size/10.0)
        draw.line([(xs,ys),(xs,ye),(xe,ye),(xe,ys),(xs,ys)],
                   width=line_width, fill=(255,0,0))
        #
    #
    img_draw.save(img_file)
    #

#
if __name__ == '__main__':
    #
    print('draw target bbox ... ')
    #
    import config as meta
    #
    list_imgs = get_files_with_ext(meta.dir_images_valid, 'png')
    #
    curr = 0
    NumImages = len(list_imgs)
    #
    # valid_result save-path
    if not os.path.exists(meta.dir_results_valid): os.mkdir(meta.dir_results_valid)
    #
    for img_file in list_imgs:
        #
        txt_file = get_target_txt_file(img_file)
        #
        img_data, feat_size, target_cls, target_ver, target_hor = \
        get_image_and_targets(img_file, txt_file, meta.anchor_heights)
        #
        curr += 1
        print('curr: %d / %d' % (curr, NumImages))
        #
        filename = os.path.basename(img_file)
        arr_str = os.path.splitext(filename)
        #
        # image
        '''
        r = Image.fromarray(img_data[0][:,:,0] *255).convert('L')
        g = Image.fromarray(img_data[0][:,:,1] *255).convert('L')
        b = Image.fromarray(img_data[0][:,:,2] *255).convert('L')
        #
        file_target = os.path.join(meta.dir_results_valid, 'target_' +arr_str[0] + '.png')
        img_target = Image.merge("RGB", (r, g, b))
        img_target.save(file_target)
        '''
        
        file_target = os.path.join(meta.dir_results_valid, 'target_' +arr_str[0] + '.png')
        img_target = Image.fromarray(np.uint8(img_data[0] *255) ) #.convert('RGB')
        img_target.save(file_target)
        
        #
        # trans
        text_bbox, conf_bbox = trans_results(target_cls, target_ver, target_hor,\
                                             meta.anchor_heights, meta.threshold)
        #
        draw_text_boxes(file_target, text_bbox)
        #
    #
    print('draw end.')
    #


