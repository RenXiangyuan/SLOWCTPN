# -*- coding: utf-8 -*-


import os
import time
import random

from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import graph_util

import model_detect_def as model_def
import config

import utils as model_data



#
TRAINING_STEPS = 50000
#
LEARNING_RATE_BASE = 1e-5
MOMENTUM = 0.9
REG_LAMBDA = 0.0001
GRAD_CLIP = 100.0
#
VALID_FREQ = 10000
LOSS_FREQ = 100
#
KEEP_NEAR = 5
KEEP_FREQ = 1000
#


class CTPN():
    #
    def __init__(self):
        # model directory
        self.pb_file = os.path.join(config.model_detect_dir, config.model_detect_pb_file)
        #        
        self.sess_config = tf.ConfigProto()
        # self.sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95
        #
        self.is_train = False
        #
        self.graph = None
        self.sess = None
        #
        self.learning_rate_base = LEARNING_RATE_BASE
        #
        self.train_steps = TRAINING_STEPS
        #
        self.valid_freq = VALID_FREQ
        self.loss_freq = LOSS_FREQ
        #
        self.keep_near = KEEP_NEAR
        self.keep_freq = KEEP_FREQ
        #
        
    def prepare_for_prediction(self, pb_file_path = None):
        #
        if pb_file_path == None: pb_file_path = self.pb_file
        #
        if not os.path.exists(pb_file_path):
            print('ERROR: %s NOT exists, when load_pb_for_predict()' % pb_file_path)
            return -1
        #
        self.graph = tf.Graph()
        #
        with self.graph.as_default():
            #
            with open(pb_file_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                #
                tf.import_graph_def(graph_def, name="")
                #
            #
            # change the input/output variables
            #
            self.x = self.graph.get_tensor_by_name('x-input:0')
            # self.w = self.graph.get_tensor_by_name('w-input:0')
            #
            self.rnn_cls = self.graph.get_tensor_by_name('rnn_cls:0')
            self.rnn_ver = self.graph.get_tensor_by_name('rnn_ver:0')
            self.rnn_hor = self.graph.get_tensor_by_name('rnn_hor:0')
            #
        #
        print('graph loaded for prediction')
        #
        self.sess = tf.Session(graph = self.graph, config = self.sess_config)
        #

    def predict(self, img_file, out_dir = None):
        #
        # img_data
        img = Image.open(img_file)
        img = img.transpose(Image.ROTATE_90)
        img_data = np.array(img, dtype = np.float32)/255
        # height, width, channel
        #
        img_data = [ img_data[:,:,0:3] ]  # rgba
        img_size = img.size  # (width, height)
        w_arr = np.array([ img_size[0] ], dtype = np.int32)
        #
        img.close()
        with self.graph.as_default():              
            #
            feed_dict = {self.x: img_data}#, self.w: w_arr}
            #
            r_cls, r_ver, r_hor = self.sess.run([self.rnn_cls, self.rnn_ver, self.rnn_hor], feed_dict)
            #
            # trans
            text_bbox, conf_bbox = model_data.trans_results(r_cls, r_ver, r_hor, \
                                                            config.anchor_heights, config.threshold)
            #
            conn_bbox = model_data.do_nms_and_connection(text_bbox, conf_bbox)
            #
            if out_dir == None: return conn_bbox, text_bbox, conf_bbox
            #
            # predication_result save-path
            if not os.path.exists(out_dir): os.mkdir(out_dir)
            #
            filename = os.path.basename(img_file)
            #
            # image
            #
            file_target = os.path.join(out_dir, 'predicted_' + filename)
            img_target = Image.fromarray(np.uint8(img_data[0] *255) ) #.convert('RGB')
            # img_target = img_target.transpose(Image.ROTATE_270)
            img_target.save(file_target)
            model_data.draw_text_boxes(file_target, conn_bbox)


            #
            # file_target = os.path.join(out_dir, 'connected_' + filename)
            # img_target = Image.fromarray(np.uint8(img_data[0] *255) ) #.convert('RGB')
            # # img_target = img_target.transpose(Image.ROTATE_270)
            # img_target.save(file_target)
            # model_data.draw_text_boxes(file_target, conn_bbox)
            #
            return conn_bbox, text_bbox, conf_bbox
            #

    def create_graph_all(self, training):
        #
        self.is_train = training
        self.graph = tf.Graph()
        #
        with self.graph.as_default():
            #
            self.x = tf.placeholder(tf.float32, (1, None, None, 3), name = 'x-input')
            # self.x = tf.placeholder(tf.float32, (1, 596, 842, 3), name='x-input')
            # self.w = tf.placeholder(tf.int32, (1,), name = 'w-input') # width
            #
            # self.conv_feat, self.seq_len = model_def.conv_feat_layers(self.x, self.w, self.is_train)   # train
            # self.rnn_cls, self.rnn_ver, self.rnn_hor = model_def.rnn_detect_layers(self.conv_feat, self.seq_len, len(config.anchor_heights))
            self.conv_feat = model_def.conv_feat_layers(self.x, self.is_train)

            self.rnn_cls, self.rnn_ver, self.rnn_hor = model_def.rnn_detect_layers(self.conv_feat, None,len(config.anchor_heights))
            #
            # print(self.rnn_cls.op.name)
            #
            self.t_cls = tf.placeholder(tf.float32, (None, None, None), name = 'c-input')
            self.t_ver = tf.placeholder(tf.float32, (None, None, None), name = 'v-input')
            self.t_hor = tf.placeholder(tf.float32, (None, None, None), name = 'h-input')
            #            
            # print(self.graph.get_operations())
            #
            self.loss = model_def.detect_loss(self.rnn_cls, self.rnn_ver, self.rnn_hor, self.t_cls, self.t_ver, self.t_hor)
            #
            # print(loss.op.name)

            #
            # train
            self.global_step = tf.train.get_or_create_global_step()
            self.learning_rate = tf.get_variable("learning_rate", shape=[], dtype=tf.float32, trainable=False)
            
            #optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM, use_nesterov=True)
            #optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)              
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = MOMENTUM)

            # optimizer = tf.train.AdamOptimizer(learning_rate=tf.cond(self.loss > 9e4, lambda: 0.0, lambda: self.learning_rate), beta1=MOMENTUM)
            #

            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            grads = optimizer.compute_gradients(self.loss + l2_loss * REG_LAMBDA)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, GRAD_CLIP)
            grads_applying = zip(capped_grads, variables)


            # grads_applying = optimizer.compute_gradients(self.loss)
            # grads_applying = [(tf.clip_by_norm(grad, GRAD_CLIP),
            #                           var) for grad, var in grads_applying]
            self.train_op = optimizer.apply_gradients(grads_applying, global_step=self.global_step)
            #

            #
            print('graph defined for training') if self.is_train else print('graph defined for validation')
            #
            #print('global_step.op.name: ' + self.global_step.op.name)
            #print('train_op.op.name: ' + train_op.op.name)                
            #
            #
    
    def train_and_valid(self, data_train, data_valid, load_model = False):
        #     
        # model save-path
        if not os.path.exists(config.model_detect_dir): os.mkdir(config.model_detect_dir)
        if not os.path.exists('./explode_result'): os.mkdir('./explode_result')
        #                   
        # graph
        self.create_graph_all(training = True)
        #
        # restore and train
        with self.graph.as_default():
            #
            saver = tf.train.Saver()
            with tf.Session(config = self.sess_config) as sess:
                #
                tf.global_variables_initializer().run()
                sess.run(tf.assign(self.learning_rate, tf.constant(self.learning_rate_base, dtype=tf.float32)))
                #
                # restore with saved data
                ckpt = tf.train.get_checkpoint_state(config.model_detect_dir)
                #
                if load_model and ckpt and ckpt.model_checkpoint_path:
                    print("Not Retrain, Load Model")
                    saver.restore(sess, ckpt.model_checkpoint_path)                
                #
                print('begin to train ...')
                #
                # start training
                start_time = time.time()
                begin_time = start_time 
                #
                step = sess.run(self.global_step)
                #
                # train_step_half = int(self.train_steps * 0.5)
                # train_step_quar = int(self.train_steps * 0.75)
                #
                accumulate_loss = 0
                explode_draw_flag = True
                explode_count = 0
                while step < self.train_steps:
                # for _ in range(1):
                    #
                    # if step == train_step_half:
                    #     sess.run(tf.assign(self.learning_rate, tf.constant(self.learning_rate_base/10, dtype=tf.float32)))
                    # if step == train_step_quar:
                    #     sess.run(tf.assign(self.learning_rate, tf.constant(self.learning_rate_base/100, dtype=tf.float32)))
                    # #
                    # # save and validation
                    if step > 0 and step % self.valid_freq == 0:
                        #
                        print('save model to ckpt ...')
                        saver.save(sess, os.path.join(config.model_detect_dir, config.model_detect_name), \
                                   global_step = step)
                        #
                        print('validating ...')
                        model_v = CTPN()
                        model_v.validate(data_valid, step)
                        #

                    #
                    img_file = random.choice(data_train)  # list image files
                    if not os.path.exists(img_file):
                        print('image_file: %s NOT exist' % img_file)
                        continue
                    #
                    # txt_file = model_data.get_target_txt_file(img_file)

                    txt_file = model_data.get_target_json_file(img_file)

                    if not os.path.exists(txt_file):
                        print('label_file: %s NOT exist' % txt_file)
                        continue
                    #
                    # input data
                    try:
                        img_data, feat_size, target_cls, target_ver, target_hor = \
                        model_data.get_image_and_targets(img_file, txt_file, config.anchor_heights)
                    except:
                        print("image load error: ", img_file)
                        continue
                    #
                    if np.sum(target_cls) == 0 or np.sum(target_cls + 1) /2 - np.sum(target_cls) /2 * 2 == 0:
                        print("not valid image:", img_file)
                        continue

                    img_size = img_data[0].shape   # height, width, channel
                    # print("image shape", img_size)
                    #
                    w_arr = np.array([ img_size[1] ], dtype = np.int32)
                    #
                    #
                    feed_dict = {self.x: img_data,# self.w: w_arr, \
                                 self.t_cls: target_cls, self.t_ver: target_ver, self.t_hor: target_hor}
                    #
                    _, loss_value, step, lr = sess.run([self.train_op, self.loss, self.global_step, self.learning_rate],\
                                                       feed_dict)
                    # gt_feat = [target_cls, target_ver, target_hor]
                    # np.save("gt_feat.npy", gt_feat)
                    #
                    if loss_value> 1e3:
                        explode_count += 1
                        if explode_draw_flag:
                            print(img_file)
                            filename = os.path.basename(img_file)
                            file_target = os.path.join('./explode_result', 'ground_truth' + filename)
                            img_target = Image.fromarray(np.uint8(img_data[0] * 255))  # .convert('RGB')
                            # img_target = img_target.transpose(Image.ROTATE_270)
                            img_target.save(file_target)
                            import utils
                            ground_truth  = utils.get_list_contents_use_json(txt_file, img_size[0])
                            print("ground_truth: ", ground_truth[0])
                            print("image size: ", img_size)
                            model_data.draw_text_boxes(file_target, ground_truth)
                            print('step: %d, eplode loss: %g, lr: %g, %s' %
                                  (step, loss_value, lr,
                                   os.path.basename(img_file)))
                        if explode_count > 3:
                            explode_draw_flag = False
                        if explode_count > 10:
                            break
                        continue
                    accumulate_loss += loss_value
                    if step > 0 and step % self.loss_freq == 0:
                        #
                        curr_time = time.time()            
                        #
                        print('step: %d, average loss: %g, lr: %g, sect_time: %.1f, total_time: %.1f, %s' %
                              (step, accumulate_loss/self.loss_freq , lr,
                               curr_time - begin_time,
                               curr_time - start_time,
                               os.path.basename(img_file)))
                        #
                        begin_time = curr_time
                        accumulate_loss = 0
                        #                        
                    #
                    # # test
                    # print(img_file)
                    # filename = os.path.basename(img_file)
                    # file_target = os.path.join('./results_prediction', 'ground_truth' + filename)
                    # img_target = Image.fromarray(np.uint8(img_data[0] * 255))  # .convert('RGB')
                    # # img_target = img_target.transpose(Image.ROTATE_270)
                    # img_target.save(file_target)
                    # import model_detect_data
                    # ground_truth  = model_detect_data.get_list_contents_use_json(txt_file, img_size[0])
                    # print("ground_truth: ", ground_truth[0])
                    # print("image size: ", img_size)
                    # model_data.draw_text_boxes(file_target, ground_truth)
                    #
                    # break
                #
        #
    
    def validate(self, data_valid, step):
        #
        # valid_result save-path
        if not os.path.exists(config.dir_results_valid): os.mkdir(config.dir_results_valid)
        if not os.path.exists('./valid_results_prediction'): os.mkdir('./valid_results_prediction')
        #
        self.create_graph_all(training = False)
        #
        with self.graph.as_default():
            #
            saver = tf.train.Saver()
            with tf.Session(config = self.sess_config) as sess:                
                #
                tf.global_variables_initializer().run()
                #sess.run(tf.assign(self.is_train, tf.constant(False, dtype=tf.bool)))
                #
                # restore with saved data
                ckpt = tf.train.get_checkpoint_state(config.model_detect_dir)
                #
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                #
                # pb
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names = \
                                                                           ['rnn_cls','rnn_ver','rnn_hor'])
                with tf.gfile.FastGFile(self.pb_file, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                #                
                # test
                NumImages = len(data_valid)
                curr = 0
                for idx, img_file in enumerate(data_valid):
                    #
                    print(img_file)
                    #
                    txt_file = model_data.get_target_json_file(img_file)
                    #
                    # input data
                    img_data, feat_size, target_cls, target_ver, target_hor = \
                    model_data.get_image_and_targets(img_file, txt_file, config.anchor_heights)
                    #
                    img_size = img_data[0].shape   # height, width, channel
                    #
                    w_arr = np.array([ img_size[1] ], dtype = np.int32)
                    #
                    #
                    feed_dict = {self.x: img_data, #self.w: w_arr, \
                                 self.t_cls: target_cls, self.t_ver: target_ver, self.t_hor: target_hor}
                    #
                    r_cls, r_ver, r_hor, loss_value = sess.run([self.rnn_cls, self.rnn_ver, self.rnn_hor, self.loss], feed_dict)
                    #
                    #
                    curr += 1
                    print('curr: %d / %d, loss: %f' % (curr, NumImages, loss_value))
                    #
                    if idx < 20:
                        # trans

                        text_bbox, conf_bbox = model_data.trans_results(r_cls, r_ver, r_hor, \
                                                                        config.anchor_heights, config.threshold)
                        conn_bbox = model_data.do_nms_and_connection(text_bbox, conf_bbox)
                        #
                        # image
                        #
                        filename = os.path.basename(img_file)
                        # file_target = os.path.join(meta.dir_results_valid, str(step) + '_predicted_' + filename)
                        file_target = os.path.join('./valid_results_prediction/' + 'predicted_' + filename)

                        img_target = Image.fromarray(np.uint8(img_data[0] *255) ) #.convert('RGB')
                        # img_target = img_target.transpose(Image.ROTATE_270)
                        img_target.save(file_target)
                        model_data.draw_text_boxes(file_target, conn_bbox)
                    #
                    # id_remove = step - self.valid_freq * self.keep_near
                    # if id_remove % self.keep_freq:
                    #     file_temp = os.path.join(meta.dir_results_valid, str(id_remove) + '_predicted_' + filename)
                    #     if os.path.exists(file_temp): os.remove(file_temp)
                    #
                #
                print('validation finished')
                #
                #
        

#
'''
#
graph 相关的操作

add_to_collection(name,value)
as_default()
device(*args,**kwds)

with g.device('/gpu:0'):
  # All operations constructed in this context will be placed
  # on GPU 0.
  with g.device(None):
    # All operations constructed in this context will have no
    # assigned device.

# Defines a function from `Operation` to device string.
def matmul_on_gpu(n):
  if n.type == "MatMul":
    return "/gpu:0"
  else:
    return "/cpu:0"

with g.device(matmul_on_gpu):
  # All operations of type "MatMul" constructed in this context
  # will be placed on GPU 0; all other operations will be placed
  # on CPU 0.
  
finalize()
get_all_collection_keys()
get_operation_by_name(name)
get_operations()
get_tensor_by_name(name)
is_feedable(tensor)           # 作用：要是一个tensor能够被feed的话，返回True。
is_fetchable(tensor_or_op)
name_scope(*args,**kwds) 

with tf.Graph().as_default() as g:
  c = tf.constant(5.0, name="c")
  assert c.op.name == "c"
  c_1 = tf.constant(6.0, name="c")
  assert c_1.op.name == "c_1"

  # Creates a scope called "nested"
  with g.name_scope("nested") as scope:
    nested_c = tf.constant(10.0, name="c")
    assert nested_c.op.name == "nested/c"

    # Creates a nested scope called "inner".
    with g.name_scope("inner"):
      nested_inner_c = tf.constant(20.0, name="c")
      assert nested_inner_c.op.name == "nested/inner/c"

    # Create a nested scope called "inner_1".
    with g.name_scope("inner"):
      nested_inner_1_c = tf.constant(30.0, name="c")
      assert nested_inner_1_c.op.name == "nested/inner_1/c"

      # Treats `scope` as an absolute name scope, and
      # switches to the "nested/" scope.
      with g.name_scope(scope):
        nested_d = tf.constant(40.0, name="d")
        assert nested_d.op.name == "nested/d"

        with g.name_scope(""):
          e = tf.constant(50.0, name="e")
          assert e.op.name == "e"

'''

