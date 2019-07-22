# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np

from nets.network import Network
from model.config import cfg


def resnet_arg_scope(is_training=True,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


class resnetv1(Network):
    def __init__(self, num_layers=50):
        Network.__init__(self)
        self._feat_stride = [32, 16, 8, 4]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._num_layers = num_layers
        self._scope = 'resnet_v1_%d' % num_layers
        self._decide_blocks()

    def _crop_pool_layer(self, bottom, rois, name, index=0):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[index])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[index])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
            if cfg.RESNET.MAX_POOL:
                pre_pool_size = cfg.POOLING_SIZE * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                                 name="crops")
                crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
            else:
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
                                                 [cfg.POOLING_SIZE, cfg.POOLING_SIZE],
                                                 name="crops")
        return crops

    # Do the first few layers manually, because 'SAME' padding can behave inconsistently
    # for images of different sizes: sometimes 0, sometimes 1
    def _build_base(self):
        with tf.variable_scope(self._scope, self._scope):
            net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

        return net

    def _image_to_head(self, is_training, reuse=None):
        assert (0 <= cfg.RESNET.FIXED_BLOCKS <= 3)
        # Now the base is always fixed (freeze) during training
        # First build first few layers manually
        # than freeze some layers based on cfg setting
        # than train remain layers

        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net_conv = self._build_base()
        if cfg.RESNET.FIXED_BLOCKS > 0:
            with slim.arg_scope(resnet_arg_scope(is_training=False)):
                net_conv, _ = resnet_v1.resnet_v1(net_conv,
                                                  self._blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                                  global_pool=False,
                                                  include_root_block=False,
                                                  reuse=reuse,
                                                  scope=self._scope)
        if cfg.RESNET.FIXED_BLOCKS < 3:
            with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
                net_conv, _ = resnet_v1.resnet_v1(net_conv,
                                                  self._blocks[cfg.RESNET.FIXED_BLOCKS:-1],
                                                  global_pool=False,
                                                  include_root_block=False,
                                                  reuse=reuse,
                                                  scope=self._scope)


        self._act_summaries.append(net_conv)
        self._layers['head'] = net_conv

        return net_conv

    def _image_to_head_with_fpn(self, is_training, reuse=None):
        assert (0 <= cfg.RESNET.FIXED_BLOCKS <= 3)
        # Now the base is always fixed (freeze) during training
        # First build first few layers manually
        # than freeze some layers based on cfg setting
        # than train remain layers

        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net_conv = self._build_base()
            p2 = net_conv

        if cfg.RESNET.FIXED_BLOCKS > 0:
            with slim.arg_scope(resnet_arg_scope(is_training=False)):
                net_conv, endpoints1 = resnet_v1.resnet_v1(net_conv,
                                                  self._blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                                  global_pool=False,
                                                  include_root_block=False,
                                                  reuse=reuse,
                                                  scope=self._scope)

        if cfg.RESNET.FIXED_BLOCKS < 3:
            with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
                net_conv, endpoints = resnet_v1.resnet_v1(net_conv,
                                                          self._blocks[cfg.RESNET.FIXED_BLOCKS:-1],
                                                          global_pool=False,
                                                          include_root_block=False,
                                                          reuse=reuse,
                                                          scope=self._scope)

        if cfg.RESNET.FIXED_BLOCKS > 0:
            endpoints.update(endpoints1)

        for elem in endpoints.items():
            print(elem)

        p5 = endpoints[self._scope + "/block3"]
        p4 = endpoints[self._scope + "/block2"]
        p3 = endpoints[self._scope + "/block1"]


        fpn_map_list = []
        with tf.variable_scope("fpn", reuse=reuse):

            with slim.arg_scope(resnet_arg_scope(is_training=is_training)):

                p5 = resnet_utils.conv2d_same(p5, 256, 1, stride=1)
                p5_map = resnet_utils.conv2d_same(p5, 1024, 3, stride=1, scope='fpn_p5')
                fpn_map_list.append(p5_map)

                p4 = resnet_utils.conv2d_same(p4, 256, 1, stride=1)
                # p5_up = slim.convolution2d_transpose(p5, 256, 1, stride=2, padding="VALID")
                p4_shape = tf.shape(p4)
                p5_up = tf.image.resize_nearest_neighbor(p5, [p4_shape[1], p4_shape[2]])
                p4 =  p5_up + p4
                p4_map = resnet_utils.conv2d_same(p4, 1024, 3, stride=1, scope='fpn_p4')
                fpn_map_list.append(p4_map)

                p3 = resnet_utils.conv2d_same(p3, 256, 1, stride=1)
                # p4_up = slim.convolution2d_transpose(p4, 256, 1, stride=2, padding="VALID")
                p3_shape = tf.shape(p3)
                p4_up = tf.image.resize_nearest_neighbor(p4, [p3_shape[1], p3_shape[2]])
                p3 = p4_up + p3
                p3_map = resnet_utils.conv2d_same(p3, 1024, 3, stride=1, scope='fpn_p3')
                fpn_map_list.append(p3_map)

                p2 = resnet_utils.conv2d_same(p2, 256, 1, stride=1)
                # p3_up = slim.convolution2d_transpose(p3, 256, 1, stride=2, padding="VALID")
                p2_shape = tf.shape(p2)
                p3_up = tf.image.resize_nearest_neighbor(p3, [p2_shape[1], p2_shape[2]])
                p2 = p3_up + p2
                p2_map = resnet_utils.conv2d_same(p2, 1024, 3, stride=1, scope='fpn_p2')
                fpn_map_list.append(p2_map)

        self._act_summaries.append(net_conv)
        self._layers['head'] = net_conv

        return fpn_map_list

    def _head_to_tail(self, pool5, is_training, reuse=None):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):

            fc7, _ = resnet_v1.resnet_v1(pool5,
                                         self._blocks[-1:],
                                         global_pool=False,
                                         include_root_block=False,
                                         reuse=reuse,
                                         scope=self._scope)

            # average pooling done by reduce_mean
            fc7 = tf.reduce_mean(fc7, axis=[1, 2])
        return fc7

    def _decide_blocks(self):
        # choose different blocks for different number of layers
        if self._num_layers == 50:
            self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                            resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                            # use stride 1 for the last conv4 layer
                            resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
                            resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        elif self._num_layers == 101:
            self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                            resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                            # use stride 1 for the last conv4 layer
                            resnet_v1_block('block3', base_depth=256, num_units=23, stride=2),
                            resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        elif self._num_layers == 152:
            self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                            resnet_v1_block('block2', base_depth=128, num_units=8, stride=2),
                            # use stride 1 for the last conv4 layer
                            resnet_v1_block('block3', base_depth=256, num_units=36, stride=1),
                            resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        else:
            # other numbers are not supported
            raise NotImplementedError

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/conv1/weights:0'):
                self._variables_to_fix[v.name] = v
                continue

            if "fpn" in v.name:
                continue

            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                # print(v.get_shape().as_list(), var_keep_dic[v.name.split(':')[0]])
                variables_to_restore.append(v)
        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix Resnet V1 layers..')
        with tf.variable_scope('Fix_Resnet_V1') as scope:
            with tf.device("/cpu:0"):
                # fix RGB to BGR
                conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({self._scope + "/conv1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))
