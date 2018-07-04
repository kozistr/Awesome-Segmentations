from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import sys

import vgg19 as vgg19

sys.path.append("../")
import tfutil as t


seed = 1337

tf.set_random_seed(seed)
np.random.seed(seed)


class FCN:

    def __init__(self, session, batch_size=16, height=224, width=224, channel=3, n_classes=151,
                 learning_rate=1e-4):
        self.s = session
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.channel = channel
        self.image_shape = [self.height, self.width, self.channel]

        self.n_classes=n_classes

        self.lr = learning_rate

        # Placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel], name='x-image')

        self.do_rate = tf.placeholder(tf.float32, shape=[], name='do-rate')

        self.build_fcn()

    def build_fcn(self):
        vgg19_net = vgg19.VGG19(image=self.x)

        net = vgg19_net.vgg19_net['pool5']

        net = t.conv2d(net, 4096, k=7, s=1, name='conv6_1')
        net = tf.nn.relu(net, name='relu6_1')
        net = tf.nn.dropout(net, self.do_rate, name='dropout-6_1')

        net = t.conv2d(net, 4096, k=1, s=1, name='conv7_1')
        net = tf.nn.relu(net, name='relu7_1')
        net = tf.nn.dropout(net, self.do_rate, name='dropout-7_1')

        feature = t.conv2d(net, self.n_classes, k=1, s=1, name='conv8_1')

        net = t.deconv2d(feature, vgg19_net.vgg19_net['pool4'].get_shape()[3], name='deconv_1')
        net = tf.add(net, vgg19_net.vgg19_net['pool4'], name='fuse_1')