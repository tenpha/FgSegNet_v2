#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 2018

@author: longang
"""

import keras
from keras.models import Model
from keras.layers import Input, Dropout, Activation, SpatialDropout2D, Concatenate, Add, Multiply,Lambda
from keras.layers.convolutional import Conv2D, Cropping2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import concatenate, add, multiply
from my_upsampling_2d import MyUpSampling2D
from instance_normalization import InstanceNormalization
import keras.backend as K
import tensorflow as tf


def loss(y_true, y_pred):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx)
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def acc(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx)
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def loss2(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def acc2(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def slice(x,idx):
    if idx == 4:
        return Lambda(lambda l: l[:,:,:,idx:])(x)
    else:
        return Lambda(lambda l: l[:,:,:,idx:idx+1])(x)

class FgSegNet_v2_module(object):

    def __init__(self, lr, img_shape, scene, vgg_weights_path):
        self.lr = lr
        self.img_shape = img_shape
        self.scene = scene
        self.vgg_weights_path = vgg_weights_path
        self.method_name = 'FgSegNet2'

    def VGG16(self, x):

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        a = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        b = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Dropout(0.5, name='dr1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Dropout(0.5, name='dr2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Dropout(0.5, name='dr3')(x)

        return x, a, b

    def decoder(self, x, a, b):
        a = GlobalAveragePooling2D()(a)
        b = Conv2D(64, (1, 1), strides=1, padding='same')(b)
        b = GlobalAveragePooling2D()(b)

        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x1 = multiply([x, b])
        x = add([x, x1])
        x = UpSampling2D(size=(2, 2))(x)

        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x2 = multiply([x, a])
        x = add([x, x2])
        x = UpSampling2D(size=(2, 2))(x)

        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(1, 1, padding='same', activation='sigmoid')(x)
        return x

    def add_conv(self, x, channel, ksize, strides=(1, 1), padding='valid', dilation_rate=(1, 1)):
        x = Conv2D(channel, kernel_size=ksize, strides=strides, padding=padding, dilation_rate=dilation_rate)(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        return x

    def M_FPM(self, x):

        pool = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x)
        pool = Conv2D(128, (1, 1), padding='same')(pool)
        pool = InstanceNormalization()(pool)
        pool = Activation('relu')(pool)

        d1 = self.add_conv(x, 128, 3, padding='same', dilation_rate=1)
        d4 = self.add_conv(x, 128, 3, padding='same', dilation_rate=4)
        d8 = self.add_conv(x, 128, 3, padding='same', dilation_rate=8)
        d16 = self.add_conv(x, 128, 3, padding='same', dilation_rate=16)

        weight_pool = self.add_conv(pool, 16, 1, padding='same')
        weight_d1 = self.add_conv(d1, 16, 1, padding='same')
        weight_d4 = self.add_conv(d4, 16, 1, padding='same')
        weight_d8 = self.add_conv(d8, 16, 1, padding='same')
        weight_d16 = self.add_conv(d16, 16, 1, padding='same')

        levels = Concatenate()([weight_pool, weight_d1, weight_d4, weight_d8, weight_d16])

        weight_levels = Conv2D(5, kernel_size=1)(levels)
        weight_levels = Activation('softmax')(weight_levels)
        # weight_levels = Dropout(0.1)(weight_levels)
        fused_out = Add()([Multiply()([pool, slice(weight_levels,0)]),
                         Multiply()([d1,slice(weight_levels,1)]),
                         Multiply()([d4, slice(weight_levels,2)]),
                         Multiply()([d8, slice(weight_levels,3)]),
                         Multiply()([d16, slice(weight_levels,4)]),
                         ])

        fused_out = self.add_conv(fused_out,256,3, padding='same')
        fused_out = SpatialDropout2D(0.25)(fused_out)
        fused_out = self.add_conv(fused_out, 128, 3, padding='same')
        # fused_out = Dropout(0.2)(fused_out)
        return fused_out

    def initModel(self, dataset_name):
        assert dataset_name in ['CDnet', 'SBI', 'UCSD'], 'dataset_name must be either one in ["CDnet", "SBI", "UCSD"]]'
        assert len(self.img_shape) == 3
        h, w, d = self.img_shape

        net_input = Input(shape=(h, w, d), name='net_input')
        vgg_output = self.VGG16(net_input)
        model = Model(inputs=net_input, outputs=vgg_output, name='model')
        model.load_weights(self.vgg_weights_path, by_name=True)

        unfreeze_layers = ['block4_conv1', 'block4_conv2', 'block4_conv3']
        for layer in model.layers:
            if (layer.name not in unfreeze_layers):
                layer.trainable = False

        x, a, b = model.output

        # pad in case of CDnet2014
        if dataset_name == 'CDnet':
            x1_ups = {'streetCornerAtNight': (0, 1), 'tramStation': (1, 0), 'turbulence2': (1, 0)}
            for key, val in x1_ups.items():
                if self.scene == key:
                    # upscale by adding number of pixels to each dim.
                    x = MyUpSampling2D(size=(1, 1), num_pixels=val, method_name=self.method_name)(x)
                    break

        x = self.M_FPM(x)
        x = self.decoder(x, a, b)

        # pad in case of CDnet2014
        if dataset_name == 'CDnet':
            if (self.scene == 'tramCrossroad_1fps'):
                x = MyUpSampling2D(size=(1, 1), num_pixels=(2, 0), method_name=self.method_name)(x)
            elif (self.scene == 'bridgeEntry'):
                x = MyUpSampling2D(size=(1, 1), num_pixels=(2, 2), method_name=self.method_name)(x)
            elif (self.scene == 'fluidHighway'):
                x = MyUpSampling2D(size=(1, 1), num_pixels=(2, 0), method_name=self.method_name)(x)
            elif (self.scene == 'streetCornerAtNight'):
                x = MyUpSampling2D(size=(1, 1), num_pixels=(1, 0), method_name=self.method_name)(x)
                x = Cropping2D(cropping=((0, 0), (0, 1)))(x)
            elif (self.scene == 'tramStation'):
                x = Cropping2D(cropping=((1, 0), (0, 0)))(x)
            elif (self.scene == 'twoPositionPTZCam'):
                x = MyUpSampling2D(size=(1, 1), num_pixels=(0, 2), method_name=self.method_name)(x)
            elif (self.scene == 'turbulence2'):
                x = Cropping2D(cropping=((1, 0), (0, 0)))(x)
                x = MyUpSampling2D(size=(1, 1), num_pixels=(0, 1), method_name=self.method_name)(x)
            elif (self.scene == 'turbulence3'):
                x = MyUpSampling2D(size=(1, 1), num_pixels=(2, 0), method_name=self.method_name)(x)

        vision_model = Model(inputs=net_input, outputs=x, name='vision_model')
        opt = keras.optimizers.RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.)

        # Since UCSD has no void label, we do not need to filter out
        if dataset_name == 'UCSD':
            c_loss = loss2
            c_acc = acc2
        else:
            c_loss = loss
            c_acc = acc

        vision_model.compile(loss=c_loss, optimizer=opt, metrics=[c_acc])
        return vision_model
