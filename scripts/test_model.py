import keras
from keras.models import Model
from keras.layers import Input, Dropout, Activation, SpatialDropout2D, BatchNormalization, Lambda, Concatenate
from keras.layers.convolutional import Conv2D, Cropping2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import concatenate, add, multiply
from keras.optimizers import Adam
from keras.utils import get_file

from my_upsampling_2d import MyUpSampling2D
from instance_normalization import InstanceNormalization
from keras import backend as K
import tensorflow as tf
from mobilenetV2 import mobilenetV2
import numpy as np


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

class FgSegNet_v2_module(object):
    def __init__(self,lr,img_shape,scene,mobile_weights_path):
        self.lr = lr
        self.img_shape = img_shape
        self.scene = scene
        self.mobile_weights_path = mobile_weights_path
        self.method_name = 'Test_model'


    def initModel(self, dataset_name):
        OS = 8
        assert dataset_name in ['CDnet', 'SBI', 'UCSD'], 'dataset_name must be either one in ["CDnet", "SBI", "UCSD"]]'
        assert len(self.img_shape) == 3
        h, w, d = self.img_shape
        img_input = Input(shape=(h,w,d),name = 'img_input')
        x,skip1 = mobilenetV2(img_input)
        b4 = AveragePooling2D(pool_size=(int(np.ceil(h / OS)), int(np.ceil(w / OS))))(x)

        b4 = Conv2D(256, (1, 1), padding='same',
                    use_bias=False, name='image_pooling')(b4)
        b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = Activation('relu')(b4)

        b4 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(
        int(np.ceil(h / OS)), int(np.ceil(w / OS)))))(b4)

        # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = Activation('relu', name='aspp0_activation')(b0)
        x = Concatenate()([b4, b0])
        x = Conv2D(256, (1, 1), padding='same',
                   use_bias=False, name='concat_projection')(x)
        x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)

        x = Conv2D(1, (1, 1), padding='same',name='custom_logits_semantic')(x)
        x = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)))(x)
        x = Activation('sigmoid')(x)

        model = Model(img_input,x,name='DeepLab3+')

        # opt = keras.optimizers.RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.)

        # Since UCSD has no void label, we do not need to filter out
        if dataset_name == 'UCSD':
            c_loss = loss2
            c_acc = acc2
        else:
            c_loss = loss
            c_acc = acc
        weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                    self.mobile_weights_path,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
        model.compile(loss=c_loss, optimizer=Adam(self.lr), metrics=[c_acc])
        return model

