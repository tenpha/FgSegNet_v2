import keras
from keras.models import Model
from keras.layers import Input, Dropout, Activation, SpatialDropout2D, BatchNormalization, Lambda, Concatenate
from keras.layers.convolutional import Conv2D, Cropping2D, UpSampling2D, ZeroPadding2D, DepthwiseConv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import concatenate, add, multiply
from keras.optimizers import Adam
from keras.utils import get_file, conv_utils
from keras.engine import Layer
from keras.engine import InputSpec
from my_upsampling_2d import MyUpSampling2D
from instance_normalization import InstanceNormalization
from keras import backend as K
import tensorflow as tf
from mobilenetV2 import mobilenetV2
import numpy as np
from resnet import resnet50
from attention_module import attach_attention_module
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


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = InstanceNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = InstanceNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.image_data_format()
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class FgSegNet_v2_module(object):
    def __init__(self,lr,img_shape,scene,weights_path):
        self.lr = lr
        self.img_shape = img_shape
        self.scene = scene
        self.weights_path = weights_path

    def decoder(self,x,skip1):
        # encoder
        OS = 16
        input_shape = self.img_shape
        b0 = Conv2D(64, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b0 = InstanceNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        # b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = Activation('relu', name='aspp0_activation')(b0)
        # b0 = attach_attention_module(b0, attention_module='cbam_block')

        b1 = SepConv_BN(x, 64, 'aspp1', rate=4, depth_activation=True, epsilon=1e-5)
        # b1 = attach_attention_module(b1, attention_module='cbam_block')
        b2 = SepConv_BN(x, 64, 'aspp2', rate=8, depth_activation=True, epsilon=1e-5)
        # b2 = attach_attention_module(b2, attention_module='cbam_block')
        b3 = SepConv_BN(x, 64, 'aspp3', rate=16, depth_activation=True, epsilon=1e-5)
        # b3 = attach_attention_module(b3, attention_module='cbam_block')

        b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
        b4 = Conv2D(64, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
        b4 = InstanceNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        # b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = Activation('relu')(b4)
        b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(b4)
        # b4 = attach_attention_module(b4, attention_module='cbam_block')
        # b4 = MaxPooling2D(pool_size=(2,2),strides=1,padding='same')(x)
        # b4 = Conv2D(64,(1,1),padding='same',use_bias=False, name='image_pooling')(b4)
        # b4 = InstanceNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        # b4 = Activation('relu')(b4)
        x = Concatenate()([b4, b0, b1, b2, b3])
        x = attach_attention_module(x, attention_module='cbam_block')

        # decoder
        x = Conv2D(64, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
        # x = attach_attention_module(x, attention_module='cbam_block')
        x = InstanceNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        # x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = attach_attention_module(x, attention_module='cbam_block')
        x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                            int(np.ceil(input_shape[1] / 4))))(x)
        dec_skip1 = Conv2D(64, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)
        # dec_skip1 = InstanceNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        # dec_skip1 = attach_attention_module(dec_skip1, attention_module='cbam_block')
        dec_skip1 = InstanceNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        # dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation('relu')(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        # x = attach_attention_module(x, attention_module='cbam_block')

        x = SepConv_BN(x, 64, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 64, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)
        x = attach_attention_module(x, attention_module='cbam_block')
        x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)
        # x = BatchNormalization(name='feature_projection1', epsilon=1e-5)(x)
        # x = attach_attention_module(x, attention_module='cbam_block')
        x = Conv2D(1, 1, padding='same', activation='sigmoid')(x)

        return x

    def initModel(self, dataset_name):
        assert dataset_name in ['CDnet', 'SBI', 'UCSD'], 'dataset_name must be either one in ["CDnet", "SBI", "UCSD"]]'
        assert len(self.img_shape) == 3
        h, w, d = self.img_shape
        img_input = Input(shape=(h,w,d),name = 'img_input')
        rs_output = resnet50(img_input)
        rs_model = Model(img_input,rs_output,name='LLNet')

        # for layer in rs_model.layers:
        #     layer.trainable = False
        # for layer in rs_model.layers[-38:]:
        #     layer.trainable = True
        path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    self.weights_path,
                                    cache_subdir='models')
        rs_model.load_weights(path, by_name=True)

        # opt = keras.optimizers.RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.)

        x,skip1 = rs_model.output
        x = self.decoder(x,skip1)
        model = Model(img_input,x)
        # Since UCSD has no void label, we do not need to filter out

        model.compile(loss=loss, optimizer=Adam(self.lr), metrics=[acc])
        return model

