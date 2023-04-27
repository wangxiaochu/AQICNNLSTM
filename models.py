# import the necessary packages
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv3D
from keras.layers import Conv2D,UpSampling2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import TimeDistributed,Reshape
from keras.layers import concatenate
from keras.layers import MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D,MaxPooling3D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.applications import VGG16, ResNet50, VGG19,MobileNet,InceptionV3
from keras.optimizers import Adam
from keras import metrics
from sklearn.metrics import r2_score
import numpy as np
from keras.layers.normalization import BatchNormalization
import glob
import os

##load sequence imagefile
def load_imagefiles(imagePath,imagelist):
    Imagefiles = []
    for imagel in imagelist:
        # load the input image path
        # update the list of input images
        img=[imagePath+image for image in imagel]
        Imagefiles.append(img)
    #return the input image set
    return Imagefiles

##load single imagefile
def load_imagefiles2(imagePath,imagelist):
    Imagefiles = []
    for image in imagelist:
        # load the input image path
        # update the list of input images
        Imagefiles.append(imagePath+image)
    #return the input image set
    return Imagefiles



def transfer_cnn_multioutput(model=VGG16, input_shape=(256,256,3),mode='fc', regress=True):
    # -------------加载模型并且添加自己的层----------------------
    # 参数说明：model:选择要训练的模型
    # input_shape：输入的图像大小(width,height,depth)
    # mode：用于选择卷积之后是用全连接层还是用全局平均或最大池化
    # inlucde_top:是否包含最上方的Dense层,weights=None
    base_model = model(input_shape=input_shape, weights='imagenet', include_top=False, pooling=None)
    x = base_model.output
    if mode == 'fc':
        # FC层设定为含有256,32个参数的隐藏层
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
    elif mode == 'avg':
        # GAP层
        x = GlobalAveragePooling2D()(x)
    elif mode == 'max':
        # GMP层
        x = GlobalMaxPooling2D()(x)

    # add two fully connected layers
    outputs = Dense(32, activation='relu')(x)
    if regress:
        out_aqi = Dense(1, activation="sigmoid",name='out_aqi')(outputs)  # activation="linear"时，估计不受限制，出现异常估计sigmoid
        out_pm2 = Dense(1, activation="sigmoid",name='out_pm2')(outputs)
        out_pm10 = Dense(1, activation="sigmoid",name='out_pm10')(outputs)

    # construct the CNN of multiple outputs
    cnn = Model(inputs=base_model.input, outputs=[out_aqi,out_pm2,out_pm10])
    # for layers in base_model.layers[:]:
    #     layers.trainable = False
    return cnn



def transfer_cnnlstm2_multioutput(model=VGG16, input_shape=(1, 256, 256, 3), out_dim=1, mode='fc', regress=True):
    # -------------加载模型并且添加自己的层----------------------
    # 参数说明：model:选择要训练的模型
    # input_shape：输入的图像大小(width,height,depth)
    # mode：用于选择卷积之后是用全连接层还是用全局平均或最大池化
    # inlucde_top:是否包含最上方的Dense层
    base_model = model(input_shape=(input_shape[1],input_shape[2],input_shape[3]),weights='imagenet',include_top=False) # , pooling=None
    # base_model.summary()
    x=base_model.output
    if mode == 'fc':
        # FC层设定为含有256个参数的隐藏层
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
    elif mode == 'avg':
        # GAP层
        x = GlobalAveragePooling2D()(x)
    elif mode == 'max':
        # GMP层
        x = GlobalMaxPooling2D()(x)


    cnn = Model(base_model.input, x)
    inputs = Input(shape=input_shape)
    encoded_frames = TimeDistributed(cnn)(inputs)
    encoded_sequence = LSTM(512)(encoded_frames)
    outputs = Dense(32, activation='relu')(encoded_sequence)

    if regress:
        out_aqi = Dense(out_dim, activation="sigmoid",name='out_aqi')(outputs)  # activation="linear"时，估计不受限制，出现异常估计sigmoid
        out_pm2 = Dense(out_dim, activation="sigmoid",name='out_pm2')(outputs)
        out_pm10 = Dense(out_dim, activation="sigmoid",name='out_pm10')(outputs)

    # construct the CNN of multiple outputs
    model = Model(inputs=inputs, outputs=[out_aqi,out_pm2,out_pm10])

    return model
