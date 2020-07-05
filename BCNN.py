import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Flatten, Input, Conv2D, concatenate
from tensorflow.heras.layers import BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import sys


def scheduler(epoch):
  learning_rate_init = 0.001
  if epoch > 55:
    learning_rate_init = 0.0002
  if epoch > 70:
    learning_rate_init = 0.00005
  return learning_rate_init

class LossWeightsModifier(tf.keras.callbacks.Callback):
  def __init__(self, alpha, beta, gamma):
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    # customize your behavior
  def on_epoch_end(self, epoch, logs={}):
    if epoch == 13:
      K.keras.set_value(self.alpha, 0.1)
      K.set_value(self.beta, 0.8)
      K.set_value(self.gamma, 0.1)
    if epoch == 23:
      K.set_value(self.alpha, 0.1)
      K.set_value(self.beta, 0.2)
      K.set_value(self.gamma, 0.7)
    if epoch == 33:
      K.set_value(self.alpha, 0)
      K.set_value(self.beta, 0)
      K.set_value(self.gamma, 1)


class CNNTrain:
    '''Based on zhuxinqimac implementation on github, article cited in paper
    https://github.com/zhuxinqimac/B-CNN/blob/master/CIFAR_100_keras_vgg16_hierarchy_dynamic.py'''

    def __init__(self):
        self.master_classes=4
        self.sub_classes=21
        self.art_classes=45

        alpha = K.variable(value=0.98, dtype="float32", name="alpha") # A1 in paper
        beta = K.variable(value=0.01, dtype="float32", name="beta") # A2 in paper
        gamma = K.variable(value=0.01, dtype="float32", name="gamma") # A3 in paper

        img_input = Input(shape=(224,224,3), name='input')
        #--- block 1 ---
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        #--- block 2 ---
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        #--- coarse 1 branch ---
        c_1_bch = Flatten(name='c1_flatten')(x)
        c_1_bch = Dense(256, activation='relu', name='c1_fc_cifar10_1')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_bch = Dense(256, activation='relu', name='c1_fc2')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_pred = Dense(self.master_classes, activation='softmax', name='c1_predictions_cifar10')(c_1_bch)

        #--- block 3 ---
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        #--- coarse 2 branch ---
        c_2_bch = Flatten(name='c2_flatten')(x)
        c_2_bch = Dense(1024, activation='relu', name='c2_fc_cifar100_1')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)
        c_2_bch = Dense(1024, activation='relu', name='c2_fc2')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)
        c_2_pred = Dense(self.sub_classes, activation='softmax', name='c2_predictions_cifar100')(c_2_bch)

        #--- block 4 ---
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)


        #--- block 5 ---
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_bcnn')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_bcnn')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_bcnn')(x)
        x = BatchNormalization()(x)

        #--- fine block ---
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc_cifar100_1')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc_cifar100_2')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        fine_pred = Dense(self.art_classes, activation='softmax', name='predictions_cifar100')(x)

        model = Model(img_input, [c_1_pred, c_2_pred, fine_pred], name='BCNN')
        trainable_params = tf.keras.backend.count_params(model.trainable_weights)
        print("Trainable paramaters: "+str(trainable_params))

        self.model = model

