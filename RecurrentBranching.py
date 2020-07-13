import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Flatten, Input, Conv2D, concatenate
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import sys

#This new arrangement is designed to look as similar as possible to the B-CNN model
#This includes BatchNorm layers between layers. 5 blocks no top from VGG16

class RecurrentTrain:
    '''This model is based off of VGG16 with the addition of BatchNorm layers and then branching '''

    def __init__(self,label):
        self.master_classes=4
        self.sub_classes=21
        self.art_classes=45
        
        '''Three inputs to model for training, image, labels ,labels for teacher forcing'''
        input_image = Input(shape=(224,224,3),name="InputImg")

        #4 element vector for the masterCategory types
        input_master = Input(shape=(self.master_classes))

        #21 element vector for the subCategory types
        input_sub = Input(shape=(self.sub_classes))

        #Layers are named to be same as VGG16

        #will need to center and scale data.

        #--- block 1 ---
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_image)
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

        #--- block 3 ---
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        #--- block 4 ---
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        #--- block 5 masterCategory ---
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_mas')(x)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_mas')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_mas')(x1)
        x1 = BatchNormalization()(x1)

        #--- masterCategory prediction branch ---
        c_1_bch = Flatten(name='c1_flatten')(x1)
        c_1_bch = Dense(256, activation='relu', name='c1_fc_mas')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_bch = Dense(256, activation='relu', name='c1_fc2_mas')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_pred = Dense(self.master_classes, activation='softmax', name='master_output')(c_1_bch)

        #--- block 5 subCategory ---
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_sub')(x)
        x2 = BatchNormalization()(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_sub')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_sub')(x2)
        x2 = BatchNormalization()(x2)

        #--- coarse 2 branch ---
        c_2_bch = Flatten(name='c2_flatten')(x2)
        c_2_bch = Dense(1024, activation='relu', name='c2_fc_sub')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)
        c_2_bch = Dense(1024, activation='relu', name='c2_fc2_sub')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)
        preds_features = concatenate([c_2_bch,input_master])
        c_2_pred = Dense(self.sub_classes, activation='softmax', name='sub_output')(preds_features)

        #--- block 5 articleType ---
        x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_art')(x)
        x3 = BatchNormalization()(x3)
        x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_art')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_art')(x3)
        x3 = BatchNormalization()(x3)

        #--- fine block ---
        y = Flatten(name='flatten')(x3)
        y = Dense(4096, activation='relu', name='fc_art')(y)
        y = BatchNormalization()(y)
        y = Dropout(0.5)(y)
        y = Dense(4096, activation='relu', name='fc2_art')(y)
        y = BatchNormalization()(y)
        y = Dropout(0.5)(y)
        preds_features = concatenate([y,input_sub,input_master])
        fine_pred = Dense(self.art_classes, activation='softmax', name='article_output')(preds_features)

        # Model to be trained
        #modelC = Model(inputs=[input_image,input_master,input_sub], outputs=z)

            # trainable_params = tf.keras.backend.count_params(model.trainable_weights)
            # print("Trainable paramaters: "+str(trainable_params))

        # print(model.summary())
        # Compiling the model
        # KEras will automatically use categorical accuracy when accuracy is used.

        model = Model(
            inputs=[input_image,input_master,input_sub],
            outputs=[c_1_pred, c_2_pred, fine_pred],
            name="Branching_CNN")

        trainable_params= np.sum([K.count_params(w) for w in model.trainable_weights])
        #trainable_params = tf.keras.backend.count_params(model.trainable_weights)
        print("Trainable paramaters: "+str(trainable_params))

        losses = {
            "master_output": "categorical_crossentropy",
            "sub_output": "categorical_crossentropy",
            "article_output": "categorical_crossentropy"
        }
        model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss=losses,
                      metrics=['categorical_accuracy'])

        checkpoint = ModelCheckpoint("../weights/"+label+"_{epoch:02d}_epochs_{val_loss:.2f}_val_loss.h5", monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=True,mode='auto')
        self.cbks = [checkpoint]
        self.model = model

class RecurrentTest:
    '''One parameter model which is a keras model'''

    def __init__(self,label):

        self.master_classes=4
        self.sub_classes=21
        self.art_classes=45


        '''Three inputs to model for training, image, labels ,labels for teacher forcing'''
        input_image = Input(shape=(224,224,3),name="InputImg")

        #Layers are named to be same as VGG16

        #will need to center and scale data.

        #--- block 1 ---
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_image)
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

        #--- block 3 ---
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        #--- block 4 ---
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        #--- block 5 masterCategory ---
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_mas')(x)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_mas')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_mas')(x1)
        x1 = BatchNormalization()(x1)

        #--- masterCategory prediction branch ---
        c_1_bch = Flatten(name='c1_flatten')(x1)
        c_1_bch = Dense(256, activation='relu', name='c1_fc_mas')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_bch = Dense(256, activation='relu', name='c1_fc2_mas')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_pred = Dense(self.master_classes, activation='softmax', name='master_output')(c_1_bch)

        #--- block 5 subCategory ---
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_sub')(x)
        x2 = BatchNormalization()(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_sub')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_sub')(x2)
        x2 = BatchNormalization()(x2)

        #--- coarse 2 branch ---
        c_2_bch = Flatten(name='c2_flatten')(x2)
        c_2_bch = Dense(1024, activation='relu', name='c2_fc_sub')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)
        c_2_bch = Dense(1024, activation='relu', name='c2_fc2_sub')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)
        preds_features = concatenate([c_2_bch,c_1_pred])
        c_2_pred = Dense(self.sub_classes, activation='softmax', name='sub_output')(preds_features)

        #--- block 5 articleType ---
        x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_art')(x)
        x3 = BatchNormalization()(x3)
        x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_art')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_art')(x3)
        x3 = BatchNormalization()(x3)

        #--- fine block ---
        y = Flatten(name='flatten')(x3)
        y = Dense(4096, activation='relu', name='fc_art')(y)
        y = BatchNormalization()(y)
        y = Dropout(0.5)(y)
        y = Dense(4096, activation='relu', name='fc2_art')(y)
        y = BatchNormalization()(y)
        y = Dropout(0.5)(y)
        preds_features = concatenate([y,c_2_pred,c_1_pred])
        fine_pred = Dense(self.art_classes, activation='softmax', name='article_output')(preds_features)

        model = Model(
            inputs=[input_image],
            outputs=[c_1_pred, c_2_pred, fine_pred],
            name="Branching_CNN")

        losses = {
            "master_output": "categorical_crossentropy",
            "sub_output": "categorical_crossentropy",
            "article_output": "categorical_crossentropy"
        }

        trainable_params= np.sum([K.count_params(w) for w in model.trainable_weights])
        #trainable_params = tf.keras.backend.count_params(model.trainable_weights)
        print("Trainable paramaters: "+str(trainable_params))

        model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss=losses,
                      metrics=['categorical_accuracy'])

        self.model = model