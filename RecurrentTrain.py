#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Any results you write to the current directory are saved as output.

print(os.getcwd())

# start with subcategory as labels

# train_df= pd.read_csv("../input/kaggle-fashion/kaggle_fashion_product_train (2).csv").loc[[0,2,4],:]
# val_df = pd.read_csv("../input/kaggle-fashion/kaggle_fashion_product_validation (2).csv").loc[[0,2],:]
# test_df = pd.read_csv("../input/kaggle-fashion/kaggle_fashion_product_test (2).csv").loc[[0,2],:]
# print(train_df.head(5))


#get commandline rguments to know which model to train.

import sys




arg_names = ['filename','typ', 'epochs','batch']
args = dict(zip(arg_names, sys.argv))



if args.get('epochs') is not None:
    epochs = int(args['epochs'])
else:
    epochs = 20

if(args.get('batch') is not None):
    batch = int(args['batch'])
else:
    batch = 64

if(args.get('typ') is not None):
    typ=args['typ']
else:
    typ = 'vgg'


print("Using architecture of type " + typ)
print("For " + str(epochs) + " epochs.")
print("Batch size " + str(batch))


from ArticleTypeModel import ArticleType
model = ArticleType(typ).model

if(typ=='vgg'):
    target_size=(224,224)
elif(typ=='inception'):
    target_size = (299, 299)
elif(typ=='resnet'):
    #resnet
    target_size = (224, 224)


train_df = pd.read_csv("fashion_product_train_full.csv")
val_df = pd.read_csv("fashion_product_validation_full.csv")
test_df = pd.read_csv("fashion_product_test_full.csv")

print(test_df.groupby("masterCategory").count()['id'])
print(train_df.groupby("masterCategory").count()['id'])
print(val_df.groupby("masterCategory").count()['id'])




direc = '../data/fashion-dataset/images/'


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)


def get_flow_from_dataframe(generator, dataframe,
                            image_shape=target_size,batch_size=batch):

    #return a dataframeiterator which yields tuples of (x, y) where x is numpy array of batch of images, y is np array of labels.
    train_generator = generator.flow_from_dataframe(
        dataframe=dataframe,
        directory=direc,
        x_col="filepath",
        y_col='targets',
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')

    while True:
        x_1 = train_generator.next()
        #x_2 = train_generator_2.next()

        print(x_1[1].shape)
        yield [x_1[0], x_1[1][:,0], x_1[1][:,1]], x_1[1]

        #should be list of length 3 and list of length 3



# Creating objects for image augmentations
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Proving the path of training and test dataset
# Setting the image input size as (224, 224)
# We are using class mode as binary because there are only two classes in our data
# can't do flow_from_directory as it expects one directory per class

# training_set = train_datagen.flow_from_directory('training_set',target_size = (224, 224),batch_size = 32,class_mode = 'categorical')

# test_set = test_datagen.flow_from_directory('test_set',target_size = (224, 224),batch_size = 32,class_mode = 'categorical')


train_generator = get_flow_from_dataframe(train_datagen,dataframe=train_df,image_shape=target_size,batch_size=batch)

test_generator = get_flow_from_dataframe(test_datagen,dataframe=test_df,image_shape=target_size,batch_size=batch)

val_generator = get_flow_from_dataframe(test_datagen,dataframe=val_df,image_shape=target_size,batch_size=batch)



# Training the model for 10 epochs

# model.fit now supports generators no need for model.fit_generator
# apparently validation-data needs to be the actual dataset?

try:
    #STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    #STEP_SIZE_VALID = val_generator.n // val_generator.batch_size
    model.fit_generator(train_generator,
                        epochs=epochs,
                        validation_data=val_generator,)
    print("Finished training")
except ValueError as v:
    print(v)
# Saving the weights in the current directory
model.save_weights("Fashion_pretrain_recurrent"+typ+".h5")


try:
    print("Test generator n", test_generator.n)
    print("Test generator batch size", test_generator.batch_size)
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

    model.evaluate(x=test_generator,steps=STEP_SIZE_TEST)
except ValueError as v:
    print(v)


