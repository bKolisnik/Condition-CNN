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




arg_names = ['filename','model', 'epochs']
args = dict(zip(arg_names, sys.argv))

if(args.get('model') is not None):
    modelT=args['model']
else:
    modelT = 'master'

if args.get('epochs') is not None:
    epochs = int(args['epochs'])
else:
    epochs = 20

print("Building model " + modelT)
print("For " + str(epochs) + " epochs.")

if(modelT=='master'):
    from MasterCategoryModel import Master
    model = Master().model

elif(modelT=='sub'):
    from SubCategoryModel import SubCategory
    model = SubCategory().model
elif(modelT=='article'):
    #model is articleType
    from ArticleTypeModel import ArticleType
    model = ArticleType().model




train_df = pd.read_csv("kaggle_fashion_product_train.csv").loc[0:1000, :]
val_df = pd.read_csv("kaggle_fashion_product_validation.csv").loc[3000:3600, :]
test_df = pd.read_csv("kaggle_fashion_product_test.csv").loc[0:1000, :]


def check_files_existence(directory, df):
    for row in df.itertuples():

        if os.path.isfile(direc + row.filepath):
            pass
        else:
            print("File path not exist {}".format(row.filepath))


direc = '../data/fashion-dataset/images/'
check_files_existence(direc, train_df)
check_files_existence(direc, val_df)
check_files_existence(direc, test_df)
'''

'''


# Creating objects for image augmentations
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Proving the path of training and test dataset
# Setting the image input size as (224, 224)
# We are using class mode as binary because there are only two classes in our data
# can't do flow_from_directory as it expects one directory per class

# training_set = train_datagen.flow_from_directory('training_set',target_size = (224, 224),batch_size = 32,class_mode = 'categorical')

# test_set = test_datagen.flow_from_directory('test_set',target_size = (224, 224),batch_size = 32,class_mode = 'categorical')

# Training the model for 5 epochs
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=direc,
    x_col="filepath",
    y_col="subCategory",
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical')
val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=direc,
    x_col="filepath",
    y_col="subCategory",
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical')
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=direc,
    x_col="filepath",
    y_col="subCategory",
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical')

# Training the model for 10 epochs

# model.fit now supports generators no need for model.fit_generator
# apparently validation-data needs to be the actual dataset?

try:
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = val_generator.n // val_generator.batch_size
    model.fit_generator(train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=STEP_SIZE_VALID)
    print("Finished training")
except ValueError as v:
    print(v)
# Saving the weights in the current directory
if modelT=='master':
    model.save_weights("Fashion_pretrain_resnet50_MasterCategory.h5")
elif modelT=='sub':
    model.save_weights("Fashion_pretrain_resnet50_SubCategory.h5")
elif modelT=='article':
    model.save_weights("Fashion_pretrain_resnet50_ArticleType.h5")

try:
    print("Test generator n", test_generator.n)
    print("Test geenrator batch size", test_generator.batch_size)
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    model.evaluate_generator(generator=test_generator,
                             steps=STEP_SIZE_TEST)

    model.evaluate()
except ValueError as v:
    print(v)


