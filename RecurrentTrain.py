#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to   in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Any results you write to the current directory are saved as output.
from tensorflow.keras.utils import to_categorical

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
    epochs = 6

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


from RecurrentModelFull import RecurrentTrain
model = RecurrentTrain(typ).model

if(typ=='vgg'):
    target_size=(224,224)
elif(typ=='inception'):
    target_size = (299, 299)
elif(typ=='resnet'):
    #resnet
    target_size = (224, 224)


train_df = pd.read_csv("fashion_product_train.csv")
val_df = pd.read_csv("fashion_product_validation.csv")
test_df = pd.read_csv("fashion_product_test.csv")


lblmapsub = {'Bags': 0, 'Belts': 1, 'Bottomwear': 2, 'Dress': 3, 'Eyewear': 4, 'Flip Flops': 5, 'Fragrance': 6, 'Headwear': 7, 'Innerwear': 8, 'Jewellery': 9, 'Lips': 10, 'Loungewear and Nightwear': 11, 'Nails': 12, 'Sandal': 13, 'Saree': 14, 'Shoes': 15, 'Socks': 16, 'Ties': 17, 'Topwear': 18, 'Wallets': 19, 'Watches': 20}

lblmaparticle = {'Backpacks': 0, 'Belts': 1, 'Bra': 2, 'Briefs': 3, 'Capris': 4, 'Caps': 5, 'Casual Shoes': 6, 'Clutches': 7, 'Deodorant': 8, 'Dresses': 9, 'Earrings': 10, 'Flats': 11, 'Flip Flops': 12, 'Formal Shoes': 13, 'Handbags': 14, 'Heels': 15, 'Innerwear Vests': 16, 'Jackets': 17, 'Jeans': 18, 'Kurtas': 19, 'Kurtis': 20, 'Leggings': 21, 'Lipstick': 22, 'Nail Polish': 23, 'Necklace and Chains': 24, 'Nightdress': 25, 'Pendant': 26, 'Perfume and Body Mist': 27, 'Sandals': 28, 'Sarees': 29, 'Shirts': 30, 'Shorts': 31, 'Socks': 32, 'Sports Shoes': 33, 'Sunglasses': 34, 'Sweaters': 35, 'Sweatshirts': 36, 'Ties': 37, 'Tops': 38, 'Track Pants': 39, 'Trousers': 40, 'Tshirts': 41, 'Tunics': 42, 'Wallets': 43, 'Watches': 44}

lblmapmaster = {'Accessories': 0, 'Apparel': 1, 'Footwear': 2, 'Personal Care': 3}

train_df['masterCategory'].replace(lblmapmaster,inplace=True)
test_df['masterCategory'].replace(lblmapmaster,inplace=True)
val_df['masterCategory'].replace(lblmapmaster,inplace=True)

train_df['subCategory'].replace(lblmapsub,inplace=True)
test_df['subCategory'].replace(lblmapsub,inplace=True)
val_df['subCategory'].replace(lblmapsub,inplace=True)

train_df['articleType'].replace(lblmaparticle,inplace=True)
test_df['articleType'].replace(lblmaparticle,inplace=True)
val_df['articleType'].replace(lblmaparticle,inplace=True)

#to_categorical converts class vector to binary matrix.

onehot_master = to_categorical(train_df['masterCategory'].values)
train_df['masterCategoryOneHot'] = onehot_master.tolist()
onehot_master = to_categorical(val_df['masterCategory'].values)
val_df['masterCategoryOneHot'] = onehot_master.tolist()
onehot_master = to_categorical(test_df['masterCategory'].values)
test_df['masterCategoryOneHot'] = onehot_master.tolist()

onehot_master = to_categorical(train_df['subCategory'].values)
train_df['subCategoryOneHot'] = onehot_master.tolist()
onehot_master = to_categorical(val_df['subCategory'].values)
val_df['subCategoryOneHot'] = onehot_master.tolist()
onehot_master = to_categorical(test_df['subCategory'].values)
test_df['subCategoryOneHot'] = onehot_master.tolist()

onehot_master = to_categorical(train_df['articleType'].values)
train_df['articleTypeOneHot'] = onehot_master.tolist()
onehot_master = to_categorical(val_df['articleType'].values)
val_df['articleTypeOneHot'] = onehot_master.tolist()
onehot_master = to_categorical(test_df['articleType'].values)
test_df['articleTypeOneHot'] = onehot_master.tolist()





print(train_df.head())





direc = '../data/fashion-dataset/images/'


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')

val = test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')


def get_flow_from_dataframe(g, dataframe,
                            image_shape=target_size,batch_size=batch):

    '''
    #return a dataframeiterator which yields tuples of (x, y) where x is numpy array of batch of images, y is np array of labels.
    g = generator.flow_from_dataframe(
        dataframe=dataframe,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')
    '''
    while True:
        x_1 = g.next()
        #x_2 = train_generator_2.next()


        #these are both numpy ndarrays
        #print(type(x_1[1][0]))
        #print(type(x_1[1][1]))
        #the y must not be a list of numpy arrays but rather a true numpy array.
        #y = np.split(x_1[1],[4,26],axis=1)

        #type of x_1[1] is a list! it is a list of 3 np arrays this may be the problem.
        #print(x_1[1][0].shape)
        #print(x_1[1][1].shape)
        #print(x_1[1][2].shape)

        #y = np.concatenate(x_1[1],axis=1)
        #print(y.shape)
        yield [x_1[0], x_1[1][0], x_1[1][1]], x_1[1]

        #should be list of length 3 and list of length 3


'''
# Creating objects for image augmentations
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
'''
# Proving the path of training and test dataset
# Setting the image input size as (224, 224)
# We are using class mode as binary because there are only two classes in our data
# can't do flow_from_directory as it expects one directory per class

# training_set = train_datagen.flow_from_directory('training_set',target_size = (224, 224),batch_size = 32,class_mode = 'categorical')

# test_set = test_datagen.flow_from_directory('test_set',target_size = (224, 224),batch_size = 32,class_mode = 'categorical')


train_generator = get_flow_from_dataframe(train,dataframe=train_df,image_shape=target_size,batch_size=batch)

#test_generator = get_flow_from_dataframe(test_datagen,dataframe=test_df,image_shape=target_size,batch_size=batch)

val_generator = get_flow_from_dataframe(val,dataframe=val_df,image_shape=target_size,batch_size=batch)



# Training the model for 10 epochs

# model.fit now supports generators no need for model.fit_generator
# apparently validation-data needs to be the actual dataset?

try:
    STEP_SIZE_TRAIN = train.n // train.batch_size
    STEP_SIZE_VALID = val.n // val.batch_size
    history = model.fit_generator(train_generator,
                        epochs=epochs,
                        validation_data=val_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_steps=STEP_SIZE_VALID)
    print("Finished training")
except ValueError as v:
    print(v)
# Saving the weights in the current directory
model.save_weights("Fashion_pretrain_recurrent_"+epochs+"_"+typ+".h5")

#json = json.dumps(history.history)
#json.dump(history_dict, open(, 'w'))
#f = open("dict.json","w")
#f.write(json)
#f.close()


# summarize history for loss
plt.plot(history.history['master_output_loss'])
plt.plot(history.history['val_master_output_loss'])
plt.plot(history.history['sub_output_loss'])
plt.plot(history.history['val_sub_output_loss'])
plt.plot(history.history['article_output_loss'])
plt.plot(history.history['val_article_output_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train master', 'val master', 'train sub', 'val sub', 'train article', 'val article'], loc='upper left')
plt.show()
plt.savefig('branching'+"_"+epochs+"_"+typ+'_loss.png', bbox_inches='tight')

pd.DataFrame.from_dict(history.history).to_csv('historyRecurrent_'+epochs+'_'+typ+'.csv',index=False)


'''
try:
    #print("Test generator n", test_generator.n)
    ##print("Test generator batch size", test_generator.batch_size)
    #STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

    model.evaluate(x=test_generator)
except ValueError as v:
    print(v)
'''

