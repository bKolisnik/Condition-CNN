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




arg_names = ['filename', 'epochs','batch']
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
    typ = 'inception'

print("Building a masterCategory model for AIDEAL")
print("Using architecture of type " + typ)
print("For " + str(epochs) + " epochs.")
print("Batch size " + str(batch))


from MasterCategoryModel import Master
model = Master(typ).model

if(typ=='vgg'):
    target_size=(224,224)
elif(typ=='inception'):
    target_size = (299, 299)
elif(typ=='resnet'):
    #resnet
    target_size = (224, 224)





# Creating objects for image augmentations
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

direc = '../data/train/'

# training_set = train_datagen.flow_from_directory('training_set',target_size = (224, 224),batch_size = 32,class_mode = 'categorical')

# test_set = test_datagen.flow_from_directory('test_set',target_size = (224, 224),batch_size = 32,class_mode = 'categorical')

# Training the model for 5 epochs
train_generator = train_datagen.flow_from_directory(
    directory='../data/train/',
    target_size=target_size,
    batch_size=batch,
    class_mode='categorical')
val_generator = test_datagen.flow_from_directory(
    directory='../data/val/',
    target_size=target_size,
    batch_size=batch,
    class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
    directory='../data/test/',
    target_size=target_size,
    batch_size=batch,
    class_mode='categorical')

print(train_generator.class_indices)
print(test_generator.class_indices)
print(val_generator.class_indices)

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

model.save_weights("AIDEAL_pretrain_MasterCategory_"+typ+".h5")

try:
    print("Test generator n", test_generator.n)
    print("Test generator batch size", test_generator.batch_size)
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

    model.evaluate(x=test_generator,steps=STEP_SIZE_TEST)
except ValueError as v:
    print(v)


