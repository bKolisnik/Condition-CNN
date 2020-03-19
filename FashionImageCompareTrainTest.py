
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

'''This module takes in a trained model and compares it's predictions on each dataset
then output confusion matrices for each dataset so you can see why the model is performing well on train but poorly on test.'''

direc = '../data/fashion-dataset/images/'
arg_names = ['filename','model','batch']
args = dict(zip(arg_names, sys.argv))

if(args.get('model') is not None):
    modelT=args['model']
else:
    modelT = 'masterCategory'

if(args.get('batch') is not None):
    batch = int(args['batch'])
else:
    batch = 64

print("Testing model " + modelT)
print("Batch size " + str(batch))

if(modelT=='masterCategory'):
    from MasterCategoryModel import Master
    model = Master().model

elif(modelT=='subCategory'):
    from SubCategoryModel import SubCategory
    model = SubCategory().model
elif(modelT=='articleType'):
    #model is articleType
    from ArticleTypeModel import ArticleType
    model = ArticleType().model

if modelT=='masterCategory':
    model.load_weights("Fashion_pretrain_resnet50_MasterCategory.h5")
elif modelT=='subCategory':
    model.load_weights("Fashion_pretrain_resnet50_SubCategory.h5")
elif modelT=='articleType':
    model.load_weights("Fashion_pretrain_resnet50_ArticleType.h5")

test_df = pd.read_csv("kaggle_fashion_product_test.csv")
train_df = pd.read_csv("kaggle_fashion_product_train.csv")
val_df = pd.read_csv("kaggle_fashion_product_validation.csv")



test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=direc,
        x_col="filepath",
        y_col=modelT,
        target_size=(224, 224),
        batch_size=batch,
        class_mode='categorical')

train_generator = test_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=direc,
    x_col="filepath",
    y_col=modelT,
    target_size=(224, 224),
    batch_size=batch,
    class_mode='categorical')
val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=direc,
    x_col="filepath",
    y_col=modelT,
    target_size=(224, 224),
    batch_size=batch,
    class_mode='categorical')



#do the predictions

#returns a numpy array of predictions
train_pred = model.predict(x=train_generator)
train_labels = train_df[modelT].values
print("Train Confusion Matrix")
print(tf.math.confusion_matrix(train_labels,train_pred))


test_pred = model.predict(x=test_generator)
test_labels = test_df[modelT].values
print("Test Confusion Matrix")
print(tf.math.confusion_matrix(train_labels,train_pred))

val_pred = model.predict(x=val_generator)
val_labels = val_df[modelT].values
print("Validation Confusion Matrix")
print(tf.math.confusion_matrix(train_labels,train_pred))