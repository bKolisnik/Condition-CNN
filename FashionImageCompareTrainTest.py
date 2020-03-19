
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
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

def save_predictions():
    train_pred = model.predict(x=train_generator)
    np.save('train_pred.npy',train_pred)

    test_pred = model.predict(x=test_generator)
    np.save('test_pred.npy', test_pred)

    val_pred = model.predict(x=val_generator)
    np.save('val_pred.npy', val_pred)

#save_predictions()


#predict returns the 2d array where each sample returns a vector of predictions.
train_pred = np.load('train_pred.npy')
train_labels = train_df[modelT].values

#label_map has class name keys and index of the output vector as values
train_label_map = train_generator.class_indices
inv_train_label_map = dict(zip(train_label_map.values(),train_label_map.keys()))

print(train_label_map)
tr_pr = [None]*len(train_labels)
for i in range(0,len(train_labels)):
    tr_pr[i] = inv_train_label_map(np.argmax(train_pred[i]))
    #tr_l[i] = data_dict[train_labels[i]]


print("Train Confusion Matrix")
print(tf.math.confusion_matrix(train_labels,tr_pr))


test_pred = np.load('test_pred.npy')
test_labels = test_df[modelT].values


te_pr = [None]*len(test_labels)
for i in range(0,len(test_labels)):
    tr_pr[i] = inv_train_label_map(np.argmax(test_pred[i]))


print("Test Confusion Matrix")
print(tf.math.confusion_matrix(test_labels,te_pr))

val_pred = np.load('val_pred.npy')
val_labels = val_df[modelT].values

#va_pr = np.zeros(len(val_pred))

val_pr = [None]*len(val_labels)
for i in range(0,len(test_labels)):
    tr_pr[i] = inv_train_label_map(np.argmax(val_pred[i]))
print("Validation Confusion Matrix")
print(tf.math.confusion_matrix(val_labels,va_pr))