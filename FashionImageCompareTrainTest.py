
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import sys
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os.path
from sklearn.metrics import confusion_matrix

import cv2


'''This module takes in a trained model and compares it's predictions on each dataset
then output confusion matrices for each dataset so you can see why the model is performing well on train but poorly on test.'''

direc = '../data/fashion-dataset/images/'
arg_names = ['filename','model','typ','batch']
args = dict(zip(arg_names, sys.argv))

if(args.get('model') is not None):
    modelT=args['model']
else:
    modelT = 'masterCategory'

if(args.get('typ') is not None):
    typ=args['typ']
else:
    typ = 'inception'

if(args.get('batch') is not None):
    batch = int(args['batch'])
else:
    batch = 64

print("Testing model " + modelT)
print("Batch size " + str(batch))
print("Type is "+typ)

if(modelT=='masterCategory'):
    from MasterCategoryModel import Master
    model = Master(typ).model

elif(modelT=='subCategory'):
    from SubCategoryModel import SubCategory
    model = SubCategory(typ).model
elif(modelT=='articleType'):
    #model is articleType
    from ArticleTypeModel import ArticleType
    model = ArticleType(typ).model

if(typ=='vgg'):
    target_size=(224,224)
elif(typ=='inception'):
    target_size = (299, 299)
elif(typ=='resnet'):
    #resnet
    target_size = (224, 224)

if modelT=='masterCategory':
    model.load_weights("Fashion_pretrain_resnet50_MasterCategory_"+typ+".h5")
elif modelT=='subCategory':
    model.load_weights("Fashion_pretrain_SubCategory_"+typ+".h5")
elif modelT=='articleType':
    model.load_weights("Fashion_pretrain_ArticleType+"+typ+".h5")


test_df = pd.read_csv("fashion_product_test.csv")
train_df = pd.read_csv("fashion_product_train.csv")
val_df = pd.read_csv("fashion_product_validation.csv")



#test_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=direc,
        x_col="filepath",
        y_col=modelT,
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')
'''
train_generator = test_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=direc,
    x_col="filepath",
    y_col=modelT,
    target_size=target_size,
    batch_size=batch,
    class_mode='categorical')
    '''
val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=direc,
    x_col="filepath",
    y_col=modelT,
    target_size=target_size,
    batch_size=batch,
    class_mode='categorical')



#do the predictions


def save_predictions():
    '''
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    train_pred = model.predict(x=train_generator)

    np.save('train_pred_'+modelT+"_"+typ+'.npy',train_pred)
    '''
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    test_pred = model.predict(x=test_generator)

    np.save('test_pred_'+modelT+"_"+typ+'.npy',test_pred)

    STEP_SIZE_VAL = val_generator.n // val_generator.batch_size
    val_pred = model.predict(x=val_generator)

    np.save('val_pred_'+modelT+"_"+typ+'.npy', val_pred)

#check if predictions already made

if not os.path.isfile('val_pred_'+modelT+"_"+typ+'.npy'):
    save_predictions()


'''
imgs=[]
full_size_image = cv2.imread(direc+"12431.jpg")
resized = cv2.resize(full_size_image, target_size, interpolation=cv2.INTER_CUBIC)
imgs.append(resized)

full_size_image = cv2.imread(direc+"13066.jpg")
resized = cv2.resize(full_size_image, target_size, interpolation=cv2.INTER_CUBIC)
imgs.append(resized)

full_size_image = cv2.imread(direc+"17779.jpg")
resized = cv2.resize(full_size_image, target_size, interpolation=cv2.INTER_CUBIC)
imgs.append(resized)

#imgs = np.array([resized])
imgs = np.array(imgs)

test_pred = model.predict(x=imgs)
print(test_pred)

print(np.argmax(test_pred,axis=-1))

'''

'''
#predict returns the 2d array where each sample returns a vector of predictions.
train_pred = np.load('train_pred_'+modelT+"_"+typ+'.npy')
#train_labels = train_df[modelT].values

train_labels = train_generator.classes
train_pred = np.argmax(train_pred, axis=-1)

acc = sum(train_pred==train_labels)/len(train_pred)
print("Training Accuracy is "+str(acc))

confusion_matrix(train_labels,train_pred)


'''
test_pred = np.load('test_pred_'+modelT+"_"+typ+'.npy')
#rem = test_generator.n % test_generator.batch_size
#test_labels = test_generator.classes[:-rem]
test_labels = test_generator.classes
print(test_generator.class_indices)
print("# labels:",len(test_labels))
print("Test Labels: ")
print(test_labels[:10])
test_pred = np.argmax(test_pred, axis=-1)
print("# predictions:",len(test_pred))
print("Test predictions: ")
print(test_pred[:10])
acc = sum(test_pred==test_labels)/len(test_pred)
print("Testing Accuracy is "+str(acc))


val_pred = np.load('val_pred_'+modelT+"_"+typ+'.npy')
val_labels = val_generator.classes
#rem = val_generator.n % val_generator.batch_size
#val_labels = val_generator.classes[:-rem]
print(val_labels[:10])
val_pred = np.argmax(val_pred, axis=-1)
print(val_pred[:10])
np.savetxt("val_pred_max_"+typ+".txt",val_pred)
acc = sum(val_pred==val_labels)/len(val_pred)
print("Validation Accuracy is "+str(acc))


'''
#label_map has class name keys and index of the output vector as values
train_label_map = train_generator.class_indices
inv_train_label_map = dict(zip(train_label_map.values(),train_label_map.keys()))

print(train_label_map)
tr_pr = [None]*len(train_labels)
tr_l = [None]*len(train_labels)
for i in range(0,len(train_labels)):
    #tr_pr[i] = inv_train_label_map[np.argmax(train_pred[i])]
    tr_pr[i] = np.argmax(train_pred[i])
    tr_l[i] = train_label_map[train_labels[i]]


print("Train Confusion Matrix")
train_mat = pd.DataFrame(tf.math.confusion_matrix(tr_l,tr_pr).numpy())
print(train_mat)

test_pred = np.load('test_pred_'+modelT+"_"+typ+'.npy')
test_labels = test_df[modelT].values


te_pr = [None]*len(test_labels)
te_l = [None]*len(test_labels)
for i in range(0,len(test_labels)):
    #tr_pr[i] = inv_train_label_map[np.argmax(test_pred[i])]
    te_pr[i] = np.argmax(test_pred[i])
    te_l[i] = train_label_map[test_labels[i]]

print("Test Confusion Matrix")
test_mat = pd.DataFrame(tf.math.confusion_matrix(te_l,te_pr).numpy())
print(test_mat)


val_pred = np.load('val_pred_'+modelT+"_"+typ+'.npy')
val_labels = val_df[modelT].values

#va_pr = np.zeros(len(val_pred))

val_pr = [None]*len(val_labels)
val_l = [None]*len(val_labels)
for i in range(0,len(val_labels)):
    #tr_pr[i] = inv_train_label_map[np.argmax(val_pred[i])]
    val_pr[i] = np.argmax(val_pred[i])
    val_l[i] = train_label_map[val_labels[i]]

print("Validation Confusion Matrix")
val_mat = pd.DataFrame(tf.math.confusion_matrix(val_l,val_pr).numpy())
print(val_mat)
'''