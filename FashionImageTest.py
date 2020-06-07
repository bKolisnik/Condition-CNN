
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

direc = '../data/fashion-dataset/images/'
arg_names = ['filename','model','typ','batch','path']
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

if(args.get('path') is not None):
    path = args['path']
else:
    if modelT == 'masterCategory':
        path = "Fashion_pretrain_MasterCategory_" + typ + ".h5"
    elif modelT == 'subCategory':
        path = "Fashion_pretrain_SubCategory_" + typ + ".h5"
    else:
        #modelT == 'articleType'
        path = "Fashion_pretrain_ArticleType_" + typ + ".h5"

print("Testing model " + modelT)
print("Batch size " + str(batch))
print("Type is "+typ)
print("Path to weights is: " + path)

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

model.load_weights(path)
#load the weights


# Compiling the model
#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
test_df = pd.read_csv("fashion_product_test.csv")
val_df = pd.read_csv("fashion_product_validation.csv")


test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=direc,
        x_col="filepath",
        y_col=modelT,
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')

val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=direc,
    x_col="filepath",
    y_col=modelT,
    target_size=target_size,
    batch_size=batch,
    class_mode='categorical')


print("Test generator n",test_generator.n)
print("Test generator batch size",test_generator.batch_size)
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
print(model.evaluate(x=test_generator,
        steps=STEP_SIZE_TEST))

print("Validation Generator n",val_generator.n)
print("Validation generator batch size",val_generator.batch_size)
STEP_SIZE_VAL=val_generator.n//val_generator.batch_size
print(model.evaluate(x=val_generator,
        steps=STEP_SIZE_VAL,verbose=1))


#get model predictions into 1D tensor
#get true labels into 1D tensor




