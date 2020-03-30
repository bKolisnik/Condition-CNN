from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.utils import to_categorical

import json

val_df = pd.read_csv("fashion_product_validation_full.csv")
test_df = pd.read_csv("fashion_product_test_full.csv")


lblmapsub = {'Bags': 0, 'Belts': 1, 'Bottomwear': 2, 'Dress': 3, 'Eyewear': 4, 'Flip Flops': 5, 'Fragrance': 6, 'Headwear': 7, 'Innerwear': 8, 'Jewellery': 9, 'Lips': 10, 'Loungewear and Nightwear': 11, 'Nails': 12, 'Sandal': 13, 'Saree': 14, 'Shoes': 15, 'Socks': 16, 'Ties': 17, 'Topwear': 18, 'Wallets': 19, 'Watches': 20}

lblmaparticle = {'Backpacks': 0, 'Belts': 1, 'Bra': 2, 'Briefs': 3, 'Capris': 4, 'Caps': 5, 'Casual Shoes': 6, 'Clutches': 7, 'Deodorant': 8, 'Dresses': 9, 'Earrings': 10, 'Flats': 11, 'Flip Flops': 12, 'Formal Shoes': 13, 'Handbags': 14, 'Heels': 15, 'Innerwear Vests': 16, 'Jackets': 17, 'Jeans': 18, 'Kurtas': 19, 'Kurtis': 20, 'Leggings': 21, 'Lipstick': 22, 'Nail Polish': 23, 'Necklace and Chains': 24, 'Nightdress': 25, 'Pendant': 26, 'Perfume and Body Mist': 27, 'Sandals': 28, 'Sarees': 29, 'Shirts': 30, 'Shorts': 31, 'Socks': 32, 'Sports Shoes': 33, 'Sunglasses': 34, 'Sweaters': 35, 'Sweatshirts': 36, 'Ties': 37, 'Tops': 38, 'Track Pants': 39, 'Trousers': 40, 'Tshirts': 41, 'Tunics': 42, 'Wallets': 43, 'Watches': 44}

lblmapmaster = {'Accessories': 0, 'Apparel': 1, 'Footwear': 2, 'Personal Care': 3}


test_df['masterCategory'].replace(lblmapmaster,inplace=True)
val_df['masterCategory'].replace(lblmapmaster,inplace=True)


test_df['subCategory'].replace(lblmapsub,inplace=True)
val_df['subCategory'].replace(lblmapsub,inplace=True)


test_df['articleType'].replace(lblmaparticle,inplace=True)
val_df['articleType'].replace(lblmaparticle,inplace=True)

#to_categorical converts class vector to binary matrix.


onehot_master = to_categorical(val_df['masterCategory'].values)
val_df['masterCategoryOneHot'] = onehot_master.tolist()
onehot_master = to_categorical(test_df['masterCategory'].values)
test_df['masterCategoryOneHot'] = onehot_master.tolist()


onehot_master = to_categorical(val_df['subCategory'].values)
val_df['subCategoryOneHot'] = onehot_master.tolist()
onehot_master = to_categorical(test_df['subCategory'].values)
test_df['subCategoryOneHot'] = onehot_master.tolist()


onehot_master = to_categorical(val_df['articleType'].values)
val_df['articleTypeOneHot'] = onehot_master.tolist()
onehot_master = to_categorical(test_df['articleType'].values)
test_df['articleTypeOneHot'] = onehot_master.tolist()

direc = '../data/fashion-dataset/images/'
arg_names = ['filename','typ','batch']
args = dict(zip(arg_names, sys.argv))


if(args.get('batch') is not None):
    batch = int(args['batch'])
else:
    batch = 64

if(args.get('typ') is not None):
    typ=args['typ']
else:
    typ = 'vgg'
print("Batch size " + str(batch))
print("Type is "+typ)

from RecurrentModelFull import RecurrentTest
from RecurrentModelFull import RecurrentTrain
#model = RecurrentTest(typ).model

model2 = RecurrentTrain(typ).model

model = RecurrentTest(typ).model

#print(model.summary())
#print(model2.summary())

'''
print("model 1 weights")
for layer in model.get_weights():
    print(layer.shape)
'''
#print(model.get_weights().shape)
#print(model2.get_weights().shape)


if(typ=='vgg'):
    target_size=(224,224)
elif(typ=='inception'):
    target_size = (299, 299)
elif(typ=='resnet'):
    #resnet
    target_size = (224, 224)





model2.load_weights("Fashion_pretrain_recurrent_"+typ+".h5")
#load the weights into model 2 then change the ordering of the weights

'''
print(model.summary())

names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

for name, weight in zip(names, weights):
    print(name, weight.shape)
'''
print(model2.summary())
names = [weight.name for layer in model2.layers for weight in layer.weights]
weights = model2.get_weights()

for name, weight in zip(names, weights):
    print(name, weight.shape)


weights = model2.get_weights()



#swap block 1 and 3
temp1 = weights[-24]
temp2 = weights[-23]
weights[-24] = weights[-20]
weights[-23] = weights[-19]
weights[-20] = temp1
weights[-19] = temp2

#put block 2 in storage, put block 6 into block 2
temp1 = weights[-22]
temp2 = weights[-21]
weights[-22] = weights[-14]
weights[-21] = weights[-13]

#put block 9 int oblok 6
weights[-14] = weights[-8]
weights[-13] = weights[-7]

#put block 7 into block 9
weights[-8] = weights[-12]
weights[-7] = weights[-11]

#put block 5 into block 7
weights[-12] = weights[-16]
weights[-11] = weights[-15]

#put temp stored block 2 into block 5
weights[-16] = temp1
weights[-15] = temp2


#put block 8 into temp storage
temp1 = weights[-10]
temp2 = weights[-9]

#put block 10 into block 8
weights[-10] = weights[-6]
weights[-9] = weights[-5]

#put block 11 into block 10
weights[-6] = weights[-4]
weights[-5] = weights[-3]

#put block 2 from tmep storage into block 11
weights[-4] = temp1
weights[-3] = temp2


model.set_weights(weights)

print(model.summary())




test_datagen = ImageDataGenerator(rescale=1. / 255)


val = test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')


test = test_datagen.flow_from_dataframe(
        dataframe=test_df,
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

        #print(y.shape)
        yield [x_1[0], x_1[1][0], x_1[1][1]], x_1[1]


val_generator = get_flow_from_dataframe(val,dataframe=val_df,image_shape=target_size,batch_size=batch)
test_generator = get_flow_from_dataframe(test,dataframe=test_df,image_shape=target_size,batch_size=batch)


print("Test generator n",test.n)
print("Test generator batch size",test.batch_size)
STEP_SIZE_TEST=test.n//test.batch_size
print(model.evaluate(x=test_generator,
        steps=STEP_SIZE_TEST))

print("Validation Generator n",val.n)
print("Test generator batch size",val.batch_size)
STEP_SIZE_VAL=val.n//val.batch_size
print(model.evaluate(x=val_generator,
        steps=STEP_SIZE_VAL,verbose=1))
