from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)




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

print("Testing model " + modelT)
print("Batch size " + str(batch))
print("Type is "+typ)

from RecurrentModelFull import RecurrentTest
model = RecurrentTest(typ).model

if(typ=='vgg'):
    target_size=(224,224)
elif(typ=='inception'):
    target_size = (299, 299)
elif(typ=='resnet'):
    #resnet
    target_size = (224, 224)


model.load_weights("Fashion_pretrain_recurrent_"+typ+".h5")
#load the weights



test_datagen = ImageDataGenerator(rescale=1. / 255)


val = test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')


test_generator = test_datagen.flow_from_dataframe(
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
test_generator = get_flow_from_dataframe(val,dataframe=test_df,image_shape=target_size,batch_size=batch)


print("Test generator n",test_generator.n)
print("Test generator batch size",test_generator.batch_size)
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
print(model.evaluate(x=test_generator,
        steps=STEP_SIZE_TEST))

print("Validation Generator n",val_generator.n)
print("Test generator batch size",val_generator.batch_size)
STEP_SIZE_VAL=val_generator.n//val_generator.batch_size
print(model.evaluate(x=val_generator,
        steps=STEP_SIZE_VAL,verbose=1))