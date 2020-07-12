import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import get_file
import sys
from datetime import datetime

arg_names = ['filename','model', 'epochs','batch']
args = dict(zip(arg_names, sys.argv))


if args.get('epochs') is not None:
    epochs = int(args['epochs'])
else:
    epochs = 20

if(args.get('batch') is not None):
    batch = int(args['batch'])
else:
    batch = 128

if(args.get('model') is not None):
    model_type=args['model']
else:
    model_type = 'Recurrent'

#Loading data
train_df = pd.read_csv("fashion_product_train.csv")
val_df = pd.read_csv("fashion_product_validation.csv")
test_df = pd.read_csv("fashion_product_test.csv")


if(model_type=='Recurrent' or model_type=='BCNN'):

    lblmapsub = {'Bags': 0, 'Belts': 1, 'Bottomwear': 2, 'Dress': 3, 'Eyewear': 4, 'Flip Flops': 5, 'Fragrance': 6, 'Headwear': 7, 'Innerwear': 8, 'Jewellery': 9, 'Lips': 10, 'Loungewear and Nightwear': 11, 'Nails': 12, 'Sandal': 13, 'Saree': 14, 'Shoes': 15, 'Socks': 16, 'Ties': 17, 'Topwear': 18, 'Wallets': 19, 'Watches': 20}
    lblmaparticle = {'Backpacks': 0, 'Belts': 1, 'Bra': 2, 'Briefs': 3, 'Capris': 4, 'Caps': 5, 'Casual Shoes': 6, 'Clutches': 7, 'Deodorant': 8, 'Dresses': 9, 'Earrings': 10, 'Flats': 11, 'Flip Flops': 12, 'Formal Shoes': 13, 'Handbags': 14, 'Heels': 15, 'Innerwear Vests': 16, 'Jackets': 17, 'Jeans': 18, 'Kurtas': 19, 'Kurtis': 20, 'Leggings': 21, 'Lipstick': 22, 'Nail Polish': 23, 'Necklace and Chains': 24, 'Nightdress': 25, 'Pendant': 26, 'Perfume and Body Mist': 27, 'Sandals': 28, 'Sarees': 29, 'Shirts': 30, 'Shorts': 31, 'Socks': 32, 'Sports Shoes': 33, 'Sunglasses': 34, 'Sweaters': 35, 'Sweatshirts': 36, 'Ties': 37, 'Tops': 38, 'Track Pants': 39, 'Trousers': 40, 'Tshirts': 41, 'Tunics': 42, 'Wallets': 43, 'Watches': 44}
    lblmapmaster = {'Accessories': 0, 'Apparel': 1, 'Footwear': 2, 'Personal Care': 3}

    #Map classes
    train_df['masterCategory'].replace(lblmapmaster,inplace=True)
    test_df['masterCategory'].replace(lblmapmaster,inplace=True)
    val_df['masterCategory'].replace(lblmapmaster,inplace=True)

    train_df['subCategory'].replace(lblmapsub,inplace=True)
    test_df['subCategory'].replace(lblmapsub,inplace=True)
    val_df['subCategory'].replace(lblmapsub,inplace=True)

    train_df['articleType'].replace(lblmaparticle,inplace=True)
    test_df['articleType'].replace(lblmaparticle,inplace=True)
    val_df['articleType'].replace(lblmaparticle,inplace=True)

    #Convert the 3 labels to one hots in train, test, val
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

#----------get VGG16 pre-trained weights--------
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                         WEIGHTS_PATH,
                         cache_subdir='../weights')


#----------globals---------
print(train_df.head())
direc = '../data/fashion-dataset/images/'
target_size=(224,224)
TODAY = str(datetime.date(datetime.now()))

#Do additional transformations to support BatchNorm, Featurewise center and scal so each feature roughly N(0,1)
#Try with and without rescale
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True)
val_datagen = ImageDataGenerator(rescale=1. / 255,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True)

def get_flow_from_dataframe(g, dataframe,image_shape=target_size,batch_size=batch):
    while True:
        x_1 = g.next()

        yield [x_1[0], x_1[1][0], x_1[1][1]], x_1[1]

def train_BCNN(label, model, cbks):
    model.load_weights(weights_path, by_name=True)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')
    try:
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = val_generator.n // val_generator.batch_size
        history = model.fit_generator(train_generator,
                            epochs=epochs,
                            validation_data=val_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_steps=STEP_SIZE_VALID,
                            callbacks=cbks)
        print("Finished training")
        #Save training as csv
        pd.DataFrame.from_dict(history.history).to_csv("../history/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'.csv',index=False)
    
        # plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        plt.savefig("../plots/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'_loss.png', bbox_inches='tight')

    except ValueError as v:
        print(v)

    # Saving the weights in the current directory
    model.save_weights("../weights/"+label+"_"+str(epochs)+"_epochs_"+TODAY+".h5")


def train_recurrent(label, model,cbks):
    model.load_weights(weights_path, by_name=True)
    train = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')
    val = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')

    train_generator = get_flow_from_dataframe(train,dataframe=train_df,image_shape=target_size,batch_size=batch)
    val_generator = get_flow_from_dataframe(val,dataframe=val_df,image_shape=target_size,batch_size=batch)
    try:
        STEP_SIZE_TRAIN = train.n // train.batch_size
        STEP_SIZE_VALID = val.n // val.batch_size
        history = model.fit_generator(train_generator,
                            epochs=epochs,
                            validation_data=val_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_steps=STEP_SIZE_VALID,
                            callbacks=cbks)
        print("Finished training")
        #Save training as csv
        pd.DataFrame.from_dict(history.history).to_csv("../history/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'.csv',index=False)

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
        plt.savefig("../plots/"+label+"_"+str(epochs)+"_epochs_"+TODAY+"_loss.png", bbox_inches='tight')
    except ValueError as v:
        print(v)

    # Saving the weights in the current directory
    model.save_weights("../weights/"+label+"_"+str(epochs)+"_epochs_"+TODAY+".h5")     
#def BCNN_train():

def train_baseline(label, model,cbks):
    model.load_weights(weights_path, by_name=True)
    '''label is masterCategory, subCategory, or, articleType'''
    y = label
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=direc,
        x_col="filepath",
        y_col=y,
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        x_col="filepath",
        y_col=y,
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')
    try:
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = val_generator.n // val_generator.batch_size
        history = model.fit_generator(train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            epochs=epochs,
                            validation_data=val_generator,
                            validation_steps=STEP_SIZE_VALID,
                            callbacks=cbks)
        print("Finished training")
        #Save training as csv
        pd.DataFrame.from_dict(history.history).to_csv("../history/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'.csv',index=False)
    
        # plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        plt.savefig("../plots/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'_loss.png', bbox_inches='tight')

    except ValueError as v:
        print(v)

    # Saving the weights in the current directory
    model.save_weights("../weights/"+label+"_"+str(epochs)+"_epochs_"+TODAY+".h5")


if(model_type == 'Recurrent'):
    from RecurrentBranching import RecurrentTrain
    recurrent = RecurrentTrain(model_type)
    model = recurrent.model
    cbks = recurrent.cbks
    train_recurrent(model_type, model, cbks)
elif(model_type=='BCNN'):
    from BCNN import BCNN
    bcnn = BCNN(model_type)
    model = bcnn.model
    cbks = bcnn.cbks
    train_BCNN(model_type, model, cbks)
elif(model_type == 'articleType'):
    from articleType import ArticleType
    articletype = ArticleType(model_type)
    model = articletype.model
    cbks = articletype.cbks
    train_baseline(model_type, model, cbks)
elif(model_type == 'subCategory'):
    from subCategory import SubCategory
    subcategory = SubCategory(model_type)
    model = subcategory.model
    cbks = subcategory.cbks
    train_baseline(model_type,model,cbks)
else:
    #masterCategory
    from masterCategory import MasterCategory
    mastercategory = MasterCategory(model_type)
    model = mastercategory.model
    cbks = mastercategory.cbks
    train_baseline(model_type, model,cbks)
    

