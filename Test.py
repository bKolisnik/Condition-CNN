import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import get_file
import sys
from datetime import datetime
from os import path
import tensorflow.keras.backend as K

arg_names = ['filename','model','filepath', 'batch']
args = dict(zip(arg_names, sys.argv))

#positional arguments required for test
model_type=args['model']
batch = int(args['batch'])
#batch = 128
weights_path = args['filepath']

test_df = pd.read_csv("fashion_product_test.csv")

#check if results csv exists. if it does not then create it. append each result test run as a new label 
#row contains the label (BCNN, Recurrent etc), the weights path, the three test accuracies (or 1 if it is a baseline cnn model)
# and the timestamp when it was tested
exists = path.exists("../testing/test_results.csv")
if(not exists):
    dtypes = np.dtype([
          ('Model', str),
          ('Weights Path', str),
          ('masterCategory Accuracy %', np.float64),
          ('subCategory Accuracy %', np.float64),
          ('articleType Accuracy %', np.float64),
          ('Trainable params', np.float64),
          ('Timestamp', np.datetime64),
          ])
    #df = pd.DataFrame(columns=['Model','Weights Path', 'masterCategory Accuracy %','subCategory Accuracy %','articleType Accuracy %','Timestamp'])
    data = np.empty(0, dtype=dtypes)
    df = pd.DataFrame(data)
else:
    types = {'Model':str,'Weights Path': str, 'masterCategory Accuracy %': np.float64, 'subCategory Accuracy %': np.float64,
    'articleType Accuracy %': np.float64, 'Trainable params': np.float64}
    df = pd.read_csv("../testing/test_results.csv",dtype=types, parse_dates=['Timestamp'])



if(model_type=='Recurrent' or model_type=='BCNN' or model_type=='Condition'or model_type=='ConditionPlus'):

    lblmapsub = {'Bags': 0, 'Belts': 1, 'Bottomwear': 2, 'Dress': 3, 'Eyewear': 4, 'Flip Flops': 5, 'Fragrance': 6, 'Headwear': 7, 'Innerwear': 8, 'Jewellery': 9, 'Lips': 10, 'Loungewear and Nightwear': 11, 'Nails': 12, 'Sandal': 13, 'Saree': 14, 'Shoes': 15, 'Socks': 16, 'Ties': 17, 'Topwear': 18, 'Wallets': 19, 'Watches': 20}
    lblmaparticle = {'Backpacks': 0, 'Belts': 1, 'Bra': 2, 'Briefs': 3, 'Capris': 4, 'Caps': 5, 'Casual Shoes': 6, 'Clutches': 7, 'Deodorant': 8, 'Dresses': 9, 'Earrings': 10, 'Flats': 11, 'Flip Flops': 12, 'Formal Shoes': 13, 'Handbags': 14, 'Heels': 15, 'Innerwear Vests': 16, 'Jackets': 17, 'Jeans': 18, 'Kurtas': 19, 'Kurtis': 20, 'Leggings': 21, 'Lipstick': 22, 'Nail Polish': 23, 'Necklace and Chains': 24, 'Nightdress': 25, 'Pendant': 26, 'Perfume and Body Mist': 27, 'Sandals': 28, 'Sarees': 29, 'Shirts': 30, 'Shorts': 31, 'Socks': 32, 'Sports Shoes': 33, 'Sunglasses': 34, 'Sweaters': 35, 'Sweatshirts': 36, 'Ties': 37, 'Tops': 38, 'Track Pants': 39, 'Trousers': 40, 'Tshirts': 41, 'Tunics': 42, 'Wallets': 43, 'Watches': 44}
    lblmapmaster = {'Accessories': 0, 'Apparel': 1, 'Footwear': 2, 'Personal Care': 3}

    #Map classes
    test_df['masterCategory'].replace(lblmapmaster,inplace=True)
    test_df['subCategory'].replace(lblmapsub,inplace=True)
    test_df['articleType'].replace(lblmaparticle,inplace=True)

    #Convert the 3 labels to one hots in train, test, val
    onehot_master = to_categorical(test_df['masterCategory'].values)
    test_df['masterCategoryOneHot'] = onehot_master.tolist()

    onehot_master = to_categorical(test_df['subCategory'].values)
    test_df['subCategoryOneHot'] = onehot_master.tolist()

    onehot_master = to_categorical(test_df['articleType'].values)
    test_df['articleTypeOneHot'] = onehot_master.tolist()

#----------globals---------
direc = '../data/fashion-dataset/images/'
target_size=(224,224)
TODAY = str(datetime.date(datetime.now()))

test_datagen = ImageDataGenerator(rescale=1. / 255,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True)

#----------test methods---------
def test_multi(label, model):
    model.load_weights(weights_path, by_name=True)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')

    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    #x can be a generator returning retunring (inputs, targets)
    #if x is a generator y should not be specified
    score = model.evaluate(x=test_generator,
        steps=STEP_SIZE_TEST)

    print(f"score is {score}")
    return score

def test_articleType(label, model):
    model.load_weights(weights_path, by_name=True)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=direc,
        x_col="filepath",
        y_col='articleType',
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    score = model.evaluate(x=test_generator,
        steps=STEP_SIZE_TEST)

    print(f"score is {score}")
    return score

def test_subCategory(label, model):
    model.load_weights(weights_path, by_name=True)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=direc,
        x_col="filepath",
        y_col='subCategory',
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    score = model.evaluate(x=test_generator,
        steps=STEP_SIZE_TEST)

    print(f"score is {score}")
    return score

def test_masterCategory(label, model):
    model.load_weights(weights_path, by_name=True)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=direc,
        x_col="filepath",
        y_col='masterCategory',
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    score = model.evaluate(x=test_generator,
        steps=STEP_SIZE_TEST)

    print(f"score is {score}")
    return score


score = 0
params = 0
masterCategory_accuracy = np.nan
subCategory_accuracy = np.nan
articleType_accuracy = np.nan

if(model_type == 'Recurrent'):
    from RecurrentBranching import RecurrentTest
    model = RecurrentTest(model_type).model
    score = test_multi(model_type, model)
    params= np.sum([K.count_params(w) for w in model.trainable_weights])
    masterCategory_accuracy = score[4]
    subCategory_accuracy = score[5]
    articleType_accuracy = score[6]
elif(model_type == 'Condition'):
    from ConditionCNN import ConditionTest
    model = ConditionTest(model_type).model
    score = test_multi(model_type, model)
    params= np.sum([K.count_params(w) for w in model.trainable_weights])
    masterCategory_accuracy = score[4]
    subCategory_accuracy = score[5]
    articleType_accuracy = score[6]
elif(model_type == 'ConditionPlus'):
    from ConditionCNNPlus import ConditionPlusTest
    model = ConditionPlusTest(model_type).model
    score = test_multi(model_type, model)
    params= np.sum([K.count_params(w) for w in model.trainable_weights])
    masterCategory_accuracy = score[4]
    subCategory_accuracy = score[5]
    articleType_accuracy = score[6]
elif(model_type=='BCNN'):
    from BCNN import BCNN
    bcnn = BCNN(model_type)
    model = bcnn.model
    score = test_multi(model_type, model)
    params= np.sum([K.count_params(w) for w in model.trainable_weights])

    #score [0:4] are the losses for the branches
    masterCategory_accuracy = score[4]
    subCategory_accuracy = score[5]
    articleType_accuracy = score[6]

elif(model_type == 'articleType'):
    from articleType import ArticleType
    model = ArticleType(model_type).model
    score = test_articleType(model_type, model)
    params= np.sum([K.count_params(w) for w in model.trainable_weights])
    articleType_accuracy = score[1]

elif(model_type == 'subCategory'):
    from subCategory import SubCategory
    model = SubCategory(model_type).model
    score = test_subCategory(model_type,model)
    params= np.sum([K.count_params(w) for w in model.trainable_weights])
    subCategory_accuracy = score[1]
else:
    #masterCategory
    from masterCategory import MasterCategory
    model = MasterCategory(model_type).model
    score = test_masterCategory(model_type, model)
    params= np.sum([K.count_params(w) for w in model.trainable_weights])
    masterCategory_accuracy = score[1]

    
df.loc[df.index.max()+1] = [model_type, weights_path, masterCategory_accuracy, subCategory_accuracy, articleType_accuracy, params,np.datetime64('now')]
df.to_csv("../testing/test_results.csv", index=False)