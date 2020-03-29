import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Flatten, Input, Conv2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import sys


# print(tf.test.is_gpu_available())

# keras verison is 2.2.4-tf



class RecurrentTrain:
    '''One parameter model which is a keras model'''

    def __init__(self,typ):

        #final_input = Input(shape=(224, 224, 3))

        '''Three inputs to model for training, image, labels ,labels for teacher forcing'''
        input_image = Input(shape=(224,224,3))

        #4 element vector
        input_master = Input(shape=(4))

        #21 element vector
        input_sub = Input(shape=(21))

        # Download the architecture of ResNet50 with ImageNet weights
        if (typ == 'vgg'):
            base_model = VGG16(include_top=False, weights='imagenet',input_shape= (224,224,3))
        elif (typ == 'inception'):
            base_model = InceptionV3(include_top=False, weights=None)
        elif (typ == 'resnet'):
            base_model = ResNet50(include_top=False, weights=None)

        #rip out last couple layers of VGG
        last_maxpoolconfig = base_model.layers[-1].get_config()
        last_maxpoolconfig['name']='maxpoolA'
        print(last_maxpoolconfig)
        last_convconfig = base_model.layers[-2].get_config()
        last_convconfig['name'] = 'last_convA'
        seclast_convconfig = base_model.layers[-3].get_config()
        seclast_convconfig['name']='seclast_convA'

        maxpoolA = MaxPooling2D.from_config(last_maxpoolconfig)
        seclast_convA = Conv2D.from_config(seclast_convconfig)
        last_convA = Conv2D.from_config(last_convconfig)

        last_maxpoolconfig['name'] = 'maxpoolB'
        last_convconfig['name'] = 'last_convB'
        seclast_convconfig['name'] = 'seclast_convB'


        maxpoolB = MaxPooling2D.from_config(last_maxpoolconfig)
        seclast_convB = Conv2D.from_config(seclast_convconfig)
        last_convB = Conv2D.from_config(last_convconfig)

        last_maxpoolconfig['name'] = 'maxpoolC'
        last_convconfig['name'] = 'last_convC'
        seclast_convconfig['name'] = 'seclast_convC'

        maxpoolC = MaxPooling2D.from_config(last_maxpoolconfig)
        seclast_convC = Conv2D.from_config(seclast_convconfig)
        last_convC = Conv2D.from_config(last_convconfig)

        #freeze majority of the base conv network
        if (typ == 'vgg'):
            for layer in base_model.layers[:-3]:
                layer.trainable = False

        for layer in base_model.layers[:-4:-1]:
            base_model.layers.pop()

        out = base_model(input_image)
        x = seclast_convA(out)
        x = last_convA(x)
        x = maxpoolA(x)

        # Adding a Global Average Pooling layer
        x = GlobalAveragePooling2D()(x)

        x = Dense(1024, activation='relu')(x)
        x = Dense(4, activation='softmax',name="master_output")(x)


        # Model to be trained
        #modelA = Model(inputs=input_image, outputs=x)
        #print(modelA.summary())

        y = seclast_convB(out)
        y = last_convB(y)
        y = maxpoolB(y)

        # Adding a Global Average Pooling layer
        y = GlobalAveragePooling2D()(y)

        y = Dense(1024,activation='relu')(y)
        #combinedB = concatenate([y, modelA.output])
        combinedB = concatenate([y,input_master])

        y = Dense(21, activation='softmax',name="sub_output")(combinedB)

        # Model to be trained
        #modelB = Model(inputs=[input_image,input_master], outputs=y)

        z = seclast_convC(out)
        z = last_convC(z)
        z = maxpoolC(z)

        # Adding a Global Average Pooling layer
        z = GlobalAveragePooling2D()(z)

        z = Dense(1024, activation='relu')(z)
        #combinedC = concatenate([z, modelB.output])

        combinedC = concatenate([z, input_sub, input_master])
        z = Dense(45, activation='softmax',name="article_output")(combinedC)

        # Model to be trained
        #modelC = Model(inputs=[input_image,input_master,input_sub], outputs=z)

            # trainable_params = tf.keras.backend.count_params(model.trainable_weights)
            # print("Trainable paramaters: "+str(trainable_params))

        # print(model.summary())
        # Compiling the model
        # KEras will automatically use categorical accuracy when accuracy is used.

        model = Model(
            inputs=[input_image,input_master,input_sub],
            outputs=[x, y, z])

        losses = {
            "master_output": "categorical_crossentropy",
            "sub_output": "categorical_crossentropy",
            "article_output": "categorical_crossentropy"
        }
        model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss=losses,
                      metrics=['categorical_accuracy'])

        self.model = model



class RecurrentTest:
    '''One parameter model which is a keras model'''

    def __init__(self,typ):

        #final_input = Input(shape=(224, 224, 3))

        '''Three inputs to model for training, image, labels ,labels for teacher forcing'''
        #no longer use the label inputs just the image.
        input_image = Input(shape=(224,224,3))

        #4 element vector, not used
        input_master = Input(shape=(4))

        #21 element vector, not used
        input_sub = Input(shape=(21))

        # Download the architecture of ResNet50 with ImageNet weights
        if (typ == 'vgg'):
            base_model = VGG16(include_top=False, weights='imagenet',input_shape= (224,224,3))
        elif (typ == 'inception'):
            base_model = InceptionV3(include_top=False, weights=None)
        elif (typ == 'resnet'):
            base_model = ResNet50(include_top=False, weights=None)

        #rip out last couple layers of VGG
        last_maxpoolconfig = base_model.layers[-1].get_config()
        last_maxpoolconfig['name']='maxpoolA'
        print(last_maxpoolconfig)
        last_convconfig = base_model.layers[-2].get_config()
        last_convconfig['name'] = 'last_convA'
        seclast_convconfig = base_model.layers[-3].get_config()
        seclast_convconfig['name']='seclast_convA'

        maxpoolA = MaxPooling2D.from_config(last_maxpoolconfig)
        seclast_convA = Conv2D.from_config(seclast_convconfig)
        last_convA = Conv2D.from_config(last_convconfig)

        last_maxpoolconfig['name'] = 'maxpoolB'
        last_convconfig['name'] = 'last_convB'
        seclast_convconfig['name'] = 'seclast_convB'


        maxpoolB = MaxPooling2D.from_config(last_maxpoolconfig)
        seclast_convB = Conv2D.from_config(seclast_convconfig)
        last_convB = Conv2D.from_config(last_convconfig)

        last_maxpoolconfig['name'] = 'maxpoolC'
        last_convconfig['name'] = 'last_convC'
        seclast_convconfig['name'] = 'seclast_convC'

        maxpoolC = MaxPooling2D.from_config(last_maxpoolconfig)
        seclast_convC = Conv2D.from_config(seclast_convconfig)
        last_convC = Conv2D.from_config(last_convconfig)

        #freeze majority of the base conv network
        if (typ == 'vgg'):
            for layer in base_model.layers[:-3]:
                layer.trainable = False

        for layer in base_model.layers[:-4:-1]:
            base_model.layers.pop()

        out = base_model(input_image)
        x = seclast_convA(out)
        x = last_convA(x)
        x = maxpoolA(x)

        # Adding a Global Average Pooling layer
        x = GlobalAveragePooling2D()(x)

        x = Dense(1024, activation='relu')(x)
        x = Dense(4, activation='softmax',name="master_output")(x)


        # Model to be trained
        #modelA = Model(inputs=input_image, outputs=x)
        #print(modelA.summary())

        y = seclast_convB(out)
        y = last_convB(y)
        y = maxpoolB(y)

        # Adding a Global Average Pooling layer
        y = GlobalAveragePooling2D()(y)

        y = Dense(1024,activation='relu')(y)
        #use predictions for masterCategory
        combinedB = concatenate([y,x])

        y = Dense(21, activation='softmax',name="sub_output")(combinedB)

        # Model to be trained
        #modelB = Model(inputs=[input_image,input_master], outputs=y)

        z = seclast_convC(out)
        z = last_convC(z)
        z = maxpoolC(z)

        # Adding a Global Average Pooling layer
        z = GlobalAveragePooling2D()(z)

        z = Dense(1024, activation='relu')(z)
        #Use the predictions of the subCategory and masterCategory to inform decision.
        tf.keras.backend.print_tensor(x)
        tf.keras.backend.print_tensor(y)
        print(tf.keras.backend.shape(x))
        combinedC = concatenate([z, y, x])
        z = Dense(45, activation='softmax',name="article_output")(combinedC)

        # Model to be trained
        #modelC = Model(inputs=[input_image,input_master,input_sub], outputs=z)

            # trainable_params = tf.keras.backend.count_params(model.trainable_weights)
            # print("Trainable paramaters: "+str(trainable_params))

        # print(model.summary())
        # Compiling the model
        # KEras will automatically use categorical accuracy when accuracy is used.

        model = Model(
            inputs=[input_image,input_master,input_sub],
            outputs=[x, y, z])

        losses = {
            "master_output": "categorical_crossentropy",
            "sub_output": "categorical_crossentropy",
            "article_output": "categorical_crossentropy"
        }
        model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss=losses,
                      metrics=['categorical_accuracy'])

        self.model = model

if __name__ == "__main__":

    typ = sys.argv[1]

    model = RecurrentTrain(typ).model
    print(model.summary())