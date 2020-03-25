# Importing the important libraries
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import sys


# print(tf.test.is_gpu_available())

# keras verison is 2.2.4-tf



class ArticleType:
    '''One parameter model which is a keras model'''

    def __init__(self,typ):

        # Download the architecture of ResNet50 with ImageNet weights
        if (typ == 'vgg'):
            base_model = VGG16(include_top=False, weights='imagenet')
        elif (typ == 'inception'):
            base_model = InceptionV3(include_top=False, weights=None)
        elif (typ == 'resnet'):
            base_model = ResNet50(include_top=False, weights=None)

        #base_model = ResNet50(include_top=False, weights=None)
        # Taking the output of the last convolution block in ResNet50
        x = base_model.output


        # Adding a Global Average Pooling layer
        x = GlobalAveragePooling2D()(x)
        #x = Flatten()(x)

        #print(x.shape)
        #print(x.output_shape)
        # Adding a fully connected layer having 1024 neurons
        #x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        # Adding a fully connected layer having 45 neurons which will
        # 1 for each class of subcategory
        # only 44 classes that work in the dataset.
        # 22 classes with edited dataset
        # 21
        predictions = Dense(45, activation='softmax')(x)

        # Model to be trained
        model = Model(inputs=base_model.input, outputs=predictions)
        #print(model.summary())

        if (typ == 'vgg'):
            for layer in base_model.layers[:-3]:
                layer.trainable = False

            # trainable_params = tf.keras.backend.count_params(model.trainable_weights)
            # print("Trainable paramaters: "+str(trainable_params))

        print("Layers: " + str(len(base_model.layers)))
        # print(model.summary())
        # Compiling the model
        # KEras will automaticall use categorical accuracy when accuracy is used.
        model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        self.model = model

if __name__ == "__main__":

    typ = sys.argv[1]

    model = ArticleType(typ).model
    print(model.summary())