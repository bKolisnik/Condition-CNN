# Importing the important libraries
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


# print(tf.test.is_gpu_available())

# keras verison is 2.2.4-tf



class Master:
    '''One parameter model which is a keras model'''

    def __init__(self):
        # Download the architecture of ResNet50 with ImageNet weights
        base_model = ResNet50(include_top=False, weights='imagenet')

        # Taking the output of the last convolution block in ResNet50
        x = base_model.output

        # Adding a Global Average Pooling layer
        x = GlobalAveragePooling2D()(x)

        # Adding a fully connected layer having 1024 neurons
        x = Dense(1024, activation='relu')(x)

        # Adding a fully connected layer having 45 neurons which will
        # 1 for each class of subcategory
        # only 44 classes that work in the dataset.
        # 22 classes with edited dataset
        # 21
        predictions = Dense(21, activation='softmax')(x)

        # Model to be trained
        model = Model(inputs=base_model.input, outputs=predictions)
        #print(model.summary())

        # Training only top layers i.e. the layers which we have added in the end

        for layer in base_model.layers:
            layer.trainable = False

        # Compiling the model
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model