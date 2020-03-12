
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
arg_names = ['filename','model']
args = dict(zip(arg_names, sys.argv))

if(args.get('model') is not None):
    modelT=args['model']
else:
    modelT = 'master'



print("Testing model " + modelT)


if(modelT=='master'):
    from MasterCategoryModel import Master
    model = Master().model

elif(modelT=='sub'):
    from SubCategoryModel import SubCategory
    model = SubCategory().model
elif(modelT=='article'):
    #model is articleType
    from ArticleTypeModel import ArticleType
    model = ArticleType().model

if modelT=='master':
    model.load_weights("Fashion_pretrain_resnet50_MasterCategory.h5")
elif modelT=='sub':
    model.load_weights("Fashion_pretrain_resnet50_SubCategory.h5")
elif modelT=='article':
    model.load_weights("Fashion_pretrain_resnet50_ArticleType.h5")
#load the weights


# Compiling the model
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])




test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=direc,
        x_col="filepath",
        y_col="subCategory",
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical')


print("Test generator n",test_generator.n)
print("Test geenrator batch size",test_generator.batch_size)
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
print(model.evaluate_generator(generator=test_generator,
        steps=STEP_SIZE_TEST))