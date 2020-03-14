
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
arg_names = ['filename','model','batch']
args = dict(zip(arg_names, sys.argv))

if(args.get('model') is not None):
    modelT=args['model']
else:
    modelT = 'masterCategory'

if(args.get('batch') is not None):
    batch = int(args['batch'])
else:
    batch = 64

print("Testing model " + modelT)
print("Batch size " + str(batch))

if(modelT=='masterCategory'):
    from MasterCategoryModel import Master
    model = Master().model

elif(modelT=='subCategory'):
    from SubCategoryModel import SubCategory
    model = SubCategory().model
elif(modelT=='articleType'):
    #model is articleType
    from ArticleTypeModel import ArticleType
    model = ArticleType().model

if modelT=='masterCategory':
    model.load_weights("Fashion_pretrain_resnet50_MasterCategory.h5")
elif modelT=='subCategory':
    model.load_weights("Fashion_pretrain_resnet50_SubCategory.h5")
elif modelT=='articleType':
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
        batch_size=batch,
        class_mode='categorical')


print("Test generator n",test_generator.n)
print("Test generator batch size",test_generator.batch_size)
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
print(model.evaluate_generator(generator=test_generator,
        steps=STEP_SIZE_TEST))