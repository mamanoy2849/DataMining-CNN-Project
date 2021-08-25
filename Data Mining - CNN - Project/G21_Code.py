# -*- coding: utf-8 -*-
"""
@author: Jake Murkin
"""



#import relevant libraries
import pandas as pd
import cv2
import os
from glob import glob 
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from IPython.display import clear_output


# Load DataSet from folder
# Load csv document that indicates (with 0 and 1) which photo
# is positive or negative
# Set train path and test path

path = "D:/CNN - Cancer/" 
labels = pd.read_csv(path + 'train_labels.csv')
train_path = path + 'train/'
test_path = path + 'test/'


# Create a dataframe which contains every training examples path, id and label:
# without it's path

df = pd.DataFrame({'path': glob(os.path.join(train_path,'*.tif'))})
df['id'] = df.path.map(lambda x: ((x.split("n")[2].split('.')[0])[1:]))
df = df.merge(labels, on = "id")
df.head(3)



# Function that reads image by its path, using open-cv
def readImage(path):
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    return rgb_img

# Choose 4 random positve and negative examples, find their respective path then display them in a subplot:
# positive gets all labels that are true
# negative gets all labels that are false
positive_indices = list(np.where(df["label"] == True)[0])
negative_indices = list(np.where(df["label"] == False)[0])
rand_pos_inds = random.sample(positive_indices, 4)
rand_neg_inds = random.sample(negative_indices, 4)

fig, ax = plt.subplots(2,4, figsize=(20,8))
fig.suptitle('Histopathologic scans of lymph node sections',fontsize=20, fontweight='bold')

for i in range(0, 4):
    ax[0,i].imshow(readImage(df.iloc[rand_pos_inds[i],0]))
    ax[0,i].set_title("Positive Example", fontweight='bold')
    
    ax[1,i].imshow(readImage(df.iloc[rand_neg_inds[i],0]))
    ax[1,i].set_title("Negative Example", fontweight='bold')

# Increasing the size of the image results in a much higher performance.
# batch_size determines the number of samples to be loaded for computation
# and the epoch determines the number of times that you what Keras to pass 
# through all your data. In essence, if you set your epoch=2 and batch_size=32 it means
# that Keras will go through all your data twice with splitting your data
# in mini-batches with 32 samples of your data.

IMG_SIZE = 196
BATCH_SIZE = 32


test_list = os.listdir(test_path)
train_list = os.listdir(train_path)

print("There are " + str(len(train_list)) + " training examples.")
print("There are " + str(len(test_list)) + " test examples.")


# Going to split 20% of the training set into a validation set.
df['label'] = df['label'].astype(str)
train, valid = train_test_split(df, test_size=0.2, stratify = df['label'])



# Generate batches of tensor image data with real-time data augmentation.
# The data will be looped over (in batches).
# ImageDataGenerator accepts the original data, randomly transforms it, 
# and returns only the new, transformed data.

train_datagen = ImageDataGenerator(rescale=1./255,
                                  vertical_flip = True,
                                  horizontal_flip = True,
                                  rotation_range=90,
                                  zoom_range=0.2, 
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.05,
                                  channel_shift_range=0.1)

test_datagen = ImageDataGenerator(rescale = 1./255) 




train_generator = train_datagen.flow_from_dataframe(dataframe = train, 
                                                    directory = None,
                                                    x_col = 'path', 
                                                    y_col = 'label',
                                                    target_size = (IMG_SIZE,IMG_SIZE),
                                                    class_mode = "binary",
                                                    batch_size=BATCH_SIZE,
                                                    shuffle = True)



valid_generator = test_datagen.flow_from_dataframe(dataframe = valid,
                                                   directory = None,
                                                   x_col = 'path',
                                                   y_col = 'label',
                                                   target_size = (IMG_SIZE,IMG_SIZE),
                                                   class_mode = 'binary',
                                                   batch_size = BATCH_SIZE,
                                                   shuffle = False)

# Creating the model

from keras.applications.resnet50 import ResNet50

dropout_fc = 0.2

# Get conv from ResNet50
# con_base will conatin all of the layers and pooling that resnet50 has.
conv_base = ResNet50(weights = 'imagenet', include_top = False, input_shape = (IMG_SIZE,IMG_SIZE,3))


# conv_base is a call for ResNet50 function that includes all the layers:
  #   def ResNet50(include_top=True,
  #            weights='imagenet',
  #            input_tensor=None,
  #            input_shape=None,
  #            pooling=None,
  #            classes=1000,
  #            **kwargs):
  # """Instantiates the ResNet50 architecture."""

  # def stack_fn(x):
  #   x = stack1(x, 64, 3, stride1=1, name='conv2')
  #   x = stack1(x, 128, 4, name='conv3')
  #   x = stack1(x, 256, 6, name='conv4')
  #   return stack1(x, 512, 3, name='conv5')

  # return ResNet(stack_fn, False, True, 'resnet50', include_top, weights,
  #               input_tensor, input_shape, pooling, classes, **kwargs)
  
my_model = Sequential()
my_model.add(conv_base)
my_model.add(Flatten())
my_model.add(Dense(256, use_bias=False))
my_model.add(BatchNormalization())
my_model.add(Activation("relu"))
my_model.add(Dropout(dropout_fc))
my_model.add(Dense(1, activation = "sigmoid"))
my_model.summary()



# As we're using ResNet50 trained on ImageNet, 
# we're going to need to train the last few layers instead of the just the last one.
# Cell images are quite different to what you see on ImageNet.

conv_base.Trainable=True
set_trainable=False
for layer in conv_base.layers:
    if layer.name == 'res5a_branch2a':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
from keras import optimizers

my_model.compile(optimizers.Adam(0.001), loss = "binary_crossentropy", metrics = ["accuracy"])       
        


train_step_size = train_generator.n // train_generator.batch_size
valid_step_size = valid_generator.n // valid_generator.batch_size



earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=2, restore_best_weights=True)
reduce = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)



# steps_per_epoch determined the number of batches
# in each epoch than to visit all your samples in each epoch set the steps_per_epoch
# as follows.
history = my_model.fit_generator(train_generator,
                                     steps_per_epoch = train_step_size,
                                     epochs = 10,
                                     validation_data = valid_generator,
                                     validation_steps = valid_step_size,
                                     callbacks = [reduce, earlystopper],
                                     verbose = 1)


##save Model
my_model.save('Cancer_Prediction.h5')

# from keras.models import load_model
# model = load_model('Cancer_Prediction.h5')



# Analysis
# Now that our model has been trained, it is time to plot some training graphs 
# to see how our accuracies and losses varied over epochs.

epochs = [i for i in range(1, len(history.history['loss'])+1)]

fig1, ax1 = plt.subplots(1,1, figsize=(20,8))

plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")
plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")
plt.legend(loc='best')
plt.title('training')
plt.xlabel('epoch')
plt.savefig("training.png", bbox_inches='tight')
plt.show()

plt.plot(epochs, history.history['accuracy'], color='blue', label="training_accuracy")
plt.plot(epochs, history.history['val_accuracy'], color='red',label="validation_accuracy")
plt.legend(loc='best')
plt.title('validation')
plt.xlabel('epoch')
plt.savefig("validation.png", bbox_inches='tight')
plt.show()



# For his predictions he's going to use Test Time Augmentation. 
# For each test image he will augment it 5 ways and average the prediction. 
# He has also used ensemble learning by averaging the results of 3 versions of this model, 
# due to this he was able to achieve his highest leaderboard score of 0.964.


#Testing
testdf = pd.DataFrame({'path': glob(os.path.join(test_path, '*.tif'))})
testdf['id'] = testdf.path.map(lambda x: ((x.split("/")[2].split('.')[0])[5:]))
testdf.head(3)



# ImageDataGenerator accepts the original data, randomly transforms it, 
# and returns only the new, transformed data.
tta_datagen = ImageDataGenerator(rescale=1./255, #Normalise
                                 vertical_flip = True,
                                 horizontal_flip = True,
                                 rotation_range=90,
                                 zoom_range=0.2, 
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.05,
                                 channel_shift_range=0.1)


tta_steps = 5
submission = pd.DataFrame()
for index in range(0, len(testdf)):
    data_frame = pd.DataFrame({'path': testdf.iloc[index,0]}, index=[index])
    data_frame['id'] = data_frame.path.map(lambda x: ((x.split("/")[2].split('.')[0])[5:]))
    img_path = data_frame.iloc[0,0]
    test_img = cv2.imread(img_path)
    test_img = cv2.resize(test_img,(IMG_SIZE,IMG_SIZE))
    test_img = np.expand_dims(test_img, axis = 0)  
    predictionsTTA = []
    for i in range(0, tta_steps):
        preds = my_model.predict_generator(tta_datagen.flow_from_dataframe(dataframe = data_frame,
                                                                           directory = None,
                                                                           x_col = 'path',
                                                                           target_size = (IMG_SIZE, IMG_SIZE),
                                                                           class_mode = None,
                                                                           batch_size = 1,
                                                                           shuffle = False), steps = 1)
        predictionsTTA.append(preds)
    clear_output()
    prediction_entry = np.array(np.mean(predictionsTTA, axis=0))
    data_frame['label'] = prediction_entry
    submission = pd.concat([submission, data_frame[['id', 'label']]])
submission.set_index('id')
submission.head(3)
submission.to_csv('submission.csv', index=False, header=True)
