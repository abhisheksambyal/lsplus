# Usage:
# python baseline.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/1/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv
# python baseline.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/2/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv
# python baseline.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/3/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv

# python baseline.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/1/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv
# python baseline.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/2/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv
# python baseline.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/3/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score
import random
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
import sys
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPool2D
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras
from keras import optimizers
from IPython.display import clear_output
from classification_models.keras import Classifiers
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from metrics import BrierScore, ECELoss, AdaptiveECELoss, ClasswiseECELoss, ClasswiseAccuracy, BrierScore
from reliabilityplots import reliability_plot

# import tensorflow_probability as tfp


ResNet18, preprocess_input = Classifiers.get('resnet18')
ResNext50, preprocess_input = Classifiers.get('resnext50')
ResNet101, preprocess_input = Classifiers.get('resnet101')
ResNet34, preprocess_input = Classifiers.get('resnet34')
ResNext101, preprocess_input = Classifiers.get('resnext101')
DenseNet201, preprocess_input = Classifiers.get('densenet201')
MobileNet, preprocess_input = Classifiers.get('mobilenet')



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-epochs", "--epochs", type=int, required=True)
parser.add_argument("-test", "--test", type=int, required=True)
parser.add_argument("-bestmodelpath", "--bestmodelpath", help='full path best_model', type=str, required=True)
parser.add_argument("-gpu", "--gpu", type=str, required=True)
parser.add_argument("-csvfilename", "--csvfilename", type=str, required=True)
parser.add_argument("-model", "--model", help='[resnet34,resnet50]', type=str, required=True)
args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu



# Ref: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) # Notice here
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


IMG_SIZE=224
BATCH_SIZE=8


if args.model=='resnet34':
    m_name='Resnet34'
elif args.model=='resnet50':
    m_name='Resnet50'
    
data_dir="../MHIST/"
num_classes=2

train = pd.read_csv(data_dir+'train_labels.csv')
train['label'] = train['label'].astype(str)
val = pd.read_csv(data_dir+'val_labels.csv')
val['label'] = val['label'].astype(str)
test=pd.read_csv(data_dir+'test_labels.csv')
test['label'] = test['label'].astype(str)
print(f"Train: {train.shape} \nVal: {val.shape} \nTest: {test.shape}")

train_datagen = ImageDataGenerator(rescale=1./255,
                                #   vertical_flip = True,
                                  horizontal_flip = True,
                                #   rotation_range=20,
                                  zoom_range=0.2, 
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.05,
                                  channel_shift_range=0.1)

test_datagen = ImageDataGenerator(rescale = 1./255) 


# In[15]:



train_generator = train_datagen.flow_from_dataframe(dataframe = train, 
                                                    directory = None,
                                                    x_col = 'path', 
                                                    y_col = 'label',
                                                    target_size = (IMG_SIZE,IMG_SIZE),
                                                    class_mode = "categorical",
                                                    batch_size=BATCH_SIZE,
                                                    seed = 110318,
                                                    shuffle = True)


# In[16]:


valid_generator = test_datagen.flow_from_dataframe(dataframe = val,
                                                   directory = None,
                                                   x_col = 'path',
                                                   y_col = 'label',
                                                   target_size = (IMG_SIZE,IMG_SIZE),
                                                   class_mode = 'categorical',
                                                   batch_size = 1,
                                                   shuffle = False)

test_generator = test_datagen.flow_from_dataframe(dataframe = test,
                                                   directory = None,
                                                   x_col = 'path',
                                                   y_col = 'label',
                                                   target_size = (IMG_SIZE,IMG_SIZE),
                                                   class_mode = 'categorical',
                                                   batch_size = 1,
                                                   shuffle = False)




dropout_fc = 0.5
inputs = tf.keras.layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))

if args.model=='resnet34':
    conv_base=ResNet34(input_shape=(224, 224, 3),include_top=False, weights='imagenet',input_tensor=inputs)
elif args.model=='resnet50':
    conv_base=tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),include_top=False, weights='imagenet',input_tensor=inputs)
# conv_base=tf.keras.applications.resnet.ResNet152(input_shape=(224, 224, 3),include_top=False, weights='imagenet',input_tensor=inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(conv_base.output)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
logits = tf.keras.layers.Dense(num_classes, name="logits")(x)  # This layer outputs logits
predictions = tf.keras.layers.Activation('softmax', name="classification")(logits)  # Softmax for predictions
teacher = tf.keras.Model(inputs=conv_base.input, outputs=[logits, predictions])



teacher.compile(optimizers.Adam(0.001), loss = keras.losses.CategoricalCrossentropy(from_logits=True), metrics = ["accuracy"])
train_step_size = train_generator.n // train_generator.batch_size
valid_step_size = valid_generator.n // valid_generator.batch_size

    
filepath = args.bestmodelpath
# filepath="v2_best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_classification_accuracy', verbose=1, save_best_only=True, mode='max')

# earlystopper = EarlyStopping(monitor='val_accuracy', patience=30, verbose=2, restore_best_weights=True)
callbacks_list = [checkpoint]
# earlystopper = EarlyStopping(monitor='val_loss', patience=50, verbose=2, restore_best_weights=True)
# reduce = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)

if args.test!=1:
    print('Running the fit function')
    history = teacher.fit(train_generator,
                        steps_per_epoch = train_step_size,
                        epochs =  args.epochs,
                        validation_data = valid_generator,
                        validation_steps = valid_step_size,
                        callbacks = callbacks_list,
                        verbose = 2)
else:
    print('Not running the fit function')





    
print("--------------------------------------------------- test ----------------------------")
print("Loading model")
teacher.load_weights(filepath)


print("Computing metrics for test data")
data=[]
testLabels=[]
for i in range(len(test_generator)):
    data.extend(test_generator[i][0])
    testLabels.extend(test_generator[i][1])
data=np.array(data)
testLabels=np.array(testLabels)

t1_logits, t1=teacher.predict(data)


# Saving numpy files
numpy_dir = os.path.join(args.model, args.bestmodelpath.split('/')[1], "numpy_files")
if not os.path.exists(numpy_dir):
    os.makedirs(numpy_dir)

np.save(f'{numpy_dir}/test_testlabels_{sys.argv[0][:-3].lower()}.npy', testLabels)
np.save(f'{numpy_dir}/test_logits_{sys.argv[0][:-3].lower()}.npy', t1_logits)
np.save(f'{numpy_dir}/test_probabilities_{sys.argv[0][:-3].lower()}.npy', t1)

print('Numpy files saved. Done!')
