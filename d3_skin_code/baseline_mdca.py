# Usage:
# python baseline_mdca.py -epochs 200 -test 0 -model resnet34 -bestmodelpath skin_resnet34/1/mdca_best_model.hdf5 -gpu 0 -csvfilename skin_resnet34/metrics.csv



import os, itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score
import random
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

from metrics import *
from reliabilityplots import reliability_plot




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
parser.add_argument("-lr", "--lr", type=float, default=0.001,required=False)
parser.add_argument("-gpu", "--gpu", type=str, required=True)
parser.add_argument("-model", "--model", help='[resnet34,resnet50]', type=str, required=True)
parser.add_argument("-csvfilename", "--csvfilename", type=str, required=True)
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
num_classes=7
if args.model=='resnet34':
    m_name='Resnet34'
elif args.model=='resnet50':
    m_name='Resnet50'
# data_dir="../chaoyang-data/"
data_dir="../ISIC_2018/"



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
    conv_base=ResNet34(input_shape=(224, 224, 3),include_top=False,weights='imagenet',input_tensor=inputs)
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


teacher.compile(optimizers.Adam(args.lr), loss = cross_entropy_with_mdca_loss, metrics = ["accuracy"])
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












    
    
    
    
    
    
print("----------------- valid ----------------------------")
teacher.load_weights(filepath)

teacher.evaluate(valid_generator)
data=[]
testLabels=[]
for i in range(len(valid_generator)):
    data.extend(valid_generator[i][0])
    testLabels.extend(valid_generator[i][1])
data=np.array(data)
testLabels=np.array(testLabels)


t1_logits,t1=teacher.predict(data)

val_auc = roc_auc_score(testLabels, t1, average='macro')
print('Teacher ROC AUC:', val_auc)

ce_loss = tf.keras.losses.CategoricalCrossentropy()
val_ce = ce_loss(testLabels, t1).numpy()

brier_loss =  BrierScore()
val_brier_loss = brier_loss(t1_logits, testLabels).numpy()


t = (t1 == t1.max(axis=1)[:, None]).astype(int)

val_acc = accuracy_score(testLabels, t)
print("VAL Teacher acc", val_acc)


testLabels = np.argmax(testLabels, axis=1)
t = np.argmax(t, axis=1)

val_class_accuracies = ClasswiseAccuracy(testLabels, t)
for class_label, accuracy in val_class_accuracies.items():
    print(f"Class {class_label} Accuracy: {accuracy:.4f}")

ece_class =  ECELoss()
aece_class =  AdaptiveECELoss()
sce_class = ClasswiseECELoss()

val_ece = ece_class(t1_logits, testLabels).numpy()
val_aece = aece_class(t1_logits, testLabels).numpy()
val_sce = sce_class(t1_logits, testLabels)
val_precision = precision_score(testLabels, t,average='macro')
val_recall = recall_score(testLabels, t,average='macro')
val_f1 = f1_score(testLabels, t,average='macro')

print("Expected Calibration Error", val_ece)
print("Adaptive Expected Calibration Error", val_aece)
print("Classwise Expected Calibration Error", val_sce)
print("Precision:", val_precision)
print("Recall:", val_recall)
print("F1-Score:", val_f1)
print("CE:", val_ce)
print("Brier-Loss:", val_brier_loss)

print("--------------------------------------------------- test ----------------------------")
teacher.evaluate(test_generator)

data=[]
testLabels=[]
for i in range(len(test_generator)):
    data.extend(test_generator[i][0])
    testLabels.extend(test_generator[i][1])
data=np.array(data)
testLabels=np.array(testLabels)

t1_logits, t1=teacher.predict(data)

test_auc = roc_auc_score(testLabels, t1, average='macro')
print('Teacher ROC AUC:', test_auc)

ce_loss = tf.keras.losses.CategoricalCrossentropy()
test_ce = ce_loss(testLabels, t1).numpy()

brier_loss =  BrierScore()
test_brier_loss = brier_loss(t1_logits, testLabels).numpy()

t = (t1 == t1.max(axis=1)[:, None]).astype(int)

test_acc = accuracy_score(testLabels, t)
print(f'{m_name} Teacher acc {test_acc}')






from util import make_hist_plot
histograms_dir = "histogram_plots"
c_pred, in_pred = make_hist_plot(t1, testLabels, histograms_dir, 'isic', m_name, sys.argv[0][:-3].upper(), test_acc, filename='mdca')
# exit()


from metrics import save_retention_curve_values
retention_curve_dir = "retention_curve_files"
if not os.path.exists(retention_curve_dir):
    os.makedirs(retention_curve_dir)
save_retention_curve_values(testLabels, t1, m_name, save_dir=retention_curve_dir, net='mdca')
# exit()




testLabels = np.argmax(testLabels, axis=1)
t = np.argmax(t, axis=1)

test_class_accuracies = ClasswiseAccuracy(testLabels, t)
for class_label, accuracy in test_class_accuracies.items():
    print(f"Class {class_label} Accuracy: {accuracy:.4f}")

ece_class =  ECELoss()
aece_class =  AdaptiveECELoss()
sce_class = ClasswiseECELoss()

test_ece = ece_class(t1_logits, testLabels).numpy()
test_aece = aece_class(t1_logits, testLabels).numpy()
test_sce = sce_class(t1_logits, testLabels)
test_precision = precision_score(testLabels, t,average='macro')
test_recall = recall_score(testLabels, t,average='macro')
test_f1 = f1_score(testLabels, t,average='macro')

print("Expected Calibration Error", test_ece)
print("Adaptive Expected Calibration Error", test_aece)
print("Classwise Expected Calibration Error", test_sce)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1-Score:", test_f1)
print("CE:", test_ce)
print("Brier-Loss:", test_brier_loss)



model_folder_name, run_id = args.bestmodelpath.split('/')[0], args.bestmodelpath.split('/')[1]
new_path = os.path.join(model_folder_name, 'reliability_plots')
if not os.path.exists(new_path):
    os.mkdir(new_path)
    
# Reliability plots
reliability_plot(np.max(t1, axis=1), t, testLabels, filename=f'{new_path}/{model_folder_name}_{run_id}_baseline_test') #confs, preds, labels



'''val_metrics_dict = {
    'Dataset': 'Chaoyang',
    'Name': sys.argv[0][:-3].upper(),
    'Arch': m_name,
    'Vacc': round(val_acc, 4),
    'Vauc': round(val_auc, 4),
    'Vprecision': round(val_precision, 4),
    'Vrecall': round(val_recall, 4),
    'Vf1': round(val_f1, 4),
    'Vece': round(val_ece, 4),
    'Vaece': round(val_aece, 4),
    'Vsce': round(val_sce, 4),
    'Vce': round(val_ce, 4),
    'Vbrierloss': round(val_brier_loss, 4),
    'VaccK0': round(val_class_accuracies[0], 4),
    'VaccK1': round(val_class_accuracies[1], 4),
    'VaccK2': round(val_class_accuracies[2], 4),
    'VaccK3': round(val_class_accuracies[3], 4),
    
    'Tacc': round(test_acc, 4),
    'Tauc': round(test_auc, 4),
    'Tprecision': round(test_precision, 4),
    'Trecall': round(test_recall, 4),
    'Tf1': round(test_f1, 4),
    'Tece': round(test_ece, 4),
    'Taece': round(test_aece, 4),
    'Tsce': round(test_sce, 4),
    'Tce': round(test_ce, 4),
    'Tbrierloss': round(test_brier_loss, 4),
    'TaccK0': round(test_class_accuracies[0], 4),
    'TaccK1': round(test_class_accuracies[1], 4),
    'TaccK2': round(test_class_accuracies[2], 4),
    'TaccK3': round(test_class_accuracies[3], 4),
}'''

val_metrics_dict = {
    'Dataset': 'ISIC2018',
    'Name': sys.argv[0][:-3].upper(),
    'Arch': m_name,
    'Vacc': round(val_acc, 4),
    'Vauc': round(val_auc, 4),
    'Vprecision': round(val_precision, 4),
    'Vrecall': round(val_recall, 4),
    'Vf1': round(val_f1, 4),
    'Vece': round(val_ece, 4),
    'Vaece': round(val_aece, 4),
    'Vsce': round(val_sce, 4),
    'Vce': round(val_ce, 4),
    'Vbrierloss': round(val_brier_loss, 4),
    'VaccK0': round(val_class_accuracies[0], 4),
    'VaccK1': round(val_class_accuracies[1], 4),
    'VaccK2': round(val_class_accuracies[2], 4),
    'VaccK3': round(val_class_accuracies[3], 4),
    'VaccK4': round(val_class_accuracies[4], 4),
    'VaccK5': round(val_class_accuracies[5], 4),
    'VaccK6': round(val_class_accuracies[6], 4),
    
    'Tacc': round(test_acc, 4),
    'Tauc': round(test_auc, 4),
    'Tprecision': round(test_precision, 4),
    'Trecall': round(test_recall, 4),
    'Tf1': round(test_f1, 4),
    'Tece': round(test_ece, 4),
    'Taece': round(test_aece, 4),
    'Tsce': round(test_sce, 4),
    'Tce': round(test_ce, 4),
    'Tbrierloss': round(test_brier_loss, 4),
    'TaccK0': round(test_class_accuracies[0], 4),
    'TaccK1': round(test_class_accuracies[1], 4),
    'TaccK2': round(test_class_accuracies[2], 4),
    'TaccK3': round(test_class_accuracies[3], 4),
    'TaccK4': round(test_class_accuracies[4], 4),
    'TaccK5': round(test_class_accuracies[5], 4),
    'TaccK6': round(test_class_accuracies[6], 4),
}

print(val_metrics_dict)

# metrics_file_name = 'metrics.csv'
metrics_file_name = args.csvfilename
print(f'Saving Metrics to {metrics_file_name} CSV')

if os.path.exists(metrics_file_name):
    test_df = pd.read_csv(metrics_file_name) # or pd.read_excel(filename) for xls file
    if test_df.empty: # will return True if the dataframe is empty or False if not.
        print('FIle is not empty')
        df = pd.DataFrame([val_metrics_dict.values()], columns=val_metrics_dict.keys())
        df.to_csv(metrics_file_name, mode='a', index=False)
        print(f'Appending {metrics_file_name}')
    else:
        df = pd.DataFrame([val_metrics_dict.values()], columns=val_metrics_dict.keys())
        df.to_csv(metrics_file_name, mode='a', index=False, header=False)
        print(f'Appending {metrics_file_name}')
else:
    df = pd.DataFrame([val_metrics_dict.values()], columns=val_metrics_dict.keys())
    df.to_csv(metrics_file_name, mode='a', index=False)
    print(f'Creating new {metrics_file_name}')