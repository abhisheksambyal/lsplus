# Usage:
# python distiller.py -epochs 100 -test 1 -model resnet34 -vanillamodelpath resnet34/1/vanilla_best_model.hdf5 -bestmodelpath resnet34/1 -m LS -gpu 0 -csvfilename resnet34/metrics.csv


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
tf.config.experimental_run_functions_eagerly(True)
from sklearn.utils import shuffle

from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score
from metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss, ClasswiseAccuracy, BrierScore
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
parser.add_argument("-m", "--method", help='[LS,LS_T,ours,ours_T, ours_V, ours_LS, ours_alpha05]', type=str, required=True)

parser.add_argument("-epochs", "--epochs", type=int, required=True)
parser.add_argument("-test", "--test", type=int, required=True)
parser.add_argument("-vanillamodelpath", "--vanillamodelpath", help='full path best_model', type=str, required=True)
parser.add_argument("-bestmodelpath", "--bestmodelpath", help='full path best_model', type=str, required=True)
parser.add_argument("-gpu", "--gpu", type=str, required=True)
parser.add_argument("-csvfilename", "--csvfilename", type=str, required=True)
parser.add_argument("-model", "--model", help='[resnet34,resnet50]', type=str, required=True)

args = parser.parse_args()

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

net = args.method
folder_model = os.path.join(args.bestmodelpath, args.method)
# folder_model = net
if os.path.isdir(folder_model):
    print("Folder present")
else:
    os.makedirs(folder_model)


if args.model=='resnet34':
    m_name='Resnet34'
elif args.model=='resnet50':
    m_name='Resnet50'
    
data_dir="../chaoyang-data/"
num_classes=4

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
IMG_SIZE = 224
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
student = tf.keras.Model(inputs=conv_base.input, outputs=[logits, predictions])




# Loading the basline model before training
vanillafilepath = args.vanillamodelpath
student.load_weights(vanillafilepath)




def get_best_model(folder_name):
    files=os.listdir(folder_name)
    matching = [s for s in files if 'index' in s]
    #print(matching)
    final_matching = [s[:-6] for s in matching]
    # print(final_matching)
    split_matching = [float(s.split(':')[1]) for s in final_matching]
    
    
    split_matching.sort()
    index=(split_matching[-1])
    final_file= [s for s in final_matching if str(index) in s]
    return final_file
#saved_model/V1_alpha_0.2_beta_0.2_run_1/student1

def delete_old_files(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))



global_train_accuracy_s1=[]
global_val_accuracy_s1=[]
global_train_eceloss=[]
global_val_eceloss=[]
global_train_loss_s1=[]
global_val_loss_s1=[]

class CustomEarlyStopping(keras.callbacks.Callback):

    def __init__(self, patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = 0
        

    def on_epoch_end(self, epoch, logs=None):
        train_current1 = logs.get("accuracy")
        global_train_accuracy_s1.append(np.round(train_current1,4))
        current1 = logs.get("val_accuracy")
        global_val_accuracy_s1.append(np.round(current1,4))
       
        
        
        ece_loss= logs.get("ece_loss")
        global_train_eceloss.append(np.round(ece_loss,6))
        ece_val_loss= logs.get("val_ece_loss")
        global_val_eceloss.append(np.round(ece_val_loss,6))
        train_s1_loss = logs.get("student_loss")
        global_train_loss_s1.append(np.round(train_s1_loss,6))
        s1_loss = logs.get("val_student_loss")
        global_val_loss_s1.append(np.round(s1_loss,6))
  
     
        if epoch%1==0:
          plt.figure(figsize=(20, 6))
          plt.subplot(2,3,1)
          max_acc_s1=np.max(np.array(global_val_accuracy_s1))
          step_s1=np.argmax(np.array(global_val_accuracy_s1))
          plt.plot(global_train_accuracy_s1, color="r", label="Train")
          plt.plot(global_val_accuracy_s1, color="b", label="Val")
       
          plt.xlabel('Epoch')
          plt.ylabel('Accuracy')
          plt.title(f'Validation accuracy:-{max_acc_s1} @ {step_s1}')
          # plt.title('Accuracy')
          plt.legend()

          plt.subplot(2, 3, 2)

          min_loss_s1=np.array(global_val_loss_s1)[step_s1]
          print("Student loss",min_loss_s1)
          plt.plot(global_train_loss_s1, color="r", label="Train")
          plt.plot(global_val_loss_s1, color="b", label="Val")
     
          plt.xlabel('Epoch')
          plt.ylabel('Loss')
          plt.title(f'Validation loss:- {min_loss_s1} @ {step_s1}')
          plt.legend()
          
          
          
          plt.subplot(2, 3, 3)
          min_loss_s1=np.array(global_val_eceloss)[step_s1]
          print("ECE loss",min_loss_s1)
          plt.plot(global_train_eceloss, color="r", label="Train")
          plt.plot(global_val_eceloss, color="b", label="Val")
        
          plt.xlabel('Epoch')
          plt.ylabel('ECE loss')
          plt.title(f'ECE loss:- {min_loss_s1} @ {step_s1}')
          plt.legend()
          print("SAVING............")
          plt.savefig(folder_model+"/loss_curves.jpg", dpi=300, bbox_inches='tight')
          plt.close()
        
        current = logs.get("val_accuracy")
        
        # print(current1)
        if np.greater(current, self.best):
            print(f'\n value of model  improved from  {np.round(self.best,4)} to {np.round(current,4)}')
            self.best = current
            self.best = np.round(self.best,4)
            self.wait = 0
            # Record the best weights if current results is better (less).

            self.best_weights = self.model.get_weights()
            delete_old_files(folder_model)
            print(f'\n Model save at {self.best}\n')
            self.model.save_weights(folder_model+"/student@:"+str(self.best))
            # print(f'value of S1 IOU improved from  {self.best1} and S2 IOU {self.best2}')
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
            else:
                print(f'value of model not improved  {np.round(self.best,4)}')
                

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

class Distiller(keras.Model):
    def __init__(self, student):
        super().__init__()
        self.student = student
        self.flag=0

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        ece_loss,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.ece_loss = ece_loss
        self.alpha = alpha
        self.temperature = temperature
    def call(self, input):
      logits,pred = self.student(input)
  
      return logits,pred

    def train_step(self, data):
        # Unpack data
        x, y = data
        # print("processing....")
        y_ = y.numpy()
        # print("done")
        # Forward pass of teacher

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_logits,student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            ece = self.ece_loss(student_logits,tf.argmax(y, axis=1, output_type=tf.dtypes.int32))
            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            if net == 'LS':
                y_uniform = np.array([.25, .25, .25, .25])
                y_uniform = tf.convert_to_tensor(y_uniform)
                if self.flag!=1:
                    print(f'***********************{net} selected***************************************\n')
                    self.flag=1
                distillation_loss = self.distillation_loss_fn(
                    y_uniform, # probability
                    tf.nn.softmax(student_logits, axis=1),
                )
                
                loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss

            elif net == 'ours_alpha05':
                # y_update = None
                
                c = np.array([0.8319, 0.6548, 0.9882, 0.7264]) # vanilla Average of 3 runs of the baseline (R34)
                # c = np.array([0.8679, 0.5556, 0.9787, 0.6916]) # vanilla Average of 3 runs of the baseline (R50)
                # c = np.array([0.8589, 0.5953, 0.9787, 0.6667]) # vanilla Average of 3 runs of the baseline (R152)
                
                y_update = np.where(y == 1, c, ((1-c[np.argmax(y, axis=1).reshape(-1, 1)])/(len(c)-1)))
                y_update = tf.convert_to_tensor(y_update)
                
                if self.flag!=1:
                    print(f'***********************{net} selected***************************************\n')
                    self.flag=1
                
                distillation_loss = self.distillation_loss_fn(
                    y_update, # probability
                    # student_predictions,
                    tf.nn.softmax(student_logits, axis=1),
                )
                
                loss = 0.5 * student_loss + 0.5 * distillation_loss
                
            elif net == 'ours_V':
                # y_update = None
                
                # c = np.array([0.8319, 0.6548, 0.9882, 0.7264]) # vanilla Average of 3 runs of the baseline (R34)
                # c = np.array([0.8679, 0.5556, 0.9787, 0.6916]) # vanilla Average of 3 runs of the baseline (R50)
                # c = np.array([0.8589, 0.5953, 0.9787, 0.6667]) # vanilla Average of 3 runs of the baseline (R152)
                
                y_update = np.where(y == 1, c, ((1-c[np.argmax(y, axis=1).reshape(-1, 1)])/(len(c)-1)))
                y_update = tf.convert_to_tensor(y_update)
                
                if self.flag!=1:
                    print(f'***********************{net} selected***************************************\n')
                    self.flag=1
                
                distillation_loss = self.distillation_loss_fn(
                    y_update, # probability
                    # student_predictions,
                    tf.nn.softmax(student_logits, axis=1),
                )
            elif net == 'ours_LS':
                # y_update = None
                
                # c = np.array([0.8319, 0.6548, 0.9882, 0.7264]) # vanilla Average of 3 runs of the baseline (R34)
                # c = np.array([0.8739, .6151, 0.9598, 0.6717]) # LS Average of 3 runs of the baseline (R50)
                # c = np.array([0.8679, 0.5556, 0.9787, 0.6916]) # LS Average of 3 runs of the baseline (R152)
                
                y_update = np.where(y == 1, c, ((1-c[np.argmax(y, axis=1).reshape(-1, 1)])/(len(c)-1)))
                y_update = tf.convert_to_tensor(y_update)
                
                if self.flag!=1:
                    print(f'***********************{net} selected***************************************\n')
                    self.flag=1
                
                distillation_loss = self.distillation_loss_fn(
                    y_update, # probability
                    # student_predictions,
                    tf.nn.softmax(student_logits, axis=1),
                )

            # FOR LS
            # loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss
            
            # FOR ours
            # loss = 0.5 * student_loss + 0.5 * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss, "ece_loss":ece}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_logits,y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)
        ece = self.ece_loss(y_logits, tf.argmax(y, axis=1, output_type=tf.dtypes.int32))
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss,"ece_loss": ece})
        return results


train_step_size = train_generator.n // train_generator.batch_size
valid_step_size = valid_generator.n // valid_generator.batch_size

# filepath="student_best_model.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# earlystopper = EarlyStopping(monitor='val_accuracy', patience=30, verbose=2, restore_best_weights=True)
callbacks_list = [CustomEarlyStopping(patience=50)]

distiller = Distiller(student=student)
distiller.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
#         optimizer=tf.keras.optimizers.SGD(
#         learning_rate=0.0001, momentum=0.9, nesterov=False, name="SGD"),
    metrics=['accuracy'],
    student_loss_fn=keras.losses.CategoricalCrossentropy(),
    distillation_loss_fn=keras.losses.KLDivergence(),
    ece_loss = ECELoss(),
    temperature=4,
)

if args.test!=1:
    print('Running the fit function')
    history=distiller.fit(train_generator,
                steps_per_epoch = train_step_size,
                epochs = args.epochs,
                validation_data = valid_generator,
                validation_steps = valid_step_size,
                callbacks =  callbacks_list,)
    hist_df = pd.DataFrame(history.history) 
    hist_df.to_csv(folder_model+'/history.csv')
else:
    print('Not running the fit function')



print("--------------------------------------------------- valid ----------------------------")
folder_model = os.path.join(args.bestmodelpath, args.method)
path_s=folder_model+'/'+get_best_model(folder_model)[0]
distiller.load_weights(path_s)

distiller.evaluate(valid_generator)
data=[]
testLabels=[]
for i in range(len(valid_generator)):
    data.extend(valid_generator[i][0])
    testLabels.extend(valid_generator[i][1])
data=np.array(data)
testLabels=np.array(testLabels)


t1_logits,t1=distiller.predict(data)

val_auc = roc_auc_score(testLabels, t1, average='macro')
print('Teacher ROC AUC:', val_auc)

ce_loss = tf.keras.losses.CategoricalCrossentropy()
val_ce = ce_loss(testLabels, t1).numpy()

brier_loss =  BrierScore()
val_brier_loss = brier_loss(t1_logits, testLabels).numpy()


t = (t1 == t1.max(axis=1)[:, None]).astype(int)

val_acc = accuracy_score(testLabels, t)
print("VAL Teacher acc", val_acc)

plot_labels = ['normal', 'serrated', 'adenocarcinoma', 'adenoma']

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
distiller.evaluate(test_generator)

data=[]
testLabels=[]
for i in range(len(test_generator)):
    data.extend(test_generator[i][0])
    testLabels.extend(test_generator[i][1])
data=np.array(data)
testLabels=np.array(testLabels)

t1_logits, t1=distiller.predict(data)

test_auc = roc_auc_score(testLabels, t1, average='macro')
print('Teacher ROC AUC:', test_auc)

ce_loss = tf.keras.losses.CategoricalCrossentropy()
test_ce = ce_loss(testLabels, t1).numpy()

brier_loss =  BrierScore()
test_brier_loss = brier_loss(t1_logits, testLabels).numpy()

t = (t1 == t1.max(axis=1)[:, None]).astype(int)

test_acc = accuracy_score(testLabels, t)
print("Resnet 18 Teacher acc", test_acc)




# from util import make_hist_plot
# histograms_dir = "histogram_plots"
# c_pred, in_pred = make_hist_plot(t1, testLabels, histograms_dir, 'd1', m_name, args.method.upper(), test_acc, filename=net)
# # exit()


# from metrics import save_retention_curve_values
# retention_curve_dir = "retention_curve_files"
# if not os.path.exists(retention_curve_dir):
#     os.makedirs(retention_curve_dir)
# save_retention_curve_values(testLabels, t1, m_name, save_dir=retention_curve_dir, net=net)
# # exit()




plot_labels = ['normal', 'serrated', 'adenocarcinoma', 'adenoma']

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
reliability_plot(np.max(t1, axis=1), t, testLabels, filename=f'{new_path}/{model_folder_name}_{run_id}_{args.method}_test') #confs, preds, labels



val_metrics_dict = {
    'Dataset': 'Chaoyang',
    'Name': args.method.upper(),
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