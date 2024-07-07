import pandas as pd
import os
import sklearn.model_selection as skl

data_dir = '../MHIST'
df = pd.read_csv(f'{data_dir}/annotations.csv')
df_train = df[df["Partition"]=='train']
df_test = df[df["Partition"]=='test']

train, val = skl.train_test_split(df_train[['Image Name', 'Majority Vote Label']], test_size=0.20, random_state=42, shuffle=True, stratify=df_train['Majority Vote Label'])

images_dir = '../MHIST/images/'
train['Image Name'] = images_dir + train['Image Name']
train.rename(columns={"Majority Vote Label": "label"}, inplace=True)
train.rename(columns={"Image Name": "path"}, inplace=True)

val['Image Name'] = images_dir + val['Image Name']
val.rename(columns={"Majority Vote Label": "label"}, inplace=True)
val.rename(columns={"Image Name": "path"}, inplace=True)

df_test = df_test[['Image Name', 'Majority Vote Label']]
df_test['Image Name'] = images_dir + df_test['Image Name']
df_test.rename(columns={"Majority Vote Label": "label"}, inplace=True)
df_test.rename(columns={"Image Name": "path"}, inplace=True)

train.to_csv(f'{data_dir}/train_labels.csv', index=False)
val.to_csv(f'{data_dir}/val_labels.csv', index=False)
df_test.to_csv(f'{data_dir}/test_labels.csv', index=False)

print(f'train_labels.csv, val_labels.csv, test_labels.csv is saved in directory "{data_dir}"')
