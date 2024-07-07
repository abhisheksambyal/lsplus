# python kde_metrics_evaluate.py -model resnet34 -method baseline -bestmodelpath resnet34/1/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics_ecekde.csv

import torch
import numpy as np
import sys
import os
import pandas as pd
import ece_kde

from sklearn.metrics import accuracy_score


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-bestmodelpath", "--bestmodelpath", help='full path best_model', type=str, required=True)
parser.add_argument("-gpu", "--gpu", type=str, required=True)
parser.add_argument("-csvfilename", "--csvfilename", type=str, required=True)
parser.add_argument("-model", "--model", help='[resnet34,resnet50]', type=str, required=True)
parser.add_argument("-method", "--method", help='[baseline, ls, baseline_fl, baseline_dca, baseline_mdca, ours_alpha05]', type=str, required=True)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu


if args.model=='resnet34':
    m_name='Resnet34'
elif args.model=='resnet50':
    m_name='Resnet50'


numpy_dir = os.path.join(args.model, args.bestmodelpath.split('/')[1], "numpy_files")

test_label = np.load(f'{numpy_dir}/test_testlabels_{args.method}.npy')
test_logits = np.load(f'{numpy_dir}/test_logits_{args.method}.npy')
test_probabilities = np.load(f'{numpy_dir}/test_probabilities_{args.method}.npy')


t = (test_probabilities == test_probabilities.max(axis=1)[:, None]).astype(int)
val_acc = accuracy_score(test_label, t)
print("VAL Teacher acc", val_acc)



test_label = torch.tensor(np.argmax(test_label, axis=1))
test_logits = torch.tensor(test_logits)
test_probabilities = torch.tensor(test_probabilities)

print(test_label.shape, test_logits.shape, test_probabilities.shape)

# bw = ece_kde.get_bandwidth(test_probabilities, device='cpu')

# print(bw)
# print(torch.sum(torch.isnan(test_probabilities).long()))
# if args.method == 'baseline_mdca':
#     print('In mdca')
#     bw=torch.tensor(0.0010)

# exit()
bw=0.001
ecekde_error = ece_kde.get_ece_kde(test_probabilities, test_label, bw, p=1, mc_type='canonical', device='cpu')

# ecekde_error = ecekde_error.numpy()
# print(type(ecekde_error), ecekde_error.shape)
# # print
# # print(round(ecekde_error.numpy(), 4))
# # exit()
print(f'Running ECE KDE for {args.model}, {args.method}')




val_metrics_dict = {
    'Dataset': 'MHIST',
    'Name': args.method.lower(),
    'Arch': m_name,
    'Tacc': round(val_acc, 4),
    'Tecekde': round(ecekde_error.item(), 4),
}


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