import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def make_hist_plot(t1, testLabels, histograms_dir, dataset, m_name, method, test_acc, filename):
    
    print('Starting now')
    correct_pred, incorrect_pred = [], []
    pred_prob = np.max(t1, axis=1)
    pred_class = np.argmax(t1, axis=1)
    gt_labels = np.argmax(testLabels, axis=1)
    correct_instance_mask = (gt_labels == pred_class)
    correct_pred = pred_prob * correct_instance_mask
    correct_pred = correct_pred[correct_pred!=0]

    incorrect_pred = pred_prob * ~correct_instance_mask
    incorrect_pred = incorrect_pred[incorrect_pred!=0]

    print(len(correct_pred)/(len(correct_pred)+len(incorrect_pred)))
    print(len(t1))

    print(np.average(correct_pred), np.average(incorrect_pred))
    print(f'# Correct Pred: {len(correct_pred)}, Incorrect Pred: {len(incorrect_pred)}')

    # histograms_dir = "histogram_plots"
    if not os.path.exists(histograms_dir):
        os.makedirs(histograms_dir)
        
        
    histogram_pred_dict = {
    'Dataset': dataset,
    'Name': method,
    'Arch': m_name,
    'correctpredictions': np.array(correct_pred),
    'incorrectpredictions': np.array(incorrect_pred),
    
}
    
    np.savetxt(f'{histograms_dir}/{dataset}_{m_name}_{method}_correct.txt', np.array(correct_pred), fmt='%f')
    np.savetxt(f'{histograms_dir}/{dataset}_{m_name}_{method}_incorrect.txt', np.array(incorrect_pred), fmt='%f')
    
    
    # # metrics_file_name = 'metrics.csv'
    # metrics_file_name = f'{histograms_dir}'
    # print(f'Saving Metrics to {metrics_file_name} CSV')

    # if os.path.exists(metrics_file_name):
    #     test_df = pd.read_csv(metrics_file_name) # or pd.read_excel(filename) for xls file
    #     if test_df.empty: # will return True if the dataframe is empty or False if not.
    #         print('FIle is not empty')
    #         df = pd.DataFrame([histogram_pred_dict.values()], columns=histogram_pred_dict.keys())
    #         df.to_csv(metrics_file_name, mode='a', index=False)
    #         print(f'Appending {metrics_file_name}')
    #     else:
    #         df = pd.DataFrame([histogram_pred_dict.values()], columns=histogram_pred_dict.keys())
    #         df.to_csv(metrics_file_name, mode='a', index=False, header=False)
    #         print(f'Appending {metrics_file_name}')
    # else:
    #     df = pd.DataFrame([histogram_pred_dict.values()], columns=histogram_pred_dict.keys())
    #     df.to_csv(metrics_file_name, mode='a', index=False)
    #     print(f'Creating new {metrics_file_name}')
        
    
    return (correct_pred, incorrect_pred)

# def plotting_histograms():
#         # https://realpython.com/python-histograms/

#     bins_used=15
#     # An "interface" to matplotlib.axes.Axes.hist() method
#     n, bins, patches = plt.hist(x=correct_pred, bins=bins_used, color='#0504aa', alpha=0.7, rwidth=0.85)
#     plt.grid(axis='y', alpha=0.75)
#     plt.xlabel('Confidence')
#     plt.ylabel('Frequency')
#     plt.title(f'Avg. Conf. [C]{np.average(correct_pred)*100:.2f}, Acc: {test_acc*100:.2f}; [C: {len(correct_pred)}, InC: {len(incorrect_pred)}]', fontsize=12)
#     maxfreq = n.max()
#     # Set a clean upper y-axis limit.
#     plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#     plt.savefig(f'{histograms_dir}/{m_name}_histogram_correct_{filename}.png', bbox_inches='tight', dpi=300)
#     plt.show()
#     plt.close()


#     # An "interface" to matplotlib.axes.Axes.hist() method
#     n, bins, patches = plt.hist(x=incorrect_pred, bins=bins_used, color='#0504aa', alpha=0.7, rwidth=0.85)
#     plt.grid(axis='y', alpha=0.75)
#     plt.xlabel('Confidence')
#     plt.ylabel('Frequency')
#     plt.title(f'Avg. Conf. [InC]{np.average(incorrect_pred)*100:.2f}, Err: {100-(test_acc*100):.2f}; [C: {len(correct_pred)}, InC: {len(incorrect_pred)}]', fontsize=12)
#     maxfreq = n.max()
#     # Set a clean upper y-axis limit.
#     plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#     plt.savefig(f'{histograms_dir}/{m_name}_histogram_incorrect_{filename}.png', bbox_inches='tight', dpi=300)
#     plt.show()
#     plt.close()
    