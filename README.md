# Label Smoothing Plus: ```"LS+: Informed Label Smoothing for Improving Calibration in Medical Image Classification"```

<!-- [![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2006.08437-B31B1B.svg)](https://arxiv.org/abs/2002.09437)
[![Pytorch 1.5](https://img.shields.io/badge/pytorch-1.5.1-blue.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/torrvision/focal_calibration/blob/main/LICENSE) -->

This repository contains the code and pretrained models for *LS+: Informed Label Smoothing for Improving Calibration in Medical Image Classification*, which has been accepted in MICCAI 2024.

If the code or the paper has been useful in your research, please add a citation to our work:

```
@article{lsplus,
  title={LS+: Informed Label Smoothing for Improving Calibration in Medical Image Classification},
  author={Sambyal, Abhishek Singh and Niyaz, Usma and Shrivastava, Saksham and Krishnan, Narayanan C and Bathula, Deepti R.},
  booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2024}
}
```

### Dependencies (TO-DO)

The code is based on Tensorflow and requires a few further dependencies, listed in [tf.yml](distiller.yml) and [tf1.yml](baselines.yml). Please create these two conda envirnoments using the following command:
* ```conda env create -f tf.yml```
* ```conda env create -f tf1.yml```

### Directory Structure
```
├── ls_plus/
    ├── 01_retention_curves.ipynb
    ├── 02_all_histogram_plots/
    ├── 02_histogram_viz.ipynb
    ├── baselines.yml
    ├── chaoyang-data/
    ├── d1_ablation/
    ├── d1_chaoyang_code/
    ├── d2_ablation/
    ├── d2_mhist_code/
    ├── d3_skin_code/
    ├── distiller.yml
    ├── ISIC_2018/
    ├── MHIST/
    ├── r34_retention_curves.png
    ├── r50_retention_curves.png
    └── README.md
```
### Datasets

Datasets can be downloaded from here:
1. Chaoyang: https://bupt-ai-cz.github.io/HSA-NRL/
2. MHIST: https://bmirds.github.io/MHIST/
3. ISIC: https://challenge.isic-archive.com/data/#2018

* *Chaoyang*: Copy ```train/``` and ```test/``` folders from the downloaded Chaoyang dataset into the ```chaoyang-data/``` directory.
* *MHIST*: Copy ```images/``` folder from downloaded MHIST dataset to ```MHIST/``` directory.
* *ISIC 2018*: Copy all ```ISIC2018_Task3_*/``` folders from downloaded ISIC dataset into ```ISIC_2018/``` directory.

### Pretrained models

<!-- All the pretrained models for all the datasets can be [downloaded from here](http://www.robots.ox.ac.uk/~viveka/focal_calibration/). -->
Link to all pretrained models will be available soon.

### Training/Evaluation

You should use ```bash_eval.py``` shell script to train different approaches on different datasets available inside each code directories ```d1_chaoyang_code/```, ```d2_mhist_code/```, ```d3_skin_code/```

Example [Python command to run baseline (HL)]:
```
python baseline.py -epochs 200 -test 0 -model resnet34 -bestmodelpath resnet34/1/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv; 
``` 

```
--model: model to train (resnet34/resnet50)
--bestmodelpath: path to save trained model (or directory path)
--gpu: choose gpu number to run you code. Default: 0.
--csvfilename: CSV filename to store all the metrics values.
```
* Please check ```bash_eval.py``` script for the correct command pertaining to the method.
* ```-test 0``` will train the model and run the evaluation code to generate outputs (metrics, plots).

### Questions

If you have any questions or doubts, please feel free to open an issue in this repository or reach out to us at the email addresses provided in the paper.

### Ablation Results
#### Results with Temperature Scaling
The tables below contains the calibration results obtained after appliying temperature scaling on the models mentioned in the paper. These are additional results and are not added in the paper.

<table>
<tr><th>ResNet34, Chaoyang</th><th>ResNet50, Chaoyang</th></tr>
<tr><td>

| Model | ECE   | ACE   | cwECE | NLL   | Brier |
|----------------|-------|-------|-------|-------|-------|
| HL        | 9.47  | 9.47  | 5.53  | 63.51 | 28.67 |
| LS             | 4.05  | 4.53  | 3.3   | 50.58 | 26.69 |
| FL-3           | 4.09  | 4.23  | 4.52  | 50.26 | 26.95 |
| DCA            | 6.9   | 6.76  | 4.1   | 53.71 | 26.79 |
| MDCA           | 8.64  | 8.47  | 5.37  | 71.92 | 28.47 |
| Ours           | 2.89  | 2.97  | 3.44  | 49.71 | 25.65 |

</td><td>

| Model | ECE   | ACE   | cwECE | NLL   | Brier |
|----------------|-------|-------|-------|-------|-------|
| HL        | 7.2   | 7.09  | 5.14  | 64.36 | 29.12 |
| LS             | 3.55  | 4.2   | 3.65  | 53.72 | 27.87 |
| FL-3           | 3.01  | 3.19  | 3.95  | 53.3  | 27.52 |
| DCA            | 12.09 | 12.01 | 6.62  | 77.96 | 31.99 |
| MDCA           | 9.63  | 9.5   | 6.15  | 75.39 | 31.83 |
| Ours           | 2.87  | 3.0   | 4.02  | 52.78 | 27.06 |

</td></tr> </table>

<table>
<tr><th>ResNet34, MHIST </th><th>ResNet50, MHIST</th></tr>
<tr><td>

| Model | ECE   | ACE   | cwECE | NLL   | Brier |
|----------------|-------|-------|-------|-------|-------|
| HL        | 15.78 | 15.72 | 16.73 | 83.69 | 37.57 |
| LS             | 6.58  | 6.73  | 7.98  | 45.78 | 29.73 |
| FL-3           | 8.08  | 8.45  | 8.15  | 45.59 | 29.76 |
| DCA            | 6.94  | 6.56  | 7.59  | 47.92 | 30.69 |
| MDCA           | 10.31 | 10.00 | 10.34 | 54.57 | 30.22 |
| Ours           | 6.22  | 5.96  | 7.67  | 45.02 | 28.34 |

</td><td>

| Model | ECE   | ACE   | cwECE | NLL   | Brier |
|----------------|-------|-------|-------|-------|-------|
| HL        | 9.43  | 9.41  | 9.62  | 54.59 | 33.11 |
| LS             | 5.37  | 5.83  | 6.48  | 45.75 | 28.7  |
| FL-3           | 5.9   | 6.1   | 7.96  | 49.13 | 32.32 |
| DCA            | 6.59  | 6.43  | 8.2   | 54.85 | 30.42 |
| MDCA           | 7.44  | 7.41  | 8.14  | 54.6  | 32.2  |
| Ours           | 3.19  | 3.69  | 5.44  | 42.28 | 26.73 |

</td></tr> </table>


In the above tables, Hard Labels (**HL**), Label Smoothing (**LS**), FL (**FL-3 denotes focal loss with ```gamma = 3```**), Difference between Confidence and Accuracy (**DCA**), Multi-Class Difference in Confidence and Accuracy (**MDCA**) and Ours (**LS+**).
