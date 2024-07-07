# Copy pretrained models of all methods for MHIST dataset.

1. Copy all the directories (pretrained models) [1, 2, 3] of ResNet34 and ResNet50 from d2_mhist_code/[resnet34, resnet50]/[1, 2, 3]
2. Using the pretrained model you will generate numpy_files directory which will give you the ece scores.

# Directory Structure
├── baseline_dca.py
├── baseline_fl.py
├── baseline_mdca.py
├── baseline.py
├── bash_eval.sh
├── distiller.py
├── metrics.py
├── README.md
├── reliabilityplots.py
├── resnet34
│   ├── 1 (Copy pretrained models from d2_mhist_code/resnet34)
        ├── numpy_files/
│   ├── 2 (Copy pretrained models from d2_mhist_code/resnet34)
        ├── numpy_files/
│   ├── 3 (Copy pretrained models from d2_mhist_code/resnet34)
        ├── numpy_files/
    ├── metrics_ecekde.csv
├── resnet50
│   ├── 1 (Copy pretrained models from d2_mhist_code/resnet50)
        ├── numpy_files/
│   ├── 2 (Copy pretrained models from d2_mhist_code/resnet50)
        ├── numpy_files/
│   ├── 3 (Copy pretrained models from d2_mhist_code/resnet50)
        ├── numpy_files/
    ├── metrics_ecekde.csv
