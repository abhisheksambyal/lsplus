# # Resnet 34
# # Model 1
python distiller.py -epochs 100 -test 0 -model resnet34 -vanillamodelpath resnet34/1/vanilla_best_model.hdf5 -bestmodelpath resnet34/1 -m LS -gpu 0 -csvfilename resnet34/metrics.csv
# Model 2
python distiller.py -epochs 100 -test 0 -model resnet34 -vanillamodelpath resnet34/2/vanilla_best_model.hdf5 -bestmodelpath resnet34/2 -m LS -gpu 0 -csvfilename resnet34/metrics.csv
# Model 3
python distiller.py -epochs 100 -test 0 -model resnet34 -vanillamodelpath resnet34/3/vanilla_best_model.hdf5 -bestmodelpath resnet34/3 -m LS -gpu 0 -csvfilename resnet34/metrics.csv



# # # Resnet 50
# # Model 1
python distiller.py -epochs 100 -test 0 -model resnet50 -vanillamodelpath resnet50/1/vanilla_best_model.hdf5 -bestmodelpath resnet50/1 -m LS -gpu 1 -csvfilename resnet50/metrics.csv
# Model 2
python distiller.py -epochs 100 -test 0 -model resnet50 -vanillamodelpath resnet50/2/vanilla_best_model.hdf5 -bestmodelpath resnet50/2 -m LS -gpu 1 -csvfilename resnet50/metrics.csv
# Model 3
python distiller.py -epochs 100 -test 0 -model resnet50 -vanillamodelpath resnet50/3/vanilla_best_model.hdf5 -bestmodelpath resnet50/3 -m LS -gpu 1 -csvfilename resnet50/metrics.csv

