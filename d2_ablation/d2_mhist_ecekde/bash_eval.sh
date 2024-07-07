# # Resnet 34
# # Model 1
# python baseline.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/1/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv; 
# python distiller.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/1 -m LS -gpu 0 -csvfilename resnet34/metrics.csv; 
# python baseline_dca.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/1/dca_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv;
# python baseline_mdca.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/1/mdca_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv; 
# python distiller.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/1 -m ours_alpha05 -gpu 0 -csvfilename resnet34/metrics.csv;

# # Model 2
# python baseline.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/2/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv; 
# python distiller.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/2 -m LS -gpu 0 -csvfilename resnet34/metrics.csv; 
# python baseline_dca.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/2/dca_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv; 
# python baseline_mdca.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/2/mdca_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv; 
# python distiller.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/2 -m ours_alpha05 -gpu 0 -csvfilename resnet34/metrics.csv;

# # Model 3
# python baseline.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/3/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv; 
# python distiller.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/3 -m LS -gpu 0 -csvfilename resnet34/metrics.csv; 
# python baseline_dca.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/3/dca_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv; 
# python baseline_mdca.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/3/mdca_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv; 
# python distiller.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/3 -m ours_alpha05 -gpu 0 -csvfilename resnet34/metrics.csv;

# conda deactivate; conda activate tf1; 
python baseline_fl.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/1/fl_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv; 
python baseline_fl.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/2/fl_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv; 
python baseline_fl.py -epochs 200 -test 1 -model resnet34 -bestmodelpath resnet34/3/fl_best_model.hdf5 -gpu 0 -csvfilename resnet34/metrics.csv;


# # Resnet 50
# Model 1
# python baseline.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/1/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv; 
# python distiller.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/1 -m LS -gpu 0 -csvfilename resnet50/metrics.csv; 
# python baseline_dca.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/1/dca_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv; 
# python baseline_mdca.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/1/mdca_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv; 
# python distiller.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/1 -m ours_alpha05 -gpu 0 -csvfilename resnet50/metrics.csv;

# # Model 2
# python baseline.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/2/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv; 
# python distiller.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/2 -m LS -gpu 0 -csvfilename resnet50/metrics.csv; 
# python baseline_dca.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/2/dca_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv; 
# python baseline_mdca.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/2/mdca_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv; 
# python distiller.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/2 -m ours_alpha05 -gpu 0 -csvfilename resnet50/metrics.csv;

# # Model 3
# python baseline.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/3/vanilla_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv; 
# python distiller.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/3 -m LS -gpu 0 -csvfilename resnet50/metrics.csv; 
# python baseline_dca.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/3/dca_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv; 
# python baseline_mdca.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/3/mdca_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv; 
# python distiller.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/3 -m ours_alpha05 -gpu 0 -csvfilename resnet50/metrics.csv;


# conda deactivate; conda activate tf1; 
python baseline_fl.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/1/fl_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv;
python baseline_fl.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/2/fl_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv;
python baseline_fl.py -epochs 200 -test 1 -model resnet50 -bestmodelpath resnet50/3/fl_best_model.hdf5 -gpu 0 -csvfilename resnet50/metrics.csv;
