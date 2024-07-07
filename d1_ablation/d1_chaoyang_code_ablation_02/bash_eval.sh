# # Resnet 34
sleep 3h; python distiller.py -epochs 200 -test 0 -model resnet34 -bestmodelpath resnet34/1 -m ours_const04 -gpu 1 -csvfilename resnet34/metrics.csv;
sleep 3h 1m; python distiller.py -epochs 200 -test 0 -model resnet34 -bestmodelpath resnet34/2 -m ours_const04 -gpu 1 -csvfilename resnet34/metrics.csv;
sleep 3h 2m; python distiller.py -epochs 200 -test 0 -model resnet34 -bestmodelpath resnet34/3 -m ours_const04 -gpu 0 -csvfilename resnet34/metrics.csv;

# # # Resnet 50
sleep 3h 3m; python distiller.py -epochs 200 -test 0 -model resnet50 -bestmodelpath resnet50/1 -m ours_const04 -gpu 0 -csvfilename resnet50/metrics.csv;
sleep 3h 4m; python distiller.py -epochs 200 -test 0 -model resnet50 -bestmodelpath resnet50/2 -m ours_const04 -gpu 0 -csvfilename resnet50/metrics.csv;
sleep 3h 5m; python distiller.py -epochs 200 -test 0 -model resnet50 -bestmodelpath resnet50/3 -m ours_const04 -gpu 0 -csvfilename resnet50/metrics.csv;


# # # Resnet 34
# python distiller.py -epochs 200 -test 0 -model resnet34 -bestmodelpath resnet34/1 -m ours_const06 -gpu 0 -csvfilename resnet34/metrics.csv;
# python distiller.py -epochs 200 -test 0 -model resnet34 -bestmodelpath resnet34/2 -m ours_const06 -gpu 0 -csvfilename resnet34/metrics.csv;
# python distiller.py -epochs 200 -test 0 -model resnet34 -bestmodelpath resnet34/3 -m ours_const06 -gpu 0 -csvfilename resnet34/metrics.csv;

# # # # Resnet 50
# python distiller.py -epochs 200 -test 0 -model resnet50 -bestmodelpath resnet50/1 -m ours_const06 -gpu 0 -csvfilename resnet50/metrics.csv;
# python distiller.py -epochs 200 -test 0 -model resnet50 -bestmodelpath resnet50/2 -m ours_const06 -gpu 0 -csvfilename resnet50/metrics.csv;
# python distiller.py -epochs 200 -test 0 -model resnet50 -bestmodelpath resnet50/3 -m ours_const06 -gpu 0 -csvfilename resnet50/metrics.csv;



# # Resnet 34
python distiller.py -epochs 200 -test 0 -model resnet34 -bestmodelpath resnet34/1 -m ours_const08 -gpu 0 -csvfilename resnet34/metrics.csv;
python distiller.py -epochs 200 -test 0 -model resnet34 -bestmodelpath resnet34/2 -m ours_const08 -gpu 0 -csvfilename resnet34/metrics.csv;
python distiller.py -epochs 200 -test 0 -model resnet34 -bestmodelpath resnet34/3 -m ours_const08 -gpu 0 -csvfilename resnet34/metrics.csv;

# # # Resnet 50
python distiller.py -epochs 200 -test 0 -model resnet50 -bestmodelpath resnet50/1 -m ours_const08 -gpu 0 -csvfilename resnet50/metrics.csv;
python distiller.py -epochs 200 -test 0 -model resnet50 -bestmodelpath resnet50/2 -m ours_const08 -gpu 0 -csvfilename resnet50/metrics.csv;
python distiller.py -epochs 200 -test 0 -model resnet50 -bestmodelpath resnet50/3 -m ours_const08 -gpu 0 -csvfilename resnet50/metrics.csv;

