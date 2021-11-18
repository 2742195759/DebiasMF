cd /home/data/cvpods && python setup.py install && cd - && pods_train --num-gpus 1 --resume --eval DATASETS.TEST "'selectionbias_coat_test',"
