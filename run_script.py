import os
from tools.utilize import *

# validation
epochs = [40]
uniform_mode = 'ranking'
dataset_names = ['unit']
infer_ranges = ['ixi', 'brats2021', 'all']
models = [['ixi', 'pd', 't2'], ['brats2021', 't1', 'flair'], ['brats2021', 't2', 'flair'], ['brats2021', 't1', 't2']]
methods = ['normal', 'complex', 'combined']
gpu = 0

for epoch in epochs:
    for name in dataset_names:
        for infer_range in infer_ranges:
            for modalties in models: 
                for method in methods:
                    run_script = 'python kaid.py -d {} -s {} -t {} --validate --uniform-mode {} --dataset-name {} --method {} --infer-range {} --dataset-epochs {} -g {}'.format(
                        modalties[0], modalties[1], modalties[2], uniform_mode, name, method, infer_range, epoch, gpu)
                    print(run_script)
                    with open('{}/log_kaid_{}_{}_epoch.txt'.format('work_dir', uniform_mode, epoch), 'a') as f:
                        print(run_script, file=f)
                    os.system(run_script)