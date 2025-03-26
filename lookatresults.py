import os
from glob import glob
import yaml

results = sorted(glob('sweeps/*_zf/predictions_evaluation.yaml'))

best_mse=10000
best_result = None

for result_fp in results:
    with open(result_fp, 'r') as f:
        result = yaml.safe_load(f)
    result = result['overall']
    if result < best_mse:
        best_mse = result
        best_result = result_fp
        
print(best_result, best_mse)