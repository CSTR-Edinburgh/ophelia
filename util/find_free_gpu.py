#!/usr/bin/python

import subprocess
import re

sp = subprocess.Popen(['nvidia-smi', 'pmon', '-c', '1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out_str = sp.communicate()

print out_str

lines = out_str[0].split('\n')
lines = [line for line in lines if not line.startswith('#')] # filter comments
#lines = [line.replace('-', '') for line in lines]            # remove - placeholders used to mark columns for idle GPUs
lines = [re.split('\s+', line.strip(' ')) for line in lines] # split on whitespace
lines = [line for line in lines if line != ['']]               # remove empty lines

free_gpu = -1
for line in lines:
    print line
    gpu_id = line[0]
    process_info = ''.join(line[1:])
    print gpu_id
    print process_info
    if process_info.replace('-', '') == '': ## does it contain only  - placeholders used to mark columns for idle GPUs?
        free_gpu = gpu_id
        break
print free_gpu

# out_dict = {}

# for item in out_list:
#     try:
#         key, val = item.split(':')
#         key, val = key.strip(), val.strip()
#         out_dict[key] = val
#     except:
#         pass

# pprint.pprint(out_dict)
