
PYTHON=python


## Generic script for submitting any tensorflow job to GPU
# usage: submit.sh [scriptname.py script_arguments ... ]

## Location of this script: assume gpu_lock.py is in same place -
SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )


gpu_id=$(python2 $SCRIPTPATH/find_free_gpu.py)

### paths for tensorflow (works on hynek)
# export LD_LIBRARY_PATH=/opt/cuda-8.0.44/extras/CUPTI/lib64/:/opt/cuda-8.0.44/:/opt/cuda-8.0.44/lib64:/opt/cuDNN-7.0/:/opt/cuDNN-6.0_8.0/:/opt/cuda/:/opt/cuDNN-6.0_8.0/lib64:/opt/cuDNN-6.0/lib6
export LD_LIBRARY_PATH=/opt/cuda-8.0.44/extras/CUPTI/lib64/:/opt/cuda-8.0.44/:/opt/cuda-8.0.44/lib64:/opt/cuDNN-7.0/:/opt/cuDNN-6.0_8.0/:/opt/cuda/:/opt/cuDNN-6.0_8.0/lib64:/opt/cuDNN-6.0/lib6:/opt/cuda-9.0.176.1/lib64/:/opt/cuda-9.1.85/lib64/:/opt/cuDNN-7.1_9.1/lib64
export CUDA_HOME=/opt/cuda-8.0.44/

export KERAS_BACKEND=tensorflow
export CUDA_VISIBLE_DEVICES=$gpu_id

if [ $gpu_id -gt -1 ]; then
    
    $PYTHON $@
    
else
    echo 'Let us wait! No GPU is available!'

fi
