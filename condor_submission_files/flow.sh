#!/bin/bash

# Set the root prefix for Micromamba
export MAMBA_ROOT_PREFIX="/net/data_cms3a-1/daumann/micromamba"

# Add micromamba binary to PATH if needed
export PATH="/net/data_cms3a-1/daumann/micromamba/bin:$PATH"

# Optionally, set the PyTorch CUDA allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Initialize the current shell with micromamba's shell hook
eval "$(/net/data_cms3a-1/daumann/micromamba/bin/micromamba shell hook --shell bash)"

# Activate the environment
micromamba activate flow_corrections_env

# Set LD_LIBRARY_PATH to include the cuDNN (and other) libraries from the environment
export LD_LIBRARY_PATH="/net/data_cms3a-1/daumann/micromamba/envs/one-flow/lib:$LD_LIBRARY_PATH"

# Log the node/machine on which the job is running
echo "Job running on node: $(hostname)"

# Use a default argument if none is provided
if [ -z "$1" ]; then
  echo "No argument provided; defaulting to 0."
  arg=1
else
  arg="$1"
fi

# Run the training script (passing the argument)
cd ..
python -u main.py "$arg"