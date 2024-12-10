# Llama Inference on Multiple Nodes



This repos uses pipeline-parallelism to do large-model inferencing. 
If the mode is too large to fit into one single node (normally with 8 GPU cards), this repo may help.

Since it's pipeline parallel, all cards work sequentially, the inference delay increases with the model size. For Llama-3.1-405B-Instruct, with 32 910b Npu, inference time for one token is 1.5s.

Device allocation uses memory-balancing mode with granuarity of 1 layer.

## Install

pip install transformers safetensors

- for huawei NPU
    install Ascend toolkit, torch, torch_npu

- for Nvidia GPU
    install cuda and torch

## Tested Env
- model:
    - meta-llama/Llama-3.1-405B-Instruct
    - meta-llama/Meta-Llama-3-8B-Instruct

- platform:
    - 4 Nodes, 32 Huawei NPUs (torch-2.1.0, torch_npu-2.1.0)
    - 1 Nodes, 2 Nvidia A100


## model download

follow huggingface guide to download models

## Usage: 
```
torchrun --nnodes=4 --node_rank=0 --nproc_per_node=8 --master_addr ipaddr --master_port 12355 main.py
torchrun --nnodes=4 --node_rank=1 --nproc_per_node=8 --master_addr ipaddr --master_port 12355 main.py
torchrun --nnodes=4 --node_rank=2 --nproc_per_node=8 --master_addr ipaddr --master_port 12355 main.py
torchrun --nnodes=4 --node_rank=3 --nproc_per_node=8 --master_addr ipaddr --master_port 12355 main.py
```
