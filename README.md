# LlamaMultiNode

llama inference on multiple-nodes

tested Llama3.1-405B-Instruct on 4 Nodes, 32 NPUs

```
torchrun --nnodes=4 --node_rank=0 --nproc_per_node=8 --master_addr ipaddr --master_port 12355 main.py
torchrun --nnodes=4 --node_rank=1 --nproc_per_node=8 --master_addr ipaddr --master_port 12355 main.py
torchrun --nnodes=4 --node_rank=2 --nproc_per_node=8 --master_addr ipaddr --master_port 12355 main.py
torchrun --nnodes=4 --node_rank=3 --nproc_per_node=8 --master_addr ipaddr --master_port 12355 main.py
```
