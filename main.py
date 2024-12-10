import torch
import torch.distributed as dist
import torch_npu

import safetensors
import transformers
import json
import os
import time

import argparse
from typing import List, Optional, Tuple, Union

import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer,  Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast


argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
argparser.add_argument("--device", type=str, default="npu", choices=["npu", "cuda"])

args = argparser.parse_args()


max_context_length = 100
max_output_tokens = 500

#model_id = "meta-llama/Llama-3.1-405B-Instruct"
#model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model_id = args.model
backend = "nccl" if args.device == "cuda" else "hccl"


def log(*args, **kwargs):
    print(rank, *args, **kwargs)

dist.init_process_group(
        backend=backend,       # Use NCCL for GPU communication, Gloo for CPU
        # init_method="env://", # Use environment variables for setup
        # world_size=world_size,
        # rank=rank,
    )

if not dist.is_initialized():
    log("Failed to initialize process group")
    exit(1)

rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
world_size = dist.get_world_size()
backend = dist.get_backend()

device_count = torch.npu.device_count()
device = f'{args.device}:{local_rank%device_count}'
log("rank", rank, 'local_rank', local_rank,  "world_size", world_size, "backend", backend, "device", device)





class LLamaForCausalLMForMultiNodeInference(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)

        # should get multinode info here


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
    

        # this is basically rewrite forward of llama model
        m = self.base_model # llama model

        if rank == 0:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = m.embed_tokens(input_ids)


        if rank == 0:
            inputs_shape = torch.tensor(inputs_embeds.shape, device=device, dtype=torch.int64)
        else:
            inputs_shape = torch.empty(3, device=device, dtype=torch.int64)

        dist.broadcast(inputs_shape, src=0)

        if rank != 0:
            inputs_embeds = torch.empty(tuple(inputs_shape.tolist()), device=device, dtype=torch.float32)

        dist.broadcast(inputs_embeds, src=0)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = m._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = m.rotary_emb(hidden_states, position_ids)
        # rotary_emb is valid only in rank 0
        dist.broadcast(position_embeddings[0], src=0)
        dist.broadcast(position_embeddings[1], src=0)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        # all_self_attns = () if output_attentions else None
        # next_decoder_cache = None


        for i,decoder_layer in enumerate(m.layers[: m.config.num_hidden_layers]):
            
            if device_map['layers'][i] == rank:
                # if output_hidden_states:
                #     all_hidden_states += (hidden_states,)
                if i > 0 and device_map['layers'][i-1] != rank:
                    # log(rank, 'hidden_states', hidden_states)
                    dist.recv(hidden_states, src=device_map['layers'][i-1], tag=device_map['layers'][i-1])

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

                hidden_states = layer_outputs[0]

                if i+1 < len(m.layers) and device_map['layers'][i+1] != rank:
                    # log(rank, 'hidden_states', hidden_states)
                    dist.send(hidden_states, dst=device_map['layers'][i+1], tag=rank)
            else:
                pass


        if rank == world_size - 1:
            hidden_states = m.norm(hidden_states)

        dist.broadcast(hidden_states, src=world_size-1)


        
        # hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss

        if rank == 0:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
            # log(rank, 'logits', logits)
        else: 
            logits = None

        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            # loss=loss,
            logits=logits,
            # past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )
    





#    dist.destroy_process_group()

def broadcast_variant_tensor(tensor, dim, src):
    if rank == src:
        shape = torch.tensor(tensor.shape, device=device, dtype=torch.int64)
        dist.boradcast(shape, src=src)
    else:
        shape = torch.empty(dim, device=device, dtype=torch.int64)
        dist.broadcast(shape, src=src)
        tensor = torch.empty(tuple(shape.tolist()), device=device, dtype=torch.int64)
            
    dist.broadcast(tensor, src=0)
    return tensor

@torch.no_grad()
def generate_response(model, tokenizer, input_ids, max_output, terminators):
        # log('start generate')
        response = []
        past_key_values = DynamicCache()
        terminated_tensor = torch.zeros(1, device=device)
        # use cache. 
        cache_len = 0
        for _ in range(max_output):
            #sync input_ids
            if rank == 0:
                input_ids_shape = torch.tensor(input_ids.shape, device=device, dtype=torch.int64)
                dist.broadcast(input_ids_shape, src=0)
            else:
                input_ids_shape = torch.empty(2, device=device, dtype=torch.int64)
                dist.broadcast(input_ids_shape, src=0)
                input_ids = torch.empty(tuple(input_ids_shape.tolist()), device=device, dtype=torch.int64)
            
            dist.broadcast(input_ids, src=0)
            

            len = input_ids.size(1) + cache_len;#.get_seq_length()
            position_ids = torch.arange(cache_len, len).unsqueeze(0).to(input_ids.device)
            attention_mask = torch.ones((1, len), dtype=torch.int64).to(input_ids.device)
            cache_position = position_ids.squeeze(0)
            
            # cache_position shoulb be provided, in the model
            # it won't be generated correctly since some
            # rank don't have correct past_key_values seq_length
            output = model(input_ids, 
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            cache_position=cache_position,
                            return_dict=True, 
                            use_cache=True,
                            past_key_values=past_key_values
                            )
            # log('past kv', past_key_values.get_seq_length())
            # past_key_values = output.past_key_values

            # only the main rank do the generation

            cache_len += input_ids.size(1)

            if rank == 0:
                next_tokens = torch.argmax(output.logits, dim=-1)
                next_tokens = next_tokens[:, -1:]
                # del output
                # del input_ids
                # input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                # cache_len += input_ids.size(1)
                input_ids = next_tokens


                token = next_tokens[0][0].item()

                # borad terminated to all. to control the loop
                if token in terminators:
                    terminated_tensor += 1
                else:
                    word = tokenizer.decode(token)
                    response.append(word)
                    print(word, end='', flush=True)

            dist.broadcast(terminated_tensor, src=0)
            terminated = terminated_tensor.item()
            if terminated:
                break
            
            # torch.cuda.empty_cache()
            # log('mem ', torch.cuda.memory_allocated()/1e9, torch.cuda.memory_reserved()/1e9)

        if rank == 0:
            print('')
        
        # log('end generate')
        return response



def prepare_input(messages, tokenizer, device):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        attention_mask=True,
        padding=True,
    )
    input_ids = input_ids[:, -max_context_length:]
    return input_ids.to(device)


def chat(max_output=None):
    if max_output is None:
        max_output = max_output_tokens
        
    messages = []
    while True:
        

            exit_flag = torch.zeros(1).to(device)
            if rank == 0:
                prompt = input("User: ")
                if prompt.lower() == "exit":
                    exit_flag = exit_flag + 1
                    
                dist.broadcast(exit_flag, src=0)

                if exit_flag.item() == 1:
                    log("User exiting chat...")
                    break
                messages.append({"role": "user", "content": prompt})
                input_ids = prepare_input(messages, tokenizer, device)
                
                start_time = time.time()
                response = generate_response(
                    model, 
                    tokenizer, 
                    input_ids, 
                    max_output, 
                    terminators
                )
                if len(response) > 0:
                    processing_time = time.time() - start_time
                    log(f"Response time: {processing_time:.2f} seconds",  processing_time/len(response), "seconds per token")

                if response:  # Only append if we got a valid response
                    messages.append({
                        'role': 'assistant',
                        'content': response
                    })
            else:
                input_ids = None
                # wait for prompt from rank 0
                dist.broadcast(exit_flag, src=0)
                if exit_flag.item() == 1:
                    log("User exiting chat...")
                    break

                response = generate_response(
                    model, 
                    tokenizer, 
                    input_ids, 
                    max_output, 
                    terminators
                )




def calculate_device_map(model, node_num):
    
    layers_num = len(model.base_model.layers)

    # embd layer is on gpu 0
    device_map = {
        'embed_tokens': 0,
        'lm_head': 0,
        'norm': world_size-1,
        
        'layers': [0, 0],
        
        "world_size": world_size,
        "layers_num": layers_num,
    }
    
    allocated_elements = np.zeros(world_size, dtype=np.int32)

    #allocate all layers into world_size - 1 gpus

    # layers for each node


    # embed_tokens and lm_head 
    # norm
    # lm_head
    # layers

    state_dict = model.state_dict()
    def calc_mem_size(prefix):
        size = 0
        for name in state_dict:
            if name.startswith(prefix):
                size += state_dict[name].numel()
        return size

    

    allocated_elements[0] = calc_mem_size("model.embed_tokens") + calc_mem_size("lm_head")
    allocated_elements[world_size-1] = calc_mem_size("model.norm")

    one_layer_size = calc_mem_size("model.layers.0")

    node_layer_num = np.zeros(node_num, dtype=np.int32)

    for i in range(layers_num):
        idx = np.argmin(allocated_elements)
        allocated_elements[idx] += one_layer_size
        node_layer_num[idx] += 1

    print("layers for nodes: ", node_layer_num)

    node_index = 0
    layers_in_node = 0
    for i in range(layers_num):
        if layers_in_node < node_layer_num[node_index]:
            device_map['layers'].append(node_index)
            layers_in_node +=1
        else:
            layers_in_node =1
            node_index += 1
            device_map['layers'].append(node_index)

    
    # layers_per_node = (layers_num-2)// (node_num - 1)
    # device_map['layers_per_node']   = layers_per_node
    # remainder = (layers_num-2) % (node_num - 1)

    # node_index = 1
    # layers_in_node = 0
    # for i in range(2,layers_num):
    #     if layers_in_node < layers_per_node:
    #         device_map['layers'].append(node_index)
    #         layers_in_node += 1
    #     elif layers_in_node == layers_per_node and (node_index-1) < remainder:
    #         device_map['layers'].append(node_index)
    #         layers_in_node += 1
    #     else:
    #         node_index += 1
    #         layers_in_node = 1
    #         device_map['layers'].append(node_index)


    # first layer be in rank 0
    # device_map['layers'][0] = 0

    return device_map



def load_meta_model():
    with torch.device("meta"):
        model = LLamaForCausalLMForMultiNodeInference.from_pretrained(model_id)        
    return model


model = load_meta_model()
config = model.config
# log(config)
device_map = calculate_device_map(model, world_size)

log(device_map)




def load_state_dict(model_id, keys):
    safe_tensors_index_path = transformers.utils.cached_file(model_id, "model.safetensors.index.json")
    with open(safe_tensors_index_path, "r") as f:
        safe_tensors_index = json.load(f)
    folder = os.path.dirname(safe_tensors_index_path)
    safe_tensors_weight_map = safe_tensors_index['weight_map']
    buffered_files = {}
    def load_tensors_from_checkpoint(file, key):
        if file not in buffered_files:
            buffered_files[file] = safetensors.safe_open(os.path.join(folder, file), framework="pt")
        return buffered_files[file].get_tensor(key)
    state_dict = dict()
    for k in keys:
        st_file = safe_tensors_weight_map[k]
        tensor = load_tensors_from_checkpoint(st_file, k)
        if tensor is None:
            log("None tensor", k, st_file)
            continue
        state_dict[k] = tensor
        # log(k, st_file, tensor.shape)
    return state_dict




def load_module_state_dict(module, prefix):
    module.to_empty(device=device)
    sd = module.state_dict()
    sd_disk = load_state_dict(model_id, map(lambda k: prefix + "." + k, sd.keys()))


    for k in sd.keys():
        if k in sd:
            k2 = prefix + "." + k
            sd[k].copy_(sd_disk[k2])
            log(rank, 'loaded', k2, sd[k].device)
        else:
            log("Missing key", k)

    # module.load_state_dict(sd_disk)

def init_rope(model):
        for l in model.base_model.layers:
            m = l.self_attn.rotary_emb
            inv_freq, attention_scale = m.rope_init_fn(m.config, None, **m.rope_kwargs)
            m.inv_freq = inv_freq
            m.attention_scale = attention_scale
        
        m = model.base_model.rotary_emb
        inv_freq, attention_scale = m.rope_init_fn(m.config, None, **m.rope_kwargs)
        m.inv_freq = inv_freq
        m.attention_scale = attention_scale

#load self part of the model


def init_rope_module(m):
    m.to_empty(device=device)
    inv_freq, attention_scale = m.rope_init_fn(m.config, device, **m.rope_kwargs)
    m.inv_freq = inv_freq
    m.attention_scale = attention_scale

def load_model():

    # state_dict = load_state_dict(model_id)

    if rank == 0:
        load_module_state_dict(model.base_model.embed_tokens, "model.embed_tokens")
        load_module_state_dict(model.lm_head, "lm_head")
        init_rope_module(model.base_model.rotary_emb)

    # load layers
    for i, l in enumerate(model.base_model.layers):
        if device_map['layers'][i] == rank:
            load_module_state_dict(l, f"model.layers.{i}")
            init_rope_module(l.self_attn.rotary_emb)

    if rank == world_size - 1:
        load_module_state_dict(model.base_model.norm, "model.norm")


load_model()


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


def setup_model():
    meta = load_meta_model()
    sd = load_state_dict(model_id)
    meta.to_empty(device="cpu")
    meta.load_state_dict(sd)


    init_rope(meta)
    return meta

chat(500)

dist.destroy_process_group()
