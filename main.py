import torch
import torch_npu
import safetensors
import transformers
import json
import os
from transformers import LlamaForCausalLM, TextIteratorStreamer,AutoTokenizer, AutoConfig,  Cache, DynamicCache
from transformers.utils import ModelOutput
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.processing_utils import Unpack
import threading

import torch.distributed as dist


# def load_config(model_id):
#     path = transformers.utils.cached_file(model_id, "config.json")
#     with open(path, "r") as f:
#         return json.load(f)
from typing import List, Optional, Tuple, Union

# rank = int(os.environ["RANK"])         # Unique rank of the process
# world_size = int(os.environ["WORLD_SIZE"]) # Total number of processes
backend = "hccl"

dist.init_process_group(
        backend=backend,       # Use NCCL for GPU communication, Gloo for CPU
        # init_method="env://", # Use environment variables for setup
        # world_size=world_size,
        # rank=rank,
    )

if not dist.is_initialized():
    print("Failed to initialize process group")
    exit(1)

rank = dist.get_rank()
world_size = dist.get_world_size()
backend = dist.get_backend()

print("rank", rank, "world_size", world_size, "backend", backend)

device = f'npu:{rank%8}'



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
        
        # # dist.barrier()
        # print(rank, 'forward')
        # # print all the arguments
        # print("input_ids", input_ids)
        # print("attention_mask", attention_mask)
        # print("position_ids", position_ids)
        # print("past_key_values", past_key_values)
        # print("inputs_embeds", inputs_embeds)
        # print("labels", labels)
        # print("use_cache", use_cache)
        # print("output_attentions", output_attentions)
        # print("output_hidden_states", output_hidden_states)
        # print("return_dict", return_dict)
        # print("cache_position", cache_position)
        # print("num_logits_to_keep", num_logits_to_keep)
        # print("kwargs", kwargs)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        
        if False:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )

        else:
            # this is basically rewrite forward of llama model
            m = self.base_model # llama model

            #use_cache = use_cache if use_cache is not None else self.config.use_cache
            use_cache = False

            if rank == 0:
                if (input_ids is None) ^ (inputs_embeds is not None):
                    raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            
            if inputs_embeds is None:
                if rank == 0:
                    inputs_embeds = m.embed_tokens(input_ids)
                    
            
            
            if rank == 0:
                # print(rank, inputs_embeds.shape, inputs_embeds.device, inputs_embeds.dtype)
                inputs_shape = torch.tensor(inputs_embeds.shape, device=device, dtype=torch.int64)
            else:
                inputs_shape = torch.empty(3, device=device, dtype=torch.int64)

            #print(rank, 'inputs shape (before broadcast)', inputs_shape)
            dist.broadcast(inputs_shape, src=0)
            # print(rank, 'inputs shape', inputs_shape)

            if rank != 0:
                inputs_embeds = torch.empty(tuple(inputs_shape.tolist()), device=device, dtype=torch.float32)
                # print(rank, inputs_embeds.shape, inputs_embeds.device, inputs_embeds.dtype)

            dist.broadcast(inputs_embeds, src=0)
            # print(rank, inputs_embeds.shape, inputs_embeds.device, inputs_embeds.dtype)
            # print(rank, 'inputs_embeds', inputs_embeds)

            # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = False
            if use_cache and not isinstance(past_key_values, Cache):
                return_legacy_cache = True
                if past_key_values is None:
                    past_key_values = DynamicCache()
                else:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                    print(
                        "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                        "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                        "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                    )

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
            all_self_attns = () if output_attentions else None
            next_decoder_cache = None

            # print(rank, 'hidden_states', hidden_states)
            # print(rank, 'causal_mask', causal_mask)
            # print(rank, 'position_ids', position_ids)
            # print(rank, 'cache_position', cache_position)
            # print(rank, 'position_embeddings', position_embeddings)

            for i,decoder_layer in enumerate(m.layers[: m.config.num_hidden_layers]):
                
                if device_map['layers'][i] == rank:
                    if output_hidden_states:
                        all_hidden_states += (hidden_states,)
                    
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
                

                dist.broadcast(hidden_states, src=rank)

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            if rank == world_size - 1:
                hidden_states = m.norm(hidden_states)

            dist.broadcast(hidden_states, src=world_size-1)
            # print(rank, 'hidden_states', hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = next_decoder_cache if use_cache else None
            if return_legacy_cache:
                next_cache = next_cache.to_legacy_cache()

            if not return_dict:
                outputs= tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            else:
                outputs = BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=next_cache,
                    hidden_states=all_hidden_states,
                    attentions=all_self_attns,
                )

        
        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss

        if rank == 0:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
            # print(rank, 'logits', logits)
        else: 
            logits = None

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    





#    dist.destroy_process_group()





def calculate_device_map(model, node_num):
    
    layers_num = len(model.base_model.layers)

    # embd layer is on gpu 0
    device_map = {
        'head': 0,
        'tail': world_size-1,
        'layers': [],
        "world_size": world_size,
        "layers_num": layers_num,
    }
    
    #allocate all layers into world_size - 1 gpus

    # layers for each node
    layers_per_node = (layers_num + node_num - 2)// (node_num - 1)
    device_map['layers_per_node']   = layers_per_node

    for i in range(1, world_size):
        device_map[i] = []
        for j in range(layers_per_node*(i-1), layers_per_node * i):
            if j < layers_num:
                device_map['layers'].append(i)


    return device_map


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

def load_meta_model():
    with torch.device("meta"):
        model = LLamaForCausalLMForMultiNodeInference.from_pretrained(model_id)        
    return model


model = load_meta_model()
config = model.config
print(config)
device_map = calculate_device_map(model, world_size)

print(device_map)




def load_state_dict(model_id):
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
    for k in safe_tensors_weight_map.keys():
        st_file = safe_tensors_weight_map[k]
        tensor = load_tensors_from_checkpoint(st_file, k)
        if tensor is None:
            print("None tensor", k, st_file)
            continue
        state_dict[k] = tensor
        # print(k, st_file, tensor.shape)
    return state_dict


# def prepare_input():
#     messages=[{"role": "user", "content": "hello"}]
#     input_ids = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         return_tensors="pt",
#         attention_mask = True,
#         padding = True,
#     )
#     return input_ids

# def compare_models(m1, m2):    
#     messages=[{"role": "user", "content": "hello"}]
#     input_ids = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         return_tensors="pt",
#         attention_mask = True,
#         padding = True,
#     )    
#     o1 = m1(input_ids.to(m1.device))
#     o2 = m2(input_ids.to(m2.device))
#     print(o1.equal(o2))

def load_model_hf(device):
    with torch.device(device):
        model = LLamaForCausalLMForMultiNodeInference.from_pretrained(model_id)
        model.eval()
    return model


def load_module_state_dict(module, prefix, state_dict):
    module.to_empty(device=device)
    sd = module.state_dict()
    for k in sd.keys():
        if k in sd:
            k2 = prefix + "." + k
            sd[k].copy_(state_dict[k2])
            print(rank, 'loaded', k2, sd[k].device)
        else:
            print("Missing key", k)

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

    state_dict = load_state_dict(model_id)

    if rank == 0:
        load_module_state_dict(model.base_model.embed_tokens, "model.embed_tokens", state_dict)
        load_module_state_dict(model.lm_head, "lm_head", state_dict)
        init_rope_module(model.base_model.rotary_emb)

    else:
        # load layers
        for i, l in enumerate(model.base_model.layers):
            if device_map['layers'][i] == rank:
                load_module_state_dict(l, f"model.layers.{i}", state_dict)
                init_rope_module(l.self_attn.rotary_emb)

    if rank == world_size - 1:
        
        load_module_state_dict(model.base_model.norm, "model.norm", state_dict)


load_model()

def run_partial_model(model):
    while True:
        model()

def test_inference(model):
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        messages = []
        messages.append({"role": "user", "content": "hello"})
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            attention_mask = True,
            padding = True,
        ).to(device)

        output = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        print("generate output", tokenizer.decode(output[0], skip_special_tokens=True))
    else:
        run_partial_model(model)

def chat(model):
    stopped = False
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]



        messages = []
        
        while True:
            # get intput
            prompt = input("User: ")
            if prompt == "exit":
                stopped = True
                break
            messages.append({"role": "user", "content": prompt})
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                attention_mask = True,
                padding = True,
            ).to(device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            
            thread = threading.Thread(
                target = lambda :  model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                streamer=streamer
            ), daemon=True)
            thread.start()
            
            response = ""
            for w in streamer:
                print(w, end="", flush=True)
                response += w
            print("")
            messages.append({"role": "assistant", "content": response})
    else:
        while not stopped:
            run_partial_model(model)



def setup_model():
    meta = load_meta_model()
    sd = load_state_dict(model_id)
    meta.to_empty(device="cpu")
    meta.load_state_dict(sd)


    init_rope(meta)
    return meta

chat(model)

# test_inference(model)

