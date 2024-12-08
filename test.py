import torch  
import time
from transformers import LlamaForCausalLM, AutoTokenizer, DynamicCache

CONFIG = {
    'model_id': "meta-llama/Meta-Llama-3-8B-Instruct",
    'device': 'cuda',
    'max_context_length': 100,
    'max_output_tokens': 500,
}

def initialize_model(model_id, device):
    try:
        with torch.device(device):
            model = LlamaForCausalLM.from_pretrained(model_id)
            model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise

def prepare_input(messages, tokenizer, device):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        attention_mask=True,
        padding=True,
    )
    input_ids = input_ids[:, -CONFIG['max_context_length']:]
    return input_ids.to(device)

@torch.no_grad()
def generate_response(model, tokenizer, input_ids, max_output, terminators):
    try:        
        response = []
        past_key_values = DynamicCache()
        for _ in range(max_output):
            # len = input_ids.size(1) + past_key_values.get_seq_length()
            # position_ids = torch.arange(past_key_values.get_seq_length(), len).unsqueeze(0).to(input_ids.device)
            # attention_mask = torch.ones((1, len), dtype=torch.int64).to(input_ids.device)
            # cache_position = position_ids.squeeze(0)
            
            output = model(input_ids, 
                            # attention_mask=attention_mask,
                            # position_ids=position_ids,
                            # cache_position=cache_position,
                            return_dict=True, 
                            use_cache=True,
                            past_key_values=past_key_values)
            # print('past kv', past_key_values.get_seq_length())
            # past_key_values = output.past_key_values
            next_tokens = torch.argmax(output.logits, dim=-1)
            next_tokens = next_tokens[:, -1:]
            # del output
            # del input_ids
            input_ids = next_tokens
            token = next_tokens[0][0].item()
            if token in terminators:
                break
            # torch.cuda.empty_cache()
            # print('mem ', torch.cuda.memory_allocated()/1e9, torch.cuda.memory_reserved()/1e9)
            word = tokenizer.decode(token)
            response.append(word)
            print(word, end='', flush=True)
        print('')
        # del past_key_values
        torch.cuda.empty_cache()
        return response
    except torch.cuda.OutOfMemoryError:
        print("Error: GPU out of memory. Try reducing the context length.")
        return []
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return []

@torch.no_grad()
def generate_response_nocache(model, tokenizer, input_ids, max_output, terminators):
    try:        
        response = []
        for _ in range(max_output):
            output = model(input_ids, return_dict=True)
            next_tokens = torch.argmax(output.logits, dim=-1)
            next_tokens = next_tokens[:, -1:]
            input_ids = torch.concat([input_ids, next_tokens], dim=-1)
            token = next_tokens[0][0].item()
            if token in terminators:
                break
            word = tokenizer.decode(token)
            response.append(word)
            print('mem ', torch.cuda.memory_allocated()/1e9, torch.cuda.memory_reserved()/1e9)
            print(word, end='', flush=True)
        print('')
        return response
    except torch.cuda.OutOfMemoryError:
        print("Error: GPU out of memory. Try reducing the context length.")
        return []
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return []

from transformers import TextIteratorStreamer
import threading
def start():
  messages = []


  while True:
    # get intput
    prompt = input("User: ")
    if prompt == "exit":
        break
    messages.append({"role": "user", "content": prompt})

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        attention_mask = True,
        padding = True,
    ).to(model.device)



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

    messages.append({"role": "assistant", "content": response})

def chat(max_output=None):
    if max_output is None:
        max_output = CONFIG['max_output_tokens']
        
    messages = []
    while True:
        try:
            prompt = input("User: ")
            if prompt.lower() == "exit":
                break
            
            messages.append({"role": "user", "content": prompt})
            input_ids = prepare_input(messages, tokenizer, CONFIG['device'])
            
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
                print(f"Response time: {processing_time:.2f} seconds",  processing_time/len(response), "seconds per token")

            if response:  # Only append if we got a valid response
                messages.append({
                    'role': 'assistant',
                    'content': response
                })
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"Error in chat loop: {str(e)}")
            continue

# Initialize model and tokenizer
model, tokenizer = initialize_model(CONFIG['model_id'], CONFIG['device'])
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

