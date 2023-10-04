# Modify inference.py for batch generation and instruction data format.

import fire
import os
import sys
import time
import json

import torch
from torch.utils.data import Dataset
from transformers import LlamaTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model

# This prompt dict is from alpaca_dataset.py
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class ValidationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        if item.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(item)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(item)
        
        return {"prompts": prompt, "labels": item["output"]}



def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    batch_size: int: 8 # Validation batch size
    **kwargs
):
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"
        with open(prompt_file, "r") as f:
            data = json.load(f)
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    # Padding to left is needed for batch generation.
    tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side="left")


    """
    As described here https://github.com/facebookresearch/llama-recipes/blob/main/docs/inference.md
    we need to add pad token manually for batch inference.
    """
    tokenizer.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )
    model.resize_token_embeddings(model.config.vocab_size + 1)

    
    val_data = ValidationDataset(data)
    val_dataloader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=True,
            drop_last=False
        )
    
    translations = []
    for data_batch in val_dataloader:
        """
        We pad to the longest sequence in the batch and not truncate at all because we are confident
        they have a reasonable lenght.
        """
        batch = tokenizer(data_batch["prompts"], padding=True, truncation=False, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs 
            )
        e2e_inference_time = (time.perf_counter()-start)*1000
        #print(f"the inference time is {e2e_inference_time} ms")

        # Could use batch decode here but I want to process each one separately.
        for ix, output in enumerate(outputs):
            prediction = tokenizer.decode(output[len(batch["input_ids"][ix]):], skip_special_tokens=True)
            print(prediction)
            translations.append(prediction)
    

    with open("pred.txt", "w") as f:
        f.write("\n".join(translations))
                
if __name__ == "__main__":
    fire.Fire(main)
