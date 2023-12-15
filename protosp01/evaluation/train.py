from dataclasses import dataclass, field
from typing import List, Optional

import json
import os

from tqdm import tqdm
import pandas as pd
from datasets import load_dataset, Dataset
from accelerate import Accelerator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, PeftConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from api_key import HF_TOKEN
from prompt_template import PROMPT_TEMPLATES
from run import get_list_of_selections

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

def write_answer_extract(list_skills):
    # process list of extracted skills to write is as demonstration
    if len(list_skills) == 0:
        return "None"
    else:
        return "\n".join(list_skills)

def get_lm_prompt(example, args):
    prompt = ""
    instruction_field = 'all'
    instruction = PROMPT_TEMPLATES[instruction_field]['instruction'][args.prompt_type]
    
    question = "Sentence: " + str(example['sentence'])

    prompt = f"### Instruction: {instruction}\n\n### Sentence: {question}\n\n### Answer:"
    if args.train:
        if args.prompt_type == "extract":
            answer = write_answer_extract(example['list_extracted_skills'])
        else:
            answer = str(example['sentence_with_tags'])
        prompt += answer

    return prompt


def create_datasets(args):
    data_list = json.load(open(args.processed_data_dir + 'train.json'))
    data_list = [{'id': sample['id'], 'text': get_lm_prompt(sample, args)} for sample in data_list]

    dataset = Dataset.from_list(data_list)

    return dataset

def run_training(dataset, args):
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit
        )
        # Copy the model to each device
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    training_args = TrainingArguments(
        output_dir=args.ckpt_path,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        report_to=args.report_to,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )

    if args.use_peft:
        peft_config = LoraConfig(
            r=args.peft_lora_r,
            lora_alpha=args.peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules,
        )
    else:
        peft_config = None

    # if "llama" in args.model.lower():
    #     tokenizer = AutoTokenizer.from_pretrained(args.model, token=HF_TOKEN, use_fast=False)
    #     tokenizer.pad_token = tokenizer.eos_token
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # if getattr(tokenizer, "pad_token", None) is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map=device_map,
        max_memory=get_max_memory(),
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        token=HF_TOKEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    response_template_with_context = "\n### Answer:"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:] 
    print(response_template_ids)
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=args.seq_length,
        train_dataset=dataset,
        dataset_text_field=args.text_field,
        peft_config=peft_config,
        data_collator=data_collator
    )

    trainer.train()

    trainer.model.save_pretrained(os.path.join(args.ckpt_path, "final_checkpoint/"))

def run_inference(dataset, args):
    args.ckpt_path = os.path.join(args.ckpt_path, "final_checkpoint")
    # loading model and tokenizer
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit
        )
        # Copy the model to each device
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt_path,
        quantization_config=quantization_config,
        device_map=device_map,
        max_memory=get_max_memory(),
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        token=HF_TOKEN,
    )
    
    model = PeftModel.from_pretrained(model, args.ckpt_path)
        
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if os.path.exists(args.save_path) and args.start_from_saved:
        df = pd.read_json(args.save_path)
    else:
        df = pd.DataFrame(columns= list(dataset.columns) + ['model', 'prompt', 'model_output', 'list_of_selection'])
    print(f'saving to {args.save_path}')
        
    ids_all = dataset['id']
    ids_done = df['id']
    ids_left = list(set(ids_all) - set(ids_done))

    for id in tqdm(ids_left,total=len(ids_left)):
        index_sample = dataset[dataset['id'] == id].index[0]
        row = dataset.iloc[index_sample]
        row_to_save = {}
        for key, value in row.items():
            row_to_save[key] = value
        prompt = get_lm_prompt(row, args)
        with torch.autocast(model.device, dtype=torch.bfloat16):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,

            )
            model_output = tokenizer.decode(
                outputs[0][inputs['input_ids'].size(1):], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        
        list_of_selections = get_list_of_selections(model_output, row['tokens'], args.prompt_type)

        row_to_save['model'] = args.model
        row_to_save['prompt'] = prompt
        row_to_save['model_output'] = model_output
        
        row_to_save['list_of_selection'] = list_of_selections
        df.loc[len(df)] = row_to_save
        df.to_json(args.save_path, orient='records', indent=4, force_ascii=False)
    return