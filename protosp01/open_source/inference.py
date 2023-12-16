import argparse
import time
import os
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pathlib
import uuid
import ipdb
import sys

sys.path.append("..")
TASK_TARGET_MAP = {"dialogue": "final_turn", "summarization": "summary", "stance": "instance" , "intent": "intent", "safety": "action", "translation": "translation"}

from collections import defaultdict
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


MODELS_MAP = {
    "gpt2": "gpt2",
    "llama7": "/mnt/nlpdata1/share/models/llama_hf/7B",
    "llama13": "/mnt/nlpdata1/share/models/llama_hf/13B",
    "llama33": "/mnt/nlpdata1/share/models/llama_hf/33B",
    "alpaca": "/mnt/nlpdata1/share/models/alpaca_hf/7B",
    "t5": "t5-11b",
    "flan-t5": "google/flan-t5-xxl", #11B
    "flan-alpaca": "declare-lab/flan-alpaca-gpt4-xl",
    "vicuna": "/mnt/nlpdata1/share/models/vicuna_hf/13B",
    "stable-vicuna": "/mnt/nlpdata1/share/models/stable_vicuna_hf/13B",
    "mt0":"bigscience/mt0-xxl", # 13B
    "mt5":"google/mt5-xxl",
    "bloomz7": "bigscience/bloomz-7b1"
}


class ModelWrapper():
    def __init__(self, model_name, logger = None):
        self.letters_biases = defaultdict(lambda: 0)
        self.model_name = model_name
        self.logger = logger
        if model_name.startswith("llama") or model_name =="alpaca" or model_name == "bloomz7" or "vicuna" in model_name:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.engine = MODELS_MAP[model_name]
            self.model = AutoModelForCausalLM.from_pretrained(self.engine, device_map="auto")
            print("Model loaded: ", self.model.hf_device_map)
            self.tokenizer = AutoTokenizer.from_pretrained(self.engine)
            print("Tokenizer loaded")
            self.model.eval()
            self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")
        elif model_name == "gpt2":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.engine = MODELS_MAP[model_name]
            self.model = AutoModelForCausalLM.from_pretrained(self.engine).to(self.device)
            print("Model loaded: ", self.model.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.engine)
            print("Tokenizer loaded")
            self.model.eval()
            self.yes_token_id = self.tokenizer.get_vocab()["yes"]
            self.no_token_id = self.tokenizer.get_vocab()["no"]
        elif model_name in ["t5","flan-t5", "flan-alpaca", "mt0", "mt5"]:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.engine = MODELS_MAP[model_name]
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.engine, device_map="auto")
            print("Model loaded: ", self.model.hf_device_map)            
            self.tokenizer = AutoTokenizer.from_pretrained(self.engine)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Tokenizer loaded")
            self.model.eval()
            self.yes_token_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
            self.no_token_id = self.tokenizer.encode("no", add_special_tokens=False)[0]

    def get_model_name(self):
        return self.model_name
                    
    def generate_answer(self, prompt, max_tokens=1, temperature=0.5, top_k=20, top_p =0.95):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        outputs = self.model.generate(input_ids, max_new_tokens = max_tokens, temperature = temperature, top_p = top_p, do_sample = False, num_return_sequences=1, min_new_tokens = 1, early_stopping = True, return_dict_in_generate=True, output_scores=True)
        yes_logit_scores = outputs.scores[0][:,self.yes_token_id]
        no_logit_scores = outputs.scores[0][:,self.no_token_id]
        #output = "yes" if torch.softmax(no_logit_scores, dim=0) <  torch.softmax(yes_logit_scores, dim=0) else "no"
        proba_output = "yes" if no_logit_scores <  yes_logit_scores else "no"
        #output = decoded_output.split("Answer:")[-1][:3].strip()
        #print(output, yes_logit_scores, no_logit_scores)

        input_length = inputs.input_ids.shape[1]
        if self.model_name in ["t5","flan-t5", "flan-alpaca", "mt0", "mt5"]:
            generated_tokens = outputs.sequences[0]
        else:
            generated_tokens = outputs.sequences[:, input_length:][0]
        decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return proba_output, decoded_output

def generate_unique_id():
    return str(uuid.uuid4()).split("-")[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to evaluation data in json", required=True)
    parser.add_argument("--model", type=str, help="Model to use for evaluation", default="gpt2")
    parser.add_argument("--temperature", type=float, help="Temperature for generation", default=0.5)
    parser.add_argument("--max_tokens", type=int, help="Max tokens for generation", default=1)
    parser.add_argument("--top_p", type=float, help="Top p for generation", default=0.95)
    parser.add_argument("--output-dir", type=str, help="Output directory for evaluation results", default="outputs")
    parser.add_argument("--demo", type=str, help="Only on a small sample set", default="False")
    parser.add_argument("--demo_size", type=int, help="Size of the demo set", default=5)
    parser.add_argument("--oracle", type=str, help="Adding the CK to the prompt", default="False")
    
    args = parser.parse_args()
    data = read_json(args.datapath)
    if args.demo == "True":
        data = data[:args.demo_size]

    task = args.datapath.split("/")[0]

    cot = False
    if 'cot' in args.datapath:
        cot = True
        args.max_tokens = 100


    predictions = {"highest_proba": [],
                "generated_output": []}
    references = []
    scores_per_situation = {"highest_proba": {},
                "generated_output": {}}

    model = ModelWrapper(args.model)

    for sample in tqdm(data, total=len(data)):
        target = sample[TASK_TARGET_MAP[task]]
        knowledge_prompt = " None "
        if "knowledge" in target:
            for kg in target["knowledge"]:
                knowledge_prompt = kg["verbalized"]

        if args.oracle == "True" and "knowledge" in target:
            prompt =sample["prompt"].split("\nAnswer:")[0] + "\n Relevant knowledge : None. \n\nAnswer: "+  sample["prompt"].split("\nAnswer:")[1]
            sample["prompt"] = prompt +" Relevant knowledge : "+ knowledge_prompt + "\n Answer:"
        
        proba_output, decoded_output = model.generate_answer(sample["prompt"]+'.', temperature= args.temperature, max_tokens = args.max_tokens, top_p=args.top_p)
        ref = 1 if sample["answer"].strip().lower() == "yes" else 0
        references.append(ref)
        res_dict = {"highest_proba": proba_output,
                "generated_output": decoded_output}
        sample['cot'] = decoded_output

        for method, response in res_dict.items():
            label = 0
            if cot:
                match = re.search("<Answer>(?P<pred>.*)</Answer>", response)
                sample[method] = match["pred"].strip().lower() if match else response.strip().lower()
            else:
                sample[method] = response.strip().lower()
            # sample can be correct only if yes or no is generated
            if sample[method] in ["yes", "no"]:
                pred = 1 if response.strip().lower() == "yes" else 0
                sample["correct_"+method] = ref == pred
            else:
                pred=2
                sample["correct_"+method] = 0
            
            predictions[method].append(pred)
            if sample["correct_"+method] == True: 
                label = 1 
            else: 
                label  = 0 

            if sample["data_id"] in scores_per_situation[method]: 
                scores_per_situation[method][sample["data_id"]].append(label)
            else: 
                scores_per_situation[method][sample["data_id"]] = [label]
          
    
    outputs = {
        "metadata": {
            "datapath": args.datapath,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
        },
        "metrics_highest_proba": {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "macro-f1": 0
        },
        "metrics_generated_output": {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "macro-f1": 0
        },
        "data": data
    }

    outputs["metrics_generated_output"]["part_without_yes_no"] = sum([elem == 2 for elem in predictions["generated_output"]])/len(predictions["generated_output"])

    for method in ["highest_proba", "generated_output"]:
        for i in range(len(predictions[method])):
            if predictions[method][i] == 2:
                predictions[method][i] = 1 if references[i] == 0 else 0

        outputs["metrics_"+method]["accuracy"] = accuracy_score(references, predictions[method])
        outputs["metrics_"+method]["precision"] = precision_score(references, predictions[method])
        outputs["metrics_"+method]["recall"] = recall_score(references, predictions[method])
        outputs["metrics_"+method]["f1"] = f1_score(references, predictions[method])
        outputs["metrics_"+method]["macro-f1"] = f1_score(references, predictions[method], average="macro")
        count = 0
        for i in scores_per_situation[method]: 
            if 0 in scores_per_situation[method][i]:
                count +=1
        outputs["metrics_"+method]["exact_match"]= (len(scores_per_situation[method]) - count)/len(scores_per_situation[method])

    
        
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    datapath = pathlib.Path(args.datapath)
    if args.oracle == "True":
        output_path = os.path.join(args.output_dir, f"{datapath.stem}_{args.model}_oracle_{generate_unique_id()}.json")
    elif cot:
        output_path = os.path.join(args.output_dir, f"{datapath.stem}_{args.model}_cot_{generate_unique_id()}.json")
    else:
        output_path = os.path.join(args.output_dir, f"{datapath.stem}_{args.model}_{generate_unique_id()}.json")
    print(output_path)
    write_json(outputs, output_path)

if __name__ == "__main__":
    main()