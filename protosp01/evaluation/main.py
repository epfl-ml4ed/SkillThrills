import pandas as pd
from tqdm import tqdm
import argparse
import os
import torch
import json
from preprocess import *
from run import run_openai
from train import run_training, run_inference, create_datasets
import random
from evaluate_src import *
from demo_retrieval_utils import embed_demo_dataset

random.seed(1234)

def download(args, split):
    dataset = load_skills_data(args.dataset_name, split)
    dataset.to_json(args.raw_data_dir + '/' + split + '.json', orient='records', indent=4, force_ascii=False)
    print(f'Saved {args.dataset_name} dataset to {args.raw_data_dir}, with {len(dataset)} examples.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='gnehm', help='Dataset name to use. Default is gnehm. Options are green, skillspan, fijo, sayfullina, kompetencer')
    parser.add_argument('--prompt_type', type=str, default='ner')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--raw_data_dir', type=str, default='../../data/annotated/raw/')
    # run parameters
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--knn', action='store_true', help='Use KNN retrieval instead of random sampling for demonstrations')
    parser.add_argument('--process', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--start_from_saved', action='store_true', help='Start from saved results instead of running inference again.')
    parser.add_argument('--exclude_empty', action='store_true', help='Exclude examples that have no skills in them.')
    parser.add_argument('--shots', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--sample', type=int, default=0, help='number of samples to perform inference on, for debugging.')
    parser.add_argument('--exclude_failed', action='store_true', help='whether to exclude previous failed attempt') 

    # train parameters
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default="ckpts/")
    parser.add_argument('--text_field', type=str, default="text", help="the text field of the dataset")
    parser.add_argument('--report_to', type=str, default="none", help="use 'wandb' to log with wandb")
    parser.add_argument('--learning_rate', type=float, default=1.41e-5, help="the learning rate")
    parser.add_argument('--batch_size', type=int, default=4, help="the batch size")
    parser.add_argument('--seq_length', type=int, default=2048, help="Input sequence length")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="the number of gradient accumulation steps")
    parser.add_argument('--load_in_8bit', action='store_true', help="load the model in 8 bits precision", default=False)
    parser.add_argument('--load_in_4bit', action='store_true', help="load the model in 4 bits precision", default=False)
    parser.add_argument('--use_peft', action='store_true', help="Whether to use PEFT or not to train adapters", default=False)
    parser.add_argument('--trust_remote_code', action='store_true', help="Enable `trust_remote_code`", default=False)
    parser.add_argument('--output_dir', type=str, default="output", help="the output directory")
    parser.add_argument('--peft_lora_r', type=int, default=64, help="the r parameter of the LoRA adapters")
    parser.add_argument('--peft_lora_alpha', type=int, default=16, help="the alpha parameter of the LoRA adapters")
    parser.add_argument('--logging_steps', type=int, default=25, help="the number of logging steps")
    parser.add_argument('--use_auth_token', action='store_true', help="Use HF auth token to access the model", default=True)
    parser.add_argument('--num_train_epochs', type=int, default=1, help="the number of training epochs")
    parser.add_argument('--max_steps', type=int, default=-1, help="the number of training steps")
    parser.add_argument('--save_steps', type=int, default=100, help="Number of updates steps before two checkpoint saves")
    parser.add_argument('--save_total_limit', type=int, default=10, help="Limits total number of checkpoints.")
    parser.add_argument('--mixed_precision', type=str, default="bf16", help="Mixed precision training")
    parser.add_argument('--target_modules', nargs='+', type=str, help="Target modules for LoRA adapters", default=None)
    
    args = parser.parse_args()
    if "gpt" in args.model:
        args.save_path = args.save_path + '/' + args.model + '/' + args.dataset_name + '_' + args.prompt_type + '_' + str(args.shots) + '-shots.json'
    else:
        args.save_path = args.save_path + '/' + args.model.split('/')[-1] + '/' + args.dataset_name + '_' + args.prompt_type + 'json'
    if args.knn:
        args.save_path = args.save_path.replace('.json', '_knn.json')
    args.raw_data_dir = args.raw_data_dir + args.dataset_name + '/'
    args.processed_data_dir = args.raw_data_dir.replace('raw', 'processed')
    args.embeddings_dir = args.raw_data_dir.replace('raw', 'embeddings')
    if not os.path.exists(args.processed_data_dir):
        os.makedirs(args.processed_data_dir)
    if not os.path.exists(args.embeddings_dir):
        os.makedirs(args.embeddings_dir)
    # if args.model not in ["gpt-3.5-turbo", "gpt-4"] or "Llama-2-7b" not in args.model:
    #     raise Exception("model not supported")
    
    if args.prompt_type == 'ner': # Gold is 'sentence_with_tags'
        args.gold_column = 'sentence_with_tags'
    elif args.prompt_type == 'extract': # Gold is 'list_extracted_skills'
        args.gold_column = 'list_extracted_skills'
    return args


def main():
    args = parse_args()
    
    # Download dataset if not already stored
    if not os.path.exists(args.raw_data_dir):
            os.makedirs(args.raw_data_dir)
    for split in ['train', 'test']:
        if not os.path.exists(args.raw_data_dir + '/' + split + '.json'):
            print(f'Downloading {args.dataset_name} dataset, {split} split...')
            download(args, split)

    # Process the dataset
    for split in ['train', 'test']:
        processed_path = args.processed_data_dir + split + '.json'
        if not os.path.exists(processed_path) or args.process:
            print(f'Processing {args.dataset_name} dataset...')
            dataset = preprocess_dataset(args, split)
    args.data_path = args.processed_data_dir + '/test.json'
    args.demo_path = args.processed_data_dir + '/train.json'

    if args.knn:
        # Embed the dataset (train for demos, and test)
        for split in ['train', 'test']:
            emb_save_path = args.embeddings_dir + '/' + split + '.pt'
            if not os.path.exists(emb_save_path):
                print(f'Generating {split} set embeddings for {args.dataset_name} dataset...')
                source_dataset = json.load(open(args.processed_data_dir + split + '.json'))
                if len(source_dataset) > 500 and split == 'train':
                    source_dataset = random.sample(source_dataset, 500)
                dataset_texts = [sample["sentence"] for sample in source_dataset]
                dataset_ids = [sample["id"] for sample in source_dataset]
                dataset_embed = embed_demo_dataset(dataset_texts, args.dataset_name)
                embeddings_dict = {'embeddings': dataset_embed, 'ids': dataset_ids}
                torch.save(embeddings_dict, emb_save_path)
                print(f'Saved {args.dataset_name} dataset embeddings to {emb_save_path}.')

    if args.run:
        # Load dataset
        dataset = pd.read_json(args.data_path)
        if args.exclude_empty:
            dataset['has_item'] = dataset.apply(lambda row: len(row['skill_spans'])>0, axis=1)
            dataset = dataset[dataset['has_item'] == True]
            dataset.drop(columns=['has_item'], inplace=True)
        print(f"Loaded {len(dataset)} examples from {args.dataset_name} dataset.")
        if len(dataset['id']) != len(set(dataset['id'].values.tolist())):
            raise Exception("The ids are not unique")
        # Run inference
        if args.sample != 0:
            sample_size = min(args.sample, len(dataset))
            dataset = dataset.sample(sample_size, random_state=1450).reset_index(drop=True)
        if "gpt" in args.model:
            run_openai(dataset, args)
        else:
            run_inference(dataset, args)
    
    if args.eval:
        print(f'Evaluating {args.save_path}...')
        all_metrics = eval(args.save_path)

    if args.train:
        if 'llama' in args.model:
            args.ckpt_path = args.ckpt_path + args.model.split('/')[-1] + '/'
        args.ckpt_path = args.ckpt_path + args.dataset_name + '_' + args.prompt_type + '/'
        dataset = create_datasets(args)
        run_training(dataset, args)
        

if __name__ == "__main__":
    main()