import pandas as pd
from tqdm import tqdm
import argparse
import os
from preprocess import *
from run import run_openai
from evaluate_src import *


def download(args):
    dataset = load_skills_data(args.dataset_name, args.split)
    dataset.to_json(args.data_path, orient='records', indent=4)
    print(f'Saved {args.dataset_name} dataset to {args.data_path}, with {len(dataset)} examples.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--exclude_empty', action='store_true', help='Exclude examples that have no skills in them.')
    parser.add_argument('--sample_longest', action='store_true', help='Sample only the longest examples.')
    parser.add_argument('--dataset_name', default='gnehm', help='Dataset name to use. Default is gnehm.Options are green, skillspan, fijo, sayfullina, kompetencer')
    parser.add_argument('--shots', type=int, default=1)
    parser.add_argument('--prompt_type', type=str, default='ner')
    parser.add_argument('--split', type=str, default='test', help='Train, test or validation split to use for the dataset. Default is test.')
    parser.add_argument('--data_path', type=str, default='../../data/annotated/raw/')
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--sample', type=int, default=10, help='number of samples to perform inference on, for debugging.')

    args = parser.parse_args()
    args.save_path = args.save_path + args.dataset_name + '_' + args.model + '_' + args.prompt_type + '_' + str(args.shots) + '-shots.json'
    args.data_path = args.data_path + args.dataset_name + '.json'
    if args.model not in ["gpt-3.5-turbo", "gpt-4"]:
        raise Exception("model not supported")
    return args


def main():
    args = parse_args()
    
    # Download dataset if not already stored
    if not os.path.exists(args.data_path):
        download(args)
    
    if args.run:
        # Process the dataset
        dataset = preprocess_dataset(args)
        print(f"Loaded {len(dataset)} examples from {args.dataset_name} dataset.")

        if len(dataset['id']) != len(set(dataset['id'].values.tolist())):
            raise Exception("The ids are not unique")
        # Run inference
        run_openai(dataset, args)
    
    if args.eval:
        eval(args)
        

if __name__ == "__main__":
    main()