import pandas as pd
import argparse
import openai
import time
from openai.error import (
    RateLimitError,
    ServiceUnavailableError,
    APIError,
    APIConnectionError,
)
import os
from tqdm import tqdm
import json
import numpy as np
import ipdb
import random
import pathlib
import re
import tiktoken
import difflib
from split_words import Splitter

from prompt_template import PROMPT_TEMPLATES
from utils import *

# fmt: off

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--datapath", type=str, help="Path to source data", default = "platform_data/vacancies")
    parser.add_argument("--datapath", type=str, help="Path to source data", default = "../../data/raw/vacancies.json")
    # parser.add_argument("--taxonomy", type=str, help="Path to taxonomy file in csv format", default = "taxonomy_files/taxonomy_V4.csv")
    parser.add_argument("--taxonomy", type=str, help="Path to taxonomy file in csv format", default = "../../data/taxonomy/taxonomy_V4.csv")
    parser.add_argument("--openai_key", type=str, help="openai keys", default = API_KEY)
    parser.add_argument("--model", type=str, help="Model to use for generation", default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, help="Temperature for generation", default=0.3)
    parser.add_argument("--max_tokens", type=int, help="Max tokens for generation", default=40)
    parser.add_argument("--top_p", type=float, help="Top p for generation", default=1)
    parser.add_argument("--frequency_penalty", type=float, help="Frequency penalty for generation", default=0)
    parser.add_argument("--presence_penalty", type=float, help="Presence penalty for generation", default=0)
    parser.add_argument("--output_path", type=str, help="Output for evaluation results", default="../results/")
    parser.add_argument("--num-samples", type=int, help="Last N elements to evaluate (the new ones)", default=0)
    parser.add_argument("--do-extraction", action="store_true", help="Wether to do the extraction or directly the matching")
    parser.add_argument("--do-matching", action="store_true", help="Wether to do the matching or not")
    parser.add_argument("--debug", action="store_true", help="Keep only one sentence per job offer / course to debug")
    parser.add_argument("--detailed", action="store_true", help="Generate detailed output")
    parser.add_argument("--ids", type=str, help="Path to a file with specific ids to evaluate", default=None)
    
    # fmt: on


    args = parser.parse_args()
    if args.datapath.split('/')[-1]=='vacancies.json':
        args.data_type = 'job'
    elif args.datapath.split('/')[-1]=='learning_opportunities.json':
        args.data_type = 'course'
    else:
        print('Error: Data source unknown')

    args.api_key = API_KEY #args.openai_key
    args.output_path = args.output_path + args.data_type + '_' + args.model + '.json'
    print("Output path", args.output_path)

    if args.ids is not None:
        with open(args.ids, 'r') as f:
            ids = f.read().splitlines()
        if 'vacancies' in ids[0]:
            args.data_type = 'job'
        elif 'learning_opportunities' in ids[0]:
            args.data_type = 'course'
        ids = [int(id.split('/')[-1]) for id in ids]
        print("Evaluating only ids:", ids)
        args.output_path = args.output_path.replace('.json', '_ids.json')

    # Load data
    if args.num_samples > 0:
        data = read_json(args.datapath, lastN=args.num_samples)
        data = data[0][-args.num_samples:]
    else:
        data = read_json(args.datapath)[0]
    
    data = pd.DataFrame.from_records(data)
    if args.data_type == 'job':
        data['fulltext'] = data['name'] + ".\n" + data['description']
    elif args.data_type == 'course':
        data = data[data['active']==True]
        data['fulltext'] = data['learning_targets_description'].fillna('') + data['name'] +data['key_benefits'].fillna('') + data['intro'].fillna('')
    # TODO select best columns for each data type
    # TODO filter the ones with too small descriptions
    # TODO merge sentences if they are too short

    # get number of words in each description
    # data['desc_len'] = data['description'].apply(lambda x: len(x.split()))

    if args.ids is not None:
        data = data[data['id'].isin(ids)]
    else:
        # apply language detection
        data['language'] = data['fulltext'].apply(detect_language)
        print(data['language'].value_counts())
        data = data[data['language']=='de']

    print("loaded data:", len(data), "elements")
    data = data.to_dict('records')

    

    # We create two files:
    # 1. results_detailed.json: contains a list of jobs/courses ids
    # each job / course has a list of sentence, each sentence has all extraction details
    # 2. results_clean.json: contains a list of jobs/courses ids
    # each job / course has only a list of skills, certifications, languages, technologies
    extraction_cost = 0
    matching_cost = 0
    detailed_results_dict = {}
    for i, item in tqdm(enumerate(data)): # item is job or course in dictionary format
        sentences = split_sentences(item['fulltext'])
        if args.debug:
            #sentences = [sent for sent in sentences if len(sent.split())<80]
            #if len(sentences)==0:
            #    continue
            sentences = [random.choice(sentences)]
        sentences_res_list = [{"sentence": sentence} for sentence in sentences]
        if len(sentences_res_list)==0:
            continue
        
        # extract skills
        if args.do_extraction:
            print("Starting extraction")
            api = OPENAI(args, sentences_res_list)
            sentences_res_list, cost = api.do_prediction("extraction")
            extraction_cost += cost

        # load taxonomy
        taxonomy, skill_names, skill_definitions = load_taxonomy(args)

        # select candidate skills from taxonomy
        if 'extracted_skills' in sentences_res_list[0]:
            splitter = Splitter()
            max_candidates = 10
            for i, sample in enumerate(sentences_res_list):
                sample = select_candidates_from_taxonomy(sample, taxonomy, skill_names, skill_definitions, splitter, max_candidates)
                sentences_res_list[i] = sample

        # match skills with taxonomy
        if args.do_matching and 'skill_candidates' in sentences_res_list[0]:
            print("Starting matching")
            api = OPENAI(args, sentences_res_list)
            sentences_res_list, cost = api.do_prediction("matching")
            matching_cost += cost
            # TODO remove "None"
            # TODO remove duplicates
            # TODO output only level 2 or rather skill id?

        # Do exact match with technologies, languages, certifications
        tech_certif_lang = pd.read_csv('../../data/taxonomy/tech_certif_lang.csv')
        tech_alternative_names = pd.read_csv('../../data/taxonomy/technologies_alternative_names.csv', sep='\t')
        certification_alternative_names = pd.read_csv('../../data/taxonomy/certifications_alternative_names.csv', sep='\t')
        sentences_res_list = exact_match(sentences_res_list, tech_certif_lang, tech_alternative_names, certification_alternative_names)
        # TODO find a way to correctly identify even common strings (eg 'R')!
        # Idem for finding C on top of C# and C++
        # Also issue with .Net not found (because it's written Microsoft .NET Framework)
        # and SQL not found (because it's written SQL Server)
        # one thing to do: find the element only if there's a space before or after or both
        # + propose alternative names for all technologies
        # same for "Microsoft Azure software" --> Azure not found
        
        detailed_results_dict[item['id']] = sentences_res_list
    
    if args.debug:
        args.output_path = args.output_path.replace('.json', '_debug.json')
    if args.detailed:
        write_json(detailed_results_dict, args.output_path.replace('.json', '_detailed.json'))
        
    # Output final
    if not args.debug:
        categs = ['Technologies', 'Technologies_alternative_names', 'Certifications', 'Certification_alternative_names','Languages']
        clean_output_dict = {}
        for item_id, detailed_res in detailed_results_dict.items():
            clean_output = {categ: [] for categ in categs}
            clean_output['skills'] = []
            for i, sample in enumerate(detailed_res):
                for cat in categs:
                    clean_output[cat].extend(sample[cat])
                if 'matched_skills' in sample:
                    for skill in sample['matched_skills']:
                        clean_output['skills'].append(sample['matched_skills'][skill])
                        # TODO output Level 2 or id!
            clean_output_dict[item_id] = clean_output
        write_json(clean_output_dict, args.output_path.replace('.json', '_clean.json'))
    print("Done")
    print("Extraction cost ($):", extraction_cost)
    print("Matching cost ($):", matching_cost)
    print("Total cost ($):", extraction_cost + matching_cost)

    if args.detailed:
        print("Saved detailed results in", args.output_path.replace('.json', '_detailed.json'))
    print("Saved clean results in", args.output_path.replace('.json', '_clean.json'))



if __name__ == "__main__":
    main()
