import openai
import pandas as pd
from tqdm import tqdm
import ast
import ipdb
import numpy as np
import json
import os
import re
import random
import string
from demo_retrieval_utils import knn_demo_retrieval

random.seed(1234)
np.random.seed(1234)

# this section sets the necessary api key to use the openai api
import sys
from api_key import API_KEY
from prompt_template import PROMPT_TEMPLATES
openai.api_key = API_KEY

def write_answer_extract(list_skills):
    # process list of extracted skills to write is as demonstration
    if len(list_skills) == 0:
        return "None"
    else:
        return "\n".join(list_skills)


def get_knn_demonstrations(test_sentence_id, args):
    demos_files = json.load(open(args.processed_data_dir + 'train.json'))
    demos_ids = knn_demo_retrieval(test_sentence_id, args)
    demos = [sample for sample in demos_files if sample['id'] in demos_ids]
    return demos

def get_prompt(dataset, args, id, demos):
    instruction_field = 'all'
    instruction = PROMPT_TEMPLATES[instruction_field]['instruction'][args.prompt_type]
    # TODO have specify prompt template for all datasets
    messages = [{"role": "system", "content": instruction}]
    
    row_index = dataset[dataset['id'] == id].index[0]
    row = dataset.iloc[row_index]

    indexes = dataset['id'].values.tolist()
    indexes.remove(id)

    if args.knn:
        demos_knn = get_knn_demonstrations(row['id'], args)
        demos = demos_knn + demos
        random.shuffle(demos)

    for example in demos:
        question = "Sentence: " + str(example['sentence'])
        messages.append({"role": "user", "content": question})
        if args.prompt_type == 'extract':
            answer = write_answer_extract(example['list_extracted_skills'])
        else:
            answer = str(example['sentence_with_tags'])
        messages.append({"role": "assistant", "content": answer})

    question = "Sentence: " + row['sentence']
    messages.append({"role": "user", "content": question})

    return messages

def get_list_of_selections(model_output, sentence, prompt_type):
    if prompt_type == 'ner':
        return get_list_of_selections_ner(model_output, sentence)
    elif prompt_type == 'extract':
        return get_list_of_selections_extract(model_output, sentence)
    else:
        raise Exception('prompt type not supported')

def get_list_of_selections_extract(model_output, sentence):
    # model_output is a list of strings separated by \n. Sentence is the list of tokens of the original sentence.
    list_of_selections = ['O']*len(sentence)
    if "None" in model_output:
        return list_of_selections
    model_output = [str(item) for item in model_output.split('\n')]
    for skill in model_output:
        skill_tokens = skill.split()
        if skill_tokens[0] not in sentence:
            skill_tokens[0] = [token for token in sentence if skill_tokens[0] in token][0]
        skill_index = sentence.index(skill_tokens[0])
        list_of_selections[skill_index] = 'B'
        for i in range(1, len(skill_tokens)):
            list_of_selections[skill_index + i] = 'I'
    return list_of_selections

def get_list_of_selections_ner(model_output, sentence):
    model_output = postprocess_ner_prompt(sentence, model_output)
    list_of_selections = []
    model_output = model_output.split()
    in_span = False
    for token in model_output:
        if not in_span:
            if '@@' in token and '##' not in token:
                in_span = True
                list_of_selections.append("B")
                continue
            elif '@@' in token and '##' in token:
                list_of_selections.append("B")
                continue
            else:  
                list_of_selections.append("O")
                continue
        
        if in_span:
            if '##' in token:
                in_span = False
                list_of_selections.append("I")
                continue
            else:
                list_of_selections.append("I")
                continue

    return list_of_selections

def check_format_response(original, generated, prompt_type):
    """
    Check if the generated response is in the correct format. If not, return feedback to the user.
    """
    feedback = ''
    if prompt_type == 'ner':
        # check for missing words. TODO check for skills exact match inside tags?
        generated = postprocess_ner_prompt(original, generated)
        generated_clean = generated.replace('@@', ' ').replace('##', ' ')
        #generated_clean = generated_clean.translate(str.maketrans('', '', string.punctuation))
        #original = original.translate(str.maketrans('', '', string.punctuation))
        generated_words = generated_clean.split()
        original_words = original.split()
        if len(generated_words) != len(original_words):
            feedback = 'You didn\'t correctly replicate the given sentence. Make sure the sentence stays the same, even if there are no skills to highlight, including punctuation and spacing. Don\'t add any extra words or punctuation to the sentence except for the ## and @@ tags. Don\'t add nor remove any space.'
        # and that you highlight all the skills and competencies that are required from the candidate, by surrounding them with tags \'@@\' and \'##\'
    elif prompt_type == 'extract':
        if generated=="None":
            feedback = ''
        else:
            missing_skills = 0
            extracted_skills = generated.split('\n')
            for skill in extracted_skills:
                if skill not in original:
                    missing_skills += 1
            if missing_skills > 0:
                feedback = 'Some of the skills you extracted are not written the same way as in the sentence, or are absent from the sentence. Make sure to correctly replicate all skills in the input sentence, and provide them with one skill per line, without adding anything.'
    return feedback

def postprocess_ner_prompt(original, generated):
    # TODO correct this, it stays stuck in while loop
    # Compare with original and fix any errors in the generated string

    # It's okay if the sentence is not perfectly replicated, just need to check if all the sentence was covered (no words missing) and all extracted skills are the same as in the original sentence.
    # deal with added punctuation by the model at the end of the sentence
    if original[-1] not in string.punctuation and generated[-1] in string.punctuation:
        generated = generated[:-1]

    pattern = r"(\w)([.,!?;'])|([.,!?;'])(\w)"

    # add spaces around punctuation
    cleaned_generation = re.sub(pattern, r'\1 \2 \3 \4', generated)
    # remove duplicated spaces
    cleaned_generation = re.sub(r'\s+', ' ', cleaned_generation)
    

    """
    original_fixed = []
    generated_fixed = []
    original_idx = 0
    generated_idx = 0
    while original_idx < len(original) and generated_idx < len(generated):
        original_char = original[original_idx]
        generated_char = generated[generated_idx]
        
        # Check if the characters match
        if original_char == generated_char:
            original_fixed.append(original_char)
            generated_fixed.append(generated_char)
            original_idx += 1
            generated_idx += 1
        else:
            # Check if the generated character is a punctuation and if it has a space behind it in the original
            if generated_char == "#" or generated_char == "@":
                generated_fixed.append(generated_char)
                generated_idx += 1

            elif generated_char in punctuation_list and original_char == ' ' and original[original_idx + 1] == generated_char:
                generated_fixed.append(' ')
                generated_fixed.append(generated_char)
                original_fixed.append(original_char)
                original_fixed.append(original[original_idx + 1])
                generated_idx += 1
                original_idx += 2

            elif original_char == ' ' and original[original_idx - 1] == '(':
                generated_fixed.append(' ')
                original_idx += 1
            
            else:
                mismatch = True
                next
    
    # Append any remaining characters in the original and generated strings
    original_fixed.extend(original[original_idx:])
    generated_fixed.extend(generated[generated_idx:])
    
    # Join the fixed characters to create the fixed strings
    original_fixed_str = ''.join(original_fixed)
    generated_fixed_str = ''.join(generated_fixed)

    if len(original.split()) != len(generated_fixed_str.split()):
        mismatch = True
    """    
    return cleaned_generation


def run_openai(dataset, args):
    if os.path.exists(args.save_path) and args.sample == 0 and args.start_from_saved:
        df = pd.read_json(args.save_path)
    else:
        df = pd.DataFrame(columns= list(dataset.columns) + ['model', 'prompt', 'model_output', 'list_of_selection'])
    print(f'saving to {args.save_path}')
        
    ids_all = dataset['id']
    ids_done = df['id']
    ids_left = list(set(ids_all) - set(ids_done))

    # sample demos from train set
    demos_dataset = json.load(open(args.processed_data_dir + 'train.json'))
    demos_with_skills = [sample for sample in demos_dataset if len(sample['skill_spans']) > 0]
    demos_without_skills = [sample for sample in demos_dataset if len(sample['skill_spans']) == 0]
    demos = random.sample(demos_with_skills, args.shots) + random.sample(demos_without_skills, args.shots)
    random.shuffle(demos)
    
    for id in tqdm(ids_left,total=len(ids_left)):
        index_sample = dataset[dataset['id'] == id].index[0]
        row = dataset.iloc[index_sample]
        row_to_save = {}
        for key, value in row.items():
            row_to_save[key] = value
        messages = get_prompt(dataset, args, id, demos)
        response = openai.ChatCompletion.create(model=args.model, messages=messages, temperature=0)
        model_output = response['choices'][0]['message']['content']

        feedback = check_format_response(row['sentence'], model_output, args.prompt_type)
        trys_count = 0
        if feedback != '':
            print(feedback)
            print(row['sentence'])
            print(model_output)
            # If the model fails to generate the output correctly, try again up to 5 times
            # update the prompt with a new message targeting the specific issue
            while feedback != '' and trys_count < 5:
                print("Re-trying...", str(trys_count))
                messages.append({"role": "assistant", "content": model_output})
                messages.append({"role": "user", "content": feedback})
                response = openai.ChatCompletion.create(model=args.model, messages=messages, temperature=0)
                model_output = response['choices'][0]['message']['content']
                feedback = check_format_response(row['sentence'], model_output, args.prompt_type)
                print(model_output)
                trys_count += 1
        if trys_count == 5:
            continue
        else:
            list_of_selections = get_list_of_selections(model_output, row['tokens'], args.prompt_type)

            row_to_save['model'] = args.model
            row_to_save['prompt'] = messages
            row_to_save['model_output'] = model_output
            
            row_to_save['list_of_selection'] = list_of_selections
            df.loc[len(df)] = row_to_save
        df.to_json(args.save_path, orient='records', indent=4, force_ascii=False)