import openai
import pandas as pd
from tqdm import tqdm
import ast
import ipdb
import os
from numpy import random
import string

# this section sets the necessary api key to use the openai api
import sys
from api_key import API_KEY
from prompt_template import PROMPT_TEMPLATES
openai.api_key = API_KEY

def write_answer_extract(list_skills):
    # process list of extracted skills to write is as demonstration
    if len(list_skills) == 0:
        return "None"
    answer = ""
    for skill in list_skills:
        answer += "\n".join(skill)
    return answer

def get_prompt(dataset, args, id):
    instruction_field = args.dataset_name if args.dataset_name == 'gnehm' else 'all'
    
    instruction = PROMPT_TEMPLATES[instruction_field]['instruction'][args.prompt_type]
    # TODO have a common prompt template for all datasets
    prompt = [{"role": "system", "content": instruction}]
    
    row_index = dataset[dataset['id'] == id].index[0]
    row = dataset.iloc[row_index]

    indexes = dataset['id'].values.tolist()
    indexes.remove(id)

    ids_of_shots = random.choice(indexes, args.shots, replace=False)
    ids_of_shots = list(ids_of_shots) 
    # TODO take shots from train set instead of test set

    content = ""
    for id_example in ids_of_shots:
        row_example_index = dataset[dataset['id'] == id_example].index[0]
        row_example = dataset.iloc[row_example_index]
        if isinstance(row_example[args.gold_column], list):
            answer = write_answer_extract(row_example[args.gold_column])
        else:
            answer = str(row_example[args.gold_column])

        content += "Sentence: " + str(row_example['sentence']) + "\n"
        content += "Answer: " + str(answer) + "\n"

    content += "Sentence: " + row['sentence'] + "\n"
    content += "Answer: "
    # print(content)
    # input('press enter to continue')
    prompt.append({"role": "user", "content": content})
    # TODO change to user / system with multiple messages
    return prompt, ids_of_shots

def get_list_of_selections(model_output, sentence, prompt_type):
    if prompt_type == 'ner':
        return get_list_of_selections_ner(model_output)
    elif prompt_type == 'extract':
        return get_list_of_selections_extract(model_output, sentence)
    else:
        raise Exception('prompt type not supported')

def get_list_of_selections_extract(model_output, sentence):
    # model_output is a list of strings. Sentence is the list of tokens of the original sentence.
    list_of_selections = ['O']*len(sentence)
    if "None" in model_output:
        return list_of_selections
    for skill in model_output:
        skill_tokens = skill.split()
        if skill_tokens[0] not in sentence:
            print(model_output, sentence)
            continue
        skill_index = sentence.index(skill_tokens[0])
        list_of_selections[skill_index] = 'B'
        for i in range(1, len(skill_tokens)):
            list_of_selections[skill_index + i] = 'I'
    return list_of_selections

def get_list_of_selections_ner(model_output):
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


def postprocess(original, generated, prompt_type):
    if prompt_type == 'ner':
        return postprocess_ner_prompt(original, generated)
    elif prompt_type == 'extract':
        return postprocess_extract_prompt(generated)
    else:
        raise Exception('prompt type not supported')

def postprocess_extract_prompt(generated):
    # it should be a list of strings. Deal with exceptions and formatting errors.
    if "None" in generated:
        return generated
    generated = [str(item) for item in generated.split('\n')]
    return generated

def postprocess_ner_prompt(original, generated):
    # Compare with original and fix any errors in the generated string
    punctuation_list = [',', '.', '/', '|', '?', ')', '(']
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
                # generated_fixed.append(generated_char)
                # original_fixed.append(original_char)
                # generated_idx += 1
                # original_idx += 1
            
            else:
                print(original)
                print(''.join(original_fixed))
                print('-'*10)
                print(generated)
                print(''.join(generated_fixed))

                print('letter mismatch')
                raise Exception('not fixable')
    
    # Append any remaining characters in the original and generated strings
    original_fixed.extend(original[original_idx:])
    generated_fixed.extend(generated[generated_idx:])
    
    # Join the fixed characters to create the fixed strings
    original_fixed_str = ''.join(original_fixed)
    generated_fixed_str = ''.join(generated_fixed)


    # all_punctuations = list(string.punctuation)
    # original_test = original_fixed
    # generated_test = generated_fixed

    # for punc in all_punctuations:
    #     original_test = original_test.replace(punc, '')
    #     generated_test = generated_test.replace(punc, '')

    if len(original.split()) != len(generated_fixed_str.split()):
        print(original)
        print(original_fixed_str)
        print(len(original.split()))
        print('-'*10)
        print(generated)
        print(generated_fixed_str)
        print(len(generated_fixed_str.split()))
        print('length mismatch')
        raise Exception('not fixable')    
    
    return generated_fixed_str


def run_openai(dataset, args):
    # this code is only ok for gnehm at the moment
    if os.path.exists(args.save_path) and args.sample == 0:
        df = pd.read_json(args.save_path)
    else:
        df = pd.DataFrame(columns= list(dataset.columns) + ['model', 'shots', 'ids_of_shots', 'model_output', 'list_of_selection'])
    print(f'saving to {args.save_path}')
        
    ids_all = dataset['id']
    ids_done = df['id']
    ids_left = list(set(ids_all) - set(ids_done))

    trys_count = 0
    i = 0

    progressbar = tqdm(total=len(ids_left))
    while i < len(ids_left):
        try:
            id = ids_left[i]
            index_sample = dataset[dataset['id'] == id].index[0]
            row = dataset.iloc[index_sample]
            row_to_save = {}
            for key, value in row.items():
                row_to_save[key] = value
            prompt, ids_of_shots = get_prompt(dataset, args, id)
            response = openai.ChatCompletion.create(model=args.model, messages=prompt, temperature=0)
            model_output = response['choices'][0]['message']['content']
            model_output = postprocess(row['sentence'], model_output, args.prompt_type)
            # len_original = len(row['sentence'].split())
            # len_output = len(model_output.split())
            # if len_original != len_output:
            #     continue
                
            list_of_selections = get_list_of_selections(model_output, row['tokens'], args.prompt_type)

            row_to_save['model'] = args.model
            row_to_save['shots'] = args.shots
            row_to_save['ids_of_shots'] = ids_of_shots
            row_to_save['model_output'] = model_output
            
            row_to_save['list_of_selection'] = list_of_selections
            df.loc[len(df)] = row_to_save
            i += 1
            trys_count = 0
            progressbar.update(1)
            df.to_json(args.save_path, orient='records', indent=4, force_ascii=False)
        
        # If the model fails to generate the output correctly, try again up to 5 times
        # TODO update the prompt as a new message targeting the specific issue!
        except Exception as e:
            text = str(e)
            trys_count += 1
            print(text)
            print('$'*50)
            print(f'number of trys: {trys_count}')
            print('$'*50)
            if trys_count == 5:
                i += 1
                trys_count = 0
            continue


            # if text == 'lengths are not equal':
            #     print('lengths are not equal')
            #     continue
            # else:
            #     continue
        # 