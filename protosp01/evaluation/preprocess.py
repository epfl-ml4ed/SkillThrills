from datasets import load_dataset
import pandas as pd
import ipdb

# this method just loads the dataset and drops the pos column
# can be used to get other splits of the dataset as well
def load_skills_data(dataset_name, split='test'):
    dataset_name = "jjzha/" + dataset_name
    dataset = load_dataset(dataset_name)
    dataset = pd.DataFrame(dataset[split])
    try:
        # for gnehm dataset
        dataset.drop(columns=['pos', 'idx'], inplace=True)
    except:
        pass
    dataset['idx'] = dataset.index
    dataset.rename(columns={'idx': 'id'}, inplace=True)
    return dataset

# this method fixes the skills column
# For the moment, it drops the occupation and domain tags
# and converts ICT (different in other datasets) tags to boolean values
def fix_the_skills_column_per_row(row, dataset_name):
    tags_skill = row['tags_skill']
    # THIS LINE MAY NEED TO BE CHANGED TO APPLY TO ANOTHER DATASET
    if dataset_name == 'gnehm':
        fixed_tags_skills = list(map(lambda item: True if "ICT" in item else False, tags_skill))
    if dataset_name == 'skillspan':
        fixed_tags_skills = list(map(lambda item: True if item in ['B', 'I'] else False, tags_skill))
    return fixed_tags_skills

# applies the fix_the_skills_column_per_row method to the whole dataset
def fix_the_skills_column(dataset, dataset_name):
    dataset['tags_skill_boolean'] = dataset.apply(lambda x: fix_the_skills_column_per_row(x, dataset_name), axis=1)
    return dataset

def sample_longest_examples(dataset, sample_size):
    dataset = dataset.sort_values(by='tokens', key=lambda col: col.str.len(), ascending=False)
    if sample_size != 0:
        dataset = dataset[:sample_size]
    return dataset


# adds beginning and end tags to the words that are tagged as skills
def add_tags_to_words(words, boolean_values, begin_tag='@@', end_tag='##'):
    result = []
    in_span = False

    for word, is_true in zip(words, boolean_values):
        if is_true:
            if not in_span:
                result.append(begin_tag)
                in_span = True
            result.append(word)
        else:
            if in_span:
                result.append(end_tag)
                in_span = False
            result.append(word)

    if in_span:
        result.append(end_tag)

    return result


# calls the add_tags_to_words method on the whole dataset
def add_golden_answer_column(dataset, prompt_type):
    if prompt_type == 'ner':
        dataset['tokens_with_tags'] = dataset.apply(lambda row: add_tags_to_words(row['tokens'], row['tags_skill_boolean']), axis=1)
        dataset = concat_tokens(dataset)
    elif prompt_type == 'extract':
        raise Exception("Not implemented yet")
    return dataset


# just a concatenation method on the list of tokens
def concat_tokens(dataset):
    dataset['sentence'] = dataset.apply(lambda row: ' '.join(row['tokens']), axis=1)
    dataset['sentence_with_tags'] = dataset.apply(lambda row: ' '.join(row['tokens_with_tags']), axis=1)
    dataset['sentence_with_tags'] = dataset.apply(lambda row: row['sentence_with_tags'].replace('@@ ', '@@'), axis=1)
    dataset['sentence_with_tags'] = dataset.apply(lambda row: row['sentence_with_tags'].replace(' ##', '##'), axis=1)
    return dataset


def drop_long_examples(dataset, max_length=512):
    dataset = dataset[dataset['tokens'].map(len) < max_length]
    return dataset

# this method performs all the required preprocessing on the dataset
def preprocess_dataset(args):
    dataset = pd.read_json(args.data_path)
    
    dataset = fix_the_skills_column(dataset, args.dataset_name)
    dataset = drop_long_examples(dataset)

    # filter only the examples that have at least one skill
    if args.exclude_empty:
        dataset['has_item'] = dataset.apply(lambda row: any(row['tags_skill_boolean']), axis=1)
        dataset = dataset[dataset['has_item'] == True]
        dataset.drop(columns=['has_item'], inplace=True)

    if args.sample_longest:
        dataset = sample_longest_examples(dataset, args.sample)
    else:
        dataset = dataset.sample(args.sample, random_state=1450).reset_index(drop=True)
    
    dataset = dataset.sample(frac = 1, random_state=1450).reset_index(drop=True)
    dataset = add_golden_answer_column(dataset, args.prompt_type)
   
    return dataset
    