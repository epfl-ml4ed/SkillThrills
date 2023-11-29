#%%
import pandas as pd
import argparse
import os
import tiktoken
import codecs
import json
import re
from utils import *

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, help="Language to keep", default="all")
parser.add_argument("--input_path", type=str, help="Input directory", default="../data/raw/coco_courses/courses_latest.csv")
parser.add_argument("--output_dir", type=str, help="Output directory", default="../data/processed")
parser.add_argument("--model", type=str, help="Model to use for generation", default="gpt-3.5-turbo")

args = parser.parse_args([])

# fmt: on
def longest_sentence(sentences):
    return max(sentences, key=len)


def num_tokens_from_string(stringl):
    encoding = tiktoken.encoding_for_model(args.model)
    return len(encoding.encode(stringl))


def decode_unicode(text):
    return codecs.decode(text, "unicode_escape")

def clean_non_ascii(text):
    pattern = re.compile("[^\x00-\x7F]+")
    return pattern.sub("", text)

def convert_to_list(value):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value

def join_strings(x):
    if isinstance(x, list):
        return " ".join(x)
    elif isinstance(x, str):
        return x
    else:
        return ""
    
    
def main():

    # %%

    data = pd.read_csv("../data/raw/coco_courses/course_latest.csv", index_col=0, encoding="utf-8")

    # %% 
    # drop non-IT/mgmt courses
    first_level_certain = ['development', 'it-and-software', 'office-productivity']
    second_level_to_add = [
        "project-management",
        "data-and-analytics",
        "web-design",
        "user-experience",
        "search-engine-optimization",
        "analytics-and-automation",
    ]

    df_1 = data[data["first_level_category"].isin(first_level_certain)]
    df_2 = data[data["second_level_category"].isin(second_level_to_add)]

    data = pd.concat([df_1, df_2], ignore_index=True)

    #%%
    # Apply the conversion function to multiple columns in the DataFrame
    text_cols = ["short_description", "long_description", "objectives", "requirements", "target_audience"]
    for col in text_cols:
        data[col] = data[col].apply(convert_to_list)

    for col in text_cols:
        data[col] = data[col].apply(join_strings)

    #%%
    acq_data = data.copy()
    acq_data["fulltext"] = (
        acq_data["short_description"].fillna("")
        + acq_data["long_description"].fillna("")
        + acq_data["objectives"].fillna("")
    )
    acq_data["skill_type"] = "to_acquire"

    req_data = data.copy()
    req_data["fulltext"] = req_data["requirements"].fillna(
        ""
    ) + req_data["target_audience"].fillna("")
    req_data["skill_type"] = "required"

    # create data with both required and to acquire skills sorted by course_id
    data = pd.concat([acq_data, req_data], ignore_index=True)
    data.sort_values(by=["course_id", "skill_type"], inplace=True)

    #%%
    # heuristics to clean fulltext
    data["fulltext"] = data["fulltext"].apply(replace_html_tags)
    data["fulltext"] = data["fulltext"].apply(decode_unicode)
    data["fulltext"] = data["fulltext"].apply(clean_non_ascii)

    def num_words_from_string(stringl):
        return len(stringl.split())
    
    # find which rows have fulltext < 100 words
    print("num rows before dropping short text:", len(data))
    print("num rows with short text:", len(data[data["fulltext"].apply(num_words_from_string) < 50]))

    data = drop_short_text(data, "fulltext", 50)

    # %%
    language_dict = {
        'italian': 'it',
        'english': 'en',
        'spanish': 'es',
        'portuguese': 'pt',
        'french': 'fr',
        'turkish': 'tr',
        'polish': 'pl',
        'german': 'de',
        'arabic': 'ar',
        'russian': 'ru',
        'korean': 'ko',
        'hindi': 'hi',
        'vietnamese': 'vi',
        'dutch': 'nl',
        'chinese': 'zh',
        'indonesian': 'id',
        'urdu': 'ur',
        'persian': 'fa',
        'japanese': 'ja',
        'thai': 'th',
        'marathi': 'mr',
        'croatian': 'hr',
        'greek': 'el',
        'telugu': 'te',
        'serbian': 'sr',
        'estonian': 'et',
        'czech': 'cs',
        'ukranian': 'uk',
        'malay': 'ms',
        'azeri': 'az',
        'hungarian': 'hu',
        'norwegian': 'no',
        'afrikaans': 'af',
        'bulgarian': 'bg',
        'hebrew': 'he',
    }

    # replace the language column with the language code
    data["language"] = data["language"].apply(lambda x: language_dict[x])
    # rename course_id to id
    data.rename(columns={"course_id": "id"}, inplace=True)

    #%%
    language = "en" if args.language == "all" else args.language

    # get tokens of each sentence and filter out rows with too long sentences
    data["sentences"] = data["fulltext"].apply(split_sentences, language=language)
    data["longest_sentence"] = data["sentences"].apply(longest_sentence)
    data["num_tokens"] = data["longest_sentence"].apply(num_tokens_from_string)

    data = data[data["num_tokens"] <= 1500]

    drop_new_cols = ["sentences", "longest_sentence", "num_tokens"]
    data.drop(columns=drop_new_cols, inplace=True)

    # %%
    if args.language != "all":
        data = data[data["language"] == args.language]
        print(f"num rows after filtering by language ({args.language}):", len(data))

    # save data to disk
    output_path = os.path.join(args.output_dir, f"course_coco_{args.language}.csv")
    data.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
