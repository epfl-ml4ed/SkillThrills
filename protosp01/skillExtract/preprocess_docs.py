# %%
import pandas as pd
import argparse
import os
import tiktoken

from prompt_template import PROMPT_TEMPLATES
from utils import *

# fmt: off

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="Language to keep", default="all")
    parser.add_argument("--input_dir", type=str, help="Input directory", default="../data/raw")
    parser.add_argument("--output_dir", type=str, help="Output directory", default="../data/processed")
    parser.add_argument("--model", type=str, help="Model to use for generation", default="gpt-3.5-turbo")
    parser.add_argument("--datatype", type=str, help="Type of data to process", default="job_evl")
    
    args = parser.parse_args()
    # fmt: on
    encoding = tiktoken.encoding_for_model(args.model)

    def longest_sentence(sentences):
        return max(sentences, key=len)


    def num_tokens_from_string(stringl):
        return len(encoding.encode(stringl))



    # %%
    # data loading from json
    if args.datatype == "job_evl":
        datapath = os.path.join(args.input_dir, "vacancies.json")
    elif args.datatype == "course_evl":
        datapath = os.path.join(args.input_dir, "learning_opportunities.json")
    else:
        raise ValueError(
            "Invalid datatype, only 'job_evl' and 'course_evl' are supported now"
        )

    data = read_json(datapath)[0]
    data = pd.DataFrame.from_records(data)

    # %%
    # create "fulltext" column we are interested in
    if args.datatype == "job_evl":
        data["fulltext"] = data["name"] + "\n" + data["description"]
        print("num total jobs:", len(data))

    elif args.datatype == "course_evl":
        data = data[data["active"] == True]
        keep_ids = {1, 5, 9}
        data = data[data["study_ids"].apply(lambda x: bool(set(x) & keep_ids))]

        print("num total courses with ids 1,5,9:", len(data))

        # drop cols that are not needed
        drop_cols = [
            "type",
            "active",
            "pricing_description",
            "ects_points",
            "average_effort_per_week",
            "total_effort",
            "application_process_description",
            "required_number_years_of_experience",
            "certificate_type",
            "currency",
            "pricing_type",
            "transaction_type",
        ]
        data.drop(columns=drop_cols, inplace=True)

        # to keep an indicator for the type of skill (to acquire or prereq), we will run the skill extraction/matching pipeline twice
        acq_data = data.copy()
        acq_data["fulltext"] = (
            acq_data["name"]
            + acq_data["intro"].fillna("")
            + acq_data["key_benefits"].fillna("")
            + acq_data["learning_targets_description"].fillna("")
            + acq_data["structure_description"].fillna("")
        )
        acq_data["skill_type"] = "to_acquire"

        req_data = data.copy()
        req_data["fulltext"] = req_data["admission_criteria_description"].fillna(
            ""
        ) + req_data["target_group_description"].fillna("")
        req_data["skill_type"] = "required"

        data = pd.concat([acq_data, req_data], ignore_index=True)
        # replace every 10 tags with a period to avoid too long sentences


    # detect language
    data["language"] = data["fulltext"].apply(detect_language)
    print(data["language"].value_counts())

    # heuristics to clean fulltext
    data["fulltext"] = data["fulltext"].apply(replace_html_tags)
    data = drop_short_text(data, "fulltext", 100)

    # %%
    # get tokens of each sentence and filter out rows with too long sentences
    language = "de" if args.language == "all" else args.language
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
    output_path = os.path.join(args.output_dir, f"{args.datatype}_{args.language}.csv")
    data.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
