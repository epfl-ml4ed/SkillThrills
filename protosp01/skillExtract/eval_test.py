# %%
import pandas as pd
import os
import sys
import argparse

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

# %%
from utils import *

# fmt: off

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, help="Path to source data", default = "../data/annotated/CVTest_V4.csv")
parser.add_argument("--taxonomy", type=str, help="Path to taxonomy file in csv format", default = "../data/taxonomy/taxonomy_V4.csv")
parser.add_argument("--api_key", type=str, help="openai keys", default = API_KEY)
parser.add_argument("--model", type=str, help="Model to use for generation", default="gpt-3.5-turbo")
parser.add_argument("--temperature", type=float, help="Temperature for generation", default=0)
parser.add_argument("--max_tokens", type=int, help="Max tokens for generation", default=40)
parser.add_argument("--top_p", type=float, help="Top p for generation", default=1)
parser.add_argument("--frequency_penalty", type=float, help="Frequency penalty for generation", default=0)
parser.add_argument("--presence_penalty", type=float, help="Presence penalty for generation", default=0)
parser.add_argument("--output_path", type=str, help="Output for evaluation results", default="results/")
# # parser.add_argument("--num-samples", type=int, help="Last N elements to evaluate (the new ones)", default=10)
parser.add_argument("--num-sentences", type=int, help="by how many sentences to split the corpus", default=2)
parser.add_argument("--do-extraction", action="store_true", help="Whether to do the extraction or directly the matching")
parser.add_argument("--do-matching", action="store_true", help="Whether to do the matching or not")
parser.add_argument("--word-emb-model", type=str, help="Word embedding model to use", default="agne/jobBERT-de")
# # parser.add_argument("--debug", action="store_true", help="Keep only one sentence per job offer / course to debug")
# # parser.add_argument("--detailed", action="store_true", help="Generate detailed output")
# # parser.add_argument("--ids", type=str, help="Path to a file with specific ids to evaluate", default=None)

# fmt: on

args = parser.parse_args()

data_type = "cv"
args.output_path = args.output_path + data_type + "_" + args.model + ".json"

taxonomy, _, _ = load_taxonomy(args)
data = pd.read_csv(args.datapath, sep=";", encoding="iso-8859-1")
print("loaded data:", len(data), "sentences")


data["Sentence"] = data["Sentence"].apply(clean_text)

sentences_col = data["Sentence"]
sentences_col = sentences_col[sentences_col.apply(lambda x: len(x.split()) > 5)]


def combine_nonempty(row, type):
    elements = [
        value
        for key, value in row.items()
        if type in key and value not in ["None", np.nan]
    ]
    return elements if elements else None


data["groundtruth_extracted"] = data.apply(
    combine_nonempty, axis=1, args=("Extracted Element",)
)
data["groundtruth_associated"] = data.apply(
    combine_nonempty, axis=1, args=("Associated Element",)
)

# for each Associated Element in groundtruth_associated, get the corresponding Type Level 2 and unique_id from taxonomy
# we have to check from the lowest level onwards (Type Level 4 -> Type Level 3 -> Type Level 2)
# if we find a match, we stop and assign the corresponding unique_id
# if we don't find a match, we assign "CHECK"


def get_id_and_level2(row, get_which):
    if row["groundtruth_associated"] is None:
        return None
    else:
        for element in row["groundtruth_associated"]:
            for level in ["Type Level 4", "Type Level 3", "Type Level 2"]:
                if element in taxonomy[level].values:
                    if get_which == "id":
                        return taxonomy[taxonomy[level] == element]["unique_id"].values[
                            0
                        ]
                    elif get_which == "level2":
                        return taxonomy[taxonomy[level] == element][
                            "Type Level 2"
                        ].values[0]
            return "ENCODING ISSUE"


data["groundtruth_unique_id"] = data.apply(get_id_and_level2, axis=1, args=("id",))
data["groundtruth_level2"] = data.apply(get_id_and_level2, axis=1, args=("level2",))


extracted_col = data["groundtruth_extracted"]
# associated col should be dictionary with unique_id and level2 and the associated element
associated_col = data.apply(
    lambda row: {
        "unique_id": row["groundtruth_unique_id"],
        "Type Level 2": row["groundtruth_level2"],
        "groundtruth_associated": row["groundtruth_associated"],
    }
    if row["groundtruth_associated"] is not None
    else None,
    axis=1,
)

word_emb = args.word_emb_model
word_emb_model = AutoModel.from_pretrained(word_emb)
word_emb_tokenizer = AutoTokenizer.from_pretrained(word_emb)


def process_chunk(ii):
    extracted_el = [
        elem
        for sublist in extracted_col[ii : ii + args.num_sentences]
        if sublist is not None
        for elem in sublist
    ]
    associated_el = [
        elem
        # for sublist in # but my col is a dictionary, so I need to get the value of the key "groundtruth_associated"
        for sublist in associated_col[ii : ii + args.num_sentences]
        if sublist is not None
        for elem in sublist["groundtruth_associated"]
    ]
    return {
        "sentence": ". ".join(sentences_col[ii : ii + args.num_sentences]),
        "groundtruth_extracted": extracted_el,
        "groundtruth_associated": associated_el,
    }


num_chunks = len(data) // args.num_sentences
sentences_res_list = [
    process_chunk(ii) for ii in range(0, len(data), args.num_sentences)
]

# %%
extraction_cost = 0
matching_cost = 0
detailed_results_dict = {}


# extract skills
if args.do_extraction:
    print("Starting extraction")
    api = OPENAI(args, sentences_res_list)
    sentences_res_list, cost = api.do_prediction("extraction")
    extraction_cost += cost

# load taxonomy
taxonomy, skill_names, skill_definitions = load_taxonomy(args)

# select candidate skills from taxonomy
if "extracted_skills" in sentences_res_list[0]:
    splitter = Splitter()
    max_candidates = 10
    for idxx, sample in enumerate(sentences_res_list):
        # sample = select_candidates_from_taxonomy(sample, taxonomy, skill_names, skill_definitions, splitter, max_candidates)
        sample = select_candidates_from_taxonomy(
            sample,
            taxonomy,
            splitter,
            word_emb_model,
            word_emb_tokenizer,
            max_candidates,
        )
        sentences_res_list[idxx] = sample

# print(sentences_res_list)

# match skills with taxonomy
if args.do_matching and "skill_candidates" in sentences_res_list[0]:
    print("Starting matching")
    api = OPENAI(args, sentences_res_list)
    sentences_res_list, cost = api.do_prediction("matching")

    matching_cost += cost

# Do exact match with technologies, languages, certifications
tech_certif_lang = pd.read_csv("../data/taxonomy/tech_certif_lang.csv")
tech_alternative_names = pd.read_csv(
    "../data/taxonomy/technologies_alternative_names.csv", sep="\t"
)
certification_alternative_names = pd.read_csv(
    "../data/taxonomy/certifications_alternative_names.csv", sep="\t"
)
sentences_res_list = exact_match(
    sentences_res_list,
    tech_certif_lang,
    tech_alternative_names,
    certification_alternative_names,
)
# # TODO find a way to correctly identify even common strings (eg 'R')! (AD: look in utils exact_match)
# # Idem for finding C on top of C# and C++
# # TODO update alternative names generation to get also shortest names (eg .Net, SQL etc) (Syrielle)
detailed_results_dict["test_CV"] = sentences_res_list

write_json(detailed_results_dict, args.output_path.replace(".json", "_detailed.json"))
# Output final
categs = [
    "Technologies",
    "Technologies_alternative_names",
    "Certifications",
    "Certification_alternative_names",
    "Languages",
]
clean_output_dict = {}
for item_id, detailed_res in detailed_results_dict.items():
    clean_output = {categ: [] for categ in categs}
    clean_output["skills"] = []
    for ii, sample in enumerate(detailed_res):
        for cat in categs:
            clean_output[cat].extend(sample[cat])
        if "matched_skills" in sample:
            for skill in sample["matched_skills"]:
                clean_output["skills"].append(sample["matched_skills"][skill])
                # TODO 1. output Level 2 or id! To do so, re-do id generation on the taxonomy to give IDs only to Level 2 elements! (refer to taxonomy v4 as well as ipynb)
                # TODO deduplicate and remove "None"
    clean_output_dict[item_id] = clean_output
write_json(clean_output_dict, args.output_path.replace(".json", "_clean.json"))
print("Done")
print("Extraction cost ($):", extraction_cost)
print("Matching cost ($):", matching_cost)
print("Total cost ($):", extraction_cost + matching_cost)
