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
import pathlib
import re
import tiktoken
import difflib
from split_words import Splitter
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from spacy_langdetect import LanguageDetector
import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


def get_lang_detector(nlp, name):
    return LanguageDetector(seed=42)  # We use the seed 42


nlp_model = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp_model.add_pipe("language_detector", last=True)


from prompt_template import PROMPT_TEMPLATES
from api_key import API_KEY

CHAT_COMPLETION_MODELS = ["gpt-3.5-turbo", "gpt-4"]
TEXT_COMPLETION_MODELS = ["text-davinci-003"]
COSTS = {
    "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},
    "gpt-4": {"input": 0.00003, "output": 0.00006},
    "text-davinci-003": {"input": 0.00002, "output": 0.00002},
}
ENCODINGS = {
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "text-davinci-003": "p50k_base",
}


def read_json(path, lastN=None):
    loaded_lines = []
    with open(path, "r", encoding="utf-8") as f:
        if lastN is None:
            lines = f.readlines()
        else:
            lines = f.readlines()[
                -lastN:
            ]  # TODO fix, doesn't wrk because it's single line json) AD: check how to read last N lines of a JSON (related to num_samples argument)
    for line in lines:
        element = json.loads(line)
        loaded_lines.append(element)
    return loaded_lines


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def detect_language(text):
    maximum = max(len(text), 150)
    doc = nlp_model(text[:maximum])
    detect_language = doc._.language
    return detect_language["language"]


def split_sentences(text):
    # sentences = re.split(r'(?<=[.!?]) +', text)
    # sentences = text.split("\n\n")  # TODO: AD test number of sentences here
    splitter = SentenceSplitter(language="de")
    sentences = splitter.split(text)
    sentences = [
        sentence.rstrip(".") for sentence in sentences if len(sentence.split()) > 5
    ]
    return sentences


def num_tokens_from_string(sentence, model):
    encoding_name = ENCODINGS[model]
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(sentence))
    return num_tokens


def compute_cost(input, output, model):
    input_len = num_tokens_from_string(input, model)
    output_len = num_tokens_from_string(output, model)
    return input_len * COSTS[model]["input"] + output_len * COSTS[model]["output"]


def chat_completion(messages, model="gpt-3.5-turbo", return_text=True, model_args=None):
    if model_args is None:
        model_args = {}

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model, messages=messages, **model_args
            )
            if return_text:
                return response["choices"][0]["message"]["content"].strip()
            return response
        except (RateLimitError, ServiceUnavailableError, APIError) as e:
            print("Timed out. Waiting for 1 minute.")
            time.sleep(60)
            continue


def text_completion(
    prompt, model="text-davinci-003", return_text=True, model_args=None
):
    if model_args is None:
        model_args = {}

    while True:
        try:
            response = openai.Completion.create(
                model=model, prompt=prompt, **model_args
            )
            if return_text:
                return response["choices"][0]["text"].strip()
            return response
        except (RateLimitError, ServiceUnavailableError, APIError) as e:
            print("Timed out. Waiting for 1 minute.")
            time.sleep(60)
            continue


class OPENAI:
    def __init__(self, args, data):
        """
        data is a list of dictionaries, each consisting of one sentence and extracted skills
        """
        openai.api_key = args.api_key
        self.args = args
        self.data = data

    def do_prediction(self, task):
        cost = self.run_gpt(task)
        print("Costs: ", task, cost)
        return self.data, cost

    def run_gpt(self, task):
        if task == "extraction":
            return self.run_gpt_df_extraction()
        elif task == "matching":
            return self.run_gpt_df_matching()

    def run_gpt_df_extraction(self):
        costs = 0
        pattern = r"@@(.*?)##"

        for idx, sample in enumerate(tqdm(self.data)):
            instruction_field = (
                "instruction_job"
                if "vacancies" in self.args.datapath
                else "instruction"
            )
            input_ = (
                PROMPT_TEMPLATES["extraction"][instruction_field]
                + "\n"
                + PROMPT_TEMPLATES["extraction"]["6shots"]
            )
            # TODO 2. nb of shots as argument ? For later experiments
            input_ += "\nSentence: " + sample["sentence"] + "\nAnswer:"
            max_tokens = self.args.max_tokens

            prediction = (
                self.run_gpt_sample(input_, max_tokens=max_tokens).lower().strip()
            )
            extracted_skills = re.findall(pattern, prediction)
            sample["extracted_skills"] = list(
                set(extracted_skills)
            )  # AD: removed duplicates
            self.data[idx] = sample

            cost = compute_cost(input_, prediction, self.args.model)
            costs += cost
        return costs

    def run_gpt_df_matching(self):
        costs = 0
        instruction_field = (
            "instruction_job" if "vacancies" in self.args.datapath else "instruction"
        )
        for idxx, sample in enumerate(tqdm(self.data)):
            # print(self.data)
            # break
            sample["matched_skills"] = {}
            for extracted_skill in sample["extracted_skills"]:
                input_ = (
                    PROMPT_TEMPLATES["matching"][instruction_field]
                    + "\n"
                    + PROMPT_TEMPLATES["matching"]["1shot"]
                )
                # TODO 1.5 having definition or not in the list of candidates ? Here we only prove the name and an example. Yes, should try, but maybe not if there are 10 candidates...
                # update as an argument - like give def or not when doing the matching then ask Marco if it helps or decreases performance
                options_dict = {
                    letter.upper(): candidate["name+example"]
                    for letter, candidate in zip(
                        list("abcdefghijklmnopqrstuvwxyz")[
                            : len(sample["skill_candidates"][extracted_skill])
                        ],
                        sample["skill_candidates"][extracted_skill],
                    )
                }
                options_string = " \n".join(
                    letter + ": " + description
                    for letter, description in options_dict.items()
                )
                input_ += f"\nSentence: {sample['sentence']} \nSkill: {extracted_skill} \nOptions: {options_string}.\nAnswer: "

                prediction = self.run_gpt_sample(input_, max_tokens=5).lower().strip()

                chosen_letter = prediction[0].upper()
                # TODO match this with the list of candidates, in case no letter was generated! (AD: try to ask it to output first line like "Answer is _")
                # Here the best way is just to change the prompt and ask the model to always output the same template, to make the extraction of the chosen option easier.
                # AD: maybe try JSON or "Answer in _ format" or with specific tags
                # AD: maybe experiment with different params (temperature)
                chosen_option = (
                    options_dict[chosen_letter]
                    if chosen_letter in options_dict
                    else "None"
                )

                for skill_candidate in sample["skill_candidates"][extracted_skill]:
                    if skill_candidate["name+example"] == chosen_option:
                        sample["matched_skills"][extracted_skill] = skill_candidate
                        break  # stop searching once matched

                self.data[idxx] = sample

                cost = compute_cost(input_, prediction, self.args.model)
                costs += cost
        return costs

    def run_gpt_sample(self, prompt, max_tokens):
        if self.args.model in CHAT_COMPLETION_MODELS:
            response = chat_completion(
                [{"role": "user", "content": prompt}],
                model=self.args.model,
                return_text=True,
                model_args={
                    "temperature": self.args.temperature,
                    "max_tokens": max_tokens,
                    "top_p": self.args.top_p,
                    "frequency_penalty": self.args.frequency_penalty,
                    "presence_penalty": self.args.presence_penalty,
                },
            )
        elif self.args.model in TEXT_COMPLETION_MODELS:
            response = text_completion(
                prompt,
                model=self.args.model,
                return_text=True,
                model_args={
                    "temperature": self.args.temperature,
                    "max_tokens": max_tokens,
                    "top_p": self.args.top_p,
                    "frequency_penalty": self.args.frequency_penalty,
                    "presence_penalty": self.args.presence_penalty,
                },
            )
        else:
            raise ValueError(f"Model {self.args.model} not supported for evaluation.")

        return response


def concatenate_cols_skillname(row):
    output = row["Type Level 2"]
    output += f": {row['Type Level 3']}" if not pd.isna(row["Type Level 3"]) else ""
    output += f": {row['Type Level 4']}" if not pd.isna(row["Type Level 4"]) else ""
    output += f": {row['Example']}" if not pd.isna(row["Example"]) else ""
    return output


# def filter_subwords(extracted_skill, skill_names, splitter):
def filter_subwords(extracted_skill, splitter):
    subwords = []
    for word in extracted_skill.split():
        subwords.extend(list(splitter.split_compound(word)[0][1:]))
    subwords = list(set(subwords))
    subwords = [word for word in subwords if len(word) > 2]
    # matched_elements = []
    # for subword in subwords:
    #     matched_elements.extend(
    #         filter(lambda item: subword in item[1], enumerate(skill_names))
    #     )
    return subwords  # matched_elements


def load_taxonomy(args):
    taxonomy = pd.read_csv(args.taxonomy, sep=",")
    taxonomy = taxonomy.dropna(subset=["Definition", "Type Level 2"])
    taxonomy["name+example"] = taxonomy.apply(concatenate_cols_skillname, axis=1)
    # taxonomy["unique_id"] = list(range(len(taxonomy)))
    skill_definitions = list(taxonomy["Definition"].apply(lambda x: x.lower()))
    skill_names = list(taxonomy["name+example"].apply(lambda x: x.lower()))

    keep_cols = [
        "unique_id",
        "ElementID",
        "Dimension",
        "Type Level 1",
        "Type Level 2",
        "Type Level 3",
        "Type Level 4",
        "Example",
        "Definition",
        "name+example",
    ]
    taxonomy = taxonomy[keep_cols]
    return taxonomy, skill_names, skill_definitions


def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        word_outputs = model(**inputs)
        embeddings = word_outputs.last_hidden_state.mean(
            dim=1
        )  # Average pooling over tokens
    return embeddings


def get_top_vec_similarity(
    extracted_skill,
    taxonomy,
    model,
    tokenizer,
    max_candidates=10,
):
    skill_vec = get_embeddings(extracted_skill, model, tokenizer)
    taxonomy["embeddings"] = taxonomy["name+example"].apply(
        lambda x: get_embeddings(x, model, tokenizer)
    )
    taxonomy["similarity"] = taxonomy["embeddings"].apply(
        lambda x: F.cosine_similarity(skill_vec, x, dim=1).item()
    )
    cut_off_score = taxonomy.sort_values(by="similarity", ascending=False).iloc[
        max_candidates
    ]["similarity"]
    taxonomy["results"] = taxonomy["similarity"].apply(
        lambda x: True if x >= cut_off_score else False
    )
    taxonomy.drop(columns=["embeddings", "similarity"], inplace=True)

    return taxonomy


# def select_candidates_from_taxonomy(
#     sample, taxonomy, skill_names, skill_definitions, splitter, max_candidates
# ):
def select_candidates_from_taxonomy(
    sample, taxonomy, splitter, model, tokenizer, max_candidates=10, method="rules"
):
    sample["skill_candidates"] = {}
    if len(sample["extracted_skills"]) > 0:
        # if method == "rules" or == "mixed" or == "embeddings" below
        for extracted_skill in sample["extracted_skills"]:
            print("extracted skill:", extracted_skill)
            # look for each extracted skill in the taxonomy
            if method == "rules" or method == "mixed":
                # First we check for matches in name+example
                print("checking for matches in name+example")
                taxonomy["results"] = taxonomy["name+example"].str.contains(
                    extracted_skill, case=False, regex=False
                )
                if taxonomy[taxonomy["results"]].empty:
                    # print("... none found")
                    print("checking for matches in definition")
                    # If no match, we check for matches in definitions
                    taxonomy["results"] = taxonomy["Definition"].str.contains(
                        extracted_skill, case=False, regex=False
                    )
                if taxonomy[taxonomy["results"]].empty:
                    # print("... none found")
                    print("checking for matches in subwords")
                    # If still no match, we check if subwords are present in name+example
                    taxonomy["results"] = False
                    for subword in filter_subwords(extracted_skill, splitter):
                        # if there is a match subwresults column + 1
                        taxonomy["results"] = taxonomy["results"] + taxonomy[
                            "name+example"
                        ].str.contains(subword, case=False, regex=False)
                if taxonomy[taxonomy["results"]].empty:
                    if method == "rules":
                        print("checking for matches in difflib")
                        matching_elements = difflib.get_close_matches(
                            extracted_skill,
                            taxonomy["name+example"],
                            cutoff=0.4,
                            n=max_candidates,
                        )
                        taxonomy["results"] = taxonomy["name+example"].isin(
                            matching_elements
                        )
                        if taxonomy[taxonomy["results"]].empty:
                            print("No candidates found for: ", extracted_skill)
                    else:
                        print("checking for highest embedding similarity")
                        taxonomy = get_top_vec_similarity(
                            extracted_skill, taxonomy, model, tokenizer
                        )
            if method == "embeddings":
                taxonomy["results"] = False
                print("checking for highest embedding similarity")
                taxonomy = get_top_vec_similarity(
                    extracted_skill, taxonomy, model, tokenizer
                )  ## AD TODO: take embedding of only the relevant subword in extracted skill

            keep_cols = [
                "unique_id",
                # "Type Level 2",
                "name+example",
            ]

            matching_df = taxonomy[taxonomy["results"]][keep_cols]

            if len(matching_df) > max_candidates:
                matching_df = matching_df.sample(n=max_candidates, random_state=42)

            sample["skill_candidates"][extracted_skill] = matching_df.to_dict("records")

            # # matching_df is results = True and only keep unique id, Type Level 2 and name+example, Definition columns
            # keep_cols_det = [
            #     "unique_id",
            #     # "Type Level 2",
            #     "name+example",
            # ]
            # keep_cols_cln = [
            #     "unique_id",
            #     "Type Level 2",
            #     # "name+example",
            # ]
            # matching_df_det = taxonomy[taxonomy["results"]][keep_cols_det]
            # matching_df_cln = taxonomy[taxonomy["results"]][keep_cols_cln]

            # if len(matching_df_det) > max_candidates:
            #     matching_df_det = matching_df_det.sample(
            #         n=max_candidates, random_state=42
            #     )

            # if len(matching_df_cln) > max_candidates:
            #     matching_df_cln = matching_df_cln.sample(
            #         n=max_candidates, random_state=42
            #     )

            # sample_det = sample.copy()
            # sample_det["skill_candidates"][extracted_skill] = matching_df_det.to_dict(
            #     "records"
            # )
            # sample_cln = sample.copy()
            # sample_cln["skill_candidates"][extracted_skill] = matching_df_cln.to_dict(
            #     "records"
            # )

        # # TODO: (Maybe try with this logic but also try embeddings (JobBERT))
        # # TODO apply rules one by one (make this more computationally efficient, don't perform all filterings beforehand)
        # # 1) look for all taxonomy skill name with exact match of extracted skill name inside
        # rule1 = list(
        #     filter(lambda item: extracted_skill in item[1], enumerate(skill_names))
        # )
        # # 2) look for all taxonomy skill definitions with exact match of extracted skill name inside
        # rule2 = list(
        #     filter(
        #         lambda item: extracted_skill in item[1],
        #         enumerate(skill_definitions),
        #     )
        # )
        # # 3)look for all taxonomy skill name with a subword of extracted skill name inside
        # rule3 = filter_subwords(extracted_skill, skill_names, splitter)
        # instructions = [rule1, rule2, rule3]

        # for instruction in instructions:
        #     matching_elements = instruction
        #     if len(matching_elements) > 0:
        #         break
        # matching_rows = taxonomy.iloc[[item[0] for item in matching_elements]]
        # if len(matching_rows) > 10:
        #     print("More than 10 candidates found for skill", extracted_skill)
        #     matching_rows = matching_rows.sample(
        #         max_candidates
        #     )  # TODO what to do with more than N extracted skills ? Find a way to make a coherent selection.
        # elif (
        #     len(matching_rows) == 0
        # ):  # TODO update now that we don't have type level 3 anymore
        #     matching_elements = difflib.get_close_matches(
        #         extracted_skill,
        #         taxonomy["name+example"],
        #         cutoff=0.4,
        #         n=max_candidates,
        #     )
        #     matching_rows = taxonomy[
        #         taxonomy["name+example"].isin(matching_elements)
        #     ]
        # if len(matching_rows) == 0:
        #     print("No candidates found for skill", extracted_skill)
        # sample["skill_candidates"][extracted_skill] = matching_rows[
        #     ["unique_id", "name+example"]
        # ].to_dict("records")
    return sample


def exact_match(
    data, tech_certif_lang, tech_alternative_names, certification_alternative_names
):
    # Create a dictionary to map alternative names to their corresponding Level 2 values
    synonym_to_tech_mapping = {}
    for _, row in tech_alternative_names.iterrows():
        alternative_names = []
        if not pd.isna(row["alternative_names_clean"]):
            alternative_names = row["alternative_names_clean"].split(", ")
        for alt_name in alternative_names:
            synonym_to_tech_mapping[alt_name] = row["Level 2"]

    synonym_to_certif_mapping = {}
    for _, row in certification_alternative_names.iterrows():
        alternative_names = []
        if not pd.isna(row["alternative_names_clean"]):
            alternative_names = row["alternative_names_clean"].split(", ")
        for alt_name in alternative_names:
            synonym_to_certif_mapping[alt_name] = row["Level 2"]

    categs = set(tech_certif_lang["Level 1"])
    word_sets = [
        set(tech_certif_lang[tech_certif_lang["Level 1"] == categ]["Level 2"])
        for categ in categs
    ]
    for sample in data:
        sentence = sample["sentence"]
        for category, word_set in zip(categs, word_sets):
            # TODO need to exclude the "#" character from being treated as a word boundary in the regular expression pattern! (for C#, same for C++?
            # AD: perhaps list out most common use cases and make an exception for them) -> look in tech_certif_lang.csv
            matching_words = re.findall(
                r"\b(?:"
                + "|".join(re.escape(word) for word in word_set).replace(r"\#", "#")
                + r")\b",
                sentence,
            )
            sample[category] = matching_words

        tech_synonym_set = list(synonym_to_tech_mapping.keys())
        matching_synonyms = re.findall(
            r"\b(?:" + "|".join(re.escape(word) for word in tech_synonym_set) + r")\b",
            sentence,
        )
        matching_tech = [synonym_to_tech_mapping[word] for word in matching_synonyms]
        sample["Technologies"].extend(matching_tech)
        sample["Technologies"] = list(set(sample["Technologies"]))
        sample["Technologies_alternative_names"] = list(set(matching_synonyms))

        certif_synonym_set = list(synonym_to_certif_mapping.keys())
        matching_synonyms = re.findall(
            r"\b(?:"
            + "|".join(re.escape(word) for word in certif_synonym_set)
            + r")\b",
            sentence,
        )
        matching_certif = [
            synonym_to_certif_mapping[word] for word in matching_synonyms
        ]
        sample["Certifications"].extend(matching_certif)
        sample["Certifications"] = list(set(sample["Certifications"]))
        sample["Certification_alternative_names"] = list(set(matching_synonyms))
    return data


def get_lowest_level(row):
    """
    Returns the lowest level of the taxonomy that is not NaN in each
    """
    for level in ["Type Level 4", "Type Level 3", "Type Level 2", "Type Level 1"]:
        value = row[level]
        if not pd.isna(value):
            return value + "_" + level
            # appending level also just in case different levels have the same name


# write something like below that does not work with splice:
def clean_skills_list(skill_name, alternative_names):
    alternative_names = alternative_names.replace("\n", ", ")
    alternative_names = (
        alternative_names.split(":")[1]
        if ":" in alternative_names
        else alternative_names
    )
    alternative_names = re.sub(r"\d+\. ", "", alternative_names)
    alternative_names = alternative_names.split(", ")
    alternative_names = [
        skill for skill in alternative_names if skill != "" and skill_name not in skill
    ]
    # remove if each skill is too long (longer than 10 words)
    alternative_names = [
        skill for skill in alternative_names if len(skill.split()) < 10
    ]
    # remove duplicates
    alternative_names = list(set(alternative_names))
    alternative_names = ", ".join(alternative_names)
    return alternative_names
