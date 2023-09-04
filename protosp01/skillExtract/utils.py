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
from spacy_langdetect import LanguageDetector
import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector


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
            ]  # TODO fix, doesn't wrk because it's single line json)
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
    sentences = text.split("\n\n")
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
        data is a list of dictionaries, each consiting of one sentence and extracted skills
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

        for i, sample in enumerate(tqdm(self.data)):
            instruction_field = (
                "instruction_job"
                if "vacancies" in self.args.datapath
                else "instruction"
            )
            input = (
                PROMPT_TEMPLATES["extraction"][instruction_field]
                + "\n"
                + PROMPT_TEMPLATES["extraction"]["6shots"]
            )
            # TODO nb of shots as argument
            input += "\nSentence: " + sample["sentence"] + "\nAnswer:"
            max_tokens = self.args.max_tokens

            prediction = (
                self.run_gpt_sample(input, max_tokens=max_tokens).lower().strip()
            )
            extracted_skills = re.findall(pattern, prediction)
            sample["extracted_skills"] = extracted_skills
            self.data[i] = sample

            cost = compute_cost(input, prediction, self.args.model)
            costs += cost
        return costs

    def run_gpt_df_matching(self):
        costs = 0
        instruction_field = (
            "instruction_job" if "vacancies" in self.args.datapath else "instruction"
        )
        for i, sample in enumerate(tqdm(self.data)):
            sample["matched_skills"] = {}
            for extracted_skill in sample["extracted_skills"]:
                input = (
                    PROMPT_TEMPLATES["matching"][instruction_field]
                    + "\n"
                    + PROMPT_TEMPLATES["matching"]["1shot"]
                )
                # TODO having definition or not in the list of candidates
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
                input += f"\nSentence: {sample['sentence']} \nSkill: {extracted_skill} \nOptions: {options_string}.\nAnswer: "

                prediction = self.run_gpt_sample(input, max_tokens=5).lower().strip()

                chosen_letter = prediction[0].upper()
                # TODO match this with the list of candidates, in case no letter was generated
                chosen_option = (
                    options_dict[chosen_letter]
                    if chosen_letter in options_dict
                    else "None"
                )

                sample["matched_skills"][extracted_skill] = chosen_option
                self.data[i] = sample

                cost = compute_cost(input, prediction, self.args.model)
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


def filter_subwords(extracted_skill, skill_names, splitter):
    subwords = []
    for word in extracted_skill.split():
        subwords.extend(list(splitter.split_compound(word)[0][1:]))
    subwords = list(set(subwords))
    subwords = [word for word in subwords if len(word) > 2]
    matched_elements = []
    for subword in subwords:
        matched_elements.extend(
            filter(lambda item: subword in item[1], enumerate(skill_names))
        )
    return matched_elements


def load_taxonomy(args):
    taxonomy = pd.read_csv(args.taxonomy, sep=",")
    taxonomy = taxonomy.dropna(subset=["Definition", "Type Level 2"])
    taxonomy["name+example"] = taxonomy.apply(concatenate_cols_skillname, axis=1)
    # taxonomy["unique_id"] = list(range(len(taxonomy)))
    skill_definitions = list(taxonomy["Definition"].apply(lambda x: x.lower()))
    skill_names = list(taxonomy["name+example"].apply(lambda x: x.lower()))
    return taxonomy, skill_names, skill_definitions


def select_candidates_from_taxonomy(
    sample, taxonomy, skill_names, skill_definitions, splitter, max_candidates
):
    sample["skill_candidates"] = {}
    if len(sample["extracted_skills"]) > 0:
        # look for each extracted skill in the taxonomy
        for extracted_skill in sample["extracted_skills"]:
            # TODO apply rules one by one
            # 1) look for all taxonomy skill name with exact match of extracted skill name inside
            rule1 = list(
                filter(lambda item: extracted_skill in item[1], enumerate(skill_names))
            )
            # 2) look for all taxonomy skill definitions with exact match of extracted skill name inside
            rule2 = list(
                filter(
                    lambda item: extracted_skill in item[1],
                    enumerate(skill_definitions),
                )
            )
            # 3)look for all taxonomy skill name with a subword of extracted skill name inside
            rule3 = filter_subwords(extracted_skill, skill_names, splitter)
            instructions = [rule1, rule2, rule3]

            for instruction in instructions:
                matching_elements = instruction
                if len(matching_elements) > 0:
                    break
            matching_rows = taxonomy.iloc[[item[0] for item in matching_elements]]
            # todo first look for it in the skill name. If not found, look for it in skill definition.
            if len(matching_rows) > 10:
                print("More than 10 candidates found for skill", extracted_skill)
                matching_rows = matching_rows.sample(
                    max_candidates
                )  # TODO what to do with more than N extracted skills ?
            elif (
                len(matching_rows) == 0
            ):  # TODO update now that ew don't have type level 3 anymore
                matching_elements = difflib.get_close_matches(
                    extracted_skill,
                    taxonomy["name+example"],
                    cutoff=0.4,
                    n=max_candidates,
                )
                matching_rows = taxonomy[
                    taxonomy["name+example"].isin(matching_elements)
                ]
            if len(matching_rows) == 0:
                print("No candidates found for skill", extracted_skill)
            sample["skill_candidates"][extracted_skill] = matching_rows[
                ["unique_id", "name+example"]
            ].to_dict("records")
    return sample


def exact_match(
    data, tech_certif_lang, tech_alternative_names, certification_alternative_names
):
    # Create a dictionary to map alternative names to their corresponding Level 2 values
    synonym_to_tech_mapping = {}
    for index, row in tech_alternative_names.iterrows():
        alternative_names = []
        if not pd.isna(row["alternative_names_clean"]):
            alternative_names = row["alternative_names_clean"].split(", ")
        for alt_name in alternative_names:
            synonym_to_tech_mapping[alt_name] = row["Level 2"]

    synonym_to_certif_mapping = {}
    for index, row in certification_alternative_names.iterrows():
        alternative_names = []
        if not pd.isna(row["alternative_names_clean"]):
            alternative_names = row["alternative_names_clean"].split(", ")
        for alt_name in alternative_names:
            synonym_to_certif_mapping[alt_name] = row["Level 2"]

    # TODO save which alternative name was matched, for each tech found.
    categs = set(tech_certif_lang["Level 1"])
    word_sets = [
        set(tech_certif_lang[tech_certif_lang["Level 1"] == categ]["Level 2"])
        for categ in categs
    ]
    for sample in data:
        sentence = sample["sentence"]
        for category, word_set in zip(categs, word_sets):
            # TODO need to exclude the "#" character from being treated as a word boundary in the regular expression pattern!
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
