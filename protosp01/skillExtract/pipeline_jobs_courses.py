# %%
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
import pickle
import datetime


# %%

from prompt_template import PROMPT_TEMPLATES
from utils import *

# %%


def main():
    # fmt: off

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to source data", default = "../data/raw/vacancies.json")
    parser.add_argument("--taxonomy", type=str, help="Path to taxonomy file in csv format", default = "../data/taxonomy/taxonomy_V4.csv")
    parser.add_argument("--api_key", type=str, help="openai keys", default = API_KEY)
    parser.add_argument("--model", type=str, help="Model to use for generation", default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, help="Temperature for generation", default=0)
    parser.add_argument("--max_tokens", type=int, help="Max tokens for generation", default=100)
    parser.add_argument("--shots", type=int, help="Number of demonstrations, max = 5", default=5)
    parser.add_argument("--top_p", type=float, help="Top p for generation", default=1)
    parser.add_argument("--frequency_penalty", type=float, help="Frequency penalty for generation", default=0)
    parser.add_argument("--presence_penalty", type=float, help="Presence penalty for generation", default=0)
    parser.add_argument("--candidates_method", type=str, help="How to select candidates: rules, mixed or embeddings. Default is embeddings", default="embeddings")
    parser.add_argument("--output_path", type=str, help="Output for evaluation results", default="results/")
    parser.add_argument("--prompt_type", type=str, help="Prompt type, from the prompt_template.py file. For now, only \"skills\" and \"wlevels\". default is wlevels.", default="wlevels")
    parser.add_argument("--num-samples", type=int, help="Last N elements to evaluate (the new ones)", default=10)
    parser.add_argument("--num-sentences", type=int, help="by how many sentences to split the corpus", default=2)
    parser.add_argument("--do-extraction", action="store_true", help="Whether to do the extraction or directly the matching")
    parser.add_argument("--do-matching", action="store_true", help="Whether to do the matching or not")
    parser.add_argument("--load-extraction", type=str, help="Path to a file with intermediate extraction results", default="")
    parser.add_argument("--word-emb-model", type=str, help="Word embedding model to use", default="agne/jobBERT-de")
    parser.add_argument("--debug", action="store_true", help="Keep only one sentence per job offer / course to debug")
    parser.add_argument("--detailed", action="store_true", help="Generate detailed output")
    parser.add_argument("--ids", type=str, help="Path to a file with specific ids to evaluate", default=None)
    parser.add_argument("--annotate", action="store_true", help="Whether to annotate the data or not")
    # fmt: on

    ###
    # Extraction checkpoints:
    # results/course_gpt-3.5-turbo_2sent_n10_V4231025_extraction.json
    ###

    args = parser.parse_args()
    if args.datapath.split("/")[-1] == "vacancies.json":
        args.data_type = "job"
    elif args.datapath.split("/")[-1] == "learning_opportunities.json":
        args.data_type = "course"
    else:
        print("Error: Data source unknown")

    nsent = f"_{args.num_sentences}sent"
    nsamp = f"_n{args.num_samples}"
    # dt = datetime.datetime.now().strftime("%y%m%d")
    # tax_v = "_" + args.taxonomy.split("/")[-1].split(".")[0].split("_")[-1]
    dt = "231025"
    tax_v = f"_{args.taxonomy.split('/')[-1].split('.')[0].split('_')[-1]}"

    args.api_key = API_KEY  # args.openai_key
    args.output_path = args.output_path + args.data_type + "_" + args.model + ".json"
    print("Output path", args.output_path)

    # Intitialize pretrained word embeddings
    word_emb = args.word_emb_model
    word_emb_model = AutoModel.from_pretrained(word_emb)
    word_emb_tokenizer = AutoTokenizer.from_pretrained(word_emb)

    emb_sh = "_rules"

    taxonomy = load_taxonomy(args)

    if args.candidates_method != "rules":
        if word_emb == "agne/jobBERT-de":
            emb_sh = "_jBd"
        elif word_emb == "agne/jobGBERT":
            emb_sh = "_jGB"

        try:
            print(f"Loading embedded taxonomy for {word_emb}")
            with open(
                f"../data/taxonomy/taxonomy{tax_v}_embeddings{emb_sh}.pkl", "rb"
            ) as f:
                emb_tax = pickle.load(f)
        except:
            print(f"Loading failed, generating embedded taxonomy for {word_emb}")
            emb_tax = embed_taxonomy(taxonomy, word_emb_model, word_emb_tokenizer)
            with open(
                f"../data/taxonomy/taxonomy{tax_v}_embeddings{emb_sh}.pkl", "wb"
            ) as f:
                pickle.dump(emb_tax, f)

    if args.candidates_method == "mixed":
        emb_sh = "_mixed"

    if args.ids is not None:
        args.num_samples = 0
        with open(args.ids, "r") as f:
            ids = f.read().splitlines()
        if "vacancies" in ids[0]:
            args.data_type = "job"
        elif "learning_opportunities" in ids[0]:
            args.data_type = "course"
        ids = [int(id.split("/")[-1]) for id in ids]
        print("Evaluating only ids:", ids)
        args.output_path = args.output_path.replace(".json", f"_ids.json")

    if args.num_samples > 0:
        data = read_json(args.datapath, lastN=args.num_samples)
        data = data[0][-args.num_samples :]
    else:
        data = read_json(args.datapath)[0]

    data = pd.DataFrame.from_records(data)
    if args.data_type == "job":
        data["fulltext"] = data["name"] + "\n" + data["description"]
        print("num jobs:", len(data))

    elif args.data_type == "course":
        data = data[data["active"] == True]
        keep_ids = {1, 5, 9}
        data = data[data["study_ids"].apply(lambda x: bool(set(x) & keep_ids))]

        print("num courses with ids 1,5,9:", len(data))

        # drop cols that are not needed
        drop_cols = [
            "type",
            "active",
            "pricing_description",
            "ects_points",
            "average_effort_per_week",
            "total_effort",
            "structure_description",
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
        )
        acq_data["skill_type"] = "to_acquire"

        req_data = data.copy()
        req_data["fulltext"] = req_data["admission_criteria_description"].fillna(
            ""
        ) + req_data["target_group_description"].fillna("")
        req_data["skill_type"] = "required"

        data = pd.concat([acq_data, req_data], ignore_index=True)
        # replace every 10 tags with a period to avoid too long sentences

    data["fulltext"] = data["fulltext"].apply(replace_html_tags)
    data = drop_short_text(data, "fulltext", 100)
    # breakpoint()

    if args.ids is not None:
        data = data[data["id"].isin(ids)]
        data_to_save = data.copy()

        data_to_save.drop(columns="fulltext", axis=1, inplace=True)
        # save the content of the ids in a separate file
        ids_content = data_to_save.to_dict("records")
        write_json(
            ids_content,
            args.output_path.replace(".json", f"{nsent}{emb_sh}{tax_v}_content.json"),
        )
    else:
        # apply language detection
        data["language"] = data["fulltext"].apply(detect_language)
        print(data["language"].value_counts())
        data = data[data["language"] == "de"]

    print("loaded data:", len(data), "elements")

    data = data.to_dict("records")

    # We create two files:
    # 1. results_detailed.json: contains a list of jobs/courses ids
    # each job / course has a list of sentence, each sentence has all extraction details
    # 2. results_clean.json: contains a list of jobs/courses ids
    # each job / course has only a list of skills, certifications, languages, technologies
    extraction_cost = 0
    matching_cost = 0
    detailed_results_dict = {}
    if args.load_extraction != "":
        args.do_extraction = False
        try:
            with open(args.load_extraction, "r") as f:
                detailed_results_dict = json.load(f)
        except:
            print(
                "Error: could not load intermediate extraction file. Try arg --do_extraction instead"
            )
            exit()

    for _, item in tqdm(enumerate(data)):  # item is job or course in dictionary format
        sentences = split_sentences(item["fulltext"])
        # breakpoint()
        if args.debug:
            # sentences = [sent for sent in sentences if len(sent.split())<80]
            # if len(sentences)==0:
            #    continue
            sentences = [random.choice(sentences)]
        sentences_res_list = []

        for ii in range(0, len(sentences), args.num_sentences):
            sentences_res_list.append(
                {
                    "sentence": ". ".join(sentences[ii : ii + args.num_sentences]),
                }
            )

        if len(sentences_res_list) == 0:
            continue

        if args.annotate:
            # export to csv
            df = pd.DataFrame(sentences_res_list)
            df.to_csv(
                args.output_path.replace(".json", f"{nsent}{nsamp}_annot.csv"),
                index=False,
            )

        # extract skills
        if args.do_extraction:
            print("Starting extraction")
            api = OPENAI(args, sentences_res_list)
            # if max(api.get_num_tokens(sentences_res_list)) > 3000:
            #     for ii, sent in enumerate(sentences_res_list):
            #         with open("diag_sentences_too_long.txt", "a") as f:
            #             f.write(f"\n\n {str(item['id'])}\n{sent['sentence']}")
            sentences_res_list, cost = api.do_prediction("extraction")
            extraction_cost += cost

        if args.load_extraction != "":
            try:
                sentences_res_list = (
                    detailed_results_dict[str(item["id"])][item["skill_type"]]
                    if args.data_type == "course"
                    else detailed_results_dict[str(item["id"])]
                )
            except:
                print(
                    f"Error: could not find {str(item['id'])} in intermediate extraction file. Try arg --do_extraction instead"
                )
                exit()

        # select candidate skills from taxonomy
        if args.do_matching and "extracted_skills" in sentences_res_list[0]:
            print("Starting candidate selection")
            splitter = Splitter()
            max_candidates = 10
            for idxx, sample in enumerate(sentences_res_list):
                sample = select_candidates_from_taxonomy(
                    sample,
                    taxonomy,
                    splitter,
                    word_emb_model,
                    word_emb_tokenizer,
                    max_candidates,
                    method=args.candidates_method,
                    emb_tax=None if args.candidates_method == "rules" else emb_tax,
                )
                sentences_res_list[idxx] = sample
            # breakpoint()

        # match skills with taxonomy
        if args.do_matching and "skill_candidates" in sentences_res_list[0]:
            print("Starting matching")
            api = OPENAI(args, sentences_res_list)
            sentences_res_list, cost = api.do_prediction("matching")
            # breakpoint()
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
            args.data_type,
        )
        # TODO find a way to correctly identify even common strings (eg 'R')! (AD: look in utils exact_match)
        # Idem for finding C on top of C# and C++
        # TODO update alternative names generation to get also shortest names (eg .Net, SQL etc) (Syrielle)
        if args.data_type == "course":
            skill_type = item["skill_type"]  # to acquire or prereq
            item_id = item["id"]  # number, first level of dict
            if item_id not in detailed_results_dict:
                detailed_results_dict[item_id] = {}
            if skill_type not in detailed_results_dict[item_id]:
                detailed_results_dict[item_id][skill_type] = sentences_res_list
            else:
                detailed_results_dict[item_id][skill_type].extend(sentences_res_list)
        else:
            detailed_results_dict[item["id"]] = sentences_res_list

    if args.debug:
        args.output_path = args.output_path.replace(
            ".json", f"{nsent}{nsamp}{emb_sh}{tax_v}_debug.json"
        )
    if args.detailed:
        detailed_results_dict_output = {
            key: remove_level_2(value) for key, value in detailed_results_dict.items()
        }
        # save detailed results
        # write_json(
        #     detailed_results_dict, args.output_path.replace(".json", "_DEBBBUGGG.json")
        # )
        # breakpoint()
        write_json(
            detailed_results_dict_output,
            args.output_path.replace(
                ".json", f"{nsent}{nsamp}{emb_sh}{tax_v}_detailed.json"
            ),
        )

    if args.do_extraction:
        write_json(
            detailed_results_dict,
            args.output_path.replace(".json", f"{nsent}{nsamp}{dt}_extraction.json"),
        )

    # Output final
    if not args.debug:
        categs = [
            "Technologies",
            "Technologies_alternative_names",
            "Certifications",
            "Certification_alternative_names",
        ]
        if args.data_type != "course":
            categs.append("Languages")
        clean_output_dict = {}

        if args.data_type == "course":
            for item_id, skill_type_dict in detailed_results_dict.items():
                for skill_type, detailed_res in skill_type_dict.items():
                    clean_output = {skill_type: {}}
                    clean_output[skill_type] = {categ: [] for categ in categs}
                    clean_output[skill_type]["skills"] = []

                    for ii, sample in enumerate(detailed_res):
                        for cat in categs:
                            clean_output[skill_type][cat].extend(sample[cat])

                        if "matched_skills" in sample:
                            for skill in sample["matched_skills"]:
                                clean_output[skill_type]["skills"].append(
                                    sample["matched_skills"][skill]
                                )
                    clean_output_dict[item_id] = clean_output
                    for key, value in clean_output_dict.items():
                        for kkey, vvalue in value.items():
                            clean_output_dict[key][kkey] = remove_namedef(vvalue)
        else:
            for item_id, detailed_res in detailed_results_dict.items():
                clean_output = {categ: [] for categ in categs}
                clean_output["skills"] = []

                for ii, sample in enumerate(detailed_res):
                    for cat in categs:
                        clean_output[cat].extend(sample[cat])

                    if "matched_skills" in sample:
                        for skill in sample["matched_skills"]:
                            clean_output["skills"].append(
                                sample["matched_skills"][skill]
                            )

                clean_output_dict[item_id] = clean_output
                for key, value in clean_output_dict.items():
                    clean_output_dict[key] = remove_namedef(value)

        write_json(
            clean_output_dict,
            args.output_path.replace(
                ".json", f"{nsent}{nsamp}{emb_sh}{tax_v}_clean.json"
            ),
        )
    print("Done")
    print("Extraction cost ($):", extraction_cost)
    print("Matching cost ($):", matching_cost)
    print("Total cost ($):", extraction_cost + matching_cost)

    if args.detailed:
        print(
            "Saved detailed results in",
            args.output_path.replace(
                ".json", f"{nsent}{nsamp}{emb_sh}{tax_v}_detailed.json"
            ),
        )
    print(
        "Saved clean results in",
        args.output_path.replace(".json", f"{nsent}{nsamp}{emb_sh}{tax_v}_clean.json"),
    )


if __name__ == "__main__":
    main()

# %%
