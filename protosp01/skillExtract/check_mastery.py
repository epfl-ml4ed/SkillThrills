# %%
import os
import pandas as pd
import random

# %%
from utils import *

# extract and match through pipeline
import argparse
import numpy as np
from split_words import Splitter

# %%

# fmt: off

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, help="Path to source data", default = "../data/taxonomy/check_mastery_per_sentence_flagged.json")
parser.add_argument("--taxonomy", type=str, help="Path to taxonomy file in csv format", default = "../data/taxonomy/taxonomy_V4.csv")
parser.add_argument("--api_key", type=str, help="openai keys", default = API_KEY)
parser.add_argument("--model", type=str, help="Model to use for generation", default="gpt-3.5-turbo")
parser.add_argument("--temperature", type=float, help="Temperature for generation", default=0)
parser.add_argument("--max_tokens", type=int, help="Max tokens for generation", default=40)
parser.add_argument("--shots", type=int, help="Number of demonstrations, max = 5", default=5)
parser.add_argument("--top_p", type=float, help="Top p for generation", default=1)
parser.add_argument("--frequency_penalty", type=float, help="Frequency penalty for generation", default=0)
parser.add_argument("--presence_penalty", type=float, help="Presence penalty for generation", default=0)
parser.add_argument("--candidates_method", type=str, help="How to select candidates: rules, mixed or embeddings. Default is rules", default="rules")
parser.add_argument("--output_path", type=str, help="Output for evaluation results", default="results/")
parser.add_argument("--prompt_type", type=str, help="Prompt type, from the prompt_template.py file. For now, only \"detailed\". default is empty.", default="")
parser.add_argument("--num-samples", type=int, help="Last N elements to evaluate (the new ones)", default=10)
parser.add_argument("--num-sentences", type=int, help="by how many sentences to split the corpus", default=2)
parser.add_argument("--do-extraction", action="store_true", help="Whether to do the extraction or directly the matching")
parser.add_argument("--do-matching", action="store_true", help="Whether to do the matching or not")
parser.add_argument("--word-emb-model", type=str, help="Word embedding model to use", default="agne/jobBERT-de")
parser.add_argument("--debug", action="store_true", help="Keep only one sentence per job offer / course to debug")
parser.add_argument("--detailed", action="store_true", help="Generate detailed output")
parser.add_argument("--ids", type=str, help="Path to a file with specific ids to evaluate", default=None)
parser.add_argument("--samplepct", type=int, help="Percentage of data to sample", default=100)

# fmt: on

args = parser.parse_args()

word_emb = args.word_emb_model
word_emb_model = AutoModel.from_pretrained(word_emb)
word_emb_tokenizer = AutoTokenizer.from_pretrained(word_emb)

# %%
# os.chdir("../data")
# assert os.getcwd().split("/")[-1] == "taxonomy", "check path"

# Reading in the data
mastery_df = pd.read_csv("../data/taxonomy/mastery_edited.csv")
mastery_dict = dict(zip(mastery_df["Level 3"], mastery_df["Level 2"]))

# %%

job_df = pd.DataFrame.from_records(read_json("../data/raw/vacancies.json")[0])
job_df["fulltext"] = job_df["name"] + "\n" + job_df["description"]

if args.samplepct < 100:
    print(f"Sampling {args.samplepct}% of data")
    job_df = job_df.sample(frac=args.samplepct / 100, random_state=42)

job_df["text_num_words"] = job_df["fulltext"].apply(lambda x: len(x.split()))
job_df = job_df[job_df["text_num_words"] > 100].drop(
    columns=["text_num_words"]
)  # 100 words

try:
    # load job_df from csv
    print("Loading german job vacancies from csv")
    job_df = pd.read_csv("../data/processed/vacancies_german.csv")
except:
    print("German job vacancies csv not found. Filtering from json...")
    print(len(job_df))
    job_df["language"] = job_df["fulltext"].apply(detect_language)
    print(job_df["language"].value_counts())
    job_df = job_df[job_df["language"] == "de"]

    # save job_df to csv
    job_df.to_csv("../data/processed/vacancies_german.csv")

keep_cols = ["id", "name", "description", "fulltext"]
job_df = job_df[keep_cols]


# %%
# set flag for if fulltext contains any values from mastery_dict key = Beginner, Intermediate, and Expert
def check_mastery(text, level, mastery_dict=mastery_dict):
    # return any(level == mastery_dict[key] for key in mastery_dict.keys() if key in text):
    for key in mastery_dict.keys():
        if key in text and level == mastery_dict[key]:
            return key
    return None


job_df["Beginner"] = job_df["fulltext"].apply(lambda x: check_mastery(x, "Beginner"))
job_df["Intermediate"] = job_df["fulltext"].apply(
    lambda x: check_mastery(x, "Intermediate")
)
job_df["Expert"] = job_df["fulltext"].apply(lambda x: check_mastery(x, "Expert"))

print("======= Overall Stats =======")
print("number of jobs:", len(job_df))
print("number of jobs with Beginner flag:", job_df["Beginner"].notnull().sum())
print("number of jobs with Intermediate flag:", job_df["Intermediate"].notnull().sum())
print("number of jobs with Expert flag:", job_df["Expert"].notnull().sum())

job_df["anyflag"] = job_df[["Beginner", "Intermediate", "Expert"]].any(axis=1)
print("number of jobs with any flag:", job_df["anyflag"].sum())

# %%
# print("======= Sample (10%) Stats =======")
# print("check json and csv")
# sample_df = job_df.sample(frac=0.1, random_state=42).copy()

sample_df = job_df.copy()
sample_dict = sample_df.to_dict("records")

results_dict = {}

# %%
for _, item in enumerate(sample_dict):
    sentences = split_sentences(item["fulltext"])
    sentences_res_list = []

    for ii in range(0, len(sentences), args.num_sentences):
        sentences_res_list.append(
            {
                "sentence": ". ".join(sentences[ii : ii + args.num_sentences]),
            }
        )

    for idx, sample in enumerate(sentences_res_list):
        sample["Beginner"] = check_mastery(sample["sentence"], "Beginner")
        sample["Intermediate"] = check_mastery(sample["sentence"], "Intermediate")
        sample["Expert"] = check_mastery(sample["sentence"], "Expert")
    results_dict[item["id"]] = sentences_res_list

# output results_dict to json
write_json(results_dict, "../data/taxonomy/check_mastery_per_sentence.json")

# keep only those examples that have at least one flag
filtered_dict = {
    key: [
        entry
        for entry in value
        if entry["Beginner"] or entry["Intermediate"] or entry["Expert"]
    ]
    for key, value in results_dict.items()
    if any(
        entry["Beginner"] or entry["Intermediate"] or entry["Expert"] for entry in value
    )
}
write_json(filtered_dict, "../data/taxonomy/check_mastery_per_sentence_flag.json")

# # %%
# # fetch 10 samples from filtered_dict with only beginner flag
# beginner_sentences = []
# intermediate_sentences = []
# expert_sentences = []
# # Separating sentences based on flags
# for key, values in filtered_dict.items():
#     for value in values:
#         beginner_flag = value.get("Beginner")
#         intermediate_flag = value.get("Intermediate")
#         expert_flag = value.get("Expert")
#         if beginner_flag:
#             beginner_sentences.append(value["sentence"])
#         if intermediate_flag:
#             intermediate_sentences.append(value["sentence"])
#         if expert_flag:
#             expert_sentences.append(value["sentence"])
# # Randomly sample 10 sentences for each flag
# beginner_samples = random.sample(beginner_sentences, min(10, len(beginner_sentences)))
# intermediate_samples = random.sample(
#     intermediate_sentences, min(10, len(intermediate_sentences))
# )
# expert_samples = random.sample(expert_sentences, min(10, len(expert_sentences)))

# # Print the samples
# print("Beginner Samples:")
# for sentence in beginner_samples:
#     print(sentence)

# print("\nIntermediate Samples:")
# for sentence in intermediate_samples:
#     print(sentence)

# print("\nExpert Samples:")
# for sentence in expert_samples:
#     print(sentence)


# breakpoint()


# %%
def get_lowest_level(entry):
    if entry["Beginner"]:
        return "Beginner"
    elif entry["Intermediate"]:
        return "Intermediate"
    elif entry["Expert"]:
        return "Expert"
    else:
        return None


results_dict_flag = {}
for job_id, details in results_dict.items():
    results_dict_flag[job_id] = [
        {
            "sentence": detail["sentence"],
            "flag": get_lowest_level(detail),
        }
        for detail in details
    ]

#
filtered_data = {}
for key, sentences in results_dict_flag.items():
    filtered_data[key] = [
        sentence for sentence in sentences if sentence["flag"] is not None
    ]

filtered_data = {key: value for key, value in filtered_data.items() if value}

# print(filtered_data)


# output results_dict_flag_subset to json
write_json(filtered_data, "../data/taxonomy/check_mastery_per_sentence_flagged.json")

# # %%
# agg_results_dict = {}

# for unique_id, result in results_dict.items():
#     num_sentences = len(result)
#     num_beginner = sum([x["Beginner"] for x in result])
#     num_intermediate = sum([x["Intermediate"] for x in result])
#     num_expert = sum([x["Expert"] for x in result])
#     agg_results_dict[unique_id] = {
#         "num_sentences": num_sentences,
#         "num_beginner": num_beginner,
#         "num_intermediate": num_intermediate,
#         "num_expert": num_expert,
#     }

# # convert to dataframe
# agg_results_df = pd.DataFrame.from_dict(agg_results_dict, orient="index")
# agg_results_df.to_csv("../data/taxonomy/check_mastery_per_job.csv")
# # # %%

# agg_results_df["total_flags"] = agg_results_df[
#     ["num_beginner", "num_intermediate", "num_expert"]
# ].sum(axis=1)

# # get distribution of total flags
# agg_results_df["pct_flagged"] = (
#     agg_results_df["total_flags"] / agg_results_df["num_sentences"]
# )

# # %%
# print("======= percentage of sentences flagged =======")
# print(agg_results_df["pct_flagged"].describe())

# %%
data_type = "ms"
args.output_path = args.output_path + data_type + "_" + args.model + ".json"

print(os.getcwd())
taxonomy, _, _ = load_taxonomy(args)
with open(args.datapath) as f:
    data = json.load(f)
print("loaded data:", len(data), "sentences")
# convert dict to list

extraction_cost = 0
matching_cost = 0
detailed_results_dict = {}


for job_id, sentences in data.items():
    # for job_id, sentences in tqdm(enumerate(data)):
    print("job id:", job_id)
    print(sentences)
    sentences_res_list = [sentence for sentence in sentences]
    print(sentences_res_list)
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
                method=args.candidates_method,
            )
            sentences_res_list[idxx] = sample

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
        # TODO find a way to correctly identify even common strings (eg 'R')! (AD: look in utils exact_match)
        # Idem for finding C on top of C# and C++
        # TODO update alternative names generation to get also shortest names (eg .Net, SQL etc) (Syrielle)
        detailed_results_dict[item["id"]] = sentences_res_list
        print(os.getcwd())

        print(args.output_path)

        results_folder = "results"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        write_json(
            detailed_results_dict, args.output_path.replace(".json", "_detailed.json")
        )

        # Output final
        if not args.debug:
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
                            clean_output["skills"].append(
                                sample["matched_skills"][skill]
                            )
                clean_output_dict[item_id] = clean_output
        write_json(clean_output_dict, args.output_path.replace(".json", "_clean.json"))
    print("Done")
    print("Extraction cost ($):", extraction_cost)
    print("Matching cost ($):", matching_cost)
    print("Total cost ($):", extraction_cost + matching_cost)
