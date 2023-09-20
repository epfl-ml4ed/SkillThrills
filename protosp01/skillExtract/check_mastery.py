# %%
import os
import pandas as pd

# %%
from utils import *

NUM_SENTENCES = 1

# %%
os.chdir("../data")
# assert os.getcwd().split("/")[-1] == "taxonomy", "check path"

# Reading in the data
mastery_df = pd.read_csv("taxonomy/mastery_edited.csv")
mastery_dict = dict(zip(mastery_df["Level 3"], mastery_df["Level 2"]))

# %%

job_df = pd.DataFrame.from_records(read_json("raw/vacancies.json")[0])
job_df["fulltext"] = job_df["name"] + "\n" + job_df["description"]

job_df["text_num_words"] = job_df["fulltext"].apply(lambda x: len(x.split()))
job_df = job_df[job_df["text_num_words"] > 100].drop(
    columns=["text_num_words"]
)  # 100 words


keep_cols = ["id", "name", "description", "fulltext"]
job_df = job_df[keep_cols]


# %%
# set flag for if fulltext contains any values from mastery_dict key = Beginner, Intermediate, and Experte
def check_mastery(text, level):
    return any(level == mastery_dict[key] for key in mastery_dict.keys() if key in text)


job_df["Beginner"] = job_df["fulltext"].apply(lambda x: check_mastery(x, "Beginner"))
job_df["Intermediate"] = job_df["fulltext"].apply(
    lambda x: check_mastery(x, "Intermediate")
)
job_df["Experte"] = job_df["fulltext"].apply(lambda x: check_mastery(x, "Experte"))

print("======= Overall Stats =======")
print("number of jobs:", len(job_df))
print("number of jobs with Beginner flag:", job_df["Beginner"].sum())
print("number of jobs with Intermediate flag:", job_df["Intermediate"].sum())
print("number of jobs with Experte flag:", job_df["Experte"].sum())

job_df["anyflag"] = job_df[["Beginner", "Intermediate", "Experte"]].any(axis=1)
print("number of jobs with any flag:", job_df["anyflag"].sum())

# %%
print("======= Sample (10%) Stats =======")
print("check json and csv")
sample_df = job_df.sample(frac=0.1, random_state=42).copy()
sample_dict = sample_df.to_dict("records")

results_dict = {}
# %%
for _, item in enumerate(sample_dict):
    sentences = split_sentences(item["fulltext"])
    sentences_res_list = []

    for ii in range(0, len(sentences), NUM_SENTENCES):
        sentences_res_list.append(
            {
                "sentence": ". ".join(sentences[ii : ii + NUM_SENTENCES]),
            }
        )

    for idx, sample in enumerate(sentences_res_list):
        sample["Beginner"] = check_mastery(sample["sentence"], "Beginner")
        sample["Intermediate"] = check_mastery(sample["sentence"], "Intermediate")
        sample["Experte"] = check_mastery(sample["sentence"], "Experte")
    results_dict[item["id"]] = sentences_res_list

# output results_dict to json
write_json(results_dict, "taxonomy/check_mastery_per_sentence.json")

# %%
# for each id in results_dict, get number of total sentences and number of sentences with each flag

agg_results_dict = {}

for unique_id, result in results_dict.items():
    num_sentences = len(result)
    num_beginner = sum([x["Beginner"] for x in result])
    num_intermediate = sum([x["Intermediate"] for x in result])
    num_experte = sum([x["Experte"] for x in result])
    agg_results_dict[unique_id] = {
        "num_sentences": num_sentences,
        "num_beginner": num_beginner,
        "num_intermediate": num_intermediate,
        "num_experte": num_experte,
    }

# convert to dataframe
agg_results_df = pd.DataFrame.from_dict(agg_results_dict, orient="index")
agg_results_df.to_csv("taxonomy/check_mastery_per_job.csv")
# %%
