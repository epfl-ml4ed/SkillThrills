# %%
import pandas as pd
import json
import os

os.getcwd()

# %%
# read in json from different folder
os.chdir("../results")
with open("job_gpt-3.5-turbo_2sent_n1500_jBd_detailed.json", "r") as f:
    results = json.load(f)

# %%
# read in taxonomy from different folder
os.chdir("../../data/taxonomy")
taxonomy = pd.read_csv("taxonomy_V4.csv")
keep_cols = [
    "unique_id",
    "Type Level 1",
    "Type Level 2",
    "Type Level 3",
    "Type Level 4",
]
taxonomy = taxonomy[keep_cols]

# %%
len(results)
matched_ids = {}
for job_id in results:
    job = results[job_id]
    for sent in job:
        matched = sent["matched_skills"]
        for skill, data in matched.items():
            if data["unique_id"] not in matched_ids:
                matched_ids[data["unique_id"]] = 1
            else:
                matched_ids[data["unique_id"]] += 1
            # matched_ids.append(data["unique_id"])

# %%
taxonomy["matched_count"] = taxonomy["unique_id"].map(matched_ids)

# %%
# group by each level to see what the most common skills are

lv1 = (
    taxonomy.groupby("Type Level 1")
    .sum()
    .sort_values(by="matched_count", ascending=False)
    .drop(columns="unique_id")
)
lv1
# %%

lv2 = (
    taxonomy.groupby(["Type Level 1", "Type Level 2"])
    .sum()
    .sort_values(by="matched_count", ascending=False)
    .drop(columns="unique_id")
)
lv2
# %%
lv3 = (
    taxonomy.groupby(["Type Level 2", "Type Level 3"])
    .sum()
    .sort_values(by="matched_count", ascending=False)
    .drop(columns="unique_id")
)
lv3

# %%
lv4 = (
    taxonomy.groupby(["Type Level 2", "Type Level 3", "Type Level 4"])
    .sum()
    .sort_values(by="matched_count", ascending=False)
    .drop(columns="unique_id")
)
lv4

# %%
