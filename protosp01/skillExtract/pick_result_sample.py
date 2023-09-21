# %%
import os
import pandas as pd
import json

# %%
print(os.getcwd())
os.chdir("results")

"""Get top N matched and extracted"""
N = 3


with open("job_gpt-3.5-turbo_detailed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# init lists to store highest n matched and extracted skills
top_n_matched = []
top_n_extracted = []

matched_key = "matched_skills"
extracted_key = "extracted_skills"


for key, sentences in data.items():
    total_matched = 0
    total_extracted = 0

    for sent in sentences:
        total_matched += len(sent[matched_key])
        total_extracted += len(sent[extracted_key])

    key_matched = (key, total_matched)
    key_extracted = (key, total_extracted)

    top_n_matched.append(key_matched)
    top_n_extracted.append(key_extracted)

top_n_matched = sorted(top_n_matched, key=lambda x: x[1], reverse=True)[:N]
top_n_extracted = sorted(top_n_extracted, key=lambda x: x[1], reverse=True)[:N]

print("top_n_matched:", top_n_matched)
print("top_n_extracted:", top_n_extracted)

# %%
