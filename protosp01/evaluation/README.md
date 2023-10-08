# Code

```
python main.py --dataset_name gnehm --download
python main.py --dataset_name gnehm --run

```







# Recent papers

https://arxiv.org/pdf/2307.03539.pdf
This work proposes an end-to-end zero-shot system for skills extraction (SE) from job descriptions based on LLMs and achieves supposedly promising results on skills extraction using ESCO.

https://arxiv.org/pdf/2209.05987.pdf
Here they propose an end-to-end system for SE based on distant supervision through literal matching and negative sampling strategies (with ESCO). It is evaluated to improve the generalization of skill extraction towards implicitly mentioned skills (so not written in the text, e.g., “being able to work in groups”  “teamwork”)

Synthetic data + real data:
https://arxiv.org/pdf/2307.10778.pdf


# Metrics
2 levels: document-level and skill-level.
(predicted list VS gold list)
With and without exact string matching
TODO how to improve?

Double check the code: is it strict or loose


# Method

Feedback: identify if the sentence is correctly replicated (ner), if the extracted skills are really in the sentence (extract)

New prompt / improvement ?

    "ner": "You are given a sentence from a job description. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.\n",

    "extract":"You are given a sentence from a job description. Extract all the skills and competencies that are required from the candidate, printing one per line. Make sure to keep the exact same words as found in the sentence. If the sentence doesn't contain any skill, output \"None\".\n"

Dataset-specific prompt

Example selections: from the train set, as separate messages. Do knn retrieval + sample some "None" examples.


https://huggingface.co/Universal-NER/UniNER-7B-all
https://universal-ner.github.io/
TODO check list of entities

# Evaluation data

1) skill extraction
Eval: 6 datasets
Fijo: French job offers, 500 examples, insurance domain, soft skills.
Gnehm: German job offers, IT domain, 25k examples
Green: English job offers, 10k examples, skills + other elements (domain, occupation, qualification, experience)
Sayfullina: English but already split into sentences, 7000 exampes, soft skills
Kompetencer: Danish job offers, 1200 examples, skills and knowledge
SkillSpan: English, IT and house domain, 12k examples, not so high quality, skills and knowledge.

(with B I O tagging --> evaluate using Span-F1, exact span matching)

--> evaluation: Using the "evaluate" package.
https://huggingface.co/spaces/evaluate-metric/seqeval



2) skill matching (using ESCO as taxonomy)
Synthetic:
https://huggingface.co/datasets/jensjorisdecorte/Synthetic-ESCO-skill-sentences
Annotated:
https://github.com/jensjorisdecorte/Skill-Extraction-benchmark/tree/main


# Limitation: 
with the ner style, we get extract spans for skills, while sometimes the whole sentence means the skill
