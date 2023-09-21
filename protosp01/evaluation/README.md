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

# Evaluation data

1) skill extraction
Eval: 6 datasets
Fijo: French job offers, 500 examples, insurance domain, soft skills.
Gnehm: German job offers, IT domain, 25k examples
Green: English job offers, 10k examples, skills + other elements (domain, occupation, qualification, experience)
Sayfullina: English but already split into sentences, 7000 exampes, soft skills
Kompetencer: Danish job offers, 1200 examples, skills and knowledge
SkillSpan: English, IT and house domain, 12k examples, not so high quality, skills and knowledge.

(with B I O tagging --> evaluate using Span-F1, exact span matching, around 60-65%)

--> evaluation: Using the "evaluate" package.
https://huggingface.co/spaces/evaluate-metric/seqeval



2) skill matching (using ESCO as taxonomy)
Synthetic:
https://huggingface.co/datasets/jensjorisdecorte/Synthetic-ESCO-skill-sentences
Annotated:
https://github.com/jensjorisdecorte/Skill-Extraction-benchmark/tree/main

