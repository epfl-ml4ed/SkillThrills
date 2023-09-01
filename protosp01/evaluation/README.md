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
2/3 high quality datasets
(with B I O tagging --> evaluate using Span-F1, exact span matching, around 60-65%)
Skillspan: https://github.com/kris927b/SkillSpan/tree/main/data/json
Green: https://github.com/acp19tag/skill-extraction-dataset/tree/main/preprocessed_data
gnehm (huge, in German, for the Swiss job market): https://aclanthology.org/2022.nlpcss-1.2.pdf?

--> evaluation pipeline:
https://huggingface.co/spaces/evaluate-metric/seqeval/blob/main/seqeval.py

2) skill matching (using ESCO as taxonomy)
Synthetic:
https://huggingface.co/datasets/jensjorisdecorte/Synthetic-ESCO-skill-sentences
Annotated:
https://github.com/jensjorisdecorte/Skill-Extraction-benchmark/tree/main

