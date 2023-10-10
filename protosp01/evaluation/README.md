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
DONE

Message system for prompting the openai api, for the demonstrations and feedback given, to simulate a conversation
DONE

New prompt / improvement (more detailed)
DONE

Example selections: from the train set, as separate messages. 
Do knn retrieval 
Sample some "None" examples
DONE

Dataset-specific prompt

There are many low-quality examples: too long, broken, with missing words. Especially saw then in gnehm. Could be captured using chatGPT.

Supervized baseline: Mike?

Argue that the BIO tagging is not adapted for the task, example:
/!\ -- TODO get many more !
Sentence: commercial savvy with excellent conceptual and analytical thinking skills and will have an
Extracted:
commercial savvy
conceptual thinking
analytical thinking
GT: analytical thinking skills
re-do the annotation: still need to refer to a specific span for expaliability!
+ only the test set?
What about NER, how do they do this? e.g. "presidents trump and Biden" ? TODO

+ amount of context to extract the skill (eg info about the company)

+ implicit

Re-do the annotation with fine-grained tags (methodo, social, factual)

https://huggingface.co/Universal-NER/UniNER-7B-all
https://universal-ner.github.io/
TODO check list of entities

TODO descriptive statistics:
- avg nb of skills per sentence
- avg span length
(syrielle)
- overlap in terms of skills between train and test (does the model identify new skills in the test set?)
(reason 1: imapct on results for fietuned modesl / reason 2: impact on the generalization of the model for skills in the train set but not test set)

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

# Mike: 
train entity linking models on the above datasets
models: Blink (encoder model bert, better) and Genre + GENIE trie
Blink is pre-trained on wikipedia but really good because there's a lot of esco sklls in wikipedia. TODO quantify overlap.
