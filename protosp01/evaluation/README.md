# Code

```
python main.py --dataset_name gnehm --download
python main.py --dataset_name gnehm --run

```


# Metrics
2 levels: document-level and skill-level.
(predicted list VS gold list)
With and without exact string matching
TODO how to improve?


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
TODO

Debug the NER pipeline
See run.py:
postprocess_ner_prompt
check_format_response
The issuse stems from the fact that most of the times, the model fails to respect the template, and then getting the extracted skills is difficult. Need to implement some rules.
TODO

Run large-scale on full datasets
TODO

There are many low-quality examples: too long, broken, with missing words. Especially saw then in gnehm. Could be captured using chatGPT.
TODO (optional)


# Evaluation data

 6 datasets
Fijo: French job offers, 500 examples, insurance domain, soft skills.
Gnehm: German job offers, IT domain, 25k examples
Green: English job offers, 10k examples, skills + other elements (domain, occupation, qualification, experience)
Sayfullina: English but already split into sentences, 7000 exampes, soft skills
Kompetencer: Danish job offers, 1200 examples, skills and knowledge
SkillSpan: English, IT and house domain, 12k examples, not so high quality, skills and knowledge.

(with B I O tagging --> evaluate using Span-F1, exact span matching)

--> evaluation: Using the "evaluate" package.
https://huggingface.co/spaces/evaluate-metric/seqeval
