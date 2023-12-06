# Pipeline targetted


<ins>remote branch:</ins> [datagen branch](https://github.com/epfl-ml4ed/SkillThrills/tree/datagen)


```shell
.
├── correct_taxonomy.ipynb # create the taxonomy containing exactly what you need
├── dataset_evaluation.py # contains evalutation classes and functions
├── evaluate.ipynb # use dataset_evaluation
├── evaluate_skill_and_matching_annotated.ipynb # first eval
├── final_evaluation.ipynb # benchmark for paper
├── frequency_skills.ipynb # compute the popularity measure
├── gen_utils.py # few utils function
├── generate_sub_taxonomy.py # deprecated
├── generation
│   ├── advanced_generation.ipynb # use generator
│   ├── assoc_skilled_dist.npy # studied assoc of skills
│   ├── data_generation.ipynb 
│   ├── feedback.py # classes used to annotate the spans of the generation
│   ├── feedback_prompt_template.py # prompt used for feedback
│   ├── frequency_vals.json # use as popularity measures of each ESCO skills
│   ├── gen_prompt_template.py
│   ├── generated
│   │   ├── SKILLSPAN # generated based on SkillSpan
│   │   └── PROTOTYPE # generated prototype
│   ├── generation_feedback.ipynb # annotate spans
│   ├── generator.py # Generator class for datagen
│   ├── new_data_gen.ipynb # use generator
│   ├── prototype_ds_generation.ipynb # use generator
│   ├── skillspan_data_generation.ipynb # use generator
│   └── tech_data_generation.ipynb # use generator
├── perf_with_retrieval.ipynb
├── ppl_inputs_simple.json # popularity measure
├── preds.json # old
├── preds_synth.json # old
├── sim_description_skill.ipynb # EDA
└── support_benchmark.py # use for final benchmark computation
```

### Compute the prediction




Everything happens in `dataset_evaluation.ipynb`. The file contains the `Predictor` class that takes as input :
- `test_domain` $\in$ `["SkillSpan-dev", "SkillSpan-test", "Proto"]`: The domain of the labels we want to find. To make sure we look for skills in the correct taxonomy.
- `train_domain` $\in$ `["SkillSpan", "Proto"]`: if we use specific support samples we need to specify where we want to get them.
- `candidate_method` : the candidate method we use for candidate selection.
- All the arguments for the underlying pipeline. `prompt_type`, `data_type` and `datapath` should not be changed for english predictions. 


You may have to put your own value for the location of `ESCO_DIR`.

According to the test domain we will load the according emb taxonomy that we use either in `ESCO_DIR + "dev_skillspan_emb.pkl"` or `ESCO_DIR + "test_skillspan_emb.pkl"` depending of the split.

If we want to use a train dataset you'll have either `"./generation/generated/SKILLSPAN/ANNOTATED_SUPPORT_SET.pkl"` for `"./generation/generated/SKILLSPAN/ANNOTATED_SUPPORT_SET.pkl"` depending on the domain itself.


Now that everything is setup we can use the prediction function :

```python
from dataset_evaluation import Predictor

predictor = Predictor(test_domain="SkillSpan-test",
                      train_domain="SkillSpan", ## no training will be used anyway
                      candidates_method="mixed")

## ds is in record format :
## [{"sentence": ..., "skills": [...]}, ...]
sp = predictor.pipeline_prediction(ds, 
                                    support_type=None) ## base prediction


## demonstrations used for matching
sp = predictor.pipeline_prediction(ds, 
                                    support_type="kNN",
                                    support_size_match=1,
                                    nb_candidates=10) 


## demonstrations used for extraction
sp = predictor.pipeline_prediction(ds, 
                                    support_type=None,
                                    support_size_extr=7) 


## demonstrations used for both
sp = predictor.pipeline_prediction(ds, 
                                    support_type="kNN",
                                    support_size_match=1,
                                    nb_candidates=10,
                                    support_size_extr=7) 
```

The shots for matching are prepare in the following function :

```
Predictor > generate_support_set > prepare_support
Predictor > generate_support_set > prepare_support_extr
```

The latter can be used to format any sample to look like the input to ChatGPT for the extraction part and the matching part.

You could use these to generate input output for LLaMa use the SkillSpans support set.


You can see an example an how to generate the dataset by looking at [this notebook](./dataprep_llama_ft.ipynb).

You can generate more samples or improve them by looking at :
- [feedack.py](./generation/feedack.py)
- [feedback prompt template](./generation/feedback_prompt_template.py)