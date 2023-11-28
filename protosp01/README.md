# Innosuisse SCESC01 Prototype

Prototype of the SubProject 01 for the SCESC Innosuisse Project.

## Environment Setup
A good starting point for a conda environment is the file [`environment_proto.yml`](../environment_proto.yml).

Note: there might still be some package dependencies that are problematic depending on the environment. 

## Data Input

### 1. Skills Data

We extract skills from 3 sources:

* resumes
* job description (="vacancy")
* courses (= "learning opportunity")

These files should be stored in the folder `data/raw/`. 
Jobs and courses are downloaded and updated regularly from EvrLearn platform, using the script [protosp01/update_platform_data.sh](protosp01/update_platform_data.sh). From the current **protosp01** directory, run:

```bash
bash update_platform_data.sh
```

#### 1.1 Pre-processing

To pre-process the job and course files, run from the current **protosp01** directory:

```shell
python skillExtract/preprocess_jobs_courses.py
```

Key arguments: 
* `--datatype` # choose between job_evl or course_evl

### 2. Taxonomy Data

We match extracted skills using an existing taxonomy of skills. All the taxonomy files are saved in the folder `data/taxonomy/`. They are updated regularly from [SP2's online Excel sheet](https://universitaetstgallen.sharepoint.com/:x:/r/sites/O365-PRJ-IWI-Research/_layouts/15/doc2.aspx?sourcedoc=%7BC9BB110D-819F-4469-9127-054ABB53EF09%7D&file=KompetenzmodellKodierbuch.xlsx&action=default&mobileredirect=true&cid=34b78d05-ea86-4ef7-b348-18d57854d510).
(*access required*)

These files should be stored in the folder data/taxonomy/.

#### 2.1 Pre-processing

From these files, we first pre-process (i.e. drop empty and assign unique ids) by running from the current **protosp01** directory:

```shell
python skillExtract/get_taxonomy_elements.py
```

This will create the following files:

* taxonomy_V4.csv: skill names, desciptions, examples, divided into levels.
* tech_certif_lang.csv: list of technical skills, certifications and languages.

#### 2.2 Extending

We then extend the taxonomy with alternative names for skills using OpenAI. From the current **protosp01** directory, run:

```shell
python skillExtract/extend_taxonomy_elements.py
```

This will create the following files, which contain the alternative names for technologies and certifications:

* technologies_alternative_names.csv
* certifications_alternative_names.csv

## Running the Prototype

From the current **protosp01** directory, run:

```shell script
python skillExtract/pipeline_jobs_courses.py
```

Key arguments for the pipeline:
- `--do-extraction` # do step 1: skill extraction
- `--candidates_method mixed` # do step 2: mixed method is rule-based+embedding-based
- `--do-matching`# do step 3: skill matching
- `--detailed` # this is just our output format (detailed output has the most info)
- `--max_tokens 1000` # for chatgpt
- `--num_sentences 2` # number of sentences to process at a time
- `--prompt_type wreqs` # choose between skills, wlevels, wreqs for skill-only, skills+mastery level and skills+mastery level+mandatory/optional extraction
- `--num-samples 0` select number of documents to process (0 for all)
- `--language de` # language of the documents to process (currently supports only de end-to-end)
- `--taxonomy ../data/taxonomy/taxonomy_V4.csv` # taxonomy file to use for matching
- `--datapath ../data/processed/job_evl_all.csv` # the job or course file to process

Points to note:
* The job or course file is generated using the step 1.1 above.

* The taxonomy file is generated using the 2.1 and 2.2 above, but is already in the repo.

Sample run command:

```shell script
python skillExtract/pipeline_jobs_courses.py --num-sentences 2 --do-extraction --do-matching --detailed --candidates_method mixed --max_tokens 2000  --prompt_type wreqs --taxonomy ../data/taxonomy/taxonomy_V4.csv --datapath ../data/processed/job_evl_all.csv --num-samples 0 --language de
```


## TODOs (out of date - ignore by Alex):

high-level TODOs
look for low-level TODOs in the code.

<!-- (extraction -> candidate generation -> matching) -->
<!-- we are only doing on job postings for now -->
<!-- We are doing this for matched skills on the extracted version (after matching but on the extracted naming) -->


- Classify skills as beginner - intermediate - expert for job postings!
<!-- - * try a few functions to try it out first (ex. have extract skills ) -->

- Make the pipeline for courses and job postings work for CVs as well (anything in docx format)
<!-- - * make pipeline_cv to be able to load CVs (txt or docx or csv formats) -->

- Make evaluation code (Output metrics)
  1. BIO evaluation (see README in evaluation folder)
  2. SP2 annotated data - comparing strings between what was extracted in pipeline and what is in annotated data (like CVtest final)
    

We need to know how important each skill is for each job posting. binary: key VS optional.

TODO can we do the extraction-matching as part as one conversation with GPT? Just as additional messages for each paragraph.

Skill extraction and matching: what we want to output is Level 2 !!!

Alex 3 suggestions:
skill id + level
{ matched_skills: [{skill id: skill level}, {2: 3}] }
skill id
{matched_skills: [1skill id 1, skill id 2, skill id 2]}
skill name
{matched_skills: [“sprachkenntnisse”, skill name 2, skill name 3]}
--> At the document level

output two json: the debug one with all info then the smaller version according to Alex's suggestion.


/!\ often there is only level 2 or 3. Example column is not level 4!
+ all other tabs in the taxonomy: tecnology, language, certification.
For these ones we only do exact string matching.

TODO give job title and course name in the prompt?

TODO Excluding job postings with less than N skills / M sentences

TODO output only the skill id when generating the "clean" file. Foradd an option with the full name of the skill on top of the id for debugging

TODO in the readme: explain each element of the _detailed json file.

Data in the github repo
--> TODO update unique id to be really unique ! When there is only example and not level 3 or 4 its not a new skill

TODO split the job postings into pragraphs instead of sentences
issue: few shots example size don't fiw in contex, + entity extraction is better with not the full context.


For tech: 
-get ALL technologies that have the same alternative names ! e.g. MS Office.
-Get more alternative names for technologies using chatGPT --> Shorter names! e.g. .Net, SQL, etc...

For certifs:
- look for occurences of "Certifications" in ALL JOBS

Extract certification from English jobs

Send synonyms to Ellen for certification.
Get examples from Ellen --> pair of sentence + extracted certification

TODO send results to Marco (example of annotated paragraphs from skills and courses)


