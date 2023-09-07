# Innosuisse SCESC01 Prototype

Prototype of the SubProject 01 for the SCESC Innosuisse Project.

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

We then extend the taxonomy with alternative names for skills, using the. From the current **protosp01** directory, run:


* technologies_alternative_names.csv and certifications_alternative_names.csv: alternative names for technical skills and certifications.


They are created using notebooks/taxonomy_datasets_processing.ipynb.

## Running

```shell script
python protosp01/skillExtract/pipeline_jobs_courses.py
```


## TODOs:

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


