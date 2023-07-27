# Innosuisse SCESC03 Prototype
Prototype of the SubProject 03 for the SCESC Innosuisse Project.

Everything is subject to change. Do not hesistate to provide me with feedback/comments.

For now I generate all the files that I will require from Syrielle and Marco.


## Installation

For now the installation requires [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Will add Docker later.

```shell script
git clone https://github.com/epfl-ml4ed/protosp03.git
cd protosp03
conda env create -f environment.yaml
conda activate proto
```

## Input
We have 4 raw files as input, for now they are json but this can be changed. 

The files are in the folder data/raw/synthetic. 

They can be generated using the script [protosp03/data/make_synthetic.py](protosp03/data/make_synthetic.py) and the config file [protosp03/config/synthetic.yaml](protosp03/config/synthetic.yaml)
- **skills.json** : a dict whoses keys are the skills name and values are the skill id 
- **jobs.json** : a dict whoses keys are the job id and values are the list of skill required
- **resume.json** : a dict whoses keys are the resume id (or profile id) and values are the list of skill present
- **courses.json** : a dict whoses keys are the job id and values are the list of skill required and the list of skills provided

The [config](protosp03/config/synthetic.yaml) file provides requirements about the synthetic dataset (nb of skills, jobs, resume) and the path where the files are to be saved.  

To generate the synthetic data files:  
```shell script
python protosp03/data/make_synthetic.py --config protosp03/config/synthetic.yaml --seed 1
```

The default value for the seed is 42. 

## Pre-Processing
Data preprocessing is done with the script [protosp03/data/inverted_index.py](protosp03/data/inverted_index.py) and the config file [protosp03/config/inverted_index.yaml](protosp03/config/inverted_index.yaml).

For now the main idea of the pre-processing is simply to create inverted indexes that can be used for effeicient search. The [config](protosp03/config/inverted_index.yaml) contains the path of the directory that contains the raw files and the path of the directory where to save the inverted indexes.

To pre-process the raw data files:  

```shell script
python protosp03/data/inverted_index.py --config protosp03/config/inverted_index.yaml
```

## User-journey Version 1.0

### Running
To try the first version of the functions I can do in the user journey, run 

```shell script
python -m tests.user_journey01 --seed 1
```
 It should display this:

```shell script
Considering Profile resume_1 with the skills: {0, 1, 3}
        We are assuming that the Profile is interested in Job jobs_4 that requires the skills: {0, 3}
        The matching between the profile and the desired job is: 100%
        Printing the attractiveness of each skill of the profile and comparing to other learners:
                Skill 0 is required for 60% of the jobs on the market and 80% of the learners have it
                Skill 1 is required for 20% of the jobs on the market and 60% of the learners have it
                Skill 3 is required for 40% of the jobs on the market and 60% of the learners have it
        The overall attractiveness of the profile is: 73%
        Printing the matching of the profile with respect to each job (from most compatible to least compatible):
                Job jobs_1 has a matching of 100%
                Job jobs_4 has a matching of 100%
                Job jobs_3 has a matching of 100%
                Job jobs_0 has a matching of 66%
        Printing the matching of the profile with respect to each course (from most compatible to least compatible):
                Course course_3 has a matching of 100%
                        If the profile takes the course course_3, it will learn the skills: {4}
                        The matching with the job jobs_0 will increase from 66% to: 100%
                        The overall attractiveness of the profile will increase from 73 to: 100
                Course course_1 has a matching of 66%
                        If the profile takes the course course_1, it will learn the skills: {4}
                        The matching with the job jobs_0 will increase from 66% to: 100%
                        The overall attractiveness of the profile will increase from 73 to: 100
                Course course_4 has a matching of 66%
                        If the profile takes the course course_4, it will learn the skills: {4}
                        The matching with the job jobs_0 will increase from 66% to: 100%
                        The overall attractiveness of the profile will increase from 73 to: 100
                Course course_0 has a matching of 66%
                        If the profile takes the course course_0, it will learn the skills: {2}
                        The matching with the job jobs_0 will increase from 66% to: 66%
                        The overall attractiveness of the profile will increase from 73 to: 73
                Course course_2 has a matching of 66%
                        If the profile takes the course course_2, it will learn the skills: {2}
                        The matching with the job jobs_0 will increase from 66% to: 66%
                        The overall attractiveness of the profile will increase from 73 to: 73
```

### Some explanations

A few details about the numbers displayed here:
- For now, I am using very simple functions to compute the matchings
- The matching between a profile and a job is the percentage of skills required for the job that the profile posesses
- The overall attractiveness of the profile is the average matching between the profile and all jobs in the market.
- The matching between a profile and a course is the number of the percentage of skills required to follow the course that the profile posesses 


### TODOs:
- Add skill level
- Add Languages
- Add Programming languages
- Add certifications
- Replace skill names
- Provide ElasticSearch functions fort matching
- Change required name with prerequisite 
- Multiple matchings with and without prerequisistes
- Add ranking functions that take into account skill demand and skill offer
- Recommending multiple skills
- Adding must have and optional skills in jobs
- Add time dimension to the job market skills 
- Add ICT jobs file

### Questions:
- In learnig_opportunities, what is the difference between a course and a program?
- In learnig_opportunities, if the key 'certificate_type' is not empty, do we automatically consider the learning opportunity to provide a certificate upon completion?
