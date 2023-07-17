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
python protosp03/data/make_synthetic.py --config protosp03/config/synthetic.yaml --seed 42
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
python -m tests.user_journey01 --seed 0
```
 It should display this:

```shell script
Considering Profile resume_3 with the skills: {4}
        We are assuming that the Profile is interested in Job jobs_3 that requires the skills: {0, 2, 3, 4}
        The matching between the profile and the desired job is: 25%
        Printing the attractiveness of each skill of the profile:
                Skill 4 is required for 80% of the jobs on the market
        The overall attractiveness of the profile is: 25%
        Printing the matching of the profile with respect to each job (from most compatible to least compatible):
                Job jobs_0 has a matching of 50%
                Job jobs_4 has a matching of 33%
                Job jobs_3 has a matching of 25%
                Job jobs_1 has a matching of 20%
        Printing the matching of the profile with respect to each course (from most compatible to least compatible):
                Course course_1 has a matching of 50%
                        If the profile takes the course course_1, it will learn the skills: {3}
                        The matching with the job jobs_1 will increase from 20% to: 40%
                        The overall attractiveness of the profile will increase from 25 to: 34
                Course course_3 has a matching of 33%
                        If the profile takes the course course_3, it will learn the skills: {1}
                        The matching with the job jobs_1 will increase from 20% to: 40%
                        The overall attractiveness of the profile will increase from 25 to: 56
                Course course_4 has a matching of 33%
                        If the profile takes the course course_4, it will learn the skills: {1}
                        The matching with the job jobs_1 will increase from 20% to: 40%
                        The overall attractiveness of the profile will increase from 25 to: 56
```

### Some explanations

A few details about the numbers displayed here:
- For now, I am using very simple functions to compute the matchings
- The matching between a profile and a job is the percentage of skills required for the job that the profile posesses
- The overall attractiveness of the profile is the average matching between the profile and all jobs in the market.
- The matching between a profile and a course is the number of the percentage of skills required to follow the course that the profile posesses 