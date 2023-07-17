# SCESC03_Prototype
Prototype of the SubProject 03 for the SCESC Innosuisse Project

### Input
We have 4 raw files as input, for now they are json but this can be changed. 

The files are in the folder data/raw/synthetic. 

They can be generated using the script [protosp03/data/make_synthetic.py](protosp03/data/make_synthetic.py) and the config file [protosp03/config/synthetic.yaml](protosp03/config/synthetic.yaml)
- **skills.json** : a dict whoses keys are the skills name and values are the skill id 
- **jobs.json** : a dict whoses keys are the job id and values are the list of skill required
- **resume.json** : a dict whoses keys are the resume id (or profile id) and values are the list of skill present
- **courses.json** : a dict whoses keys are the job id and values are the list of skill required and the list of skills provided


To generate the synthetic data files:  
```shell script
python protosp03/data/make_synthetic.py --config protosp03/config/synthetic.yaml
```
 