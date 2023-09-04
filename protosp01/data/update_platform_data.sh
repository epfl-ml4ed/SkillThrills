#!bin/bash
wget https://evrlearn-staging.herokuapp.com/api/v1/proto/learning_opportunities -O "SkillThrills/data/raw/learning_opportunities.json"
wget https://evrlearn-staging.herokuapp.com/api/v1/proto/vacancies -O "/home/montario/Documents/EPFL - SkillThrills/data/raw/vacancies.json"
wget https://evrlearn-staging.herokuapp.com/api/v1/proto/job_skills -O "/home/montario/Documents/EPFL - SkillThrills/data/raw/job_skills.json"
wget https://evrlearn-staging.herokuapp.com/api/v1/proto/job_profiles -O "/home/montario/Documents/EPFL - SkillThrills/data/raw/job_profiles.json"