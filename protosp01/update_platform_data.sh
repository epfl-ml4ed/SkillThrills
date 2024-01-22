#!bin/bash
# wget https://evrlearn-production.herokuapp.com/api/v1/proto/learning_opportunities -O "../data/raw/learning_opportunities.json"
wget https://evrlearn-production.herokuapp.com/api/v1/proto/learning_opportunities -O "../data/raw/learning_opportunities.json"
wget https://evrlearn-staging.herokuapp.com/api/v1/proto/vacancies -O "../data/raw/vacancies.json"
wget https://evrlearn-staging.herokuapp.com/api/v1/proto/job_skills -O "../data/raw/job_skills.json"
wget https://evrlearn-staging.herokuapp.com/api/v1/proto/job_profiles -O "../data/raw/job_profiles.json"