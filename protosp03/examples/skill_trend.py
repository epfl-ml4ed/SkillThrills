import sys

sys.path.append("../recommendation")  # Add the src directory to the path

from market import get_skill_demand, get_skill_trend

jobs = [
    {"required_skills": {"Python": 2}, "year": 2020},
    {"required_skills": {"Python": 3}, "year": 2020},
    {"required_skills": {"Python": 3}, "year": 2021},
    {"required_skills": {"Python": 3}, "year": 2021},
    {"required_skills": {"Python": 2}, "year": 2022},
    {"required_skills": {"Python": 2}, "year": 2022},
    {"required_skills": {"Python": 3}, "year": 2022},
    {"required_skills": {"Python": 2}, "year": 2022},
    {"required_skills": {"Python": 2}, "year": 2022},
    {"required_skills": {"Python": 3}, "year": 2022},
    {"required_skills": {"Python": 3}, "year": 2022},
]

# Calculate skill demand over the years
years = [2022, 2021, 2020]
skill_demand = get_skill_demand(jobs, years)


trend1 = get_skill_trend(skill_demand, ("Python", 3), years)
print(
    f"Scenario 1 - Python 3 demand trend: {trend1}"
    # This should be 50% as the demand for Python 3 is growing
)


trend2 = get_skill_trend(skill_demand, ("Python", 2), years)
print(
    f"Scenario 2 - Python 2 demand trend: {trend2}"
    # This should be None, since there was no demand for Python 2 in 2021, to discuss with the team
)
