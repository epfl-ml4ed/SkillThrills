import sys

sys.path.append("../recommendation")  # Add the src directory to the path

from market import get_skill_demand, get_skill_supply, skill_attractiveness

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


learners = [
    {"possessed_skills": {"Python": 2}, "year": 2020},
    {"possessed_skills": {"Python": 3}, "year": 2020},
    {"possessed_skills": {"Python": 3}, "year": 2021},
    {"possessed_skills": {"Python": 3}, "year": 2021},
    {"possessed_skills": {"Python": 3}, "year": 2022},
    {"possessed_skills": {"Python": 3}, "year": 2022},
    {"possessed_skills": {"Python": 3}, "year": 2022},
    {"possessed_skills": {"Python": 3}, "year": 2022},
    {"possessed_skills": {"Python": 2}, "year": 2022},
    {"possessed_skills": {"Python": 3}, "year": 2022},
    {"possessed_skills": {"Python": 3}, "year": 2022},
]

# Calculate skill demand over the years
years = [2022, 2021, 2020]
skill_demand = get_skill_demand(jobs, years)
skill_supply = get_skill_supply(learners, years)


attract1 = skill_attractiveness(("Python", 3), years, skill_supply, skill_demand)
print(
    f"Scenario 1 - Python 3 attractiveness: {attract1}"
    # This should be 0.72 as the supply for  Python 3 is larger than the demand
)


attract2 = skill_attractiveness(("Python", 2), years, skill_supply, skill_demand)
print(
    f"Scenario 2 - Python 2 attractiveness: {attract2}"
    # This should be 2.36 as the demand for Python 2 is larger than the supply
)
