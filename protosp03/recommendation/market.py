from collections import Counter


def get_skill_demand(jobs, years):
    """Get the number of jobs that require a skill for each year.

    Args:
        jobs (list): list of jobs
        years (list): list of years

    Returns:
        dict: dictionary of Counter for each year with skills and their demand

    Example:
        jobs = [
            {"required_skills": {"Python": 2, "JavaScript": 1}, "year": 2020},
            {"required_skills": {"Python": 3}, "year": 2021},
            {"required_skills": {"JavaScript": 2}, "year": 2021}
        ]
        years = [2020, 2021]

        get_skill_demand(jobs, years) # This should output {2020: Counter({('Python', 2): 1, ('JavaScript', 1): 1}), 2021: Counter({('Python', 3): 1, ('JavaScript', 2): 1})}
    """
    skill_demand = {year: Counter() for year in years}
    for job in jobs:
        for skill, level in job["required_skills"].items():
            skill_demand[job["year"]][(skill, level)] += 1
    return skill_demand


def get_skill_trend(skill_demand, skill, years):
    """Get the trend of a skill's demand between the last two years.

    Args:
        skill_demand (Counter): Counter of skills and their demand for each year
        skill (str): skill name
        years (list): list of years

    Returns:
        float: trend of a skill's demand between the last two years

    Example:
        demand = {2020: Counter({('Python', 2): 1, ('JavaScript', 1): 1}), 2021: Counter({('Python', 3): 1, ('JavaScript', 2): 1})}

        skill = ('Python', 2)
        years = [2021, 2020]

        get_skill_trend(demand, skill, years) # This should output: -100.0
    """
    current_year = years[0]
    last_year = years[1]
    current_demand = skill_demand[current_year][skill]
    last_demand = skill_demand[last_year][skill]
    if last_demand == 0:
        return None
    return 100 * (current_demand - last_demand) / last_demand


def get_learner_trend(skill_demand, learner, years):
    """Get the trend of a learner's demand between the last two years.

    Args:
        skill_demand (Counter): Counter of skills and their demand for each year
        learner (dict): learner
        years (list): list of years

    Returns:
        dict: trend of a learner's skills demand between the last two years

    Example:
        demand = {2020: Counter({('Python', 2): 2, ('JavaScript', 1): 2}), 2021: Counter({('Python', 2): 5, ('JavaScript', 1): 1})}

        learner = {"possessed_skills": {"Python": 2, "JavaScript": 1}, "year": 2021}
        years = [2021, 2020]

        get_learner_trend(demand, learner, years) # This should output: {'Python': 150.0, 'JavaScript': -50.0}
    """
    current_year = years[0]
    last_year = years[1]

    learner_trend = dict()

    for skill, level in learner["possessed_skills"].items():
        current_demand = skill_demand[current_year][(skill, level)]
        last_demand = skill_demand[last_year][(skill, level)]
        if last_demand == 0:
            learner_trend[skill] = None
        else:
            learner_trend[skill] = 100 * (current_demand - last_demand) / last_demand

    return learner_trend


def get_skill_supply(learners, years):
    """Get the number of learners that possess a skill for each year.

    Args:
        learners (list): list of learners
        years (list): list of years

    Returns:
        dict: dictionary of Counter for each year with skills and their supply

    Example:

        learners = [
            {"possessed_skills": {"Python": 3, "JavaScript": 2}, "year": 2020},
            {"possessed_skills": {"Python": 2}, "year": 2021},
            {"possessed_skills": {"JavaScript": 3}, "year": 2021}
        ]
        years = [2020, 2021]

        get_skill_supply(learners, years) # This should output: {2020: Counter({('Python', 3): 1, ('JavaScript', 2): 1}), 2021: Counter({('Python', 2): 1, ('JavaScript', 3): 1})}
    """
    skill_supply = {year: Counter() for year in years}
    for learner in learners:
        for skill, level in learner["possessed_skills"].items():
            skill_supply[learner["year"]][(skill, level)] += 1
    return skill_supply


def get_skill_attractiveness(skill, years, skill_supply, skill_demand):
    """Calculate the attractiveness of a skill. If the value is lower than 1, the skill is not attractive (supply larger than demand).

    Args:
        skill (str): skill name
        years (list): list of years
        skill_supply (dict): dictionary of Counter for each year with skills and their supply
        skill_demand (dict): dictionary of Counter for each year with skills and their demand

    Returns:
        float: attractiveness of a skill

    Example:
        skill_supply = {2020: Counter({('Python', 3): 1, ('JavaScript', 2): 1}), 2021: Counter({('Python', 3): 1, ('JavaScript', 3): 1})}
        skill_demand = {2020: Counter({('Python', 3): 1, ('JavaScript', 1): 1}), 2021: Counter({('Python', 3): 2, ('JavaScript', 2): 1})}
        skill = ('Python', 3)
        years = [2020, 2021]

        get_skill_attractiveness(skill, years, skill_supply, skill_demand) # This should output 1.3333333333333333
    """
    skill_attractiveness = 0
    normalization_factor = 0
    for i, year in enumerate(years):
        if skill in skill_supply[year]:
            skill_attractiveness += skill_demand[year][skill] / (
                skill_supply[year][skill] * (i + 1)
            )
        normalization_factor += 1 / (i + 1)
    return skill_attractiveness / normalization_factor


def get_learner_attractiveness(learner, years, skill_supply, skill_demand):
    """Calculate the attractiveness of a learner for each skill.

    Args:
        learner (dict): learner
        years (list): list of years
        skill_supply (dict): dictionary of Counter for each year with skills and their supply
        skill_demand (dict): dictionary of Counter for each year with skills and their demand

    Returns:
        dict: attractiveness of each skill that the learner possesses

    Example:
        skill_supply = {2020: Counter({('Python', 3): 1, ('JavaScript', 2): 1}), 2021: Counter({('Python', 3): 1, ('JavaScript', 3): 1})}
        skill_demand = {2020: Counter({('Python', 3): 1, ('JavaScript', 1): 1}), 2021: Counter({('Python', 3): 2, ('JavaScript', 2): 1})}
        learner = {"possessed_skills": {"Python": 3, "JavaScript": 2}, "year": 2021}
        years = [2020, 2021]

        get_learner_attractiveness(learner, years, skill_supply, skill_demand) # This should output {'Python': 1.3333333333333333, 'JavaScript': 0.0}
    """
    learner_attractiveness = dict()
    for skill, level in learner["possessed_skills"].items():
        learner_attractiveness[skill] = get_skill_attractiveness(
            (skill, level), years, skill_supply, skill_demand
        )
    return learner_attractiveness
