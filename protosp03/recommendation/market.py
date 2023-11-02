from collections import Counter


def get_skill_demand(jobs, years, max_level=4):
    """Get the number of jobs that require a skill for each year. If a job requires a skill at a certain level, it is assumed that the skill at all higher levels are accepted as well.

    Args:
        jobs (list): list of jobs
        years (list): list of years

    Returns:
        dict: dictionary of Counter for each year with skills and their demand

    Example:
        jobs = [
            {"required_skills": {"Python": 2, "JavaScript": 1}, "year": 2020},
            {"required_skills": {"JavaScript": 2}, "year": 2021}
            ]
        years = [2020, 2021]

        get_skill_demand(jobs, years, max_level=2) # This should output {2020: Counter({('Python', 2): 1, ('JavaScript', 1): 1, ('JavaScript', 2): 1}), 2021: Counter({('JavaScript', 2): 1})}

    """
    skill_demand = {year: Counter() for year in years}
    for job in jobs:
        for skill, level in job["required_skills"].items():
            for sublevel in range(level, max_level + 1):
                skill_demand[job["year"]][(skill, sublevel)] += 1
    return skill_demand


def get_skill_trend(skill_demand, skill, mastery_level, years):
    """Get the trend of a skill's demand between the last two years.

    Args:
        skill_demand (Counter): Counter of skills and their demand for each year
        skill (str): skill name
        mastery_level (int): mastery level
        years (list): list of years

    Returns:
        float: trend of a skill's demand between the last two years

    Example:
        demand = {2020: Counter({('Python', 2): 1, ('JavaScript', 2): 1}), 2021: Counter({('Python', 3): 1, ('JavaScript', 2): 1})}

        skill = ('JavaScript', 2)
        years = [2021, 2020]

        get_skill_trend(demand, skill, years) # This should output: 100.0
    """
    current_year = years[0]
    last_year = years[1]
    current_demand = skill_demand[current_year][(skill, mastery_level)]
    last_demand = skill_demand[last_year][(skill, mastery_level)]
    if last_demand == 0:
        return 100 * current_demand
    return 100 * (current_demand - last_demand) / (last_demand)


def get_all_skills_trend(skills, mastery_levels, years, skill_demand):
    """Get the trend of all skills' demand between the last two years.

    Args:
        skills (list): list of skills
        mastery_levels (list): list of mastery levels
        years (list): list of years
        skill_demand (dict): dict of skills and their demand for each year

    Returns:
        Counter: trend of all skills' demand between the last two years
    """
    skills_trend = Counter()
    for skill in skills:
        for level in mastery_levels:
            skills_trend[(skill, level)] = get_skill_trend(
                skill_demand, skill, level, years
            )
    return skills_trend


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
    learner_trend = dict()

    for skill, level in learner["possessed_skills"].items():
        learner_trend[(skill, level)] = get_skill_trend(
            skill_demand, (skill, level), years
        )

    return learner_trend


def get_skill_supply(learners, years):
    """Get the number of learners that possess a skill for each year. If a learner possesses a skill at a certain level, it is assumed that they also possess the skill at all lower levels.

        Args:
            learners (list): list of learners
            years (list): list of years

        Returns:
            dict: dictionary of Counter for each year with skills and their supply

        Example:

            learners = [
                {"possessed_skills": {"Python": 2, "JavaScript": 2}, "year": 2020},
                {"possessed_skills": {"JavaScript": 1}, "year": 2021}
            ]
    years = [2020, 2021]

    get_skill_supply(learners, years) # This should output: {2020: Counter({('Python', 2): 1,
              ('Python', 1): 1,
              ('JavaScript', 2): 1,
              ('JavaScript', 1): 1}),
     2021: Counter({('JavaScript', 1): 1})}
    """
    skill_supply = {year: Counter() for year in years}
    for learner in learners:
        for skill, level in learner["possessed_skills"].items():
            for sublevel in range(level, 0, -1):
                skill_supply[learner["year"]][(skill, sublevel)] += 1
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

        get_skill_attractiveness(skill, years, skill_supply, skill_demand) # This should output 1.1666666666666667
    """
    skill_attractiveness = 0
    normalization_factor = 0
    for i, year in enumerate(years):
        skill_attractiveness += (skill_demand[year][skill] + 1) / (
            (skill_supply[year][skill] + 1) * (i + 1)
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
        Counter: attractiveness of each skill that the learner possesses

    Example:
        skill_supply = {2020: Counter({('Python', 3): 1, ('JavaScript', 2): 1}), 2021: Counter({('Python', 3): 1, ('JavaScript', 3): 1})}
        skill_demand = {2020: Counter({('Python', 3): 1, ('JavaScript', 1): 1}), 2021: Counter({('Python', 3): 2, ('JavaScript', 2): 1})}
        learner = {"possessed_skills": {"Python": 3, "JavaScript": 2}, "year": 2021}
        years = [2020, 2021]

        get_learner_attractiveness(learner, years, skill_supply, skill_demand) # This should output {'Python': 1.3333333333333333, 'JavaScript': 0.0}
    """
    learner_attractiveness = Counter()
    for skill, level in learner["possessed_skills"].items():
        learner_attractiveness[skill] = get_skill_attractiveness(
            (skill, level), years, skill_supply, skill_demand
        )
    return learner_attractiveness


def get_all_skills_attractiveness(
    skills, mastery_levels, years, skill_supply, skill_demand
):
    """Calculate the attractiveness of all skills for each mastery level.

    Args:
        skills (list): list of skills
        mastery_levels (list): list of mastery levels
        years (list): list of years
        skill_supply (dict): dictionary of Counter for each year with skills and their supply
        skill_demand (dict): dictionary of Counter for each year with skills and their demand

    Returns:
        Counter: attractiveness of all skills for each mastery level
    """
    skills_attractiveness = Counter()
    for skill in skills:
        for level in mastery_levels:
            skills_attractiveness[(skill, level)] = get_skill_attractiveness(
                (skill, level), years, skill_supply, skill_demand
            )
    return skills_attractiveness


def get_all_market_metrics(skills, mastery_levels, learners, jobs, years):
    """Get all market metrics.

    Args:
        skills (list): list of skills
        mastery_levels (list): list of mastery levels
        learners (list): list of learners
        jobs (list): list of jobs
        years (list): list of years

    Returns:
        tuple: tuple of skill_supply, skill_demand, skill_trends, skills_attractiveness
    """
    skill_supply = get_skill_supply(learners, years)
    skill_demand = get_skill_demand(jobs, years)
    skill_trends = get_all_skills_trend(skills, mastery_levels, years, skill_demand)
    skills_attractiveness = get_all_skills_attractiveness(
        skills, mastery_levels, years, skill_supply, skill_demand
    )
    return skill_supply, skill_demand, skill_trends, skills_attractiveness
