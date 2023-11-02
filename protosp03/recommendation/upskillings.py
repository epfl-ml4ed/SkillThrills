from copy import deepcopy

import matchings


def update_advice(
    skill,
    up_level,
    learner,
    job,
    skills_attractiveness,
    new_learner_job_match,
    up_skilling_advice,
):
    """Given a skill and level, update the up-skilling advice if the new match is better.

    Args:
        skill (str): skill name
        up_level (int): mastery level
        learner (dict): Learner's profile including possessed skills and levels.
        job (dict): Job's profile including required skills and levels.
        skills_attractiveness (dict): Attractiveness of all skills for each mastery level.
        new_learner_job_match (float): current matching between learner and job
        up_skilling_advice (tuple): up-skilling advice (skill, level)

    Returns:
        _type_: _description_
    """
    tmp_up_skilling_advice = (skill, up_level)
    tmp_learner = deepcopy(learner)
    tmp_learner["possessed_skills"][skill] = up_level
    tmp_match = matchings.learner_job_matching(tmp_learner, job)
    if tmp_match > new_learner_job_match:
        new_learner_job_match = tmp_match
        up_skilling_advice = tmp_up_skilling_advice
    elif tmp_match == new_learner_job_match:
        if (
            skills_attractiveness[tmp_up_skilling_advice]
            > skills_attractiveness[up_skilling_advice]
        ):
            new_learner_job_match = tmp_match
            up_skilling_advice = tmp_up_skilling_advice
    return new_learner_job_match, up_skilling_advice


def up_skilling_job_advice(learner, job, skills_attractiveness):
    """Return the up-skilling advice for a learner to match a job.

    Args:
        learner (dict): Learner's profile including possessed skills and levels.
        job (dict): Job's profile including required skills and levels.
        skills_attractiveness (dict): Attractiveness of all skills for each mastery level.

    Returns:
        tuple: A tuple of the skill and level to up-skill to. If no up-skilling advice is possible (perfect profile), return None.
    """
    up_skilling_advice = None
    new_learner_job_match = matchings.learner_job_matching(learner, job)
    for skill, level in job["required_skills"].items():
        if skill not in learner["possessed_skills"]:
            up_level = 1
            new_learner_job_match, up_skilling_advice = update_advice(
                skill,
                up_level,
                learner,
                job,
                skills_attractiveness,
                new_learner_job_match,
                up_skilling_advice,
            )
        elif learner["possessed_skills"][skill] < level:
            up_level = learner["possessed_skills"][skill] + 1
            new_learner_job_match, up_skilling_advice = update_advice(
                skill,
                up_level,
                learner,
                job,
                skills_attractiveness,
                new_learner_job_match,
                up_skilling_advice,
            )
    return up_skilling_advice


def up_skilling_market_advice(learner, skills_attractiveness):
    """Return the up-skilling advice for a learner to increase their overall attractiveness.

    Args:
        learner (dict): Learner's profile including possessed skills and levels.
        skills_attractiveness (dict): Attractiveness of all skills for each mastery level.

    Returns:
        tuple: A tuple of the skill and level to up-skill to.
    """
    up_skilling_advice = None

    learner_attractiveness = sum(
        [
            skills_attractiveness[skill, level]
            for skill, level in learner["possessed_skills"].items()
        ]
    )
    print(learner_attractiveness)

    return up_skilling_advice


def up_skilling_market_advice(learner, skills, skills_attractiveness):
    """Return the up-skilling advice for a learner to increase their overall attractiveness.

    Args:
        learner (dict): Learner's profile including possessed skills and levels.
        skills (list): List of all skills.
        skills_attractiveness (dict): Attractiveness of all skills for each mastery level.

    Returns:
        tuple: A tuple of the skill and level to up-skill to.
    """
    up_skilling_advice = None

    original_learner_attractiveness = sum(
        [skills_attractiveness[(s, l)] for s, l in learner["possessed_skills"].items()]
    )
    learner_attractiveness = original_learner_attractiveness

    # Up-skilling, checking how much attractiveness increases with the next level
    for skill, level in learner["possessed_skills"].items():
        if (skill, level + 1) in skills_attractiveness:
            tmp_learner = deepcopy(learner)
            tmp_learner["possessed_skills"][skill] = level + 1
            tmp_learner_attractiveness = sum(
                [
                    skills_attractiveness[(s, l)]
                    for s, l in tmp_learner["possessed_skills"].items()
                ]
            )
            if tmp_learner_attractiveness > learner_attractiveness:
                learner_attractiveness = tmp_learner_attractiveness
                up_skilling_advice = (skill, level + 1)

    # New skill, checking how much attractiveness increases with a new skill
    for skill in skills:
        if skill not in learner["possessed_skills"]:
            tmp_learner = deepcopy(learner)
            tmp_learner["possessed_skills"][skill] = 1
            tmp_learner_attractiveness = (
                original_learner_attractiveness + skills_attractiveness[(skill, 1)]
            )
            if tmp_learner_attractiveness > learner_attractiveness:
                learner_attractiveness = tmp_learner_attractiveness
                up_skilling_advice = (skill, 1)

    return up_skilling_advice
