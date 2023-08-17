import argparse
import random
import yaml
import json
import os

from itertools import product


def make_skills(config):
    """Returns a dict of skills
    Each skill is located within a hierarchy

    Args:
        config (dict): a configuration dictionary

    Returns:
        dict: dict of skills
    """
    skills = dict()

    levels = [range(1, elem + 1) for elem in config["nb_elem_per_levels"]]
    for i, level in enumerate(product(*levels)):
        skill_name = "skill_" + str(i)
        level_str = ".".join([str(elem) for elem in level])
        skills[skill_name] = {"skill_id": i, "level_str": level_str}
        skills[skill_name]["level"] = level

    return skills


def save_skills(skills, config):
    """Saves the dict of skills to a csv file

    Args:
        skills (dict): dict of skills
        config (dict): Configuration dictionary
    """
    with open(os.path.join(config["dataset_path"], "skills.json"), "w") as f:
        json.dump(skills, f, indent=4)


def get_random_skills(skills, max_skills):
    """Returns a random dict of {skills: level}
    the skills are unique and the level is between 1 and 4

    Args:
        skills (dict): dict of skills
        max_skills (int): maximum number of skills

    Returns:
        list: list of skills
    """
    nb_skills = random.randint(1, max_skills)
    random_skills = random.sample(sorted(skills), nb_skills)
    return {skill: random.randint(1, 4) for skill in random_skills}


def get_random_provided_skills(skills, required_skills, max_skills):
    """Returns a random list of [skills, level], the level is between 1 and 4
    the provided skills must either not be in the required skills or have a higher level

    Args:
        skills (dict): dict of skills
        required_skills (dict): dict of required skills: level
        max_skills (int): maximum number of skills

    Returns:
        list: list of skills
    """
    nb_skills = random.randint(1, max_skills)
    list_skills = list(skills.keys())
    provided_skills = dict()
    while len(provided_skills) < nb_skills:
        candidate_skill = random.choice(list_skills)
        candidate_level = random.randint(1, 4)
        if (
            candidate_skill not in required_skills
            and candidate_skill not in provided_skills
        ):
            provided_skills[candidate_skill] = candidate_level
        elif (
            candidate_skill in required_skills
            and candidate_level > required_skills[candidate_skill]
        ):
            provided_skills[candidate_skill] = candidate_level

    return provided_skills


def make_resumes(config, skills):
    """Returns a dict of resumes

    Args:
        config (dict): Configuration dictionary
        skills (list): list of skills

    Returns:
        dict: dict of resumes with skills
    """
    return {
        "resume_" + str(i): get_random_skills(skills, config["max_resume_skills"])
        for i in range(config["nb_resumes"])
    }


def save_resumes(resumes, config):
    """Saves the dict of resumes to a json file

    Args:
        resumes (dict): dict of resumes
        config (dict): Configuration dictionary
    """
    with open(os.path.join(config["dataset_path"], "resumes.json"), "w") as f:
        json.dump(resumes, f, indent=4)


def make_jobs(config, skills):
    """Returns a dict of jobs

    Args:
        config (dict): Configuration dictionary
        skills (list): list of skills

    Returns:
        dict: dict of jobs with skills
    """
    return {
        "jobs_" + str(i): get_random_skills(skills, config["max_job_skills"])
        for i in range(config["nb_jobs"])
    }


def save_jobs(jobs, config):
    """Saves the dict of jobs to a json file

    Args:
        jobs (dict): dict of jobs
        config (dict): Configuration dictionary
    """
    with open(os.path.join(config["dataset_path"], "jobs.json"), "w") as f:
        json.dump(jobs, f, indent=4)


def make_courses(config, skills):
    """Returns a dict of courses.
    Each courses has 2 types of skills: required and provided.
    Provided skills cannot be in the required skills.

    Args:
        config (dict): Configuration dictionary

    Returns:
        dict: dict of courses with skills
    """
    courses = {
        "course_"
        + str(i): {
            "required": get_random_skills(skills, config["max_required_course_skills"]),
            "provided": [],
        }
        for i in range(config["nb_courses"])
    }

    for course in courses:
        required_skills = courses[course]["required"]
        courses[course]["provided"] = get_random_provided_skills(
            skills, required_skills, config["max_provided_course_skills"]
        )
    return courses


def save_courses(courses, config):
    """Saves the dict of courses to a json file

    Args:
        courses (dict): dict of courses
        config (dict): Configuration dictionary
    """
    with open(os.path.join(config["dataset_path"], "courses.json"), "w") as f:
        json.dump(courses, f, indent=4)


def make_synthetic(config):
    """Creates a synthetic dataset consisting of skills, resumes, jobs and courses

    Args:
        config (dict): Configuration dictionary
    """
    random.seed(config["seed"])
    skills = make_skills(config)
    save_skills(skills, config)
    resumes = make_resumes(config, skills)
    save_resumes(resumes, config)
    jobs = make_jobs(config, skills)
    save_jobs(jobs, config)
    courses = make_courses(config, skills)
    save_courses(courses, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="protosp03/config/synthetic.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config["seed"] = args.seed
    make_synthetic(config)


if __name__ == "__main__":
    main()
