import argparse
import random
import yaml
import json
import os


def make_skills(config):
    """Creates a list of skills

    Args:
        config (dict): Configuration dictionary

    Returns:
        dict: dict of skills
    """
    return ["skill_" + str(i) for i in range(config["nb_skills"])]


def save_skills(skills, config):
    """Saves the dict of skills to a csv file

    Args:
        skills (dict): dict of skills
        config (dict): Configuration dictionary
    """
    with open(os.path.join(config["dataset_path"], "skills.json"), "w") as f:
        json.dump({skill: i for i, skill in enumerate(skills)}, f)


def get_random_skills(skills, max_skills):
    """Returns a random list of skills

    Args:
        skills (dict): dict of skills
        max_skills (int): maximum number of skills

    Returns:
        list: list of skills
    """
    nb_skills = random.randint(1, max_skills)
    return random.sample(skills, nb_skills)


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
        json.dump(resumes, f)


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
        json.dump(jobs, f)


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
        skills_candidates = [
            skill for skill in skills if skill not in courses[course]["required"]
        ]
        courses[course]["provided"] = get_random_skills(
            skills_candidates, config["max_provided_course_skills"]
        )
    return courses


def save_courses(courses, config):
    """Saves the dict of courses to a json file

    Args:
        courses (dict): dict of courses
        config (dict): Configuration dictionary
    """
    with open(os.path.join(config["dataset_path"], "courses.json"), "w") as f:
        json.dump(courses, f)


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
