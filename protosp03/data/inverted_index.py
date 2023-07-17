import argparse
import yaml
import json
import os
import pickle


def make_jobs_inverted_index(config):
    """Creates an inverted index of jobs

    Args:
        config (dict): Configuration dictionary
    """
    jobs = json.load(open(os.path.join(config["raw_dataset_path"], "jobs.json"), "r"))
    skills = json.load(
        open(os.path.join(config["raw_dataset_path"], "skills.json"), "r")
    )
    inverted_index = {}
    for job_id, job in jobs.items():
        for skill in job:
            skill_id = skills[skill]
            if skill_id not in inverted_index:
                inverted_index[skill_id] = {job_id}
            else:
                inverted_index[skill_id].add(job_id)
    with open(
        os.path.join(config["inverted_index_path"], "jobs_inverted_index.pkl"), "wb"
    ) as f:
        pickle.dump(inverted_index, f)


def make_course_provided_inverted_index(config):
    """Creates an inverted index of courses based on the skills they provide

    Args:
        config (dict): Configuration dictionary
    """
    courses = json.load(
        open(os.path.join(config["raw_dataset_path"], "courses.json"), "r")
    )
    skills = json.load(
        open(os.path.join(config["raw_dataset_path"], "skills.json"), "r")
    )
    inverted_index = {}
    for course_id, course in courses.items():
        for skill in course["provided"]:
            skill_id = skills[skill]
            if skill_id not in inverted_index:
                inverted_index[skill_id] = {course_id}
            else:
                inverted_index[skill_id].add(course_id)
    with open(
        os.path.join(
            config["inverted_index_path"], "courses_provided_inverted_index.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(inverted_index, f)


def make_course_required_inverted_index(config):
    """Creates an inverted index of courses based on the skills they require

    Args:
        config (dict): Configuration dictionary
    """
    courses = json.load(
        open(os.path.join(config["raw_dataset_path"], "courses.json"), "r")
    )
    skills = json.load(
        open(os.path.join(config["raw_dataset_path"], "skills.json"), "r")
    )
    inverted_index = {}
    for course_id, course in courses.items():
        for skill in course["required"]:
            skill_id = skills[skill]
            if skill_id not in inverted_index:
                inverted_index[skill_id] = {course_id}
            else:
                inverted_index[skill_id].add(course_id)
    with open(
        os.path.join(
            config["inverted_index_path"], "courses_required_inverted_index.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(inverted_index, f)


def make_inverted_indexes(config):
    make_jobs_inverted_index(config)
    make_course_provided_inverted_index(config)
    make_course_required_inverted_index(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config/inverted_index.yaml")
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    make_inverted_indexes(config)


if __name__ == "__main__":
    main()
