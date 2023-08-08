import argparse
import yaml
import json
import os


def make_jobs_inverted_index(config):
    """Creates an inverted index of jobs

    Args:
        config (dict): Configuration dictionary
    """
    jobs = json.load(open(os.path.join(config["raw_dataset_path"], "jobs.json"), "r"))
    inverted_index = {}
    for job_id, job in jobs.items():
        for skill, level in job.items():
            if skill not in inverted_index:
                inverted_index[skill] = {job_id: level}
            else:
                inverted_index[skill][job_id] = level
    with open(
        os.path.join(config["inverted_index_path"], "jobs_inverted_index.json"), "w"
    ) as f:
        json.dump(inverted_index, f, indent=4)


def make_profile_inverted_index(config):
    """Creates an inverted index of profiles

    Args:
        config (dict): Configuration dictionary
    """
    profiles = json.load(
        open(os.path.join(config["raw_dataset_path"], "resumes.json"), "r")
    )
    inverted_index = {}
    for profile_id, profile in profiles.items():
        for skill, level in profile.items():
            if skill not in inverted_index:
                inverted_index[skill] = {profile_id: level}
            else:
                inverted_index[skill][profile_id] = level
    with open(
        os.path.join(config["inverted_index_path"], "profiles_inverted_index.json"), "w"
    ) as f:
        json.dump(inverted_index, f, indent=4)


def make_course_provided_inverted_index(config):
    """Creates an inverted index of courses based on the skills they provide

    Args:
        config (dict): Configuration dictionary
    """
    courses = json.load(
        open(os.path.join(config["raw_dataset_path"], "courses.json"), "r")
    )
    inverted_index = {}
    for course_id, course in courses.items():
        for skill, level in course["provided"].items():
            if skill not in inverted_index:
                inverted_index[skill] = {course_id: level}
            else:
                inverted_index[skill][course_id] = level
    with open(
        os.path.join(
            config["inverted_index_path"], "courses_provided_inverted_index.json"
        ),
        "w",
    ) as f:
        json.dump(inverted_index, f, indent=4)


def make_course_required_inverted_index(config):
    """Creates an inverted index of courses based on the skills they require

    Args:
        config (dict): Configuration dictionary
    """
    courses = json.load(
        open(os.path.join(config["raw_dataset_path"], "courses.json"), "r")
    )
    inverted_index = {}
    for course_id, course in courses.items():
        for skill, level in course["required"].items():
            if skill not in inverted_index:
                inverted_index[skill] = {course_id: level}
            else:
                inverted_index[skill][course_id] = level
    with open(
        os.path.join(
            config["inverted_index_path"], "courses_required_inverted_index.json"
        ),
        "w",
    ) as f:
        json.dump(inverted_index, f, indent=4)


def make_inverted_indexes(config):
    make_jobs_inverted_index(config)
    make_profile_inverted_index(config)
    make_course_provided_inverted_index(config)
    make_course_required_inverted_index(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="protosp03/config/inverted_index.yaml"
    )
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    make_inverted_indexes(config)


if __name__ == "__main__":
    main()
