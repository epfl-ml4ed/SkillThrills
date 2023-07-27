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
    skills = json.load(
        open(os.path.join(config["raw_dataset_path"], "skills.json"), "r")
    )
    inverted_index = {}
    for job_id, job in jobs.items():
        for skill in job:
            skill_id = skills[skill]
            if skill_id not in inverted_index:
                inverted_index[skill_id] = [job_id]
            else:
                inverted_index[skill_id].append(job_id)
    with open(
        os.path.join(config["inverted_index_path"], "jobs_inverted_index.json"), "w"
    ) as f:
        json.dump(inverted_index, f)


def make_profile_inverted_index(config):
    """Creates an inverted index of profiles

    Args:
        config (dict): Configuration dictionary
    """
    profiles = json.load(
        open(os.path.join(config["raw_dataset_path"], "resumes.json"), "r")
    )
    skills = json.load(
        open(os.path.join(config["raw_dataset_path"], "skills.json"), "r")
    )
    inverted_index = {}
    for profile_id, profile in profiles.items():
        for skill in profile:
            skill_id = skills[skill]
            if skill_id not in inverted_index:
                inverted_index[skill_id] = [profile_id]
            else:
                inverted_index[skill_id].append(profile_id)
    with open(
        os.path.join(config["inverted_index_path"], "profiles_inverted_index.json"), "w"
    ) as f:
        json.dump(inverted_index, f)


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
                inverted_index[skill_id] = [course_id]
            else:
                inverted_index[skill_id].append(course_id)
    with open(
        os.path.join(
            config["inverted_index_path"], "courses_provided_inverted_index.json"
        ),
        "w",
    ) as f:
        json.dump(inverted_index, f)


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
                inverted_index[skill_id] = [course_id]
            else:
                inverted_index[skill_id].append(course_id)
    with open(
        os.path.join(
            config["inverted_index_path"], "courses_required_inverted_index.json"
        ),
        "w",
    ) as f:
        json.dump(inverted_index, f)


def inverted_index_decoder(inverted_index):
    """Decodes the inverted index, to convert the keys to integers (if they are strings representing integers) and the values to sets if the are lists

    Args:
        inverted_index (dict): Inverted index

    Returns:
        decoded_inverted_index: Decoded inverted index
    """
    decoded_inverted_index = {}
    for k, v in inverted_index.items():
        # Check if key is an integer in string format
        if k.isdigit():
            new_k = int(k)
        else:
            new_k = k

        # Check if value is a list
        if isinstance(v, list):
            new_v = set(v)
        else:
            new_v = v

        decoded_inverted_index[new_k] = new_v
    return decoded_inverted_index


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
