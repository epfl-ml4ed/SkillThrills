import argparse
import yaml
import json
import os

from collections import defaultdict


def read_job_skills(config):
    """Reads the job skills from evrlearn dataset

    Args:
        config (dict): Configuration dictionary

    Returns:
        dict: Dictionary of job skills
    """
    job_skills = json.load(
        open(os.path.join(config["raw_dataset_path"], "job_skills.json"), "r")
    )
    job_skills_dict = defaultdict(dict)
    for job in job_skills:
        id = job["id"]
        job_skills_dict[id] = job
        job_skills_dict[id].pop("id")
    return job_skills_dict


def read_job_profiles(config):
    """Reads the job profiles from evrlearn dataset

    Args:
        config (dict): Configuration dictionary

    Returns:
        dict: Dictionary of job profiles
    """
    job_profiles = json.load(
        open(os.path.join(config["raw_dataset_path"], "job_profiles.json"), "r")
    )
    job_profiles_dict = {}
    remove_keys = [
        "id",
        "level",
        "level_id",
    ]
    remove_skill_keys = [
        "id",
        "job_profile_id",
        "created_at",
        "updated_at",
    ]

    for job in job_profiles:
        id = job["id"]
        job["name"] = job["level"] + " " + job["name"]
        job_profiles_dict[id] = job
        for key in remove_keys:
            job_profiles_dict[id].pop(key)
        for skill in job_profiles_dict[id]["job_profile_skills"]:
            for key in remove_skill_keys:
                skill.pop(key)

    return job_profiles_dict


def process_raw(config):
    job_skills = read_job_skills(config)
    job_profiles = read_job_profiles(config)
    print(f"Read {len(job_skills)} job skills")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="protosp03/config/process_raw.yaml"
    )
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    process_raw(config)


if __name__ == "__main__":
    main()
