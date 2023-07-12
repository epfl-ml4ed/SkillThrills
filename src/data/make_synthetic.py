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
    return {i: "skill_" + str(i) for i in range(config["nb_skills"])}


def save_skills(skills, config):
    """Saves the dict of skills to a json file

    Args:
        skills (dict): dict of skills
        config (dict): Configuration dictionary
    """
    with open(os.path.join(config["dataset_path"], "skills.json"), "w") as f:
        json.dump(skills, f)


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


def make_synthetic(config):
    print("Making synthetic data")
    skills = make_skills(config)
    save_skills(skills, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config/synthetic.yaml")
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    make_synthetic(config)


if __name__ == "__main__":
    main()
