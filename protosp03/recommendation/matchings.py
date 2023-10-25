from collections import Counter
import sys


def skill_skill_similarity(provided_level, required_level):
    """
    Computes the similarity between two mastery levels of the same skill.

    Args:
        provided_level (int): Mastery level of the provided_skill.
        required_level (int): Mastery level of the required_skill.

    Returns:
        float: Similarity ratio for the given skill levels, ranging from 0 to 1.
    """
    return min(provided_level, required_level) / required_level


def learner_job_matching(learner, job):
    """
    Computes the compatibility score between a learner's skills and a job's required skills.

    For each required skill in the job, the function checks if the learner possesses that skill.
    If so, it calculates a similarity ratio based on the learner's mastery level and the
    job's required level for that skill. The final matching score is the average of all these
    similarity ratios for all required skills, expressed as a percentage.

    Args:
        learner (dict): Dictionary containing details about the learner.
                        - "possessed_skills": Dictionary where keys are skill names and values
                                              represent mastery levels.
                        - "year": Year associated with the learner's data.
        job (dict): Dictionary containing job requirements.
                    - "required_skills": Dictionary where keys are skill names and values
                                         represent required mastery levels.
                    - "year": Year associated with the job's data.

    Returns:
        float: Matching score between the learner and the job, ranging from 0 to 100.

    Example:
        learner = {
            "possessed_skills": {
                "Python": 3,
                "JavaScript": 1
            },
            "year": 2020
        }
        job = {
            "required_skills": {
        "Python": 2,
        "JavaScript": 3
            },
            "year": 2023
        }
        score = learner_job_matching(learner, job)
        print(score)  # This would output 66.66666
    """
    matching = 0

    # For each required skill in the job
    for skill in job["required_skills"]:
        # Check if the learner possesses the skill
        if skill in learner["possessed_skills"]:
            # Calculate similarity ratio based on mastery levels

            matching += skill_skill_similarity(
                learner["possessed_skills"][skill], job["required_skills"][skill]
            )

    # Convert total similarity into percentage form
    matching = 100 * matching / len(job["required_skills"])

    return matching


def shortest_distance(node1, node2):
    """Compute the shortest distance between two nodes based on their paths.

    Args:
        node1 (list): Path from the group to the first node.
        node2 (list): Path from the group to the second node.

    Returns:
        int: The shortest distance between the two nodes.

    Example:
        shortest_distance([2, 1, 3], [2, 1]) # This would output 1
    """

    # Find the last common ancestor
    min_len = min(len(node1), len(node2))
    last_common_index = -1
    for i in range(min_len):
        if node1[i] == node2[i]:
            last_common_index = i
        else:
            break

    # If there's no common ancestor (e.g., [], [1,2,3]), set last_common_index to maxsize
    if last_common_index == -1:
        return sys.maxsize

    # Calculate distance from last common ancestor to each node and sum them up
    distance_node1 = len(node1) - last_common_index - 1
    distance_node2 = len(node2) - last_common_index - 1
    return distance_node1 + distance_node2


def learner_job_group_matching(learner, job, groups_dict, levels_dict):
    """Match a learner's skills with job requirements grouped by skill categories.

    Args:
        learner (dict): Learner's profile including possessed skills and levels.
        job (dict): Job's profile including required skills and levels.
        groups_dict (dict): Mapping of group IDs to group names.
        levels_dict (dict): Mapping of skills to their levels and group associations.

    Returns:
        Counter: A dictionary of group names with corresponding matchings.

    Example:

     job = {
         'required_skills': {
             'Fokuswechsel': 1,
             'Organisation': 2
         },
         'year': 2022}

     learner = {
         'possessed_skills': {
             'Organisation': 2,
             'Hand/Finger-Geschwindigkeit': 4
         },
         'year': 2019}

     groups_dict = {
         2: 'Fertigkeit',
         5: 'Fachkompetenz'}


     levels_dict = {
         'Fokuswechsel': [2, 1, 8, 2],
         'Hand/Finger-Geschwindigkeit': [2, 3, 3, 2],
         'Organisation': [5, 10]}

        matchings = learner_job_group_matching(learner, job, groups_dict, levels_dict)
        matchings.most_common() # This would output [('Fachkompetenz', 50.0), ('Fertigkeit', 7.142857142857142)]

    """
    group_matchings = Counter()

    for group_id, group_name in groups_dict.items():
        matching = 0
        # For each required skill in the job
        for job_skill in job["required_skills"]:
            if levels_dict[job_skill][0] != group_id:
                continue
            # Check if the learner possesses the skill
            if job_skill in learner["possessed_skills"]:
                # Calculate similarity ratio based on mastery levels
                matching += skill_skill_similarity(
                    learner["possessed_skills"][job_skill],
                    job["required_skills"][job_skill],
                )
            else:
                # If the learner does not possess the skill, calculate similarity ratio based on the distance between the skill and the learner's skills
                learner_skills = list(learner["possessed_skills"].keys())
                min_distance = sys.maxsize
                closest_skill = None
                for learner_skill in learner_skills:
                    distance = shortest_distance(
                        levels_dict[job_skill], levels_dict[learner_skill]
                    )
                    if distance < min_distance:
                        min_distance = distance
                        closest_skill = learner_skill

                matching += skill_skill_similarity(
                    learner["possessed_skills"][closest_skill],
                    job["required_skills"][job_skill],
                ) / (min_distance + 1)

        # Convert total similarity into percentage form
        matching = 100 * matching / len(job["required_skills"])
        group_matchings[group_name] = matching
    return group_matchings
