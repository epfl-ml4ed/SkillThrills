from collections import Counter, defaultdict


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
            sim = (
                min(learner["possessed_skills"][skill], job["required_skills"][skill])
                / job["required_skills"][skill]
            )
            matching += sim

    # Convert total similarity into percentage form
    matching = 100 * matching / len(job["required_skills"])

    return matching
