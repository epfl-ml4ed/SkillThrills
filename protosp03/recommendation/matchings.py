from collections import Counter


def profile_job_match(profile, job):
    """Computes the proportion of required skills for a job that the profile posseses.

    Args:
        profile (set): set of skills that the profile has
        job (set): set of skills required for the job

    Returns:
        float: proporition of skills possesed
    """
    possesed_skills = profile.intersection(job)
    matching = 100 * len(possesed_skills) / len(job)
    return matching


def profile_alljobs_match(profile, jobs, job_inverted_index):
    """Computes a matching for all jobs based on the proportion of skills that the user possesses

    Args:
        profile (set): set of skills that the profile has
        jobs (dict): dictionnary of all jobs and the skills that they require
        job_inverted_index (dict): inverted index of jobs and skills for an efficient search

    Returns:
        dict: jobs matchings based on the proportion of skills that the user possesses
    """
    ranked_jobs = Counter()
    for skill in profile:
        for job in job_inverted_index[skill]:
            ranked_jobs[job] += 100
    for job in ranked_jobs:
        ranked_jobs[job] /= len(jobs[job])
    return ranked_jobs
