from collections import Counter, defaultdict


def profile_job_match(profile, job):
    """Computes a profile job matching score.

    Args:
        profile (set): set of skills that the profile has
        job (set): set of skills required for the job

    Returns:
        float: matching score
    """
    matching = 0
    for skill in job:
        if skill in profile:
            sim = min(profile[skill], job[skill]) / job[skill]
            matching += sim
    matching = 100 * matching / len(job)
    return matching


def profile_job_match_with_level(profile, job, skills, levels):
    """Computes a profile job matching score for each level in the hierarchy.

    Args:
        profile (set): set of skills that the profile has
        job (set): set of skills required for the job

    Returns:
        float: matching score
    """
    matching = 0
    return matching


def profile_alljobs_match(profile, jobs, job_inverted_index):
    """Computes a matching for all jobs using the proportion of skills that the user possesses

    Args:
        profile (set): set of skills that the profile has
        jobs (dict): dictionnary of all jobs and the skills that they require
        job_inverted_index (dict): inverted index of jobs and skills for an efficient search

    Returns:
        dict: jobs matchings based on the proportion of skills that the user possesses
    """
    ranked_jobs = Counter()
    for skill in profile:
        if skill in job_inverted_index:
            for job in job_inverted_index[skill]:
                ranked_jobs[job] += 100
    for job in ranked_jobs:
        ranked_jobs[job] /= len(jobs[job])
    return ranked_jobs


def profile_allcourse_requirements(profile, courses, course_required_inverted_index):
    """Computes a matching for all courses based on the proportion of skills that the user possesses
    in the required skills for the course. Remove courses that the user already has the skills for.

    Args:
        profile (set): set of skills that the profile has
        courses (dict): dictionnary of all courses and the skills that they provide
        course_required_inverted_index (dict): inverted index of courses and skills for an efficient search

    Returns:
        dict: courses matchings based on the proportion of skills that the user possesses
    """
    ranked_courses = Counter()
    for skill in profile:
        if skill in course_required_inverted_index:
            for course in course_required_inverted_index[skill]:
                ranked_courses[course] += 100

    # Create a list of courses to remove after iterating over the dictionary
    courses_to_remove = []
    for course in ranked_courses:
        for skill, level in courses[course]["provided"].items():
            if skill in profile and profile[skill] >= level:
                courses_to_remove.append(course)
                break
        else:
            ranked_courses[course] /= len(courses[course]["required"])

    for course in courses_to_remove:
        del ranked_courses[course]

    return ranked_courses
