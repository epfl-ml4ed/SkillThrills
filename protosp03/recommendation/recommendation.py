def get_course_recommendation(learner, enrollable_courses, up_skilling_advice):
    """Get the course recommendation for a learner and a given up-skilling advice.

    Args:
        learner (dict): Learner's profile including possessed skills and levels.
        enrollable_courses (list): List of all enrollable courses.
        up_skilling_advice (tuple): A tuple of the skill and level to up-skill to.

    Returns:
        dict: The course recommendation
    """
    course = None
    matching = 0
    for course in enrollable_courses:
        if up_skilling_advice in course["provided_skills"].items():
            tmp_matching = matchings.learner_course_matching(learner, course)
            if tmp_matching > matching:
                matching = tmp_matching
                course_recommendation = course
    return course
