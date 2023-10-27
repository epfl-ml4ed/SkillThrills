def up_skilling_advice(learner, job, skills_attractiveness):
    up_skilling_advice = None
    for skill, level in job["required_skills"].items():
        if skill not in learner["possessed_skills"]:
            if up_skilling_advice is None:
                up_skilling_advice = (skill, 1)
            elif (
                skills_attractiveness[(skill, level)]
                > skills_attractiveness[up_skilling_advice]
            ):
                up_skilling_advice = (skill, 1)
        elif learner["possessed_skills"][skill] < level:
            if up_skilling_advice is None:
                up_skilling_advice = (skill, learner["possessed_skills"][skill] + 1)
            elif (
                skills_attractiveness[(skill, level)]
                > skills_attractiveness[up_skilling_advice]
            ):
                up_skilling_advice = (skill, learner["possessed_skills"][skill] + 1)
    return up_skilling_advice
