def upskilling_advice(learner, job, skills_attractiveness):
    upskilling_advice = None
    for skill, level in job["required_skills"].items():
        if skill not in learner["possessed_skills"]:
            if upskilling_advice is None:
                upskilling_advice = (skill, 1)
            elif (
                skills_attractiveness[(skill, level)]
                > skills_attractiveness[upskilling_advice]
            ):
                upskilling_advice = skill
        elif learner["possessed_skills"][skill] < level:
            if upskilling_advice is None:
                upskilling_advice = (skill, 1)
            elif (
                skills_attractiveness[(skill, level)]
                > skills_attractiveness[upskilling_advice]
            ):
                upskilling_advice = (skill, learner["possessed_skills"][skill] + 1)
    return upskilling_advice
