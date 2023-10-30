import sys

sys.path.append("../recommendation")  # Add the src directory to the path

from upskillings import up_skilling_advice

learner = {"possessed_skills": {"Python": 2, "Data Analysis": 1}}

job = {"required_skills": {"Python": 3, "Data Analysis": 2}}

skills_attractiveness = {("Python", 3): 10, ("Data Analysis", 2): 8}

advice = up_skilling_advice(learner, job, skills_attractiveness)
print(
    advice
    # This should output ("Data Analysis", 2)
)


learner = {"possessed_skills": {"Python": 3, "Data Analysis": 2}}

job = {"required_skills": {"Python": 2, "Data Analysis": 2}}

skills_attractiveness = {("Python", 2): 7, ("Data Analysis", 2): 8}

advice = up_skilling_advice(learner, job, skills_attractiveness)
print(
    advice
    # Should print None since the learner already meets the job requirements
)


learner = {"possessed_skills": {"Python": 1, "Machine Learning": 1, "Statistics": 1}}

job = {"required_skills": {"Python": 3, "Machine Learning": 3, "Statistics": 3}}

skills_attractiveness = {
    ("Python", 2): 15,
    ("Machine Learning", 2): 12,
    ("Statistics", 2): 10,
}

advice = up_skilling_advice(learner, job, skills_attractiveness)
print(
    advice
    # Should print ("Python", 2) since it is the most attractive
)
