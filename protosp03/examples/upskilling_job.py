import sys

sys.path.append("../recommendation")  # Add the src directory to the path

from upskillings import up_skilling_job_advice
from recommendations import get_course_recommendation
from matchings import get_all_enrollable_courses


courses = [
    {
        "required_skills": {"Python": 1, "Data Analysis": 1},
        "provided_skills": {"Python": 2, "Data Analysis": 1},
    },
    {
        "required_skills": {"Python": 1, "Data Analysis": 2},
        "provided_skills": {"Python": 2, "Data Analysis": 2},
    },
    {
        "required_skills": {"Python": 2, "Data Analysis": 1},
        "provided_skills": {"Python": 2, "Data Analysis": 2},
    },
    {
        "required_skills": {"Python": 3, "Data Analysis": 1},
        "provided_skills": {"Python": 3, "Data Analysis": 2},
    },
    {
        "required_skills": {"Python": 2, "Data Analysis": 2},
        "provided_skills": {"Python": 3, "Data Analysis": 3},
    },
    {
        "required_skills": {"Python": 1, "Machine Learning": 1},
        "provided_skills": {"Python": 2, "Machine Learning": 1},
    },
    {
        "required_skills": {"Python": 1, "Statistics": 2},
        "provided_skills": {"Python": 2, "Statistics": 2},
    },
    {
        "required_skills": {"Python": 2, "Machine Learning": 1},
        "provided_skills": {"Python": 2, "Machine Learning": 2},
    },
    {
        "required_skills": {"Python": 3, "Statistics": 1},
        "provided_skills": {"Python": 3, "Statistics": 2},
    },
    {
        "required_skills": {"Machine Learning": 2, "Statistics": 2},
        "provided_skills": {"Machine Learning": 3, "Statistics": 3},
    },
]


learner = {"possessed_skills": {"Python": 2, "Data Analysis": 1}}

job = {"required_skills": {"Python": 3, "Data Analysis": 2}}


skills_attractiveness = {("Python", 3): 10, ("Data Analysis", 2): 8}


enrollable_courses = get_all_enrollable_courses(learner, courses, threshold=0.5)

up_skilling_advice = up_skilling_job_advice(learner, job, skills_attractiveness)

course_recommendation = get_course_recommendation(
    learner, enrollable_courses, up_skilling_advice
)

print(
    f"Learner: {learner}\nWanted Job: {job}\nOur advice: {up_skilling_advice}\nOur course recommendation: {course_recommendation}\n"
)

learner = {"possessed_skills": {"Python": 3, "Data Analysis": 2}}

job = {"required_skills": {"Python": 2, "Data Analysis": 2}}

skills_attractiveness = {("Python", 2): 7, ("Data Analysis", 2): 8}

enrollable_courses = get_all_enrollable_courses(learner, courses, threshold=0.5)

up_skilling_advice = up_skilling_job_advice(learner, job, skills_attractiveness)

course_recommendation = get_course_recommendation(
    learner, enrollable_courses, up_skilling_advice
)

print(
    f"Learner: {learner}\nWanted Job: {job}\nOur advice: {up_skilling_advice}\nOur course recommendation: {course_recommendation}\n"
)

learner = {"possessed_skills": {"Python": 1, "Machine Learning": 1, "Statistics": 1}}

job = {"required_skills": {"Python": 3, "Machine Learning": 3, "Statistics": 3}}

skills_attractiveness = {
    ("Python", 2): 15,
    ("Machine Learning", 2): 12,
    ("Statistics", 2): 10,
}


enrollable_courses = get_all_enrollable_courses(learner, courses, threshold=0.5)

up_skilling_advice = up_skilling_job_advice(learner, job, skills_attractiveness)

course_recommendation = get_course_recommendation(
    learner, enrollable_courses, up_skilling_advice
)

print(
    f"Learner: {learner}\nWanted Job: {job}\nOur advice: {up_skilling_advice}\nOur course recommendation: {course_recommendation}"
)
