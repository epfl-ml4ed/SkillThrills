import sys

sys.path.append("../recommendation")  # Add the src directory to the path

from matchings import skill_skill_similarity, learner_job_matching


learner1 = {"possessed_skills": {"Python": 5, "JavaScript": 4}, "year": 2020}
job1 = {"required_skills": {"Python": 3, "JavaScript": 2}, "year": 2023}

score1 = learner_job_matching(learner1, job1)
print(
    f"Scenario 1: {score1:.2f}%"
)  # This should be 100% as learner meets and exceeds all requirements


learner2 = {"possessed_skills": {"Python": 2, "JavaScript": 1}, "year": 2020}
job2 = {"required_skills": {"Python": 3, "JavaScript": 4}, "year": 2023}

score2 = learner_job_matching(learner2, job2)
print(
    f"Scenario 2: {score2:.2f}%"
)  # This should be 45.83% since the learner's skill levels are below the requirements


learner3 = {"possessed_skills": {"Python": 3}, "year": 2020}
job3 = {"required_skills": {"Python": 2, "JavaScript": 3}, "year": 2023}

score3 = learner_job_matching(learner3, job3)
print(
    f"Scenario 3: {score3:.2f}%"
)  # This should 50% as the learner lacks JavaScript skill altogether


learner4 = {
    "possessed_skills": {
        "Python": 3,
        "JavaScript": 3,
        "Java": 5,  # Not required by job
    },
    "year": 2020,
}
job4 = {"required_skills": {"Python": 2, "JavaScript": 3}, "year": 2023}

score4 = learner_job_matching(learner4, job4)
print(
    f"Scenario 4: {score4:.2f}%"
)  # This will be 100% because even though learner knows Java, the algorithm focuses on required skills
