import sys

sys.path.append("../recommendation")  # Add the src directory to the path

from matchings import learner_course_matching

course = dict()
course["year"] = 2021
course["required_skills"] = {"Tätigkeiten priorisieren": 4, "Erinnerungsvermögen": 4}
course["provided_skills"] = {"Reaktionsgeschwindigkeit": 4, "Führungsorientierung": 2}
learner = {
    "possessed_skills": {
        "Erinnerungsvermögen": 3,
        "Kommunikation in Präsentations- und Vortragssituationen": 3,
        "Reaktionszeit und Schnelligkeit": 4,
        "Statische Kraft": 4,
        "Zusammenarbeit in stabilen Teams": 3,
        "Kommunikation in Ambiguitätssituationen": 1,
    },
    "year": 2021,
}
print(
    learner_course_matching(learner, course)
    # This should output: 0.375
)

course["provided_skills"] = {"Statische Kraft": 2, "Erinnerungsvermögen": 2}

print(
    learner_course_matching(learner, course)
    # This should output: 0 Since the learner has all the provided skills
)
