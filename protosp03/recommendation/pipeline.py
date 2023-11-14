import os
import json
import argparse

import market
import matchings
import upskillings
import recommendations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path")
    args = parser.parse_args()

    dataset_path = args.dataset_path

    filenames = [
        "skills.json",
        "mastery_levels.json",
        "years.json",
        "learners.json",
        "jobs.json",
        "courses.json",
    ]

    data = [json.load(open(os.path.join(dataset_path, fname))) for fname in filenames]

    skills, mastery_levels, years, learners, jobs, courses = data

    (
        skill_supply,
        skill_demand,
        skill_trends,
        skills_attractiveness,
        learners_attractiveness,
    ) = market.get_all_market_metrics(skills, mastery_levels, learners, jobs, years)

    avg_learners_attractiveness = 0
    for learner_attractiveness in learners_attractiveness:
        avg_learners_attractiveness += learner_attractiveness.total()
    avg_learners_attractiveness /= len(learners_attractiveness)

    print(
        f"The average attractiveness of the learners is {avg_learners_attractiveness}."
    )

    no_recommendation = 0
    no_enrollable_courses = 0

    for learner in learners:
        # Get all enrollable courses for the learner

        enrollable_courses = matchings.get_all_enrollable_courses(
            learner, courses, threshold=0.5
        )

        # Based on the enrollable courses, get all the learnable skills for the learner

        learnable_skills = [
            (skill, level)
            for course in enrollable_courses
            for skill, level in course["provided_skills"].items()
            if skill not in learner["possessed_skills"]
            or learner["possessed_skills"][skill] < level
        ]

        # Among the learnable skills, get the up-skilling advice for the learner

        up_skilling_advice = upskillings.up_skilling_market_advice(
            learner, learnable_skills, skills_attractiveness
        )

        course_recommendation = recommendations.get_course_recommendation(
            learner, enrollable_courses, up_skilling_advice
        )

        if enrollable_courses is None:
            no_enrollable_courses += 1

        if course_recommendation is None:
            no_recommendation += 1

        else:
            for skill, level in course_recommendation["provided_skills"].items():
                if (
                    skill not in learner["possessed_skills"]
                    or learner["possessed_skills"][skill] < level
                ):
                    learner["possessed_skills"][skill] = level

    print(
        f"{no_recommendation} learners ({no_recommendation / len(learners) * 100}%) have no course recommendation."
    )

    print(
        f"{no_enrollable_courses} learners ({no_enrollable_courses / len(learners) * 100}%) have no enrollable courses."
    )

    new_learners_attractiveness = market.get_all_learners_attractiveness(
        learners, years, skill_supply, skill_demand
    )

    avg_learners_attractiveness = 0
    for learner_attractiveness in new_learners_attractiveness:
        avg_learners_attractiveness += learner_attractiveness.total()
    avg_learners_attractiveness /= len(new_learners_attractiveness)

    print(
        f"The new average attractiveness of the learners is {avg_learners_attractiveness}."
    )


if __name__ == "__main__":
    main()
