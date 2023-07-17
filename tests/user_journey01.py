import json
import random
from protosp03.recommendation import matchings
from protosp03.data.inverted_index import inverted_index_decoder


def main():
    inverted_index_required_course_path = (
        "data/inverted_index/synthetic/courses_required_inverted_index.json"
    )
    inverted_index_provided_course_path = (
        "data/inverted_index/synthetic/courses_provided_inverted_index.json"
    )
    inverted_index_job_path = "data/inverted_index/synthetic/jobs_inverted_index.json"
    resume_path = "data/raw/synthetic/resumes.json"
    course_path = "data/raw/synthetic/courses.json"
    skill_path = "data/raw/synthetic/skills.json"
    job_path = "data/raw/synthetic/jobs.json"

    course_required_inverted_index = json.load(
        open(inverted_index_required_course_path, "r"),
        object_hook=inverted_index_decoder,
    )
    course_provided_inverted_index = json.load(
        open(inverted_index_provided_course_path, "r"),
        object_hook=inverted_index_decoder,
    )
    job_inverted_index = json.load(
        open(inverted_index_job_path, "r"), object_hook=inverted_index_decoder
    )
    skills = json.load(open(skill_path, "r"))
    profiles = json.load(open(resume_path, "r"))
    profiles = {
        key: {skills[skill] for skill in value} for key, value in profiles.items()
    }
    jobs = json.load(open(job_path, "r"))
    jobs = {key: {skills[skill] for skill in value} for key, value in jobs.items()}
    courses = json.load(open(course_path, "r"))
    courses = {
        key: {
            "provided": {skills[skill] for skill in value["provided"]},
            "required": {skills[skill] for skill in value["required"]},
        }
        for key, value in courses.items()
    }

    profile_id = random.choice(list(profiles.keys()))
    profile = profiles[profile_id]
    print(f"Considering Profile {profile_id} with the skills: {profile}")
    job = random.choice(list(jobs.keys()))
    print(
        f"\tWe are assuming that the Profile is interested in Job {job} with the skills: {jobs[job]}"
    )
    matching = matchings.profile_job_match(profile, jobs[job])
    print(
        f"\tThe matching is between the profile and the desired job is: {int(matching)}%"
    )

    print(f"\tPrinting the attractiveness of each skill of the profile:")
    for skill in profile:
        attract = 100 * len(job_inverted_index[skill]) / len(jobs)
        print(
            f"\t\tSkill {skill} is required for {int(attract)}% of the jobs in the market"
        )

    ranked_jobs = matchings.profile_alljobs_match(profile, jobs, job_inverted_index)

    overall_attract = ranked_jobs.total() / len(jobs)

    print(f"\tThe overall attractiveness of the profile is: {int(overall_attract)}")
    print(f"\tPrinting the matching of the profile with respect to each job:")

    for job, matching in ranked_jobs.most_common():
        print(f"\t\tJob {job} has a matching of {int(matching)}%")

    print(f"\tPrinting the matching of the profile with respect to each course:")
    ranked_courses = matchings.profile_allcourse_requirements(
        profile,
        courses,
        course_required_inverted_index,
    )
    for course, matching in ranked_courses.most_common():
        print(f"\t\tCourse {course} has a matching of {int(matching)}%")
        tmp_profile = profile.union(courses[course]["provided"])
        new_matching = matchings.profile_job_match(tmp_profile, jobs[job])
        new_ranked_jobs = matchings.profile_alljobs_match(
            tmp_profile, jobs, job_inverted_index
        )
        new_overall_attract = new_ranked_jobs.total() / len(jobs)

        print(
            f"\t\t\tIf the profile takes the course {course}, it will have the skills: {tmp_profile}"
        )
        print(
            f"\t\t\tThe matching with the job {job} will increase from {int(matching)}% to: {int(new_matching)}%"
        )
        print(
            f"\t\t\tThe overall attractiveness of the profile will increase from {int(overall_attract)} to: {int(new_overall_attract)}"
        )

    print()


if __name__ == "__main__":
    main()
