from protosp03.recommendation import matchings


def test_profile_job_match():
    profile = {"python", "java", "sql", "c++"}
    job = {"python", "java", "sql"}
    assert matchings.profile_job_match(profile, job) == 100


def test_profile_alljobs_match():
    profile = {"python", "java", "sql"}
    jobs = {
        "job1": {"python", "java", "sql", "c++"},
        "job2": {"python", "java", "sql", "c++"},
        "job3": {"python", "java", "sql", "c++"},
    }
    job_inverted_index = {
        "python": ["job1", "job2", "job3"],
        "java": ["job1", "job2", "job3"],
        "sql": ["job1", "job2", "job3"],
        "c++": ["job1", "job2", "job3"],
    }
    assert matchings.profile_alljobs_match(profile, jobs, job_inverted_index) == {
        "job1": 75,
        "job2": 75,
        "job3": 75,
    }


def main():
    test_profile_job_match()
    test_profile_alljobs_match()


if __name__ == "__main__":
    main()
