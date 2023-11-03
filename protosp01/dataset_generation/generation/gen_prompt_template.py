PROMPT_TEMPLATE = {
    "baseline": {
        "role_instruction": "You are a hiring manager for a big company. You need to define write a job opening for different skill requirements in your company.\n",
        "instruction": "You are a hiring manager for a big company and your goal is to write the perfect sentence to describe job that uses a set of skills. You'll be given a set of skill, the job posting will reference each of them explicitely or implicitely. The job you describe must include capailities in each of these skills. No external skills should be mentionned. The description of the job should be one line long and be as specific as possible.\n",
        "shots": [
            "skills: [SLQ, relational databases]\nJob opening : ability to manage database and query using SQL.\n"
        ]
    }
}