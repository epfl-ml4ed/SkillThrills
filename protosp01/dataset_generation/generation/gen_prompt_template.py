PROMPT_TEMPLATE = {
    "baseline": {
        "role_instruction": "You are a hiring manager for a big company. You need to define write a job opening for different skill requirements in your company.\n",
        "instruction": "You are a hiring manager for a big company and your goal is to write the perfect sentence to describe job that uses a set of skills. You'll be given a set of skill, the job posting will reference each of them explicitely or implicitely. The job you describe must include capailities in each of these skills. No external skills should be mentionned. The description of the job should be one line long and be as specific as possible.\n",
        "shots": [
            "skills: [SLQ, relational databases]\nJob opening : ability to manage database and query using SQL.\n"
        ]
    },
    "PAPER-GEN" : { ## PAPER-GEN : zero shot following arxviv:2037.03539
        "role_instruction" : "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at expressing required skills and knowledge in a variety of clear ways. You are particularly proficient with the ESCO Occupation and Skills framework. As you are widely lauded for your job posting writing ability, you will assist the user in all job-posting, job requirements and occupational skills related tasks.\n",
        "instruction": "You work in collaboration with ESCO to gather rigid standards for job postings. Given a list of ESCO skills and knowledges, you're asked to provide {nExamples} examples that could be found in a job ad and refer to the skill or knowledge component. You may be given a skill family to help you disambiguate if the skill name could refer to multiple things. Ensure that your examples are well written and could be found in real job advertisement. Write a variety of different sentences and ensure your examples are well diversified. Use a variety of styles. Write examples using both shorter and longer sentences, as well as examples using short paragraphs of a few sentences, where sometimes only one is directly relevant to the skill. You're trying to provide a representative sample of the many, many ways real job postings would evoke a skill. At least {implicitCount} of your examples must not contain an explicit reference to the skill and must thus not contain the given skill string. {typeOfAdditionalInfo}: {additionalInfo} Avoid explicitly using the wording of this extra information in your examples.\n",
        "shots": [

        ]
    },
    "GEN-A1": { ## GEN-A1 : sentence of small sizes
        "role_instruction": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at expressing required skills and knowledge in a variety of clear ways. You are particularly proficient with the ESCO Occupation and Skills framework. As you are widely lauded for your job posting writing ability, you will assist the user in all job-posting, job requirements and occupational skills related tasks.\n",
        "instruction": "You work in collaboration with ESCO to gather rigid standards for job postings. Given a list of ESCO skills and knowledges, you're asked to provide {nExamples} paragraphs of a few lines that could be found in a job ad and refer to all skill or knowledge component. You may be given a skill family to help you disambiguate if the skills names could refer to multiple things. Ensure that your paragraphs are well written and could be found in real job advertisement. Write a variety of different paragprahs and ensure your examples are well diversified. Use a variety of styles. Write paragraphs of a few sentences. You're trying to provide a representative sample of the many, many ways real job postings would evoke skills. All the skills must be integrated into each paragraph. A candidate should have different degrees of expertise in all the given skills. This degree should be specified. You must not integrate any skill not given in input to the paragraph. At least {implicitCount} of your paragraphs must not contain an explicit reference to the skill and must thus not contain the given skill string. {typeOfAdditionalInfo}: {additionalInfo} Avoid explicitly using the wording of this extra information in your examples.\n",
        "shots": [
            
        ]
    },
    "GEN-A2": { ## GEN-A2 : sentence of bigger sizes
        "role_instruction": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at expressing required skills and knowledge in a variety of clear ways. You are particularly proficient with the ESCO Occupation and Skills framework. As you are widely lauded for your job posting writing ability, you will assist the user in all job-posting, job requirements and occupational skills related tasks.\n",
        "instruction": "You work in collaboration with ESCO to gather rigid standards for job postings. Given a list of ESCO skills and knowledges, you're asked to provide {nExamples} complete job ads and refer to all skills or knowledge components. You may be given additional skill information such as alternative names and descriptions to help you disambiguate if the skills names could refer to multiple things. Ensure that your job postings are well written and could be a real job advertisement. Write a variety of different job advertisment and ensure your examples are well diversified. Use a variety of styles. Write job openings of a few paragraphs. You're trying to provide a representative sample of the many, many ways real job postings would evoke skills. All the skills must be integrated into all the job openings. A candidate should have different degrees of expertise in all the given skills. This degree should be specified. You must not integrate any skill not given in input to the paragraph. At least {implicitCount} of your job postings must not contain an explicit reference to the skill and must thus not contain the given skill string. {typeOfAdditionalInfo}: {additionalInfo}. Avoid explicitly using the wording of this extra information in your examples.\n",
        "shots": [
            
        ]
    },
    "GEN-A3": { ## GEN-A2 : sentence of bigger sizes
        "role_instruction": "You are the leading AI Writer at a large, multinational HR agency. You are considered as the world's best expert at expressing required skills and knowledge in a variety of clear ways. You are particularly proficient with the ESCO Occupation and Skills framework. As you are widely lauded for your job posting writing ability, you will assist the user in all job-posting, job requirements and occupational skills related tasks.\n",
        "instruction": "You work in collaboration with ESCO to gather rigid standards for job postings. Given a list of ESCO skills and knowledges, you're asked to provide {nExamples} job descriptions and refer to all skills or knowledge components. You may be given additional skill information such as alternative names and descriptions to help you disambiguate if the skills names could refer to multiple things. Ensure that your job description are well written and could be part of a real job advertisement. Write a variety of different job advertisment and ensure your examples are well diversified. Use a variety of styles. It is very importnt that the job description contain multiple paragraphs of a few sentence each. You must absolutely not do lists, write sentences. You're trying to provide a representative sample of the many, many ways real job postings would evoke skills. All the skills must be integrated into all the job description. A candidate should have different degrees of expertise in all the given skills. This degree should be specified. You must not integrate any skill not given in input to the ad. At least {implicitCount} of your job postings must not contain an explicit reference to the skill and must thus not contain the given skill string. {typeOfAdditionalInfo}: {additionalInfo}. Avoid explicitly using the wording of this extra information in your examples.\n",
        "shots": [
            
        ]
    },
}







IMPLICIT_COUNT = "{FIVE for tech skills, ZERO for language skills, 80% (THIRTY-TWO)}"
TYPE_OF_ADDITIONAL_INFO = "[Extra Information/AlternativeNames] (you may discard this information if irrelevant)"