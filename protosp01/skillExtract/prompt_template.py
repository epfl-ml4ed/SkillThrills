PROMPT_TEMPLATES = {
    "extraction": {
        "instruction_en": "You are an expert human resource manager. You are given an extract from a job description. Highlight all the skills, competencies and tasks that are required from the candidate applying for the job, by surrounding them with tags '@@' and '##'. Make sure you don't highlight job titles, nor elements related to the company and not to the job itself.\n",
        "instruction_job": "You are an expert human resource manager. You are given a sentence from a job description in German. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.\n",
        "instruction_job_level": "You are an expert human resource manager. You are given a sentence from a job description in German. Extract all skills, competencies, and tasks that are required from the candidate applying for the job and infer the corresponding mastery skill level (beginner, intermediate, advanced, or unknown). Return the output as a json file with the skill as key and mastery level as value. \n",
        "instruction_job_detailed": "You are an expert human resource manager. You are given an extract from a job description in German. Highlight all the skills, competencies and tasks that are required from the candidate applying for the job, by surrounding them with tags '@@' and '##'. Make sure you don't highlight job titles, nor elements related to the company and not to the job itself.\n",
        "instruction_CV": "You are an expert human resource manager. Extract candidates skills in German from the following German sentence, taken from a CV.\n",
        # "instruction_course": "You are looking for an online course. Extract skills that are learned when following the course described in the following German sentence.\n",
        "instruction_course": "You are looking for an online course. You are given a sentence from a course description in German. Highlight all the skills and competencies that are learned when following the course described in the sentence, by surrounding them with tags '@@' and '##'.\n",
        "shots": [
            "Sentence: Wir suchen einen Teamleiter mit ausgeprägten Kommunikationskompetenzen, um die Zusammenarbeit und den Informationsaustausch innerhalb des Teams zu fördern.\nAnswer: Wir suchen einen Teamleiter mit ausgeprägten @@Kommunikationskompetenzen##, um die Zusammenarbeit und den Informationsaustausch innerhalb des Teams zu fördern.",
            "Sentence: Die Fähigkeit zur interdisziplinären Zusammenarbeit ist ein Schlüsselkriterium für diese Position. \n Answer: Die @@Fähigkeit zur interdisziplinären Zusammenarbeit## ist ein Schlüsselkriterium für diese Position.",
            "Sentence: Als Java Senior Software Engineer mit Erfahrung wirst du Mitglied eines Scrum-Teams. \n Answer: Als Java Senior Software Engineer mit Erfahrung wirst du Mitglied eines Scrum-Teams.",
            # "Sentence: In ihrer Rolle als Teamleiterin hat sie kontinuierlich die berufliche Entwicklung ihrer Mitarbeitenden gefördert. \nAnswer: In ihrer Rolle als Teamleiterin hat sie kontinuierlich die berufliche @@Entwicklung ihrer Mitarbeitenden## gefördert.",
            "Sentence: Er ist ein belastbarer Mitarbeiter, der in Zeiten hoher Arbeitsbelastung in der Lage war, richtige Prioritäten zu setzen und Aufgaben durchdacht zu organisieren. \nAnswer: Er ist ein belastbarer Mitarbeiter, der in Zeiten hoher Arbeitsbelastung in der Lage war, @@richtige Prioritäten zu setzen und Aufgaben durchdacht zu organisieren##.",
            "Sentence: Hochqualifizierte, flexible Mitarbeiterinnen und Mitarbeiter aus der Versicherungs- und IT-Branche entwickeln sie weiter. \nAnswer: Hochqualifizierte, flexible Mitarbeiterinnen und Mitarbeiter aus der Versicherungs- und IT-Branche entwickeln sie weiter.",
            "Sentence: Über die letzten Jahre ist es ihm gelungen, sich in einem sich schnell verändernden Umfeld kontinuierlich weiterzuentwickeln. \nAnswer: Über die letzten Jahre ist es ihm gelungen, sich in einem sich schnell verändernden Umfeld @@kontinuierlich weiterzuentwickeln##.\n",
        ],
        "shots_level": [
            "Sentence: Wir suchen einen Teamleiter mit ausgeprägten Kommunikationskompetenzen, um die Zusammenarbeit und den Informationsaustausch innerhalb des Teams zu fördern.\nAnswer: {'Kommunikationskompetenzen': 'advanced'}",
            "Sentence: Die Fähigkeit zur interdisziplinären Zusammenarbeit ist ein Schlüsselkriterium für diese Position. \n Answer: {'Fähigkeit zur interdisziplinären Zusammenarbeit': 'unknown'}",
            "Sentence: Als Java Senior Software Engineer mit Erfahrung wirst du Mitglied eines Scrum-Teams. \n Answer: {'java': 'advanced', 'software engineering': 'advanced'}",
        ],
        "shots_en": [
            "Sentence: We are looking for a team leader with strong communication skills to foster collaboration and information sharing within the team.\nAnswer: We are looking for a team leader with strong @@communication skills## to foster collaboration and information sharing within the team.",
            "Sentence: the ability to work collaboratively across disciplines is a key criterion for this position. \n Answer: @@ability to collaborate across disciplines## is a key criterion for this position.",
            "Sentence: As a Java Senior Software Engineer with experience, you will be a member of a Scrum team. \n Answer: As a Java Senior Software Engineer with experience, you will be a member of a Scrum team.",
            "Sentence: In her role as a team leader, she has continuously supported the professional development of her employees. \nAnswer: In her role as a team leader, she has continuously fostered the professional @@development of her employees##.",
            "Sentence: He is a resilient employee who has been able to set proper priorities and organize tasks thoughtfully during periods of heavy workload. \nAnswer: He is a resilient employee who has been able to set @@correct priorities and organize tasks thoughtfully## during periods of high workload.",
            "Sentence: Highly qualified, flexible employees from the insurance and IT industry develop them further. \nAnswer: Highly qualified, flexible employees from the insurance and IT industries continue to develop them.",
            "Sentence: Over the past few years, it has succeeded in continuously developing itself in a rapidly changing environment. \nAnswer: Over the past few years, he has succeeded in @@continuously developing## himself in a rapidly changing environment##.\n",

            # ...,
            # "Sentence: As a senior software engineer the candidate should be able to coach the other developpers, give valuable feedbacks. \nAnswer: As a senior software engineer the candidate should be able to @@coach the other developpers##, @@give valuable feedbacks##."
            # "Sentence: We are looking for a senior software engineer proficient in machine learning and able to use pytorch and python. \nAnswer: We are looking for a senior software engineer proficient in @@machine learning## and able to use @@pytorch## and @@python##."
            # "Sentence: 3 years of experience with back-end development, preferably using Node.js. \nAnswer: 3 years of experience with @@back-end development##, preferably using @@Node.js##",
            # "Sentence: We are looking for a team leader with pronounced communication skills to promote collaboration and information exchange within the team. \nAnswer: We are looking for a team leader with pronounced @@communication skills## to promote collaboration and information exchange within the team.\n",
            # "Sentence: The ability to collaborate interdisciplinarily is a key criterion for this position. \nAnswer: The ability to @@collaborate interdisciplinarily## is a key criterion for this position.\n",
            # "Sentence: He is a resilient employee who, during times of high workload, was able to set proper priorities and organize tasks thoughtfully. \nAnswer: He is a resilient employee who, during times of high workload, was able to set @@proper priorities and organize tasks thoughtfully##."
        ]
    },
    "matching": {
        "instruction_en" : "You are an expert human resource manager. You are given a sentence from a job description, and a skill extracted from this sentence. Choose from the list of options the one that best match the skill in the context. Answer with the associated letter.\n",
        "instruction_job": "You are an expert human resource manager. You are given a sentence from a job description in German, and a skill extracted from this sentence. Choose from the list of options the one that best match the skill in the context. Answer with the associated letter.\n",
        "instruction": "You are an expert human resource manager. From a given German sentence from a CV, and a skill extracted from this sentence, choose from the options one or several items that best match the skill in the context. Answer with the associated letter(s).\n",
        "shots_en": [
            'Sentence: Understand basic provisions of copyright and privacy. \nSkill: Data protection. \nOptions: \nA: "Respect privacy principles." \nB: "Understand data protection" \nC: "Ensure data protection in aviation operations" \nD: "Data protection." \nAnswer: "Understanding data protection", "Data protection".\n',
        ],
        "shots": [
            'Sentence: Grundlegende Bestimmungen von Urheberrecht und Datenschutz verstehen. \nSkill: Datenschutz. \nOptions: \nA: "Grundsätze des Datenschutzes respektieren" \nB: "Datenschutz verstehen" \nC: "Datenschutz im Luftfahrtbetrieb sicherstellen" \nD: "Datenschutz". \nAnswer: "Datenschutz verstehen", "Datenschutz".\n'
        ]
    },
    # "extend_taxonomy_tech": {
    #     "instruction": "I am looking for occurrences of the <SKILL_TYPE> '<NAME>' in a document. However, the author doesn't always refer to this <SKILL_TYPE> using the full name. Generate only a list of 10 other names that I could look for, separated by commas.\n",
    #     "1shot": "Skill: Microsoft Excel \n Answer: Excel, MS Excel, Microsoft Excel, Spreadsheet software by Microsoft, Microsoft's spreadsheet application, Excel program, Excel software, Microsoft's data analysis tool, Microsoft's workbook software, Spreadsheet program by Microsoft\n",
    # },
    # "extend_taxonomy_certif": {
    #     "instruction": "I am looking for occurrences of the <SKILL_TYPE> '<NAME>' in a document. However, the author doesn't always refer to this <SKILL_TYPE> using the full name. Generate only a list of 10 other names that I could look for, separated by commas.\n",
    #     "1shot": "Certification: AWS DevOps Engineer \n Answer: AWS, AWS DevOps Specialist, Amazon DevOps Engineer, AWS DevOps Practitioner, Certified AWS DevOps Professional, AWS DevOps Architect, Amazon Web Services DevOps Expert, AWS DevOps Solutions Engineer, AWS Cloud DevOps Engineer, AWS DevOps Deployment Specialist, AWS DevOps Integration Engineer\n",
    # },
}

## TODO: Think about incorporating the above used in "extend_taxonomy_elements.py" into the Generator
