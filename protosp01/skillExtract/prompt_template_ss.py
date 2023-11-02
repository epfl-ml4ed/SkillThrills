PROMPT_TEMPLATES = {
    "system_course": "You are looking for an online course.",
    "system_job": "You are an expert human resource manager. You need to analyse skills in a job posting.",
    "system_resume": "You are an expert human resource manager. You need to analyse skills in a CV.",
    "extraction": {
        "instruction_job": "You are given a sentence from a job description in German. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.\n",
        "instruction_job_level": "You are given a sentence from a job description in German. Extract all skills, competencies, and tasks that are required from the candidate applying for the job (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level (beginner, intermediate, advanced, or unknown). Return the output as only a json file with the skill as key and mastery level as value.\n",
        "instruction_job_detailed": "You are an expert human resource manager. You are given an extract from a job description in German. Highlight all the skills, competencies and tasks that are required from the candidate applying for the job, by surrounding them with tags '@@' and '##'. Make sure you don't highlight job titles, nor elements related to the company and not to the job itself.\n",
        "instruction_CV": "Extract candidates skills in German from the following German sentence, taken from a CV.\n",
        # "instruction_course": "Extract skills that are learned when following the course described in the following German sentence.\n",
        "instruction_course": "You are given a sentence from a course description in German. Highlight all the skills and competencies that are learned when following the course described in the sentence, by surrounding them with tags '@@' and '##'.\n",
        "instruction_course_level": "You are given a sentence from a job description in German. Extract all skills and competencies that are mentioned in the course description sentence (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level (beginner, intermediate, advanced, or unknown). Return the output as only a json file with the skill as key and mastery level as value.\n",
        "shots": [
            "Sentence: Wir suchen einen Teamleiter mit ausgeprägten Kommunikationskompetenzen, um die Zusammenarbeit und den Informationsaustausch innerhalb des Teams zu fördern.\nAnswer: Wir suchen einen Teamleiter mit ausgeprägten @@Kommunikationskompetenzen##, um die Zusammenarbeit und den Informationsaustausch innerhalb des Teams zu fördern.",
            "Sentence: Die Fähigkeit zur interdisziplinären Zusammenarbeit ist ein Schlüsselkriterium für diese Position. \nAnswer: Die @@Fähigkeit zur interdisziplinären Zusammenarbeit## ist ein Schlüsselkriterium für diese Position.",
            "Sentence: Als Java Senior Software Engineer mit Erfahrung wirst du Mitglied eines Scrum-Teams. \n Answer: Als Java Senior Software Engineer mit Erfahrung wirst du Mitglied eines Scrum-Teams.",
            # "Sentence: In ihrer Rolle als Teamleiterin hat sie kontinuierlich die berufliche Entwicklung ihrer Mitarbeitenden gefördert. \nAnswer: In ihrer Rolle als Teamleiterin hat sie kontinuierlich die berufliche @@Entwicklung ihrer Mitarbeitenden## gefördert.",
            "Sentence: Er ist ein belastbarer Mitarbeiter, der in Zeiten hoher Arbeitsbelastung in der Lage war, richtige Prioritäten zu setzen und Aufgaben durchdacht zu organisieren. \nAnswer: Er ist ein belastbarer Mitarbeiter, der in Zeiten hoher Arbeitsbelastung in der Lage war, @@richtige Prioritäten zu setzen und Aufgaben durchdacht zu organisieren##.",
            "Sentence: Hochqualifizierte, flexible Mitarbeiterinnen und Mitarbeiter aus der Versicherungs- und IT-Branche entwickeln sie weiter. \nAnswer: Hochqualifizierte, flexible Mitarbeiterinnen und Mitarbeiter aus der Versicherungs- und IT-Branche entwickeln sie weiter.",
            "Sentence: Über die letzten Jahre ist es ihm gelungen, sich in einem sich schnell verändernden Umfeld kontinuierlich weiterzuentwickeln. \nAnswer: Über die letzten Jahre ist es ihm gelungen, sich in einem sich schnell verändernden Umfeld @@kontinuierlich weiterzuentwickeln##.\n",
        ],
        "shots_level": [
            'Sentence: Wir suchen einen Teamleiter mit ausgeprägten Kommunikationskompetenzen, um die Zusammenarbeit und den Informationsaustausch innerhalb des Teams zu fördern.\nAnswer: {"Kommunikationskompetenzen": "advanced"}',
            'Sentence: Die Fähigkeit zur interdisziplinären Zusammenarbeit ist ein Schlüsselkriterium für diese Position. \nAnswer: {"Fähigkeit zur interdisziplinären Zusammenarbeit": "unknown"}',
            'Sentence: Als Java Senior Software Engineer mit Erfahrung wirst du Mitglied eines Scrum-Teams. \nAnswer: {"Java Senior Software Engineer": "advanced"}',
            "Sentence: Du arbeitst eng mit unserem erfahrenen Team zusammen und trägst aktiv zum Erfolg des Unternehmens bei. \nAnswer: {}",
            'Sentence: Du hast sehr gute Kenntnisse in digitaler Schaltungstechnik und Regelkreisen. \nAnswer: {"digitaler Schaltungstechnik": "advanced", "Regelkreisen": "advanced"}',
            'Sentence: Nebst guten Kenntnisse in moderner, agiler Softwareentwicklung und deren Konzepte, hast du auch noch ein grundlegendes Wissen in der Testautomatisierung. \nAnswer: {"agiler Softwareentwicklung": "advanced", "Testautomatisierung": "beginner"}',
        ],
    },
    "matching": {
        "instruction_job": "You are an expert human resource manager. You are given a sentence from a job description in German, and a skill extracted from this sentence. Choose from the list of options the one that best match the skill in the context. Answer with the associated letter.\n",
        "instruction_course": "You are looking for an online course. You are given a sentence from a course description in German, and a skill extracted from this sentence. Choose from the list of options the one that best match the skill in the context. Answer with the associated letter.\n",
        "instruction_CV": "You are an expert human resource manager. From a given German sentence from a CV, and a skill extracted from this sentence, choose from the options one or several items that best match the skill in the context. Answer with the associated letter(s).\n",
        "shots": [
            'Sentence: Grundlegende Bestimmungen von Urheberrecht und Datenschutz verstehen. \nSkill: Datenschutz. \nOptions: \nA: "Grundsätze des Datenschutzes respektieren" \nB: "Datenschutz verstehen" \nC: "Datenschutz im Luftfahrtbetrieb sicherstellen" \nD: "Datenschutz". \nAnswer: "Datenschutz verstehen", "Datenschutz".\n'
        ],
    },
    # "extend_taxonomy_tech": {
    #     "instruction": "I am lookgit ing for occurrences of the <SKILL_TYPE> '<NAME>' in a document. However, the author doesn't always refer to this <SKILL_TYPE> using the full name. Generate only a list of 10 other names that I could look for, separated by commas.\n",
    #     "1shot": "Skill: Microsoft Excel \n Answer: Excel, MS Excel, Microsoft Excel, Spreadsheet software by Microsoft, Microsoft's spreadsheet application, Excel program, Excel software, Microsoft's data analysis tool, Microsoft's workbook software, Spreadsheet program by Microsoft\n",
    # },
    # "extend_taxonomy_certif": {
    #     "instruction": "I am looking for occurrences of the <SKILL_TYPE> '<NAME>' in a document. However, the author doesn't always refer to this <SKILL_TYPE> using the full name. Generate only a list of 10 other names that I could look for, separated by commas.\n",
    #     "1shot": "Certification: AWS DevOps Engineer \n Answer: AWS, AWS DevOps Specialist, Amazon DevOps Engineer, AWS DevOps Practitioner, Certified AWS DevOps Professional, AWS DevOps Architect, Amazon Web Services DevOps Expert, AWS DevOps Solutions Engineer, AWS Cloud DevOps Engineer, AWS DevOps Deployment Specialist, AWS DevOps Integration Engineer\n",
    # },
}

## TODO: Think about incorporating the above used in "extend_taxonomy_elements.py" into the Generator