PROMPT_TEMPLATES = {
    "extraction": {
        "instruction_job": "You are an expert human resource manager. You are given a sentence from a job description in German. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.\n",
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
    },
    "matching": {
        "instruction_job": "You are an expert human resource manager. You are given a sentence from a job description in German, and a skill extracted from this sentence. Choose from the list of options the one that best match the skill in the context. Answer with the associated letter.\n",
        "instruction": "You are an expert human resource manager. From a given German sentence from a CV, and a skill extracted from this sentence, choose from the options one or several items that best match the skill in the context. Answer with the associated letter(s).\n",
        "shots": [
            'Sentence: Grundlegende Bestimmungen von Urheberrecht und Datenschutz verstehen. \nSkill: Datenschutz. \nOptions: \nA: "Grundsätze des Datenschutzes respektieren" \nB: "Datenschutz verstehen" \nC: "Datenschutz im Luftfahrtbetrieb sicherstellen" \nD: "Datenschutz". \nAnswer: "Datenschutz verstehen", "Datenschutz".\n'
        ],
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
