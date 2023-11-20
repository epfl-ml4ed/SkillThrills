########### SHOT EXAMPLES ###########
### extracts only skills marked with @@ and ##
job_shots_extr_skills = [
    "Sentence: Wir suchen einen Teamleiter mit ausgeprägten Kommunikationskompetenzen, um die Zusammenarbeit und den Informationsaustausch innerhalb des Teams zu fördern.\nAnswer: Wir suchen einen Teamleiter mit ausgeprägten @@Kommunikationskompetenzen##, um die Zusammenarbeit und den Informationsaustausch innerhalb des Teams zu fördern.",
    "Sentence: Die Fähigkeit zur interdisziplinären Zusammenarbeit ist ein Schlüsselkriterium für diese Position. \nAnswer: Die @@Fähigkeit zur interdisziplinären Zusammenarbeit## ist ein Schlüsselkriterium für diese Position.",
    "Sentence: Als Java Senior Software Engineer mit Erfahrung wirst du Mitglied eines Scrum-Teams. \nAnswer: Als Java Senior Software Engineer mit Erfahrung wirst du Mitglied eines Scrum-Teams.",
    "Sentence: Er ist ein belastbarer Mitarbeiter, der in Zeiten hoher Arbeitsbelastung in der Lage war, richtige Prioritäten zu setzen und Aufgaben durchdacht zu organisieren. \nAnswer: Er ist ein belastbarer Mitarbeiter, der in Zeiten hoher Arbeitsbelastung in der Lage war, @@richtige Prioritäten zu setzen und Aufgaben durchdacht zu organisieren##.",
    "Sentence: Hochqualifizierte, flexible Mitarbeiterinnen und Mitarbeiter aus der Versicherungs- und IT-Branche entwickeln sie weiter. \nAnswer: Hochqualifizierte, flexible Mitarbeiterinnen und Mitarbeiter aus der Versicherungs- und IT-Branche entwickeln sie weiter.",
    "Sentence: Über die letzten Jahre ist es ihm gelungen, sich in einem sich schnell verändernden Umfeld kontinuierlich weiterzuentwickeln. \nAnswer: Über die letzten Jahre ist es ihm gelungen, sich in einem sich schnell verändernden Umfeld @@kontinuierlich weiterzuentwickeln##.\n",
]

### extracts skills and their corresponding mastery levels as {skill: level} pairs
job_shots_extr_wlevels = [
    'Sentence: Wir suchen einen Teamleiter mit ausgeprägten Kommunikationskompetenzen, um die Zusammenarbeit und den Informationsaustausch innerhalb des Teams zu fördern.\nAnswer: {"Kommunikationskompetenzen": "expert"}',
    'Sentence: Die Fähigkeit zur interdisziplinären Zusammenarbeit ist ein Schlüsselkriterium für diese Position. \nAnswer: {"Fähigkeit zur interdisziplinären Zusammenarbeit": "unknown"}',
    'Sentence: Als Java Senior Software Engineer mit Erfahrung wirst du Mitglied eines Scrum-Teams. \nAnswer: {"Java Senior Software Engineer": "expert"}',
    'Sentence: Du arbeitst eng mit unserem erfahrenen Team zusammen und trägst aktiv zum Erfolg des Unternehmens bei. \nAnswer: {"arbeitst eng mit unserem erfahrenen Team zusammen": "unknown"}',
    'Sentence: Du hast sehr gute Kenntnisse in digitaler Schaltungstechnik und Regelkreisen. \nAnswer: {"digitaler Schaltungstechnik": "expert", "Regelkreisen": "expert"}',
    'Sentence: Nebst guten Kenntnisse in moderner, agiler Softwareentwicklung und deren Konzepte, hast du auch noch ein grundlegendes Wissen in der Testautomatisierung. \nAnswer: {"agiler Softwareentwicklung": "expert", "Testautomatisierung": "beginner"}',
]

### extracts skills and their corresponding mastery levels and required/optional as {skill: [level, req_status]} pairs
job_shots_extr_wreqs = [
    'Sentence: Wir suchen einen Teamleiter mit ausgeprägten Kommunikationskompetenzen, um die Zusammenarbeit und den Informationsaustausch innerhalb des Teams zu fördern.\nAnswer: {"Kommunikationskompetenzen": ["expert", "mandatory"]}',
    'Sentence: Die Fähigkeit zur interdisziplinären Zusammenarbeit ist ein Schlüsselkriterium für diese Position. \nAnswer: {"Fähigkeit zur interdisziplinären Zusammenarbeit": ["unknown", "mandatory"]}',
    'Sentence: Als Java Senior Software Engineer mit Erfahrung wirst du Mitglied eines Scrum-Teams. \nAnswer: {"Java Senior Software Engineer": ["expert", "unknown"]}',
    'Sentence: Du arbeitst eng mit unserem erfahrenen Team zusammen und trägst aktiv zum Erfolg des Unternehmens bei. \nAnswer: {"arbeitst eng mit unserem erfahrenen Team zusammen": ["unknown", "unknown"]}',
    'Sentence: Du hast sehr gute Kenntnisse in digitaler Schaltungstechnik und Regelkreisen. \nAnswer: {"digitaler Schaltungstechnik": ["expert", "unknown"], "Regelkreisen": ["expert", "unknown"]}',
    'Sentence: Nebst guten Kenntnisse in moderner, agiler Softwareentwicklung und deren Konzepte, hast du auch noch ein grundlegendes Wissen in der Testautomatisierung. \nAnswer: {"agiler Softwareentwicklung": ["intermediate", "unknown"], "Testautomatisierung": ["beginner", "unknown"]}',
]


course_shots_extr_wlevels = [
    'Sentence: Digitale Kompetenzen einfacher einschätzen und besser erlernen können.\nAnswer: {"Kompetenzen einfacher einschätzen und besser erlernen": "unknown"}',
    'Sentence: Strategisches IT-GRC: IT-Governance, IT-Risk und Compliance Management als Bindeglied zwischen Business und IT.\nAnswer: {"IT-Governance, IT-Risk und Compliance Management": "unknown", "Bindeglied zwischen Business und IT": "unknown"}',
    'Sentence: Adaptives IT-Demand und -Projektportfolio Management: Arten von ITDPPM, Methoden und Toolunterstützung.\nAnswer: {"IT-Demand und -Projektportfolio Management": "unknown"}',
    "Sentence: Die interkulturelle Dimension spielt in unserer globalisierten Zeit eine wichtige Rolle.\nAnswer: {}",
    'Sentence: Sie erweitern Ihr Handlungsrepertoire als Coach und können Menschen aus unterschiedlichsten Kontexten kultursensibel und lösungsorientiert begleiten.\nAnswer: {"Menschen aus unterschiedlichsten Kontexten kultursensibel und lösungsorientiert begleiten": "unknown"}',
    'Sentence: In unserem praxisnahen Lehrgang erfahren Sie, wie Sie solche Entwicklungsprozesse zielführend anleiten, mit wirkungsvollen Tools fördern und stimmig abschliessen.\nAnswer: {"Entwicklungsprozesse zielführend anleiten, mit wirkungsvollen Tools fördern und stimmig abschliessen": "unknown"}',
]


# send example to Marco to check on these
# ask for matching examples

job_shots_match = [
    'Sentence: Grundlegende Bestimmungen von Urheberrecht und Datenschutz verstehen. \nSkill: Datenschutz. \nOptions: \nA: "Grundsätze des Datenschutzes respektieren" \nB: "Datenschutz verstehen" \nC: "Datenschutz im Luftfahrtbetrieb sicherstellen" \nD: "Datenschutz". \nAnswer: "Datenschutz verstehen", "Datenschutz".\n'
]

course_shots_match = [
    """
    Sentence: Sie haben alle wichtigen Tools und Methoden in Ihrem Rucksack, um Entwicklungsprozesse im interkulturellen Umfeld zu begleiten, erfolgreich abzuschliessen und zu evaluieren.
    \nSkill: Entwicklungsprozesse im interkulturellen Umfeld zu begleiten, erfolgreich abzuschliessen und zu evaluieren
    \nOptions: \nA: "Kognitive Fertigkeit: Aufmerksamkeit: Fokuswechsel: Zwischen zwei oder mehreren Aktivitäten oder Informationsquellen (z. B. Sprache, Geräusche, Berührungen oder andere Quellen) hin und her wechseln."
    \nB: "Soziale Fertigkeit: Fertigkeiten, um in der Zusammenarbeit mit anderen Menschen Ziele zu erreichen"
    \nC: "Leistungsorientierung: Die Person ist bestrebt, persönliche Ziele zu erreichen und in der eigenen Arbeit kompetent zu sein."
    \nD: "Kommunikation: Die Kompetenz, erfolgreich mit Personen aus verschiedenen kulturellen Hintergründen zu kommunizieren, indem man kulturelle Unterschiede respektiert, Verständnis und Sensibilität zeigt und eine offene und inklusive Kommunikationsumgebung fördert."
    \nE: "Kollaboration: Zusammenarbeit in interdisziplinären Teams: Die Kompetenz, erfolgreich mit Mitgliedern aus verschiedenen Berufsgruppen oder Fachrichtungen zusammenzuarbeiten, indem unterschiedliche Fachkenntnisse, Perspektiven und Arbeitsweisen respektiert und integriert werden, um gemeinsame Ziele zu erreichen."
    \nF: "Kommunikation: Die Kompetenz, in Coaching- und Beratungssituationen einfühlsam auf das Gegenüber (i.d.R. Klienten) einzugehen und ihre Anliegen zu klären, um damit eine vertrauensvolle Beziehung aufzubauen und unterstützende Ratschläge zu geben."
    \nAnswer: "Kommunikation: Die Kompetenz, erfolgreich mit Personen aus verschiedenen kulturellen Hintergründen zu kommunizieren, indem man kulturelle Unterschiede respektiert, Verständnis und Sensibilität zeigt und eine offene und inklusive Kommunikationsumgebung fördert.", "Kommunikation: Die Kompetenz, in Coaching- und Beratungssituationen einfühlsam auf das Gegenüber (i.d.R. Klienten) einzugehen und ihre Anliegen zu klären, um damit eine vertrauensvolle Beziehung aufzubauen und unterstützende Ratschläge zu geben."
    """
]

### TEMP HOLDERS BELOW ###
job_shots_match = course_shots_match

course_shots_extr_skills = job_shots_extr_skills

cv_shots_extr_skills = job_shots_extr_skills
cv_shots_extr_wlevels = job_shots_extr_wlevels
cv_shots_match = job_shots_match


job_shots_tl = [
    'Ich spreche sehr gut Java und Deutsch.\n Answer: {"agiler Softwareentwicklung": ["expert", "unknown"], "Testautomatisierung": ["beginner", "unknown"]}',
]

########### INSTRUCTIONS ###########
job_inst_extr_wlevels = "You are given a sentence from a job description in German. Extract all skills, competencies, and tasks that are required from the candidate applying for the job (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level (beginner, intermediate, expert, or unknown). Return the output as only a json file with the skill as key and mastery level as value.\n"
job_inst_extr_wreqs = 'You are given a sentence from a job description in German. Extract all skills and competencies that are required from the candidate applying for the job (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level as well as if the skill or task is mandatory for the job or optional (nice-to-have). Return the output as a json file with the extracted skill as key and a list of ("mastery level", "requirement status") as the value. Mastery level should be either "expert", "intermediate", "beginner", or "unknown" and requirement status should be either "required", "optional", or "unknown" based on the context.\n'

course_inst_extr_wlevels = "You are given a sentence from a job description in German. Extract all skills and competencies that are mentioned in the course description sentence (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level (beginner, intermediate, expert, or unknown). Return the output as only a json file with the skill as key and mastery level as value.\n"
# course_inst_extr_wreqs = 'You are given a sentence from a course description in German. Extract all skills and competencies that are mentioned in the course description sentence (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level as well as if the skill or task is mandatory for the course or optional (nice-to-have). Return the output as a json file with the extracted skill as key and a list of ("mastery level", "requirement status") as the value. Mastery level should be either "expert", "intermediate", "beginner", or "unknown" and requirement status should be either "required", "optional", or "unknown" based on the context.\n'

########### PROMPT TEMPLATES ###########

PROMPT_TEMPLATES = {
    "job": {
        "system": "You are an expert human resource manager. You need to analyse skills in a job posting.",
        "extraction": {
            "skills": {
                "instruction": "You are an expert human resource manager. You are given an extract from a job description in German. Highlight all the skills, competencies and tasks that are required from the candidate applying for the job, by surrounding them with tags '@@' and '##'. Make sure you don't highlight job titles, nor elements related to the company and not to the job itself.\n",
                # "instruction": "You are given a sentence from a job description in German. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.\n"
                "shots": job_shots_extr_skills,
            },
            "wlevels": {
                "instruction": job_inst_extr_wlevels,
                "shots": job_shots_extr_wlevels,
            },
            "wreqs": {
                "instruction": job_inst_extr_wreqs,
                "shots": job_shots_extr_wreqs,
            },
        },
        "matching": {
            "instruction": "You are an expert human resource manager. You are given a sentence from a job description in German, and a skill extracted from this sentence. Choose from the list of options the one that best match the skill in the context. Answer with the associated letter.\n",
            "shots": job_shots_match,
        },
        "tech_lang": {
            "instruction": 'You are an expert human resource manager. You are given a sentence from a job description in German, and technology or language skills from this sentence. Help me double check the skill is actually the sentence and correct the skill was not accurate and the corresponding mastery skill level as well as if the skill or task is mandatory for the job or optional (nice-to-have). Return the output as a json file with the correct skill as key and a list of ("mastery level", "requirement status", "job") as the value. Mastery level should be either "expert", "intermediate", "beginner", or "unknown" and requirement status should be either "required", "optional", or "unknown" based on the context.\n',
            "shots": job_shots_tl,
        },
    },
    "course": {
        "system": "You are looking for an online course.",
        "extraction": {
            "skills": {
                "instruction": "You are given a sentence from a course description in German. Highlight all the skills and competencies that are learned when following the course described in the sentence, by surrounding them with tags '@@' and '##'.\n",
                "shots": course_shots_extr_skills,
            },
            "wlevels": {
                "instruction": course_inst_extr_wlevels,
                "shots": course_shots_extr_wlevels,
            },
            "wreqs": {
                "instruction": course_inst_extr_wlevels,  # courses don't need requirements
                "shots": course_shots_extr_wlevels,  # courses don't need requirements
            },
        },
        "matching": {
            "instruction": "You are looking for an online course. You are given a sentence from a course description in German, and a skill extracted from this sentence. Choose from the list of options the one that best match the skill in the context. Answer with the associated letter.\n",
            "shots": course_shots_match,
        },
    },
    "cv": {
        "system": "You are an expert human resource manager. You need to analyse skills in a CV.",
        "extraction": {
            "skills": {
                "instruction": "Extract candidates skills in German from the following German sentence, taken from a CV.\n",
                "shots": cv_shots_extr_skills,
            },
            "wlevels": {
                "instruction": "Extract all skills and competencies from the CV (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level (beginner, intermediate, expert, or unknown). Return the output as only a json file with the skill as key and mastery level as value.\n",
                "shots": cv_shots_extr_wlevels,
            },
        },
        "matching": {
            "instruction": "You are an expert human resource manager. From a given German sentence from a CV, and a skill extracted from this sentence, choose from the options one or several items that best match the skill in the context. Answer with the associated letter(s).\n",
            "shots": cv_shots_match,
        },
    },
}
