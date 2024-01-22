########### FEW SHOT EXAMPLES ###########
########## EXTRACTION ##########
###### JOBS ######

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

### extracts only skills marked with @@ and ## in English
en_job_shots_extr_skills = [
    "Sentence: We are looking for a team leader with strong communication skills to foster collaboration and information sharing within the team.\nAnswer: We are looking for a team leader with strong @@communication skills## to foster collaboration and information sharing within the team.",
    "Sentence: the ability to work collaboratively across disciplines is a key criterion for this position. \nAnswer: @@ability to collaborate across disciplines## is a key criterion for this position.",
    "Sentence: As a Java Senior Software Engineer with experience, you will be a member of a Scrum team. \nAnswer: As a Java Senior Software Engineer with experience, you will be a member of a Scrum team.",
    "Sentence: In her role as a team leader, she has continuously supported the professional development of her employees. \nAnswer: In her role as a team leader, she has continuously fostered the professional @@development of her employees##.",
    "Sentence: He is a resilient employee who has been able to set proper priorities and organize tasks thoughtfully during periods of heavy workload. \nAnswer: He is a resilient employee who has been able to set @@correct priorities and organize tasks thoughtfully## during periods of high workload.",
    "Sentence: Highly qualified, flexible employees from the insurance and IT industry develop them further. \nAnswer: Highly qualified, flexible employees from the insurance and IT industries continue to develop them.",
    "Sentence: Over the past few years, it has succeeded in continuously developing itself in a rapidly changing environment. \nAnswer: Over the past few years, he has succeeded in @@continuously developing## himself in a rapidly changing environment##.\n",
]

### extracts skills and their corresponding mastery levels as {skill: level} pairs in English
en_job_shots_extr_wlevels = [
    'Sentence: We are looking for a team leader with strong communication skills to foster collaboration and information sharing within the team. \nAnswer: {"communication skills": "expert"}',
    'Sentence: the ability to work collaboratively across disciplines is a key criterion for this position. \nAnswer: {"ability to collaborate across disciplines": "unknown"}',
    'Sentence: In her role as a team leader, she has continuously supported the professional development of her employees. \nAnswer: {"development of her employees": "unknown"}',
    'Sentence: He is a resilient employee who has been able to set proper priorities and organize tasks thoughtfully during periods of heavy workload. \nAnswer: {"correct priorities and organize tasks thoughtfully": "unknown"}',
    'Sentence: You have very good knowledge of digital circuit technology and control circuits. \nAnswer: {"digital circuit technology": "expert", "control loops": "expert"}',
    'Sentence: In addition to good knowledge of modern, agile software development and its concepts, you also have a basic knowledge of test automation. \nAnswer: {"agile software development": "expert", "test automation": "beginner"}',
]

###### COURSES ######
course_shots_extr_wlevels = [
    'Sentence: Digitale Kompetenzen einfacher einschätzen und besser erlernen können.\nAnswer: {"Kompetenzen einfacher einschätzen und besser erlernen": "unknown"}',
    'Sentence: Strategisches IT-GRC: IT-Governance, IT-Risk und Compliance Management als Bindeglied zwischen Business und IT.\nAnswer: {"IT-Governance, IT-Risk und Compliance Management": "unknown", "Bindeglied zwischen Business und IT": "unknown"}',
    'Sentence: Adaptives IT-Demand und -Projektportfolio Management: Arten von ITDPPM, Methoden und Toolunterstützung.\nAnswer: {"IT-Demand und -Projektportfolio Management": "unknown"}',
    "Sentence: Die interkulturelle Dimension spielt in unserer globalisierten Zeit eine wichtige Rolle.\nAnswer: {}",
    'Sentence: Sie erweitern Ihr Handlungsrepertoire als Coach und können Menschen aus unterschiedlichsten Kontexten kultursensibel und lösungsorientiert begleiten.\nAnswer: {"Menschen aus unterschiedlichsten Kontexten kultursensibel und lösungsorientiert begleiten": "unknown"}',
    'Sentence: In unserem praxisnahen Lehrgang erfahren Sie, wie Sie solche Entwicklungsprozesse zielführend anleiten, mit wirkungsvollen Tools fördern und stimmig abschliessen.\nAnswer: {"Entwicklungsprozesse zielführend anleiten, mit wirkungsvollen Tools fördern und stimmig abschliessen": "unknown"}',
]

en_course_shots_extr_wlevels = [
    'Sentence: Assess digital skills more easily and learn them better.\nAnswer: {"Assess skills more easily and learn them better": "unknown"}',
    'Sentence: Strategic IT GRC: IT governance, IT risk and compliance management as a link between business and IT.\nAnswer: {"IT governance, IT risk and compliance management": "unknown", "Link between business and IT": "unknown"}',
    'Sentence: Adaptive IT demand and project portfolio management: types of ITDPPM, methods and tool support.\nAnswer: {"IT demand and project portfolio management": "unknown"}',
    "Sentence: The intercultural dimension plays an important role in our globalized times.\nAnswer: {}",
    'Sentence: You expand your repertoire of actions as a coach and can accompany people from a wide variety of contexts in a culturally sensitive and solution-oriented manner.\nAnswer: {"Accompanying people from a wide variety of contexts in a culturally sensitive and solution-oriented manner": "unknown"}',
    'Sentence: In our practice-oriented course, you will learn how to guide such development processes in a goal-oriented manner, promote them with effective tools and conclude them in a coherent manner.\nAnswer: {"Guiding development processes in a goal-oriented manner, promoting them with effective tools and concluding them in a coherent manner": "unknown"}',
]

###### CVs ######

en_cv_shots_extr_skills = [
    "Sentence: Good logical and analytical skills • Positive attitude towards solving problems and accepting challenges • A team player & leader.\nAnswer: Good @@logical and analytical skills## • @@Positive attitude towards solving problems and accepting challenges## • A @@team player## & @@leader##.",
    "Sentence: Well-versed in analyzing and mitigating risk and finding cost-effective solutions. Excels at boosting performance and productivity by establishing realistic goals and enforcing deadlines.  Versatile IT professional with 37 years of Enterprise design and engineering methodology.\nAnswer: Well-versed in @@analyzing and mitigating risk## and @@finding cost-effective solutions##. Excels at @@boosting performance and productivity## by establishing realistic goals and enforcing deadlines.  Versatile IT professional with 37 years of @@Enterprise design and engineering methodology##.",
    "Sentence: The different projects required careful management of specific STIG. compliance and hardening for the different configurations. \nAnswer: The different projects required careful @@management of specific STIG##. compliance and hardening for the different configurations.",
    "Sentence: Highlights Excellent communication techniques Manufacturing systems integration Multidisciplinary exposure Design instruction creation Project management Complex problem solver Advanced critical thinking SharePoint Microsoft Excel \nAnswer: Highlights Excellent @@communication techniques## Manufacturing @@systems integration## @@Multidisciplinary## exposure Design instruction creation @@Project management## Complex @@problem solver## Advanced @@critical thinking## SharePoint Microsoft Excel",
    "Sentence: Group and project management experience for over twelve years. Advanced problem solving skills and expertise. \nAnswer: @@Group and project management## experience for over twelve years. Advanced @@problem solving skills## and expertise.",
    "Sentence: GPA: 4  0 B.S : Marketing , Iowa State University - City , State Marketing Microsoft Front Page Certified New Horizons - Des Moines, IA. \nAnswer: GPA: 4  0 B.S : Marketing , Iowa State University - City , State Marketing Microsoft Front Page Certified New Horizons - Des Moines, IA.",
]
en_cv_shots_extr_wlevels = [
    'Sentence: Good logical and analytical skills • Positive attitude towards solving problems and accepting challenges • A team player & leader.\nAnswer: {"logical and analytical skills": "intermediate", "Positive attitude towards solving problems and accepting challenges": "unknown", "team player": "unknown", "leader": "unknown"}',
    'Sentence: Well-versed in analyzing and mitigating risk and finding cost-effective solutions. Excels at boosting performance and productivity by establishing realistic goals and enforcing deadlines.  Versatile IT professional with 37 years of Enterprise design and engineering methodology.\nAnswer: {"analyzing and mitigating risk": "expert", "finding cost-effective solutions": "expert", "boosting performance and productivity": "expert", "Enterprise design and engineering methodology": "expert"}',
    'Sentence: The different projects required careful management of specific STIG. compliance and hardening for the different configurations. \nAnswer: {"management of specific STIG": "unknown"}',
    'Sentence: Highlights Excellent communication techniques Manufacturing systems integration Multidisciplinary exposure Design instruction creation Project management Complex problem solver Advanced critical thinking SharePoint Microsoft Excel \nAnswer: {"communication techniques": "expert", "systems integration": "unknown", "Multidisciplinary": "beginner", "Project management": "unknown", "problem solver": "expert", "critical thinking": "expert"}',
    'Sentence: Group and project management experience for over twelve years. Advanced problem solving skills and expertise. \nAnswer: {"Group and project management": "expert", "problem solving skills": "expert"}',
    "Sentence: GPA: 4  0 B.S : Marketing , Iowa State University - City , State Marketing Microsoft Front Page Certified New Horizons - Des Moines, IA. \nAnswer: {}",
]


cv_shots_extr_skills = [
    "Sentence: Gute logische und analytische Fähigkeiten • Positive Einstellung zur Problemlösung und Akzeptanz von Herausforderungen • Ein Teamplayer & Leader.\nAnswer: Gute @@logische und analytische Fähigkeiten## • @@Positive Einstellung zur Problemlösung und Akzeptanz von Herausforderungen## • Ein @@Teamplayer## & @@Leader##.",
    "Sentence: Gut vertraut mit der Analyse und Minderung von Risiken und der Suche nach kostengünstigen Lösungen. Hervorragend in der Steigerung von Leistung und Produktivität durch die Festlegung realistischer Ziele und die Durchsetzung von Fristen. Vielseitiger IT-Profi mit 37 Jahren Enterprise Design und Engineering Methodik.\nAnswer: Gut vertraut mit der @@Analyse und Minderung von Risiken## und der Suche nach kostengünstigen Lösungen##. Hervorragend in der Steigerung von Leistung und Produktivität durch die Festlegung realistischer Ziele und die Durchsetzung von Fristen. Vielseitiger IT-Profi mit 37 Jahren @@Enterprise Design und Engineering Methodik##.",
    "Sentence: Die verschiedenen Projekte erforderten ein sorgfältiges Management von spezifischen STIG. Compliance und Härtung für die verschiedenen Konfigurationen. \nAnswer: Die verschiedenen Projekte erforderten ein sorgfältiges @@Management von spezifischen STIG##. Compliance und Härtung für die verschiedenen Konfigurationen.",
    "Sentence: Highlights Hervorragende Kommunikationstechniken Integration von Fertigungssystemen Multidisziplinäre Exposition Designanweisungserstellung Projektmanagement Komplexer Problemlöser Fortgeschrittenes kritisches Denken SharePoint Microsoft Excel \nAnswer: Highlights Hervorragende @@Kommunikationstechniken## Integration von @@Fertigungssystemen## @@Multidisziplinäre## Exposition Designanweisungserstellung @@Projektmanagement## Komplexer @@Problemlöser## Fortgeschrittenes @@kritisches Denken## SharePoint Microsoft Excel",
    "Sentence: Gruppen- und Projektmanagementerfahrung seit über zwölf Jahren. Fortgeschrittene Problemlösungsfähigkeiten und Expertise. \nAnswer: @@Gruppen- und Projektmanagement## Erfahrung seit über zwölf Jahren. Fortgeschrittene @@Problemlösungsfähigkeiten## und Expertise.",
    "Sentence: GPA: 4  0 B.S : Marketing , Iowa State University - City , State Marketing Microsoft Front Page Certified New Horizons - Des Moines, IA. \nAnswer: GPA: 4  0 B.S : Marketing , Iowa State University - City , State Marketing Microsoft Front Page Certified New Horizons - Des Moines, IA.",
]
cv_shots_extr_wlevels = [
    'Sentence: Gute logische und analytische Fähigkeiten • Positive Einstellung zur Problemlösung und Akzeptanz von Herausforderungen • Ein Teamplayer & Leader.\nAnswer: {"logische und analytische Fähigkeiten": "intermediate", "Positive Einstellung zur Problemlösung und Akzeptanz von Herausforderungen": "unknown", "Teamplayer": "unknown", "Leader": "unknown"}',
    'Sentence: Gut vertraut mit der Analyse und Minderung von Risiken und der Suche nach kostengünstigen Lösungen. Hervorragend in der Steigerung von Leistung und Produktivität durch die Festlegung realistischer Ziele und die Durchsetzung von Fristen. Vielseitiger IT-Profi mit 37 Jahren Enterprise Design und Engineering Methodik.\nAnswer: {"Analyse und Minderung von Risiken": "expert", "Suche nach kostengünstigen Lösungen": "expert", "Steigerung von Leistung und Produktivität": "expert", "Enterprise Design und Engineering Methodik": "expert"}',
    'Sentence: Die verschiedenen Projekte erforderten ein sorgfältiges Management von spezifischen STIG. Compliance und Härtung für die verschiedenen Konfigurationen. \nAnswer: {"Management von spezifischen STIG": "unknown"}',
    'Sentence: Highlights Hervorragende Kommunikationstechniken Integration von Fertigungssystemen Multidisziplinäre Exposition Designanweisungserstellung Projektmanagement Komplexer Problemlöser Fortgeschrittenes kritisches Denken SharePoint Microsoft Excel \nAnswer: {"Kommunikationstechniken": "expert", "Fertigungssystemen": "unknown", "Multidisziplinäre": "beginner", "Projektmanagement": "unknown", "Problemlöser": "expert", "kritisches Denken": "expert"}',
    'Sentence: Gruppen- und Projektmanagementerfahrung seit über zwölf Jahren. Fortgeschrittene Problemlösungsfähigkeiten und Expertise. \nAnswer: {"Gruppen- und Projektmanagement": "expert", "Problemlösungsfähigkeiten": "expert"}',
    "Sentence: GPA: 4  0 B.S : Marketing , Iowa State University - City , State Marketing Microsoft Front Page Certified New Horizons - Des Moines, IA. \nAnswer: {}",
]

########## MATCHING ##########
###### JOBS ######
job_shots_match = [
    'Sentence: Grundlegende Bestimmungen von Urheberrecht und Datenschutz verstehen. \nSkill: Datenschutz. \nOptions: \nA: "Grundsätze des Datenschutzes respektieren" \nB: "Datenschutz verstehen" \nC: "Datenschutz im Luftfahrtbetrieb sicherstellen" \nD: "Datenschutz". \nAnswer: "Datenschutz verstehen", "Datenschutz".\n'
]

en_job_shots_match = [
    'Sentence: Understand basic provisions of copyright and privacy. \nSkill: Data protection. \nOptions: \nA: "Respect privacy principles." \nB: "Understand data protection" \nC: "Ensure data protection in aviation operations" \nD: "Data protection." \nAnswer: b, d.\n',
]

###### COURSES ######

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
    \nAnswer: F
    """
]

en_course_shots_match = [
    """
    Sentence: "You have all the important tools and methods in your backpack to accompany, successfully complete, and evaluate development processes in an intercultural environment."
    \nSkill: "To accompany development processes in an intercultural environment, successfully complete, and evaluate them."
    \nOptions: \nA: "Cognitive Skill: Attention: Switching focus between two or more activities or sources of information (e.g., language, sounds, touches, or other sources)."
    \nB: "Social Skill: Skills to achieve goals in collaboration with other people."
    \nC: "Performance Orientation: The person strives to achieve personal goals and to be competent in their own work."
    \nD: "Communication: The competence to successfully communicate with individuals from different cultural backgrounds by respecting cultural differences, showing understanding and sensitivity, and fostering an open and inclusive communication environment."
    \nE: "Collaboration: Collaboration in interdisciplinary teams: The competence to successfully work with members from different professional groups or disciplines by respecting and integrating different expertise, perspectives, and work methods to achieve common goals."
    \nF: "Communication: The competence to empathetically engage with the other party (usually clients) in coaching and counseling situations, clarifying their concerns to build a trusting relationship and provide supportive advice."
    \nAnswer: D
    """
]

########### TEMP HOLDERS BELOW ###########

job_shots_match = course_shots_match
en_job_shots_match = en_course_shots_match

cv_shots_match = course_shots_match
en_cv_shots_match = en_course_shots_match

course_shots_extr_skills = job_shots_extr_skills
en_course_shots_extr_skills = en_job_shots_extr_skills


job_shots_tl = [
    'Ich spreche sehr gut Java und Deutsch.\n Answer: {"agiler Softwareentwicklung": ["expert", "unknown"], "Testautomatisierung": ["beginner", "unknown"]}',
]

########### INSTRUCTIONS ###########
job_inst_extr_wlevels = "You are given a sentence from a job description in German. Extract all skills, competencies, and tasks that are required from the candidate applying for the job (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level (beginner, intermediate, expert, or unknown). Return the output as only a json file with the skill as key and mastery level as value.\n"
job_inst_extr_wreqs = 'You are given a sentence from a job description in German. Extract all skills and competencies that are required from the candidate applying for the job (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level as well as if the skill or task is mandatory for the job or optional (nice-to-have). Return the output as a json file with the extracted skill as key and a list of ("mastery level", "requirement status") as the value. Mastery level should be either "expert", "intermediate", "beginner", or "unknown" and requirement status should be either "required", "optional", or "unknown" based on the context.\n'
job_inst_extr_skills = (
    "You are an expert human resource manager. You are given an extract from a job description in German. Highlight all the skills, competencies and tasks that are required from the candidate applying for the job, by surrounding them with tags '@@' and '##'. Make sure you don't highlight job titles, nor elements related to the company and not to the job itself.\n",
)
course_inst_extr_wlevels = "You are given a sentence from a job description in German. Extract all skills and competencies that are mentioned in the course description sentence (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level (beginner, intermediate, expert, or unknown). Return the output as only a json file with the skill as key and mastery level as value.\n"
# course_inst_extr_wreqs = 'You are given a sentence from a course description in German. Extract all skills and competencies that are mentioned in the course description sentence (make sure that the extracted skills are substrings of the sentence) and infer the corresponding mastery skill level as well as if the skill or task is mandatory for the course or optional (nice-to-have). Return the output as a json file with the extracted skill as key and a list of ("mastery level", "requirement status") as the value. Mastery level should be either "expert", "intermediate", "beginner", or "unknown" and requirement status should be either "required", "optional", or "unknown" based on the context.\n'


en_job_inst_extr_skills = "You are an expert human resource manager. You are given an extract from a job description. Highlight all the skills, competencies and tasks that are required from the candidate applying for the job, by surrounding them with tags '@@' and '##'. Make sure you don't highlight job titles, nor elements related to the company and not to the job itself.\n"
en_job_inst_extr_wlevels = "You are given a sentence from a job description. Extract all skills and competencies that are required from the candidate applying for the job (make sure that the extracted skills are exact substrings of the sentence) and infer the corresponding mastery skill level (beginner, intermediate, expert, or unknown). Return the output as only a json file with the skill as key and mastery level as value.\n"

en_course_inst_extr_skills = "You are given a sentence from a course description. Highlight all the skills and competencies that are learned when following the course described in the sentence, by surrounding them with tags '@@' and '##'.\n"
en_course_inst_extr_wlevels = "You are given a sentence from a course description. Extract all skills and competencies that are mentioned in the course description sentence (make sure that the extracted skills are exact substrings of the sentence) and infer the corresponding mastery skill level (beginner, intermediate, expert, or unknown). Return the output as only a json file with the skill as key and mastery level as value.\n"

en_cv_inst_extr_skills = "You are an expert human resource manager. You are given text from a CV or resume. Highlight all the skills, competencies and tasks that the candidate has, by surrounding them with tags '@@' and '##'. Make sure you don't highlight job titles.\n"
en_cv_inst_extr_wlevels = "You are an expert human resource manager. You are given a sentence from a CV or resume. Extract all skills and competencies that are mentioned in the CV sentence (make sure that the extracted skills are exact substrings of the sentence) and infer the corresponding mastery skill level (beginner, intermediate, expert, or unknown). Return the output as only a json file with the skill as key and mastery level as value.\n"

########### PROMPT TEMPLATES ###########

PROMPT_TEMPLATES = {
    "en_job": {
        "system": "You are an expert human resource manager. You need to analyse skills in a job posting.",
        "extraction": {
            "skills": {
                "instruction": en_job_inst_extr_skills,
                "shots": en_job_shots_extr_skills,
            },
            "wlevels": {
                "instruction": en_job_inst_extr_wlevels,
                "shots": en_job_shots_extr_wlevels,
            },
        },
        "matching": {
            "instruction": "You are an expert human resource manager. You are given a sentence from a job description, and a skill extracted from this sentence. Choose from the list of options the one that best match the skill in the context. Answer with the associated letter.\n",
            "shots": en_job_shots_match,
        },
    },
    "job": {
        "system": "You are an expert human resource manager. You need to analyse skills in a job posting.",
        "extraction": {
            "skills": {
                "instruction": job_inst_extr_skills,
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
    "en_course": {
        "system": "You are looking for an online course.",
        "extraction": {
            "skills": {
                "instruction": en_course_inst_extr_skills,
                "shots": en_course_shots_extr_skills,
            },
            "wlevels": {
                "instruction": en_course_inst_extr_wlevels,
                "shots": en_course_shots_extr_wlevels,
            },
        },
        "matching": {
            "instruction": "You are looking for an online course. You are given a sentence from a course description, and a skill extracted from this sentence. Choose from the list of options the one that best match the skill in the context. Answer with the associated letter.\n",
            "shots": en_course_shots_match,
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
    "en_cv": {
        "system": "You are an expert human resource manager. You need to analyse skills in a CV.",
        "extraction": {
            "skills": {
                "instruction": en_cv_inst_extr_skills,
                "shots": en_cv_shots_extr_skills,
            },
            "wlevels": {
                "instruction": en_cv_inst_extr_wlevels,
                "shots": en_cv_shots_extr_wlevels,
            },
        },
        "matching": {
            "instruction": "You are an expert human resource manager. From a given sentence from a CV, and a skill extracted from this sentence, choose from the options one or several items that best match the skill in the context. Answer with the associated letter(s).\n",
            "shots": en_cv_shots_match,
        },
    },
}
