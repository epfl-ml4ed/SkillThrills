############ JOBS ############

job_sys = "You are a human resource expert tasked with analyzing skills in a German job posting."

job_inst_extr_wlevels = (
    "You are given sentences from a job description in German. Extract the hard and soft skills (as exact substrings of the sentence) that are required from the candidate applying for the job. "
    "Infer the corresponding skill mastery level ('beginner', 'intermediate', 'expert', or 'unknown') of each skill based on the context. "
    "Return the output as only a json file extracted substrings (as keys) and their inferred mastery levels (as values).\n"
)

en_job_shots_extr_wlevels = [
    'Sentence: We are looking for a team leader with strong communication skills to foster collaboration and information sharing within the team. \nAnswer: {"communication skills": "expert"}',
    'Sentence: You need a strong background in Python and C++ programming languages. Familiarity with Rust and Go are a plus. \nAnswer: {"Python": "expert", "C++": "expert", "Rust": "beginner", "Go": "beginner"}',
    "Sentence: You will be responsible for the development of the company’s new product. \nAnswer: {}",
    'Sentence: The ability to work collaboratively across disciplines is a key criterion for this position. \nAnswer: {"ability to collaborate across disciplines": "unknown"}',
    'Sentence: You have very good knowledge of digital circuit technology and control loops. \nAnswer: {"digital circuit technology": "expert", "control loops": "expert"}',
    'Sentence: In addition to knowledge of modern, agile software development and its concepts, you also have a basic knowledge of test automation. \nAnswer: {"agile software development": "intermediate", "test automation": "beginner"}',
]

job_shots_extr_wlevels = [
    'Sentence: Wir suchen einen Teamleiter mit ausgeprägten Kommunikationsfähigkeiten, der die Zusammenarbeit und den Informationsaustausch im Team fördert. \nAnswer: {"ommunikationsfähigkeiten": "expert"}',
    'Sentence: Sie benötigen gute Kenntnisse in den Programmiersprachen Python und C++. Vertrautheit mit Rust und Go sind ein Plus. \nAnswer: {"Python": "expert", "C++": "expert", "Rust": "beginner", "Go": "beginner"}',
    "Sentence: Sie werden für die Entwicklung des neuen Produkts des Unternehmens verantwortlich sein. \nAnswer: {}",
    'Sentence: Die Fähigkeit, interdisziplinär zusammenzuarbeiten, ist ein wichtiges Kriterium für diese Position. \nAnswer: {"Fähigkeit, interdisziplinär zusammenzuarbeiten": "unknown"}',
    'Sentence: Sie haben sehr gute Kenntnisse in der digitalen Schaltungstechnik und in Regelkreisen. \nAnswer: {"digitalen Schaltungstechnik": "expert", "Regelkreisen": "expert"}',
    'Sentence: Neben Kenntnissen über moderne, agile Softwareentwicklung und deren Konzepte verfügen Sie auch über Grundkenntnisse in der Testautomatisierung. \nAnswer: {"agile Softwareentwicklung": "intermediate", "Testautomatisierung": "beginner"}',
]

job_inst_extr_wreqs = (
    "You are given sentences from a job description in German. Extract the hard and soft skills (as exact substrings of the sentence) that are required from the candidate applying for the job. "
    "Infer the corresponding skill mastery level ('beginner', 'intermediate', 'expert', or 'unknown') of each skill as well as if the skill or task is 'required' for the job, 'optional' (nice-to-have), or 'unknown' based on the context. "
    "Return the output as only a json file extracted substrings (as keys) and a list of ['mastery level', 'requirement status'] as values.\n"
)

job_shots_extr_wreqs = [
    'Sentence: Wir suchen einen Teamleiter mit ausgeprägten Kommunikationsfähigkeiten, der die Zusammenarbeit und den Informationsaustausch im Team fördert. \nAnswer: {"ommunikationsfähigkeiten": ["expert", "required"]}',
    'Sentence: Sie benötigen gute Kenntnisse in den Programmiersprachen Python und C++. Vertrautheit mit Rust und Go sind ein Plus. \nAnswer: {"Python": ["expert", "required"], "C++": ["expert", "required"], "Rust": ["beginner", "optional"], "Go": ["beginner", "optional"]}',
    "Sentence: Sie werden für die Entwicklung des neuen Produkts des Unternehmens verantwortlich sein. \nAnswer: {}",
    'Sentence: Die Fähigkeit, interdisziplinär zusammenzuarbeiten, ist ein wichtiges Kriterium für diese Position. \nAnswer: {"Fähigkeit, interdisziplinär zusammenzuarbeiten": ["unknown", "required"]}',
    'Sentence: Sie haben sehr gute Kenntnisse in der digitalen Schaltungstechnik und in Regelkreisen. \nAnswer: {"digitalen Schaltungstechnik": ["expert", "unknown"], "Regelkreisen": ["expert", "unknown"]}',
    'Sentence: Neben Kenntnissen über moderne, agile Softwareentwicklung und deren Konzepte verfügen Sie auch über Grundkenntnisse in der Testautomatisierung. \nAnswer: {"agile Softwareentwicklung": ["intermediate", "unknown"], "Testautomatisierung": ["beginner", "unknown"]}',
]

# job_inst_match = ("Given sentences from a job description and a skill extracted from these sentences along with several skills (as 'skill name: skill definition' pairs) as options, your task is to identify the best matching option."
#                      "If the best option matches exactly to the extracted skill in context of the sentence, output the associated letter (e.g., 'a'); otherwise (if there is no match or you are not sure), output nothing.\n")

job_inst_match = (
    "Given sentences from a German job description and a skill extracted from these sentences along with several skills (as 'skill name: skill definition' pairs) as options, identify the best matched option."
    "Only if the extracted skill in context of the sentence is a perfect match to the best option, output the associated letter (e.g., 'b'); otherwise, output nothing. If you have any doubt, output nothing.\n"
)
job_shots_match = [
    """
    Sentence: Wir benötigen Kenntnisse über moderne, agile Softwareentwicklung und deren Konzepte.
    \nSkill: agile Softwareentwicklung
    \nOptions: \na: "Integrierte Entwicklungsumgebung Software: die Sammlung von Software-Entwicklungswerkzeugen zum Schreiben von Programmen, wie Compiler, Debugger, Code-Editor, Code-Highlights, verpackt in einer einheitlichen Benutzeroberfläche, wie Visual Studio oder Eclipse."
    \nb: "Agile Entwicklung: Das agile Entwicklungsmodell ist eine Methodik zur Gestaltung von Softwaresystemen und Anwendungen."
    \nAnswer: b.
    """,
    """
    Sentence: Die Rolle erfordert Kompetenz in statistischer Analyse und Datenvisualisierungstechniken.
    \nSkill: Techniken der Datenvisualisierung
    \nOptions: \na: "Datenanalyse: der Prozess des Inspektierens, Reinigens, Transformierens mit dem Ziel, nützliche Erkenntnisse zu gewinnen"
    \nb: "Datenmanagement: effizientes Handhaben und Organisieren großer Datensätze."
    \nAnswer: .
    """,
]

########### COURSES ############

course_sys = "You are a student looking for an online course in German."

course_inst_extr_wlevels = (
    "You are given sentences from a German course description. Extract hard and soft skills (as substrings of the sentence) that the course will either teach you or are mandatory prerequisites for the course. "
    # "You are given sentences from a German course description. Extract hard and soft skills (as substrings of the sentence) that the course will explicitly teach you."
    # "You are given sentences from a German course description. Extract hard and soft skills (as substrings of the sentence) that are mandatory prerequisites for the course. "
    "Infer the corresponding skill mastery level ('beginner', 'intermediate', 'expert', or 'unknown') of each skill based on the context. "
    "Return the output as only a json file extracted substrings (as keys) and their inferred mastery levels (as values).\n"
)
en_course_shots_extr_wlevels = [
    "Sentence: This course is from scratch so no prerequisites is required , but if a student has some basic knowledge of HTML , CSS and JS than it will be beneficial. Internet Access required\nAnswer: {}",
    'Sentence: Strategic IT GRC: IT governance, IT risk and compliance management as a link between business and IT.\nAnswer: {"IT governance, IT risk and compliance management": "unknown"}',
    'Sentence: We will teach you everything you need to know to level up your AWS skills. Not for certification exams and those with no experience \nAnswer: {"AWS skills": "Intermediate"}',
    'Sentence: What do you think about Spring Framework?. Why is Spring Popular? Can you give a big picture of the Spring Framework? \nAnswer: {"Spring Framework": "beginner"}',
    'Sentence: Code in C#. Make original game art in Blender. Texture 3D models in Photoshop. \nAnswer: {"C#": "unknown", "Blender": "beginner", "Photoshop": "beginner"}',
    """Sentence: Make web apps in the ELM language and 2D games!. The complete beginner's guide for web programmers and game developer \nAnswer: {"web apps in the ELM language": "beginner", "2D games": "beginner"}""",
]

course_shots_extr_wlevels = [
    "Sentence: Dieser Kurs ist von Grund auf, daher sind keine Vorkenntnisse erforderlich, aber wenn ein Student grundlegende Kenntnisse in HTML, CSS und JS hat, dann wird es vorteilhaft sein. Internetzugang erforderlich\nAnswer: {}",
    'Sentence: Strategisches IT-GRC: IT-Governance, IT-Risiko- und Compliance-Management als Bindeglied zwischen Business und IT.\nAnswer: {"IT-Governance, IT-Risiko- und Compliance-Management": "unknown"}',
    'Sentence: Wir bringen Ihnen alles bei, was Sie wissen müssen, um Ihre AWS-Kenntnisse zu verbessern. Nicht für Zertifizierungsprüfungen und diejenigen ohne Erfahrung\nAnswer: {"AWS-Kenntnisse": "intermediate"}',
    'Sentence: Was denken Sie über das Spring Framework? Warum ist Spring beliebt? Können Sie ein Gesamtbild des Spring Frameworks geben? \nAnswer: {"Spring Framework": "beginner"}',
    'Sentence: Programmieren in C#. Erstellen Sie originale Spielgrafiken in Blender. Texturieren Sie 3D-Modelle in Photoshop.\nAnswer: {"C#": "unknown", "Blender": "beginner", "Photoshop": "beginner"}',
    """Sentence: Erstellen Sie Web-Apps in der ELM-Sprache und 2D-Spiele! Der komplette Leitfaden für beginner für Webprogrammierer und Spieleentwickler\nAnswer: {"Web-Apps in der ELM-Sprache": "beginner", "2D-Spiele": "beginner"}""",
]


# course_inst_match = ("Given sentences from a course description and a skill extracted from these sentences along with several skills (as 'skill name: skill definition' pairs) as options, your task is to identify the matching option(s)."
#                         "If the extracted skill in context of the sentence exactly matches one or more of the options, output the associated letter(s) (e.g., 'a', 'b', 'c'); otherwise, output nothing.\n")
course_inst_match = (
    "Given sentences from a German course description and a skill extracted from these sentences along with several skills (as 'skill name: skill definition' pairs) as options, identify the best matched option."
    "Only if the extracted skill in context of the sentence is a perfect match to the best option, output the associated letter (e.g., 'b'); otherwise, output nothing. If you have any doubt, output nothing.\n"
)
course_shots_match = [
    """
    Sentence: "Sie werden lernen, erfolgreich ein Team in einer Hochdruckumgebung zu führen und zu managen."
    \nSkill: "Team führen und managen"
    \nOptions: \na: "Führung: Die Fähigkeit, andere zu führen und zu motivieren, um Ziele zu erreichen."
    \nb: "Sozialkompetenz: Fähigkeiten, Ziele in Zusammenarbeit mit anderen Menschen zu erreichen."
    \nc: "Kommunikation: Informationen und Ideen auf verständliche Weise an andere vermitteln."
    \nd: "Projektmanagement: Die Fähigkeit, Ressourcen zu planen, zu organisieren und zu verwalten, um die erfolgreiche Umsetzung spezifischer Projektziele und -ziele zu erreichen."
    \nAnswer: a.
    """,
    """
    Sentence: "Sie werden lernen, Spiele mit der GameMaker Studio 2 Spiele-Engine zu entwickeln."
    \nSkill: "Spiele mit dem GameMaker entwickeln"
    \nOptions: \na: Agile Entwicklung: Das agile Entwicklungsmodell ist eine Methodik zur Gestaltung von Softwaresystemen und Anwendungen."
    \nb: "Spieltestsoftware erstellen: Software entwickeln, um Online- und Land-Glücksspiele, Wetten und Lotteriespiele zu testen und zu bewerten."
    \nAnswer: .
    """,
]

########### CVs ###########
cv_sys = "You are an expert human resource manager in Switzerland tasked with finding the best candidate with the right skills."

cv_inst_extr_wlevels = (
    "You are given a sentence from a CV/resume in German from a candidate. Extract the important skills (as exact substrings of the sentence if possible) that the candidate hasly. "
    "Infer the corresponding skill mastery level ('beginner', 'intermediate', 'expert', or 'unknown') of each skill based on the context. "
    "Return the output as only a json file with the extracted skills (as keys) and their inferred mastery levels (as values).\n"
)

en_cv_shots_extr_wlevels = [
    'Sentence: The purpose of reports is to provide an end-to-end view of the entire sales order transaction, and to track the investment orders in various funds, including the status of orders, order types along with the gross value and net value to be generated from these orders. \nAnswer: {"business analysis": "unknown"}',
    "Sentence: Project 2: Project Name Nestle Client Nestle Globe Company Tech Mahindra Pvt. Ltd. \nAnswer: {}",
    'Sentence: Suggested best visualization components in dashboard to use \nAnswer: {"data visualization": "unknown"}',
    'Sentence: Proficient in Microsoft Office Suite (Word, Excel, PowerPoint, Outlook), SharePoint, and Visio. \nAnswer: {"microsoft office Suite": "expert", "sharePoint": "expert", "visio": "expert"}',
    'Sentence: The different projects required careful management of specific STIG. compliance and hardening for the different configurations. \nAnswer: {"management of specific STIG": "unknown"}',
    'Sentence: Highlights Excellent communication techniques Manufacturing systems integration Multidisciplinary exposure Design instruction creation Project management Complex problem solver Advanced critical thinking SharePoint Microsoft Excel \nAnswer: {"communication techniques": "expert", "systems integration": "unknown", "project management": "unknown", "critical thinking": "expert"}',
    'Sentence: Group and project management experience for over twelve years. Advanced problem solving skills and expertise. \nAnswer: {"Group and project management": "expert", "problem solving skills": "expert"}',
]

cv_shots_extr_wlevels = [
    'Sentence: Der Zweck von Berichten besteht darin, eine ganzheitliche Ansicht der gesamten Verkaufsbestellungstransaktion zu bieten und die Investitionsaufträge in verschiedenen Fonds zu verfolgen, einschließlich des Status der Aufträge, der Auftragsarten sowie des Brutto- und Nettowerts, der aus diesen Aufträgen generiert werden soll. \nAnswer: {"Geschäftsanalyse": "unknown"}',
    "Sentence: Projekt 2: Projektname Nestle Kunde Nestle Globe Unternehmen Tech Mahindra Pvt. Ltd. \nAnswer: {}",
    'Sentence: Schlug die besten Visualisierungskomponenten im Dashboard zur Verwendung vor \nAnswer: {"Datenvisualisierung": "unknown"}',
    'Sentence: Versiert im Umgang mit Microsoft Office Suite (Word, Excel, PowerPoint, Outlook), SharePoint und Visio. \nAnswer: {"Microsoft Office Suite": "expert", "SharePoint": "expert", "Visio": "expert"}',
    'Sentence: Die verschiedenen Projekte erforderten eine sorgfältige Verwaltung spezifischer STIG. Compliance und Absicherung für die verschiedenen Konfigurationen. \nAnswer: {"Verwaltung spezifischer STIG": "unknown"}',
    'Sentence: Highlights Hervorragende Kommunikationstechniken Integration von Fertigungssystemen Multidisziplinäre Exposition Designanweisungserstellung Projektmanagement Komplexer Problemlöser Fortgeschrittenes kritisches Denken SharePoint Microsoft Excel \nAnswer: {"Kommunikationstechniken": "expert", "Systemintegration": "unknown", "Projektmanagement": "unknown", "Kritisches Denken": "expert"}',
    'Sentence: Gruppen- und Projektmanagementerfahrung für über zwölf Jahre. Fortgeschrittene Problemlösungsfähigkeiten und Expertise. \nAnswer: {"Gruppen- und Projektmanagement": "expert", "Problemlösungsfähigkeiten": "expert"}',
]


cv_inst_match = (
    "Given sentences from a CV/resume in German and a skill extracted from these sentences along with several skills (as 'skill name: skill definition' pairs) as options, identify the best matched option."
    "Only if the extracted skill in context of the sentence is a perfect match to the best option, output the associated letter (e.g., 'b'); otherwise, output nothing. If you have any doubt, output nothing.\n"
)

cv_shots_match = [
    """
    Sentence: "Ich habe umfassende Erfahrung in Design, Entwicklung und Implementierung von Unternehmens- und Cloud-basierten Systemen."
    \nSkill: "Design, Entwicklung und Implementierung von Unternehmens- und Cloud-basierten Systemen."
    \nOptions: \na: "Planung der Migration in die Cloud: Auswahl bestehender Arbeitslasten und Prozesse für eine mögliche Migration in die Cloud und Auswahl von Migrationstools. Bestimmung einer neuen Cloud-Architektur für eine bestehende Lösung, Planung einer Strategie zur Migration bestehender Arbeitslasten in die Cloud."
    \nb: "Entwicklung mit Cloud-Diensten: Schreiben von Code, der mit Cloud-Diensten über APIs, SDKs und Cloud-CLI interagiert. Schreiben von Code für serverlose Anwendungen, Übersetzen funktionaler Anforderungen in ein Anwendungsdesign, Implementieren des Anwendungsdesigns in Anwendungscode."
    \nc: "Parrot Security OS: das Betriebssystem Parrot Security ist eine Linux-Distribution, die Penetration Cloud-Tests durchführt, Sicherheitsschwächen analysiert und potenziell unbefugten Zugriff ermöglicht."
    \nAnswer: b
    """,
    """
    Sentence: "Projekt 2: Projektname Nestle Kunde Nestle Globe Unternehmen Tech Mahindra Pvt. Ltd."
    \nSkill: "Tech Mahindra Pvt. Ltd."
    \nOptions: \na: Flottenmanagement: Überblick über die Fahrzeugflotte eines Unternehmens, um zu bestimmen, welche Fahrzeuge für die Bereitstellung von Transportdienstleistungen verfügbar und geeignet sind."
    \nb: Aufbau eines Tote-Boards: Installation und das Tote-Board, das zur Anzeige von Informationen verwendet wird, die für Tote-Wetten bei einer Veranstaltung relevant sind."
    \nc: Sandstrahlmaschinenteile: die verschiedenen Teile einer Art von Sandstrahlmaschinerie, ihre Eigenschaften und Anwendungen, wie ein Trittbrett, Strahldüse, Staubkollektor, Filter, Schleifmittel und andere."
    \nAnswer: .
    """,
]


########### PROMPT TEMPLATES ###########
PROMPT_TEMPLATES = {
    "job": {
        "system": job_sys,
        "extraction": {
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
            "instruction": job_inst_match,
            "shots": job_shots_match,
        },
    },
    "course": {
        "system": course_sys,
        "extraction": {
            "wlevels": {
                "instruction": course_inst_extr_wlevels,
                "shots": course_shots_extr_wlevels,
            },
        },
        "matching": {
            "instruction": course_inst_match,
            "shots": course_shots_match,
        },
    },
    "cv": {
        "system": cv_sys,
        "extraction": {
            "wlevels": {
                "instruction": cv_inst_extr_wlevels,
                "shots": cv_shots_extr_wlevels,
            },
        },
        "matching": {
            "instruction": cv_inst_match,
            "shots": cv_shots_match,
        },
    },
}
