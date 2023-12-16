############ JOBS ############

en_job_sys="You are a human resource expert tasked with analyzing skills in a job posting."

en_job_inst_extr_wlevels = ("You are given sentences from a job description. Extract the hard and soft skills (as exact substrings of the sentence) that are required from the candidate applying for the job. " 
                            "Infer the corresponding skill mastery level (beginner, intermediate, expert, or unknown) of each skill based on the context. "
                            "Return the output as only a json file extracted substrings (as keys) and their inferred mastery levels (as values).\n")
en_job_shots_extr_wlevels = [
    'Sentence: We are looking for a team leader with strong communication skills to foster collaboration and information sharing within the team. \nAnswer: {"communication skills": "expert"}',
    'Sentence: You need a strong background in Python and C++ programming languages. Familiarity with Rust and Go are a plus. \nAnswer: {"Python": "expert", "C++": "expert", "Rust": "beginner", "Go": "beginner"}',
    'Sentence: You will be responsible for the development of the company’s new product. \nAnswer: {}',
    'Sentence: The ability to work collaboratively across disciplines is a key criterion for this position. \nAnswer: {"ability to collaborate across disciplines": "unknown"}',
    'Sentence: You have very good knowledge of digital circuit technology and control circuits. \nAnswer: {"digital circuit technology": "expert", "control loops": "expert"}',
    'Sentence: In addition to knowledge of modern, agile software development and its concepts, you also have a basic knowledge of test automation. \nAnswer: {"agile software development": "intermediate", "test automation": "beginner"}',
]
# en_job_inst_match = ("Given sentences from a job description and a skill extracted from these sentences along with several skills (as 'skill name: skill definition' pairs) as options, your task is to identify the best matching option."
#                      "If the best option matches exactly to the extracted skill in context of the sentence, output the associated letter (e.g., 'a'); otherwise (if there is no match or you are not sure), output nothing.\n")

en_job_inst_match = ("Given sentences from a job description and a skill extracted from these sentences along with several skills (as 'skill name: skill definition' pairs) as options, identify the best matched option."
                    "Only if the extracted skill in context of the sentence is a perfect match to the best option, output the associated letter (e.g., 'b'); otherwise, output nothing. If you have any doubt, output nothing.\n")
en_job_shots_match = [
    """
    Sentence: We need nowledge of modern, agile software development and its concepts.
    \nSkill: agile software development
    \nOptions: \na: "integrated development environment software : the suite of software development tools for writing programs, such as compiler, debugger, code editor, code highlights, packaged in a unified user interface, such as visual studio or eclipse."
    \nb: "Agile development: the agile development model is a methodology to design software systems and applications."
    \nAnswer: b.
    """,
    """
    Sentence: The role requires proficiency in statistical analysis and data visualization techniques.
    \nSkill: data visualization techniques
    \nOptions: \na: "Data analysis: the process of inspecting, cleansing, transforming with the goal of discovering useful insights"
    \nb: "Data management: handling and organizing large datasets efficiently."
    \nAnswer: .
    """
]

########### COURSES ############

en_course_sys = "You are a student looking for an online course."

en_course_inst_extr_wlevels = ("You are given sentences from a course description. Extract hard and soft skills (as substrings of the sentence) that you are certain that the course will either teach you or are mandatory prerequisites the course. "
                               "Infer the corresponding skill mastery level (beginner, intermediate, expert, or unknown) of each skill based on the context. "
                               "Return the output as only a json file extracted substrings (as keys) and their inferred mastery levels (as values).\n")
en_course_shots_extr_wlevels = [
    'Sentence: This course is from scratch so no prerequisites is required , but if a student has some basic knowledge of HTML , CSS and JS than it will be beneficial. Internet Access required\nAnswer: {}',
    'Sentence: Strategic IT GRC: IT governance, IT risk and compliance management as a link between business and IT.\nAnswer: {"IT governance, IT risk and compliance management": "unknown"}',
    'Sentence: We will teach you everything you need to know to level up your AWS skills. Not for certification exams and those with no experience \nAnswer: {"AWS skills": "Intermediate"}',
    'Sentence: What do you think about Spring Framework?. Why is Spring Popular? Can you give a big picture of the Spring Framework? \nAnswer: {"Spring Framework": "beginner"}',
    'Sentence: Code in C#. Make original game art in Blender. Texture 3D models in Photoshop. \nAnswer: {"C#": "unknown", "Blender": "beginner", "Photoshop": "beginner"}',
    """Sentence: Make web apps in the ELM language and 2D games!. The complete beginner's guide for web programmers and game developer \nAnswer: {"web apps in the ELM language": "beginner", "2D games": "beginner"}""",
]

# en_course_inst_match = ("Given sentences from a course description and a skill extracted from these sentences along with several skills (as 'skill name: skill definition' pairs) as options, your task is to identify the matching option(s)."
#                         "If the extracted skill in context of the sentence exactly matches one or more of the options, output the associated letter(s) (e.g., 'a', 'b', 'c'); otherwise, output nothing.\n")
en_course_inst_match = ("Given sentences from a course description and a skill extracted from these sentences along with several skills (as 'skill name: skill definition' pairs) as options, identify the best matched option."
                    "Only if the extracted skill in context of the sentence is a perfect match to the best option, output the associated letter (e.g., 'b'); otherwise, output nothing. If you have any doubt, output nothing.\n")
en_course_shots_match = [
    """
    Sentence: "You will learn to successfully lead and manage team in a high-pressure environment."
    \nSkill: "lead and manage team"
    \nOptions: \na: "Leadership: To ability to lead and motivate others to achieve goals."
    \nb: "Social Skill: Skills to achieve goals in collaboration with other people."
    \nc: "Communication: Conveying information and ideas to others in an understandable manner."
    \nd: "Project management: The ability to plan, organize, and manage resources to bring about the successful completion of specific project goals and objectives."
    \nAnswer: a.
    """,
    """
    Sentence: "You will learn to develop games using the GameMaker Studio 2 game engine."
    \nSkill: "develop games using the gamemaker"
    \nOptions: \na: Agile development : the agile development model is a methodology to design software systems and applications."
    \nb: "create game testing software : develop software to test and evaluate online and land-based gambling, betting and lottery games."
    \nAnswer: .
    """,
]

########### CVs ###########
en_cv_sys = "You are an expert human resource manager tasked with finding the best candidate with the right skills."

en_cv_inst_extr_wlevels = ("You are given a sentence from a CV/resume from a candidate. Extract the important skills (as exact substrings of the sentence if possible) that the candidate has. "
                            "Infer the corresponding skill mastery level (beginner, intermediate, expert, or unknown) of each skill based on the context. "
                            "Return the output as only a json file with the extracted skills (as keys) and their inferred mastery levels (as values).\n")

en_cv_shots_extr_wlevels = [
    'Sentence: The purpose of reports is to provide an end-to-end view of the entire sales order transaction, and to track the investment orders in various funds, including the status of orders, order types along with the gross value and net value to be generated from these orders. \nAnswer: {"business analysis": "unknown"}',
    'Sentence: Project 2: Project Name Nestle Client Nestle Globe Company Tech Mahindra Pvt. Ltd. \nAnswer: {}',
    'Sentence: Suggested best visualization components in dashboard to use \nAnswer: {"data visualization": "unknown"}',
    'Sentence: Proficient in Microsoft Office Suite (Word, Excel, PowerPoint, Outlook), SharePoint, and Visio. \nAnswer: {"microsoft office Suite": "expert", "sharePoint": "expert", "visio": "expert"}',
    'Sentence: The different projects required careful management of specific STIG. compliance and hardening for the different configurations. \nAnswer: {"management of specific STIG": "unknown"}',
    'Sentence: Highlights Excellent communication techniques Manufacturing systems integration Multidisciplinary exposure Design instruction creation Project management Complex problem solver Advanced critical thinking SharePoint Microsoft Excel \nAnswer: {"communication techniques": "expert", "systems integration": "unknown", "project management": "unknown", "critical thinking": "expert"}',
    'Sentence: Group and project management experience for over twelve years. Advanced problem solving skills and expertise. \nAnswer: {"Group and project management": "expert", "problem solving skills": "expert"}',
]

en_cv_inst_match = ("Given sentences from a CV/resume and a skill extracted from these sentences along with several skills (as 'skill name: skill definition' pairs) as options, identify the best matched option."
                    "Only if the extracted skill in context of the sentence is a perfect match to the best option, output the associated letter (e.g., 'b'); otherwise, output nothing. If you have any doubt, output nothing.\n")

en_cv_shots_match = [
    """
    Sentence: "I have a strong background in the design, development, and implementation of enterprise and cloud-based systems."
    \nSkill: "design, development, and implementation of enterprise and cloud-based systems."
    \nOptions: \na: "plan migration to cloud : select existing workloads and processes for potential migration to the cloud and choose migration tools. determine a new cloud architecture for an existing solution, plan a strategy for migrating existing workloads to the cloud."
    \nb: "develop with cloud services : write code that interacts with cloud services by using apis, sdks, and cloud cli. write code for serverless applications, translate functional requirements into application design, implement application design into application code."
    \nc: "Parrot Security OS: the operating system parrot security is a linux distribution which performs penetration cloud testing, analysing security weaknesses for potentially unauthorised access."
    \nAnswer: b
    """,
    """
    Sentence: "Project 2: Project Name Nestle Client Nestle Globe Company Tech Mahindra Pvt. Ltd""
    \nSkill: "tech mahindra pvt. ltd"
    \nOptions: \na: manage vehicle fleet : possess an overview of the vehicle fleet of a company in order to determine what vehicles are available and suitable for the provision of transport services.
    \nb: set up tote board : install and the tote board used to display information relevant to tote betting at an event.
    \nc: sand blasting machine parts : the various parts of a type of sand blasting machinery, their qualities and applications, such as a treadle, blast nozzle, dust collecter, filter, abrasive materials and others.
    \nAnswer: .
    """
]
    

########### PROMPT TEMPLATES ###########
PROMPT_TEMPLATES = {
    "en_job": {
        "system": en_job_sys,
        "extraction": {
            "wlevels": {
                "instruction": en_job_inst_extr_wlevels,
                "shots": en_job_shots_extr_wlevels,
            },
        },
        "matching": {
            "instruction": en_job_inst_match,
            "shots": en_job_shots_match,
        },
    },

    "en_course": {
        "system": en_course_sys,
        "extraction": {
            "wlevels": {
                "instruction": en_course_inst_extr_wlevels,
                "shots": en_course_shots_extr_wlevels,
            },
        },
        "matching": {
            "instruction": en_course_inst_match,
            "shots": en_course_shots_match,
        },
    },
    "en_cv": {
        "system": en_cv_sys,
        "extraction": {
            "wlevels": {
                "instruction": en_cv_inst_extr_wlevels,
                "shots": en_cv_shots_extr_wlevels,
            },
        },
        "matching": {
            "instruction": en_cv_inst_match,
            "shots": en_cv_shots_match,
        },
    },
}
