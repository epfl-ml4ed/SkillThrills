PROMPT_TEMPLATES = {
    "all": {
        "system": "You are an expert human resource manager. You need to analyse skills required in job offers.",
        "instruction": {
            "ner": "You are given a sentence from a job description. Replicate the sentence and highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically.",
            "extract": "You are given a sentence from a job description. Extract all the skills and competencies that are required from the candidate as a list, with one skill per line. If no skill is found in the sentence, return \"None\".",
        },
    },
    "gnehm": {
        "system": "You are an expert human resource manager from Germany. You need to analyse skills required in German job offers.",
        "instruction": {
            "ner": "You are given an extract from a job description in German. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.\n",
            "extract": "You are given a sentence from a job description in German. Extract all the skills and competencies that are required from the candidate as list.\n",
        },
    },
    "skillspan": {
        "system": "You are an expert human resource manager. You need to analyse skills required in job offers.",
        "instruction": {
            "ner": "You are given a sentence from a job description. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.\n",
            "extract": "You are given a sentence from a job description. Extract all the skills and competencies that are required from the candidate as list.\n",
        },
    },
}
